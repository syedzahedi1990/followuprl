"""One-update, eight-action group-relative policy gradient.

Each prompt is forwarded once, G actions are drawn from its detached
distribution, and the same graph supplies the current-policy likelihood.
There is exactly one optimizer step after all prompt microbatches. This is
a categorical diagnostic, not unrestricted generation or a claim of exact
equivalence to TRL GRPO.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

ACTIONS = ("0", "1", "2", "3", "4", "5", "?", "!")


def policy_logprobs(logits: Tensor, *, temperature: float = 1.0,
                    exploration_epsilon: float = 0.0) -> tuple[Tensor, Tensor]:
    """Return log p and log q; q is used consistently for sampling and ratios."""
    if logits.ndim != 2 or logits.shape[-1] != len(ACTIONS):
        raise ValueError("Expected [prompts, 8] action logits.")
    if temperature <= 0 or not 0 <= exploration_epsilon < 1:
        raise ValueError("Temperature must be positive and epsilon in [0, 1).")
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    if exploration_epsilon == 0:
        return logp, logp
    logq = torch.logaddexp(logp + math.log1p(-exploration_epsilon),
                          logp.new_tensor(math.log(exploration_epsilon / 8)))
    return logp, logq


def group_advantages(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    if rewards.ndim != 2:
        raise ValueError("Rewards must have shape [prompts, group_size].")
    rewards = rewards.detach().float()
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    std = torch.sqrt((centered ** 2).mean(dim=-1, keepdim=True))
    return torch.where(std > 0, centered / (std + eps), torch.zeros_like(centered)).detach()


def clipped_objective(logq: Tensor, actions: Tensor, old_logprobs: Tensor,
                      advantages: Tensor, *, clip_epsilon: float = 0.2) -> Tensor:
    """Graph-connected per-prompt loss, including all equal-reward groups."""
    if not 0 <= clip_epsilon < 1:
        raise ValueError("clip_epsilon must lie in [0, 1).")
    current = logq.gather(-1, actions.detach())
    ratios = (current - old_logprobs.detach()).exp()
    advantage = advantages.detach()
    return -torch.minimum(ratios * advantage,
                          ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage).mean(-1)


def mixed_group_probability(p: float, group_size: int) -> float:
    if not 0 <= p <= 1 or group_size < 1:
        raise ValueError("Invalid Bernoulli probability or group size.")
    return 1 - p ** group_size - (1 - p) ** group_size


def _norm(tensors: Sequence[Tensor | None]) -> float:
    return math.sqrt(sum(float(t.detach().double().square().sum()) for t in tensors if t is not None))


def optimizer_moment_norms(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        "adam_first_moment_norm": _norm([s.get("exp_avg") for s in optimizer.state.values()]),
        "adam_second_moment_norm": _norm([s.get("exp_avg_sq") for s in optimizer.state.values()]),
        "optimizer_step_min": min((float(s.get("step", 0)) for s in optimizer.state.values()), default=0),
        "optimizer_step_max": max((float(s.get("step", 0)) for s in optimizer.state.values()), default=0),
    }


@dataclass
class StepResult:
    metrics: dict[str, Any]
    records: list[dict[str, Any]]


def training_step(
    policy: nn.Module,
    prompts: Sequence[str],
    reward_function: Callable[[int, str], float],
    optimizer: torch.optim.Optimizer,
    *,
    generator: torch.Generator,
    families: Sequence[str] | None = None,
    group_size: int = 8,
    microbatch_size: int = 1,
    temperature: float = 1.0,
    exploration_epsilon: float = 0.0,
    clip_epsilon: float = 0.2,
    entropy_coefficient: float = 0.0,
    max_grad_norm: float = 1.0,
    diagnose_family_gradients: bool = False,
) -> StepResult:
    """Accumulate complete behaviour batch, then step even when task gradient is zero.

    KL is deliberately absent from this primary beta=0 implementation. A runner
    must reject beta != 0 rather than silently choose a different reference.
    CPU sampling has its own explicitly checkpointed generator.
    """
    if not prompts or group_size < 2 or microbatch_size < 1:
        raise ValueError("Nonempty prompts, group_size>=2 and positive microbatch are required.")
    if entropy_coefficient < 0 or max_grad_norm <= 0:
        raise ValueError("Invalid entropy or clipping setting.")
    if families is not None and len(families) != len(prompts):
        raise ValueError("Family labels must align with prompts.")
    policy.eval()  # Disable dropout without disabling autograd.
    params = [p for p in policy.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Policy has no trainable parameters.")
    previous = [p.detach().clone() for p in params]
    optimizer.zero_grad(set_to_none=False)
    records: list[dict[str, Any]] = []
    all_rewards: list[Tensor] = []
    sums = {"loss": 0.0, "entropy": 0.0, "valid_probability": 0.0,
            "training_invalid_probability": 0.0}
    separate = {}
    # Accumulate component gradients only when required for diagnostics. They
    # are independent detached tensors and never replace the training gradient.
    component_grads: dict[str, list[Tensor]] = {}
    for start in range(0, len(prompts), microbatch_size):
        end = min(start + microbatch_size, len(prompts))
        batch = list(prompts[start:end])
        logits = policy.action_logits(batch)
        logp, logq = policy_logprobs(logits, temperature=temperature,
                                   exploration_epsilon=exploration_epsilon)
        samples_cpu = torch.multinomial(logq.detach().exp().cpu(), group_size,
                                        replacement=True, generator=generator)
        actions = samples_cpu.to(logq.device)
        old = logq.detach().gather(-1, actions).clone()
        rewards = torch.tensor(
            [[reward_function(start + i, ACTIONS[a]) for a in row]
             for i, row in enumerate(samples_cpu.tolist())],
            device=logq.device, dtype=torch.float32)
        if not torch.isfinite(rewards).all() or not torch.all((rewards == 0) | (rewards == 1)):
            raise ValueError("This diagnostic requires finite binary rewards.")
        advantages = group_advantages(rewards)
        per_prompt = clipped_objective(logq, actions, old, advantages,
                                       clip_epsilon=clip_epsilon)
        task_loss = per_prompt.sum() / len(prompts)
        entropy = -(logp.exp() * logp).sum(-1)
        entropy_loss = -entropy_coefficient * entropy.sum() / len(prompts)
        components: dict[str, Tensor] = {}
        if entropy_coefficient:
            components["task"] = task_loss
            components["entropy"] = entropy_loss
        if diagnose_family_gradients and families is not None:
            for label in ("affected", "protected"):
                mask = torch.tensor(
                    [(f == "belief") if label == "affected" else (f != "belief")
                     for f in families[start:end]], device=logq.device)
                components[label] = (per_prompt * mask).sum() / len(prompts)
        for name, component in components.items():
            grads = torch.autograd.grad(component, params, retain_graph=True, allow_unused=True)
            if name not in component_grads:
                component_grads[name] = [torch.zeros_like(p) for p in params]
            for accum, g in zip(component_grads[name], grads):
                if g is not None:
                    accum.add_(g.detach())
        loss = task_loss + entropy_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite finite-action objective.")
        loss.backward()
        all_rewards.append(rewards.detach().cpu())
        sums["loss"] += float(loss.detach())
        sums["entropy"] += float(entropy.detach().sum()) / len(prompts)
        sums["valid_probability"] += float(logp[:, :6].exp().detach().sum()) / len(prompts)
        sums["training_invalid_probability"] += float(logq[:, 6:].exp().detach().sum()) / len(prompts)
        for i, row in enumerate(samples_cpu.tolist()):
            records.append({
                "prompt_index": start + i,
                "actions": row,
                "raw_text": [ACTIONS[a] for a in row],
                "behaviour_logprobs": old[i].detach().cpu().tolist(),
                "training_rewards": rewards[i].detach().cpu().tolist(),
                "advantages": advantages[i].detach().cpu().tolist(),
                "p": logp[i].exp().detach().cpu().tolist(),
                "q": logq[i].exp().detach().cpu().tolist(),
            })
    total_grad_norm = _norm([p.grad for p in params])
    if not math.isfinite(total_grad_norm):
        raise FloatingPointError("Non-finite policy gradient.")
    task_grad_norm = _norm(component_grads["task"]) if "task" in component_grads else total_grad_norm
    for name, gradients in component_grads.items():
        separate[f"{name}_gradient_norm"] = _norm(gradients)
    torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
    # Every parameter must receive a graph-connected zero rather than None:
    # Adam's carried moments can legitimately move it on equal-reward batches.
    optimizer.step()
    update_norm = _norm([p.detach() - before for p, before in zip(params, previous)])
    rewards = torch.cat(all_rewards)
    all_zero = (rewards == 0).all(-1)
    all_one = (rewards == 1).all(-1)
    metrics = {
        **sums, **separate, **optimizer_moment_norms(optimizer),
        "task_gradient_norm": task_grad_norm,
        "total_gradient_norm": total_grad_norm,
        "parameter_update_norm": update_norm,
        "all_zero_fraction": float(all_zero.float().mean()),
        "all_one_fraction": float(all_one.float().mean()),
        "zero_variance_fraction": float((all_zero | all_one).float().mean()),
        "mixed_group_fraction": float((~(all_zero | all_one)).float().mean()),
        "mean_training_reward": float(rewards.mean()),
        "unique_prompts": len(prompts),
        "sampled_actions": len(prompts) * group_size,
        "kl_beta": 0.0,
        "kl_gradient_norm": 0.0,
    }
    return StepResult(metrics, records)
