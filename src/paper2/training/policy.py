"""Validated Hugging Face/PEFT policy for the *artificial* eight-action diagnostic.

This is not an unrestricted text decoder.  Each prompt produces one forward pass,
whose final-position logits are normalised over ``0..5,?,!`` by the training loop.
Full-vocabulary logits at earlier prompt positions are never requested here.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import re
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn

ACTIONS = ("0", "1", "2", "3", "4", "5", "?", "!")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class PolicyValidationError(ValueError):
    """The model interface does not implement the declared diagnostic."""


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _logits_option(model: nn.Module) -> str:
    """Inspect the actual causal LM, not PEFT's generic ``**kwargs`` wrapper."""
    actual = model.get_base_model() if hasattr(model, "get_base_model") else model
    parameters = inspect.signature(actual.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return name
    raise PolicyValidationError(
        "This causal LM has no explicit last-position-only head option; "
        "do not allocate full prompt logits and slice them afterward."
    )


def _disable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    # Qwen implements attention dropout functionally, outside nn.Dropout.
    for module in model.modules():
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    if hasattr(model, "config"):
        for key in ("attention_dropout", "hidden_dropout_prob", "hidden_dropout"):
            if hasattr(model.config, key):
                setattr(model.config, key, 0.0)


class HFActionPolicy(nn.Module):
    """Causal LM plus a checked single-token verbalizer at its chat boundary.

    ``metadata`` is safe to serialize: it contains identities and package versions,
    never process environment values or authentication material.  Checkpoints use
    named trainable tensors, while the base is reconstructed from its pinned SHA.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        max_prompt_tokens: int = 512,
        identity: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if max_prompt_tokens < 1:
            raise PolicyValidationError("max_prompt_tokens must be positive")
        if not getattr(tokenizer, "chat_template", None):
            raise PolicyValidationError("A recorded chat template is required")
        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self._head_option = _logits_option(model)
        self._validated_prompts: dict[str, tuple[list[int], str]] = {}
        self._action_ids: tuple[int, ...] | None = None
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise PolicyValidationError("Tokenizer has neither PAD nor EOS")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        _disable_dropout(self.model)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        self.metadata: dict[str, Any] = {
            "policy_kind": "hf_peft_eight_action",
            "artificial_action_space": list(ACTIONS),
            "last_position_head_option": self._head_option,
            "padding_side": "left",
            "position_ids": "attention_mask.cumsum(-1)-1; padding=0",
            "max_prompt_tokens": self.max_prompt_tokens,
            "chat_template_sha256": hashlib.sha256(
                str(self.tokenizer.chat_template).encode()
            ).hexdigest(),
            "software": {
                name: _version(name)
                for name in ("torch", "transformers", "peft", "tokenizers", "safetensors")
            },
            **dict(identity or {}),
        }
        self._record_trainable_parameters()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _record_trainable_parameters(self) -> None:
        trainable = [(name, p) for name, p in self.named_parameters() if p.requires_grad]
        if not trainable:
            raise PolicyValidationError("Policy has no trainable parameters")
        self.metadata["trainable_names"] = [name for name, _ in trainable]
        self.metadata["trainable_parameter_count"] = sum(p.numel() for _, p in trainable)
        self.metadata["total_parameter_count"] = sum(p.numel() for p in self.parameters())

    def train(self, mode: bool = True) -> "HFActionPolicy":
        super().train(mode)
        _disable_dropout(self.model)
        return self

    def _encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(encoded, list):
            encoded = list(encoded)
        return [int(token) for token in encoded]

    def validate_prompt(self, prompt: str) -> tuple[list[int], str]:
        if not isinstance(prompt, str) or not prompt:
            raise PolicyValidationError("Prompts must be nonempty strings")
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if digest in self._validated_prompts:
            return self._validated_prompts[digest]
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = self._encode(prefix)
        if not prefix_ids or len(prefix_ids) > self.max_prompt_tokens:
            raise PolicyValidationError(
                f"Formatted prompt has {len(prefix_ids)} tokens, cap "
                f"{self.max_prompt_tokens}; facts must not be truncated."
            )
        ids: list[int] = []
        for action in ACTIONS:
            complete = self._encode(prefix + action)
            if complete[:-1] != prefix_ids or len(complete) != len(prefix_ids) + 1:
                raise PolicyValidationError(
                    f"Action {action!r} does not append one token without changing "
                    "the prefix. Use another boundary or sequence verbalizers."
                )
            ids.append(complete[-1])
        if len(set(ids)) != len(ACTIONS):
            raise PolicyValidationError("Action verbalizers have colliding token IDs")
        if self._action_ids is not None and tuple(ids) != self._action_ids:
            raise PolicyValidationError("Action token IDs depend on the prompt")
        self._action_ids = tuple(ids)
        self.metadata["action_token_ids"] = list(ids)
        # Bound the cache: it stores rendered public training prompts only.
        if len(self._validated_prompts) >= 8192:
            self._validated_prompts.clear()
        self._validated_prompts[digest] = (prefix_ids, prefix)
        return prefix_ids, prefix

    def prepare_inputs(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        if not prompts:
            raise PolicyValidationError("A policy batch must contain a prompt")
        rows = [self.validate_prompt(prompt)[0] for prompt in prompts]
        width = max(map(len, rows))
        ids = torch.full(
            (len(rows), width),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, -len(row):] = torch.tensor(row, device=self.device)
            mask[index, -len(row):] = 1
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        return {"input_ids": ids, "attention_mask": mask, "position_ids": positions}

    def _last_position_logits(self, prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.prepare_inputs(prompts)
        outputs = self.model(
            **inputs,
            **{self._head_option: 1},
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        if logits.ndim != 3 or logits.shape[:2] != (len(prompts), 1):
            raise PolicyValidationError(
                f"Expected one-position logits, got {tuple(logits.shape)}"
            )
        return logits[:, 0, :]

    def action_logits(self, prompts: Sequence[str]) -> torch.Tensor:
        logits = self._last_position_logits(prompts)
        action_ids = torch.tensor(self._action_ids, device=logits.device)
        action_logits = logits.index_select(-1, action_ids).float()
        if not torch.isfinite(action_logits).all():
            raise FloatingPointError("Nonfinite action logits")
        return action_logits

    @torch.no_grad()
    def output_interface_diagnostics(
        self, prompts: Sequence[str], gold_actions: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Audit probability discarded by the artificial eight-action decoder.

        This is a first-token readout at the unchanged chat answer boundary,
        not a full generated-response score or a task-competence evaluation.
        Used only during preflight; the training objective is unchanged.
        """
        if gold_actions is not None and (
            len(gold_actions) != len(prompts) or any(a not in ACTIONS[:6] for a in gold_actions)
        ):
            raise PolicyValidationError("Gold actions must be one valid digit per prompt")
        logits = self._last_position_logits(prompts).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite full-vocabulary logits")
        token_ids = torch.tensor(self._action_ids, device=logits.device)
        logp = logits.log_softmax(-1)
        action_logp = logp.index_select(-1, token_ids)
        global_ids = logits.argmax(-1)
        restricted_ids = token_ids[logits.index_select(-1, token_ids).argmax(-1)]
        result = {
            "scope": "preflight_first_token_only_not_unrestricted_task_performance",
            "boundary": "recorded_chat_template_with_generation_prompt",
            "temperature": 1.0,
            "prompt_count": len(prompts),
            "full_vocabulary_action_mass": action_logp.exp().sum(-1).cpu().tolist(),
            "full_vocabulary_valid_digit_mass": action_logp[:, :6].exp().sum(-1).cpu().tolist(),
            "full_vocabulary_argmax_token_ids": global_ids.cpu().tolist(),
            "eight_action_argmax_token_ids": restricted_ids.cpu().tolist(),
            "global_argmax_is_valid_digit": torch.isin(global_ids, token_ids[:6]).cpu().tolist(),
        }
        if gold_actions is not None:
            gold_ids = token_ids[torch.tensor([ACTIONS.index(a) for a in gold_actions], device=logits.device)]
            result["full_vocabulary_first_token_correct"] = (global_ids == gold_ids).cpu().tolist()
            result["eight_action_ranking_correct"] = (restricted_ids == gold_ids).cpu().tolist()
            result["gold_full_vocabulary_probability"] = (
                logp.gather(-1, gold_ids[:, None]).exp().flatten().cpu().tolist()
            )
        return result

    def forward(self, prompts: Sequence[str]) -> torch.Tensor:
        return self.action_logits(prompts)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Adapter-only tensors with exact names, copied onto CPU."""
        return {
            name: value.detach().cpu().clone()
            for name, value in self.named_parameters()
            if value.requires_grad
        }

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if list(state) != list(expected):
            raise PolicyValidationError("Trainable parameter names/order do not match")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
                raise PolicyValidationError(f"Bad checkpoint tensor shape for {name}")
            if not torch.isfinite(value).all():
                raise PolicyValidationError(f"Nonfinite checkpoint tensor for {name}")
        with torch.no_grad():
            for name, value in state.items():
                expected[name].copy_(value.to(expected[name]))


def load_policy(config: Mapping[str, Any]) -> HFActionPolicy:
    """Load a pinned base/tokenizer and a new zero-dropout LoRA adapter.

    Checkpoint restoration is a separate operation.  This loader never interprets
    an absent checkpoint as permission to start the experiment from scratch.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = dict(config.get("model", {}))
    adapter = dict(config.get("adapter", {}))
    model_id = spec.get("id")
    revision = spec.get("revision")
    tokenizer_revision = spec.get("tokenizer_revision")
    if not model_id:
        raise PolicyValidationError("model.id is required")
    for key, value in (("revision", revision), ("tokenizer_revision", tokenizer_revision)):
        if not isinstance(value, str) or not _COMMIT.fullmatch(value):
            raise PolicyValidationError(f"model.{key} must be an exact 40-character SHA")
    precision = str(spec.get("precision", "bf16"))
    device = torch.device(spec.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PolicyValidationError("CUDA requested but no CUDA GPU is available")
    if precision not in ("bf16", "fp32", "float32"):
        raise PolicyValidationError("Only bf16 and fp32 are validated; FP16 needs a scaler")
    if precision == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise PolicyValidationError("BF16 profile requires a BF16-capable CUDA GPU")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    if adapter.get("dropout", 0.0) != 0.0:
        raise PolicyValidationError("The one-update diagnostic requires zero dropout")
    if adapter.get("bias", "none") != "none":
        raise PolicyValidationError("The validated profile uses LoRA bias='none'")
    targets = adapter.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
        raise PolicyValidationError("adapter.target_modules must be a nonempty name list")
    if "lm_head" in targets:
        raise PolicyValidationError("lm_head adaptation is outside the validated profile")
    local_only = bool(spec.get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=tokenizer_revision, trust_remote_code=False,
        local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=False,
        local_files_only=local_only,
        attn_implementation=spec.get("attention_backend", "eager"),
    )
    _logits_option(model)  # Fail before attaching an adapter if this API is absent.
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = set(targets) - available
    if missing:
        raise PolicyValidationError(f"Missing adapter target modules: {sorted(missing)}")
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved is not None and resolved != revision:
        raise PolicyValidationError("Resolved model revision differs from configuration")
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(adapter.get("rank", 16)),
            lora_alpha=int(adapter.get("alpha", 32)),
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
        ),
    ).to(device)
    unexpected = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected:
        raise PolicyValidationError(f"Unexpected unfrozen base parameters: {unexpected}")
    if spec.get("gradient_checkpointing", False):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return HFActionPolicy(
        model, tokenizer, max_prompt_tokens=int(spec.get("max_prompt_tokens", 512)),
        identity={
            "model_id": model_id,
            "model_revision": revision,
            "tokenizer_revision": tokenizer_revision,
            "precision": precision,
            "device": str(device),
            "attention_backend": spec.get("attention_backend", "eager"),
            "gradient_checkpointing": bool(spec.get("gradient_checkpointing", False)),
            "adapter": {
                "rank": int(adapter.get("rank", 16)),
                "alpha": int(adapter.get("alpha", 32)),
                "dropout": 0.0, "bias": "none", "target_modules": targets,
            },
            "license": spec.get("license", "unrecorded"),
        },
    )


def preflight_policy(
    config: Mapping[str, Any], prompts: Sequence[str], gold_actions: Sequence[str] | None = None
) -> dict[str, Any]:
    """Run an actual model-interface smoke, never a claim of task competence.

    Includes constructor/tokenization, padded-versus-unpadded equivalence, one
    backward pass, and ordinary generation.  It does not make an optimizer update.
    """
    if not prompts:
        raise PolicyValidationError("preflight requires representative rendered prompts")
    started = time.perf_counter()
    policy = load_policy(config)
    if policy.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(policy.device)
    policy.eval()
    with torch.no_grad():
        batch = policy.action_logits(prompts)
        singles = torch.cat([policy.action_logits([prompt]) for prompt in prompts], dim=0)
    output_interface = policy.output_interface_diagnostics(prompts, gold_actions)
    max_error = float((batch - singles).abs().max())
    tolerance = 0.08 if next(policy.parameters()).dtype == torch.bfloat16 else 2e-5
    relative_tolerance = 0.005 if next(policy.parameters()).dtype == torch.bfloat16 else 1e-5
    if not torch.allclose(batch, singles, atol=tolerance, rtol=relative_tolerance):
        raise PolicyValidationError(
            f"Padded logits differ from unpadded logits ({max_error}, tolerance {tolerance})"
        )
    probability_error = float((batch.softmax(-1).sum(-1) - 1.0).abs().max())
    policy.train()
    policy.zero_grad(set_to_none=True)
    logits = policy.action_logits(prompts)
    (-logits.log_softmax(-1)[:, 0].mean()).backward()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
        raise PolicyValidationError("Missing/nonfinite adapter gradients")
    grad_norm = math.sqrt(sum(float(p.grad.float().square().sum()) for p in trainable))
    if grad_norm == 0.0:
        raise PolicyValidationError("All adapter gradients are zero in the preflight")
    policy.zero_grad(set_to_none=True)
    policy.eval()
    generation_inputs = policy.prepare_inputs(prompts[:1])
    # Generation computes its own positions, including positions of new tokens.
    generation_inputs.pop("position_ids")
    with torch.no_grad():
        generated = policy.model.generate(
            **generation_inputs,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=policy.tokenizer.pad_token_id,
        )
    generated_count = int(generated.shape[-1] - generation_inputs["input_ids"].shape[-1])
    if generated_count < 1:
        raise PolicyValidationError("Generation produced no continuation token")
    peak = None
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
        peak = int(torch.cuda.max_memory_allocated(policy.device))
    return {
        "status": "passed",
        "scope": "model-interface validation, not a scientific training result",
        "metadata": policy.metadata,
        "output_interface": output_interface,
        "prompt_lengths": [len(policy.validate_prompt(p)[0]) for p in prompts],
        "padded_unpadded_max_abs_error": max_error,
        "padded_unpadded_tolerance": tolerance,
        "padded_unpadded_relative_tolerance": relative_tolerance,
        "probability_sum_max_abs_error": probability_error,
        "adapter_gradient_norm": grad_norm,
        "generation_tokens": generated_count,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": peak,
    }
