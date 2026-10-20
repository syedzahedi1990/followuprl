"""Bounded finite-action experiment runner.

The test-only backend exercises engineering contracts on CPU; its measurements
are never presented as language-model research results.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoints import (canonical_hash, digest, load_checkpoint, policy_state,
                          save_checkpoint, verify_checkpoint, write_checkpoint_payload)
from .finite_grpo import ACTIONS, policy_logprobs, training_step
from .toy import TinyPolicy


def is_tiny(config: dict) -> bool:
    return config.get("backend") in {"tiny", "toy"} or config.get("model", {}).get("id") in {
        "tiny-test-policy", "test-only-text-hash-linear-policy"}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_policy(config: dict, seed: int):
    if is_tiny(config):
        return TinyPolicy(seed=seed, features=int(config.get("model", {}).get("hidden_size", 16)))
    if config.get("backend") not in {None, "finite_bandit"}:
        raise ValueError("Only finite_bandit is implemented; unrestricted TRL confirmation is gated.")
    from .policy import load_policy
    return load_policy(config)


def _optimizer(policy, config):
    settings = config.get("training", {})
    if settings.get("schedule", "constant") != "constant" or settings.get("warmup_steps", 0) != 0:
        raise ValueError("The diagnostic uses constant learning rate and zero warmup.")
    if settings.get("optimizer", "adamw").lower() != "adamw":
        raise ValueError("Only the explicit AdamW optimizer is supported.")
    if settings.get("updates_per_rollout_batch", 1) != 1:
        raise ValueError("This diagnostic implements exactly one update per behaviour batch.")
    if float(settings.get("kl_beta", 0)) != 0:
        raise ValueError("Nonzero KL needs a validated common clean reference; primary runner supports beta=0.")
    return torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                             lr=float(settings.get("learning_rate", 1e-5)),
                             weight_decay=float(settings.get("weight_decay", 0)),
                             betas=(0.9, 0.999), eps=1e-8)


def _phase_seed(seed: int, phase: str) -> int:
    return seed + {"clean": 100_000, "vulnerable": 200_000, "control": 200_000,
                   "recovery": 300_000, "clean_reference": 300_000}.get(phase, 400_000)


class PromptSchedule:
    """Private deterministic permutation/cursor, with unique prompts per update."""
    def __init__(self, size: int, seed: int):
        if size < 1:
            raise ValueError("Empty training pool.")
        self.size = size
        self.generator = np.random.default_rng(seed)
        self.order = self.generator.permutation(size).tolist()
        self.cursor = 0
        self.epoch = 0

    def next(self, count: int) -> list[int]:
        if not 1 <= count <= self.size:
            raise ValueError("Unique prompts per update must not exceed pool size.")
        selected = []
        while len(selected) < count:
            if self.cursor == self.size:
                self.order = self.generator.permutation(self.size).tolist()
                self.cursor = 0
                self.epoch += 1
            candidate = self.order[self.cursor]
            self.cursor += 1
            if candidate not in selected:
                selected.append(candidate)
        return selected

    def state(self) -> dict:
        return {"size": self.size, "order": self.order, "cursor": self.cursor, "epoch": self.epoch}

    def restore(self, state: dict) -> None:
        if state["size"] != self.size or sorted(state["order"]) != list(range(self.size)):
            raise ValueError("Dataset schedule differs from saved state.")
        self.order, self.cursor, self.epoch = list(state["order"]), state["cursor"], state["epoch"]
        if not 0 <= self.cursor <= self.size:
            raise ValueError("Corrupt sampler cursor.")


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _code_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(root)): digest(path) for path in root.rglob("*.py")}
    return {"python_source_sha256": canonical_hash(files), "files": files}


def _contract(config: dict, phase: str, seed: int, task_ids: list[str]) -> str:
    return canonical_hash({
        "project": config.get("project"), "protocol": config.get("protocol_version"),
        "backend": config.get("backend"), "model": config.get("model"),
        "adapter": config.get("adapter"), "task": config.get("task"),
        "training": config.get("training"), "seed": seed, "phase": phase,
        "task_ids": task_ids, "code": _code_identity()["python_source_sha256"],
    })


def _device_sync(policy) -> None:
    if any(p.is_cuda for p in policy.parameters()):
        torch.cuda.synchronize()


def _token_ids(policy, prompts: list[str]) -> list[list[int]]:
    if hasattr(policy, "validate_prompt"):
        return [policy.validate_prompt(p)[0] for p in prompts]
    # No pretend tokenizer counts for the synthetic engineering policy.
    return [[] for _ in prompts]


def evaluate_policy(policy, tasks, *, microbatch_size: int = 4,
                    exploration_epsilon: float = 0.0) -> dict:
    from paper2.analysis.aggregate import finite_metrics
    from paper2.environments.rendering import render_prompt
    from paper2.evaluation.independent_oracle import gold_answer
    if not tasks:
        raise ValueError("Evaluation must contain tasks.")
    policy.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(tasks), microbatch_size):
            prompts = [render_prompt(t) for t in tasks[start:start + microbatch_size]]
            logp, _ = policy_logprobs(policy.action_logits(prompts), temperature=1.0)
            probabilities.extend(logp.exp().cpu().tolist())
    metrics = finite_metrics(probabilities, [gold_answer(t) for t in tasks],
                             [t.family for t in tasks])
    if exploration_epsilon:
        q = [[(1 - exploration_epsilon) * p + exploration_epsilon / 8 for p in row]
             for row in probabilities]
        metrics["training_mixture"] = finite_metrics(q, [gold_answer(t) for t in tasks],
                                                    [t.family for t in tasks])
    metrics["decoder"] = "underlying_8_action_softmax_temperature_1"
    metrics["test_only"] = bool(getattr(policy, "metadata", {}).get("test_only", False))
    return metrics


def evaluate_checkpoint(config: dict, checkpoint: str | Path, tasks) -> dict:
    from paper2.environments.rendering import render_prompt
    metadata = verify_checkpoint(checkpoint)
    policy = _build_policy(config, int(metadata["seed"]))
    if tasks and hasattr(policy, "validate_prompt"):
        policy.validate_prompt(render_prompt(tasks[0]))
    load_checkpoint(checkpoint, policy, restore_random_state=False)
    return evaluate_policy(policy, tasks,
                           microbatch_size=int(config.get("training", {}).get("microbatch_size", 1)),
                           exploration_epsilon=float(metadata.get("exploration_epsilon", 0)))


def preflight(config: dict) -> dict:
    """Real-model readiness is separate from test-only CPU engineering checks."""
    from paper2.environments.belief import Episode, Move, Task
    from paper2.environments.rendering import render_prompt
    tasks = [
        Task(Episode(2, (Move(4, True, False), Move(1, False, True))), query="A"),
        Task(Episode(1, (Move(0, False, True),)), query="B"),
    ]
    prompts = [render_prompt(task) for task in tasks]
    if not is_tiny(config):
        from paper2.experiments.budget import BudgetLedger
        from .policy import preflight_policy
        if not torch.cuda.is_available() and config.get("model", {}).get("device", "cuda") != "cpu":
            raise RuntimeError("No CUDA GPU is available; real-model GPU preflight remains unrun.")
        budget = config.get("budget", {})
        ledger = BudgetLedger(budget.get("ledger_path", "runs/budget.jsonl"))
        mode = budget.get("mode", "paid")
        if mode == "cpu_test":
            raise ValueError("Real-model validation cannot be labelled cpu_test.")
        seconds = float(budget.get("job_walltime_cap_minutes", 0)) * 60
        job_id = f"preflight:{uuid.uuid4().hex}"
        stage = budget.get("stage", "validation")
        ledger.authorize_job(config, job_id=job_id, stage=stage,
                             estimated_seconds=seconds, mode=mode)
        started = time.monotonic()
        status = "failed"
        try:
            from paper2.evaluation.independent_oracle import gold_answer
            result = preflight_policy(config, prompts, [str(gold_answer(task)) for task in tasks])
            status = "success"
            return {**result, "budget_ledger": str(ledger.path)}
        finally:
            ledger.record_usage(job_id=job_id, stage=stage, seconds=time.monotonic() - started,
                                rate_usd_per_hour=budget.get("hourly_rate_usd"),
                                mode=mode, event_id=f"{job_id}:complete")
            ledger.finish_job(job_id, status)
    policy = _build_policy(config, int(config.get("seed", 101)))
    optimizer = _optimizer(policy, config)
    started = time.monotonic()
    result = training_step(policy, prompts, lambda index, action: float(action == str(index)),
                           optimizer, generator=torch.Generator().manual_seed(10), group_size=8,
                           microbatch_size=1)
    return {"status": "passed", "test_only": True,
            "scientific_claims": False, "elapsed_seconds": time.monotonic() - started,
            "policy": policy.metadata, "step": result.metrics}


def train_phase(config: dict, phase: str, seed: int, resume: str | Path | None = None,
                parent: str | Path | None = None, arm: str | None = None) -> dict:
    """Run one bounded phase, saving complete boundaries and every evaluation node."""
    from paper2.environments.rendering import render_prompt
    from paper2.evaluation.verifiers import reward
    from paper2.experiments.budget import BudgetLedger
    from paper2.experiments.data import load_tasks
    from paper2.experiments.manifests import create_manifest, redact_config, utcnow

    if phase not in {"clean", "vulnerable", "control", "recovery", "clean_reference"}:
        raise ValueError(f"Unsupported training phase: {phase}")
    if resume and parent:
        raise ValueError("Interruption resume and scientific parent branch are distinct operations.")
    if phase != "clean" and not (resume or parent):
        raise ValueError("Non-clean phases require an explicit complete parent checkpoint.")
    settings = config.get("training", {})
    budget_config = config.get("budget", {})
    max_seconds = float(budget_config.get("job_walltime_cap_minutes", 0)) * 60
    if max_seconds <= 0:
        raise ValueError("A positive wall-time cap is required for every training command.")
    cap = int(settings.get("phase_update_cap", 100))
    if cap < 1:
        raise ValueError("phase_update_cap must be positive.")
    batch_size = int(settings.get("unique_prompts_per_update", 32))
    microbatch = int(settings.get("microbatch_size", 1))
    eval_every = int(settings.get("evaluation_every_updates", 25))
    checkpoint_every = int(config.get("checkpoint", {}).get("save_every_updates", eval_every))
    if min(eval_every, checkpoint_every, microbatch) < 1:
        raise ValueError("Evaluation/checkpoint periods and microbatch must be positive.")
    tiny = is_tiny(config)
    mode = "cpu_test" if tiny else budget_config.get("mode", "paid")
    if not tiny and mode == "cpu_test":
        raise ValueError("Real-model jobs must declare paid or colab accounting, not cpu_test.")
    arm = arm or ("clean_reference" if phase in {"control", "clean_reference"} else phase)
    exploration = float(settings.get("exploration_epsilon", 0))
    if arm in {"continue_exploration", "reset_exploration"}:
        exploration = float(settings.get("recovery_exploration_epsilon", 0.02))
    verifier = "vulnerable" if phase == "vulnerable" else "corrected"
    stage = budget_config.get("stage", "recovery" if phase == "recovery" else
                              ("validation" if tiny else "pilot"))
    root = Path(config.get("output_dir", config.get("run", {}).get("output_dir", "runs")))
    old_meta = verify_checkpoint(resume) if resume else None
    run_id = old_meta["run_id"] if old_meta else f"{phase}-{arm}-seed{seed}-{uuid.uuid4().hex[:8]}"
    restore_directory = config.get("run", {}).get("restore_directory")
    if restore_directory and not resume:
        raise ValueError("run.restore_directory is only valid for an explicit interruption resume.")
    output = (Path(restore_directory or old_meta["run_directory"]) if old_meta else root / run_id)
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(budget_config.get("ledger_path", "runs/budget.jsonl"))
    job_id = f"{run_id}:segment:{uuid.uuid4().hex[:10]}"
    segment_started = time.monotonic()
    last_accounted = segment_started
    status = "running"
    manifest = {**create_manifest(config, run_id=run_id, arm=arm, seed=seed,
                                 parent=str(parent) if parent else None,
                                 root=Path(__file__).resolve().parents[3]),
                "phase": phase, "test_only": tiny, "run_directory": str(output),
                "status": status,
                "budget_ledger": str(ledger.path), "segments": [],
                "code_identity": _code_identity()}
    if resume:
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Resume checkpoint has no retained run manifest.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("run_id") != old_meta["run_id"]:
            raise ValueError("Restored run manifest does not match checkpoint run identity.")
        if restore_directory:
            candidates = [c for c in manifest.get("checkpoints", [])
                          if Path(c["path"]).resolve() == Path(resume).resolve()]
            if not candidates or any(c.get("checksums_sha256") != digest(Path(resume) / "checksums.json")
                                     for c in candidates):
                raise ValueError("Restored checkpoint path/checksum is not declared in restored manifest.")
            if not manifest.get("restore_metadata"):
                raise ValueError("Restored manifest must include export/restore provenance.")
        boundary = int(old_meta["phase_step"])
        logged_steps = []
        steps_path = output / "steps.jsonl"
        if steps_path.exists():
            logged_steps = [json.loads(line)["step"] for line in steps_path.read_text().splitlines()]
        committed_step = max([int(manifest.get("phase_step", 0)), *logged_steps])
        if boundary < committed_step:
            raise ValueError("Resume checkpoint predates retained run logs. Preserve the later "
                             "attempt separately before recovery; stale resume cannot append a duplicate trajectory.")
    ledger.authorize_job(config, job_id=job_id, stage=stage, estimated_seconds=max_seconds, mode=mode)
    manifest["segments"].append({"job_id": job_id, "resume": str(resume) if resume else None,
                                 "started_unix": time.time()})
    manifest["started_at_utc"] = manifest.get("started_at_utc") or utcnow()
    manifest["scheduled_evaluation_points"] = sorted(set([0, cap, *range(eval_every, cap + 1, eval_every)]))
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    policy = optimizer = action_generator = schedule = None
    step = 0
    historical_updates = 0
    elapsed_before = float(old_meta.get("phase_elapsed_seconds", 0)) if old_meta else 0.0
    processed_tokens = int(old_meta.get("processed_prompt_tokens", 0)) if old_meta else 0
    generated_tokens = int(old_meta.get("sampled_action_count", 0)) if old_meta else 0
    stop_requested = False
    old_handlers = {}
    last_boundary_saved = str(resume) if resume else None
    started_training = None
    contract = None

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    def account():
        nonlocal last_accounted
        now = time.monotonic()
        ledger.record_usage(job_id=job_id, stage=stage, seconds=now - last_accounted,
                            rate_usd_per_hour=0.0 if tiny else budget_config.get("hourly_rate_usd"),
                            mode=mode, event_id=f"{job_id}:{last_accounted:.9f}")
        last_accounted = now

    def phase_elapsed():
        # Runtime axis includes setup, evaluations, and checkpoint I/O.
        return elapsed_before + time.monotonic() - segment_started

    def metadata():
        return {
            "seed": seed, "phase": phase, "arm": arm, "phase_step": step,
            "historical_updates": historical_updates, "total_updates": historical_updates + step,
            "sampler": schedule.state(), "run_id": run_id, "run_directory": str(output),
            "config": redact_config(config), "resume_contract": contract, "verifier": verifier,
            "exploration_epsilon": exploration, "parent": manifest.get("parent"),
            "phase_elapsed_seconds": phase_elapsed(),
            "processed_prompt_tokens": processed_tokens, "sampled_action_count": generated_tokens,
            "budget_ledger_offset": ledger.snapshot()["entries"],
            "reference_policy": None, "kl_beta": 0, "episode_state": None,
            "code_identity": manifest["code_identity"],
            "runtime": manifest.get("runtime"),
            "environment_lock_sha256": manifest.get("environment_lock_sha256"),
        }

    def checkpoint(kind="complete"):
        nonlocal last_boundary_saved
        path = output / ("curve" if kind == "inference" else "checkpoints") / f"step-{step:06d}"
        if path.exists():
            existing = verify_checkpoint(path)
            if existing["phase_step"] != step:
                raise ValueError("Conflicting checkpoint boundary.")
            if kind == "complete":
                last_boundary_saved = str(path)
            return str(path)
        save_checkpoint(path, policy, optimizer if kind == "complete" else None,
                        metadata=metadata(), generator=action_generator,
                        numpy_generator=schedule.generator, inference_only=kind == "inference")
        manifest["checkpoints"].append({"path": str(path), "step": step,
                                         "inference_only": kind == "inference",
                                         "checksums_sha256": digest(path / "checksums.json")})
        if kind == "complete":
            last_boundary_saved = str(path)
        return str(path)

    def evaluate(development):
        node_started = time.monotonic()
        _device_sync(policy)
        metric = evaluate_policy(policy, development, microbatch_size=microbatch,
                                 exploration_epsilon=exploration)
        _device_sync(policy)
        inference_path = checkpoint("inference")
        node = {"phase": phase, "arm": arm, "seed": seed, "step": step,
                "additional_compute_seconds": 0.0 if step == 0 else phase_elapsed(),
                "axis": "cpu_process_seconds" if tiny else "same_accelerator_process_seconds",
                "hardware_id": manifest["hardware_id"],
                "backend_id": manifest["backend_id"],
                "evaluation_seconds": time.monotonic() - node_started,
                "metrics": metric, "checkpoint": inference_path, "suite": "development",
                "processed_prompt_tokens": processed_tokens, "sampled_action_count": generated_tokens}
        manifest["evaluations"].append(node)
        _append_jsonl(output / "evaluations.jsonl", [node])

    try:
        train_tasks = load_tasks(config.get("data", {}).get("path", "data/project_a"), "train")
        development = load_tasks(config.get("data", {}).get("path", "data/project_a"), "development")
        dataset_manifest = Path(config.get("data", {}).get("path", "data/project_a")) / "manifest.json"
        if dataset_manifest.exists():
            manifest["split_manifest_sha256"] = digest(dataset_manifest)
        max_eval = int(settings.get("evaluation_max_tasks", 0))
        if max_eval:
            development = development[:max_eval]
            manifest["development_subset"] = [t.task_id for t in development]
        contract = _contract(config, phase, seed, [t.task_id for t in train_tasks])
        _seed(seed)
        policy = _build_policy(config, seed)
        if hasattr(policy, "validate_prompt"):
            policy.validate_prompt(render_prompt(train_tasks[0]))
        optimizer = _optimizer(policy, config)
        schedule = PromptSchedule(len(train_tasks), _phase_seed(seed, phase))
        action_generator = torch.Generator().manual_seed(_phase_seed(seed, phase) + 1)
        if resume:
            ledger.assert_checkpoint_offset(old_meta["budget_ledger_offset"])
            restored = load_checkpoint(resume, policy, optimizer, generator=action_generator,
                                       numpy_generator=schedule.generator, expected_contract=contract)
            if restored["phase"] != phase or restored["arm"] != arm or restored["seed"] != seed:
                raise ValueError("Resume phase/arm/seed changed.")
            schedule.restore(restored["sampler"])
            step = int(restored["phase_step"])
            historical_updates = int(restored["historical_updates"])
        elif parent:
            parent_meta = verify_checkpoint(parent)
            if arm == "rollback" and not (
                parent_meta.get("checkpoint_role") == "scientific_branch_anchor"
                and parent_meta.get("incident_parent")
            ):
                raise ValueError("Rollback must use a scientific branch anchor with incident_parent; "
                                 "directly loading clean state loses shared incident history.")
            if phase in {"vulnerable", "control"} and parent_meta.get("phase") != "clean":
                raise ValueError("Vulnerable and matched-control phases must branch from a clean checkpoint.")
            reset = arm in {"reset_optimizer", "reset_exploration"}
            load_checkpoint(parent, policy, None if reset else optimizer, restore_random_state=False)
            historical_updates = int(parent_meta["total_updates"])
            manifest["parent"] = str(parent)
            manifest["parent_checksums_sha256"] = digest(Path(parent) / "checksums.json")
            # Deliberately start a new phase schedule/RNG, never parent cursor.
        manifest["policy"] = getattr(policy, "metadata", {})
        manifest["hardware"] = {"cuda": torch.cuda.is_available(),
                                "device": torch.cuda.get_device_name() if not tiny and torch.cuda.is_available() else "cpu",
                                "precision": config.get("model", {}).get("precision")}
        manifest["hardware_id"] = manifest["hardware"]["device"]
        manifest["backend_id"] = (
            f"{config.get('backend')}:{config.get('model', {}).get('precision')}:"
            f"torch-{torch.__version__}:cuda-{torch.version.cuda}"
        )
        manifest["numerical_backend"] = {
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "attention_implementation": config.get("model", {}).get("attention_implementation", "sdpa"),
        }
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, request_stop)
        if not resume:
            evaluate(development)
            checkpoint()
        started_training = time.monotonic()
        recent_step_seconds = 0.0
        while step < cap:
            remaining = max_seconds - (time.monotonic() - segment_started)
            if stop_requested:
                status = "preemption"
                break
            if remaining <= max(2.0, recent_step_seconds * 1.25):
                status = "budget_stop"
                break
            indices = schedule.next(batch_size)
            tasks = [train_tasks[i] for i in indices]
            prompts = [render_prompt(task) for task in tasks]
            tokens = _token_ids(policy, prompts)
            _device_sync(policy)
            step_started = time.monotonic()
            result = training_step(
                policy, prompts, lambda i, response: reward(tasks[i], response, verifier=verifier),
                optimizer, generator=action_generator, families=[t.family for t in tasks],
                group_size=int(settings.get("samples_per_prompt", 8)), microbatch_size=microbatch,
                temperature=float(settings.get("temperature", 1)), exploration_epsilon=exploration,
                clip_epsilon=float(settings.get("clip_epsilon", 0.2)),
                entropy_coefficient=float(settings.get("entropy_coefficient", 0)),
                max_grad_norm=float(settings.get("max_grad_norm", 1)),
                diagnose_family_gradients=(step % int(settings.get("gradient_diagnostics_every", 25)) == 0))
            _device_sync(policy)
            recent_step_seconds = time.monotonic() - step_started
            step += 1
            processed_tokens += sum(map(len, tokens))
            generated_tokens += len(tasks) * int(settings.get("samples_per_prompt", 8))
            row = {**result.metrics, "run_id": run_id, "phase": phase, "arm": arm, "seed": seed,
                   "step": step, "complete_step_seconds": recent_step_seconds,
                   "phase_elapsed_seconds": phase_elapsed(), "prompt_ids": [t.task_id for t in tasks],
                   "processed_prompt_tokens": processed_tokens, "sampled_action_count": generated_tokens}
            _append_jsonl(output / "steps.jsonl", [row])
            rollouts = []
            for item in result.records:
                i = item["prompt_index"]
                action_ids = getattr(policy, "metadata", {}).get("action_token_ids")
                for sample, action in enumerate(item["actions"]):
                    rollouts.append({
                        "run_id": run_id, "phase": phase, "arm": arm, "seed": seed, "update": step,
                        "prompt_id": tasks[i].task_id, "episode_id": tasks[i].episode_id,
                        "group_id": f"{run_id}:{step}:{i}", "sample_index": sample,
                        "prompt_hash": hashlib.sha256(prompts[i].encode()).hexdigest(),
                        "prompt_token_ids": tokens[i], "model_revision": config.get("model", {}).get("revision"),
                        "policy_checkpoint_id": f"{run_id}:pre-update-{step - 1}",
                        "completion_token_ids": [action_ids[action]] if action_ids else [],
                        "completion_mask": [1], "raw_text": ACTIONS[action],
                        "eos_reached": False, "truncated": False,
                        "parse_status": "valid_digit" if action < 6 else "invalid_alias",
                        "parsed_fields": {"answer": action} if action < 6 else None,
                        "sampling_configuration": {"temperature": float(settings.get("temperature", 1)),
                                                   "exploration_epsilon": exploration, "group_size": len(item["actions"])},
                        "behaviour_logprobs_if_recorded": [item["behaviour_logprobs"][sample]],
                        "training_verifier_id": verifier, "training_reward": item["training_rewards"][sample],
                        "audit_evaluator_id": None, "audit_labels_when_permitted": None,
                        "wall_time": phase_elapsed(), "prompt_tokens": len(tokens[i]),
                        "completion_tokens": 1 if action_ids else 0, "test_only": tiny})
            _append_jsonl(output / "rollouts.jsonl", rollouts)
            if step % eval_every == 0 or step == cap:
                evaluate(development)
            account()
            if step % checkpoint_every == 0 or step == cap:
                checkpoint()
            manifest["status"] = "running"
            manifest["phase_step"] = step
            _atomic_json(manifest_path, manifest)
        else:
            status = "success"
        checkpoint()
        manifest["final_checkpoint"] = last_boundary_saved
    except BaseException as exc:
        status = "numerical_failure" if isinstance(exc, FloatingPointError) else "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        # An exception during an optimizer step has no trustworthy new boundary.
        # Retain the previous complete checkpoint, never serialize partial grads.
        manifest["final_checkpoint"] = last_boundary_saved
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        account()
        ledger.finish_job(job_id=job_id, status=status)
        manifest["status"] = status
        manifest["phase_step"] = step
        manifest["phase_elapsed_seconds"] = phase_elapsed()
        manifest["processed_tokens"] = processed_tokens
        manifest["generated_tokens"] = generated_tokens
        manifest["gpu_seconds"] = phase_elapsed() if not tiny and torch.cuda.is_available() else None
        manifest["ended_at_utc"] = utcnow()
        manifest["budget"] = ledger.snapshot()
        manifest["cash_usd"] = manifest["budget"]["by_job_usd"].get(job_id, 0)
        manifest["segments"][-1].update({"finished_unix": time.time(), "status": status})
        manifest["manifest_path"] = str(manifest_path)
        _atomic_json(manifest_path, manifest)
    return manifest


def create_branch(config: dict, parent: str | Path, output_dir: str | Path, arm: str,
                  seed: int, incident_parent: str | Path | None = None) -> str:
    """Transform a verified checkpoint on CPU, without allocating a GPU model."""
    from paper2.experiments.data import load_tasks
    from paper2.experiments.manifests import redact_config
    arms = {"continue", "reset_optimizer", "rollback", "continue_exploration",
            "reset_exploration", "clean_reference"}
    if arm not in arms:
        raise ValueError(f"Unknown recovery arm: {arm}")
    if arm == "rollback" and incident_parent is None:
        raise ValueError("Rollback requires incident_parent to preserve shared incident history.")
    original_checksum = digest(Path(parent) / "checksums.json")
    parent_metadata = verify_checkpoint(parent)
    incident = verify_checkpoint(incident_parent) if incident_parent else parent_metadata
    if parent_metadata.get("seed") != seed or incident.get("seed") != seed:
        raise ValueError("Scientific branches must use the paired pre-patch training seed.")
    tasks = load_tasks(config.get("data", {}).get("path", "data/project_a"), "train")
    if parent_metadata.get("inference_only"):
        raise ValueError("Scientific branches require a complete parent optimizer checkpoint.")
    if parent_metadata.get("config", {}).get("model") != config.get("model"):
        raise ValueError("Branch model configuration differs from its parent.")
    payload = torch.load(Path(parent) / "state.pt", map_location="cpu", weights_only=False)
    settings = config.get("training", {})
    expected = {"lr": float(settings.get("learning_rate", 1e-5)),
                "weight_decay": float(settings.get("weight_decay", 0)),
                "betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False}
    for group in payload["optimizer"]["param_groups"]:
        if any(group.get(key) != value for key, value in expected.items()):
            raise ValueError("Optimizer hyperparameters differ from parent; recovery arms need common settings.")
    if arm in {"reset_optimizer", "reset_exploration"}:
        payload["optimizer"]["state"] = {}
    schedule = PromptSchedule(len(tasks), _phase_seed(seed, "recovery"))
    generator = torch.Generator().manual_seed(_phase_seed(seed, "recovery") + 1)
    # These new phase RNG states are metadata for the branch anchor. The
    # authorized training process also starts this schedule explicitly; this
    # operation never initializes or queries CUDA outside its runtime budget.
    payload["rng"] = {
        "python": random.Random(_phase_seed(seed, "recovery")).getstate(),
        "numpy_global": np.random.RandomState(_phase_seed(seed, "recovery") % (2 ** 32)).get_state(),
        "numpy_private": schedule.generator.bit_generator.state,
        "torch_cpu": torch.Generator().manual_seed(_phase_seed(seed, "recovery")).get_state(),
        "torch_cuda": [],
        "action_generator": generator.get_state(),
    }
    metadata = {
        **parent_metadata,
        "seed": seed, "phase": "recovery", "arm": arm, "phase_step": 0,
        "total_updates": incident["total_updates"], "historical_updates": incident["total_updates"],
        "sampler": schedule.state(), "parent": str(parent),
        "incident_parent": str(incident_parent) if incident_parent else str(parent),
        "parent_checksums_sha256": original_checksum,
        "budget_ledger_offset": max(parent_metadata.get("budget_ledger_offset", 0),
                                   incident.get("budget_ledger_offset", 0)),
        "config": redact_config(config), "branch_operation": "new phase, shared recovery schedule",
        "checkpoint_role": "scientific_branch_anchor",
        "optimizer_treatment": "fresh" if arm in {"reset_optimizer", "reset_exploration"} else "carry",
        "exploration_epsilon": (float(config.get("training", {}).get("recovery_exploration_epsilon", 0.02))
                               if "exploration" in arm else 0.0),
    }
    # Parent-run counters/paths must not masquerade as a literal interruption resume.
    for key in ("run_id", "run_directory", "resume_contract", "phase_elapsed_seconds",
                "processed_prompt_tokens", "sampled_action_count"):
        metadata.pop(key, None)
    path = write_checkpoint_payload(output_dir, payload, metadata)
    if original_checksum != digest(Path(parent) / "checksums.json"):
        raise RuntimeError("Parent checkpoint changed during branching.")
    return str(path)


def resume_check(config: dict, output_dir: str | Path | None = None) -> dict:
    """Run N updates vs K+fresh-process+(N-K), then compare every saved tensor."""
    from paper2.experiments.budget import BudgetLedger
    tiny = is_tiny(config)
    # An omitted backend is accepted only for direct low-level test callers.
    # Any explicitly selected real model remains a real model in the workers.
    if "backend" not in config and "model" not in config:
        config = {**config, "backend": "tiny"}
        tiny = True
    output = Path(output_dir or config.get("output_dir", "runs/resume_check")) / f"resume-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True)
    worker_config = copy.deepcopy(config)
    worker_config.setdefault("training", {})
    resume_updates = int(config.get("resume_check", {}).get("updates", 6))
    resume_stop = int(config.get("resume_check", {}).get("interrupt_after", 3))
    if not 0 < resume_stop < resume_updates:
        raise ValueError("Resume check requires 0 < interrupt_after < updates.")
    worker_config["resume_check"] = {"updates": resume_updates, "interrupt_after": resume_stop}
    (output / "config.json").write_text(json.dumps(worker_config, indent=2))
    commands = [
        ["full", str(output / "full")],
        ["partial", str(output / "partial")],
        ["resume", str(output / "resumed"), str(output / "partial" / "checkpoint")],
    ]
    logs = []
    budget = config.get("budget", {})
    ledger = BudgetLedger(budget.get("ledger_path", str(output / "budget.jsonl")))
    mode = "cpu_test" if tiny else budget.get("mode", "paid")
    stage = budget.get("stage", "validation")
    cap_seconds = float(budget.get("job_walltime_cap_minutes", 5 if tiny else 0)) * 60
    admission_config = copy.deepcopy(config)
    admission_config.setdefault("budget", {}).setdefault("job_walltime_cap_minutes", 5 if tiny else 0)
    job_id = f"resume-check:{output.name}"
    ledger.authorize_job(admission_config, job_id=job_id, stage=stage,
                         estimated_seconds=cap_seconds, mode=mode)
    started = time.monotonic()
    status = "failed"
    try:
        for args in commands:
            remaining = cap_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Resume validation exceeded its walltime cap.")
            completed = subprocess.run([sys.executable, "-m", "paper2.training.resume_worker",
                                        str(output / "config.json"), *args],
                                       text=True, capture_output=True, check=False, timeout=remaining)
            logs.append({"command": args[0], "returncode": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr})
            if completed.returncode:
                _atomic_json(output / "result.json", {"status": "failed", "processes": logs})
                raise RuntimeError(f"Fresh-process resume worker failed: {completed.stderr}")
        status = "success"
    finally:
        ledger.record_usage(job_id=job_id, stage=stage, seconds=time.monotonic() - started,
                            rate_usd_per_hour=0 if tiny else budget.get("hourly_rate_usd"),
                            mode=mode, event_id=f"{job_id}:complete")
        ledger.finish_job(job_id, status)
    full = torch.load(output / "full" / "checkpoint" / "state.pt", weights_only=False)
    resumed = torch.load(output / "resumed" / "checkpoint" / "state.pt", weights_only=False)
    errors = {name: float((value - resumed["policy"][name]).abs().max())
              for name, value in full["policy"].items()}
    full_records = json.loads((output / "full" / "records.json").read_text())
    partial_records = json.loads((output / "partial" / "records.json").read_text())
    resumed_records = json.loads((output / "resumed" / "records.json").read_text())
    records_equal = full_records == partial_records + resumed_records
    full_meta = verify_checkpoint(output / "full" / "checkpoint")
    resumed_meta = verify_checkpoint(output / "resumed" / "checkpoint")
    counters_equal = (full_meta["phase_step"] == resumed_meta["phase_step"] and
                      full_meta["sampler"] == resumed_meta["sampler"])
    optimizer_equal = _nested_equal(full["optimizer"], resumed["optimizer"])
    tolerance = float(config.get("resume_check", {}).get("atol", 0.0))
    # Deterministic debug expectation remains exact; tolerances never excuse a
    # different sampler cursor, action sequence, optimizer counter, or LR.
    passed = max(errors.values(), default=0) <= tolerance and records_equal and counters_equal and optimizer_equal
    result = {"status": "passed" if passed else "failed", "test_only": tiny,
              "backend": "CPU float32 tiny policy" if tiny else config.get("model"),
              "fresh_process": True, "updates": resume_updates, "interruption_update": resume_stop,
              "max_parameter_difference": max(errors.values(), default=0),
              "next_prompts_actions_losses_equal": records_equal,
              "sampler_counters_equal": counters_equal, "optimizer_state_equal": optimizer_equal,
              "output_dir": str(output), "processes": logs, "budget": ledger.snapshot()}
    _atomic_json(output / "result.json", result)
    if not passed:
        raise RuntimeError(f"Resume equivalence failed: {result}")
    return result


def _nested_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_nested_equal(x, y) for x, y in zip(a, b))
    return a == b
