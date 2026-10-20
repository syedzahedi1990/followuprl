"""Atomic, checksummed checkpoints at complete optimizer boundaries.

torch serialization must only be loaded from trusted local runs. Checksums
detect damaged/incomplete copies; they do not authenticate an untrusted file.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

MARKER = "CHECKPOINT_COMPLETE"


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def software_identity() -> dict[str, str]:
    versions = {}
    for package in ("torch", "numpy", "transformers", "peft", "accelerate", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def trainable_names(policy: torch.nn.Module) -> list[str]:
    return [name for name, param in policy.named_parameters() if param.requires_grad]


def policy_state(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    method = getattr(policy, "trainable_state_dict", None)
    state = method() if method else policy.state_dict()
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def restore_policy(policy: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    method = getattr(policy, "load_trainable_state_dict", None)
    if method:
        method(state)
    else:
        policy.load_state_dict(state, strict=True)


def capture_rng(generator: torch.Generator, numpy_generator: np.random.Generator | None = None) -> dict:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_private": numpy_generator.bit_generator.state if numpy_generator is not None else None,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "action_generator": generator.get_state(),
    }


def restore_rng(state: dict, generator: torch.Generator,
                numpy_generator: np.random.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint cannot be resumed without CUDA.")
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device topology changed.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["action_generator"])
    if state["numpy_private"] is not None:
        if numpy_generator is None:
            raise ValueError("Checkpoint requires a private NumPy generator.")
        numpy_generator.bit_generator.state = state["numpy_private"]


def save_checkpoint(path: str | Path, policy: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None, *,
                    metadata: dict, generator: torch.Generator | None = None,
                    numpy_generator: np.random.Generator | None = None,
                    inference_only: bool = False) -> Path:
    if not inference_only and (optimizer is None or generator is None):
        raise ValueError("Complete checkpoints require optimizer and sampling generator.")
    manifest = {
        **metadata,
        "schema_version": 1,
        "inference_only": inference_only,
        "optimizer_boundary": True,
        "outstanding_rollout_buffer": None,
        "parameter_names": trainable_names(policy),
        "policy_metadata": getattr(policy, "metadata", {}),
        "software": software_identity(),
    }
    payload = {"policy": policy_state(policy)}
    if not inference_only:
        payload.update({
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng(generator, numpy_generator),
            "scheduler": {"kind": "constant", "lr": [g["lr"] for g in optimizer.param_groups]},
            "grad_scaler": None,
        })
    return write_checkpoint_payload(path, payload, manifest)


def write_checkpoint_payload(path: str | Path, payload: dict, manifest: dict) -> Path:
    """Atomically write an already validated local payload, without a model load."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Immutable checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    try:
        torch.save(payload, temporary / "state.pt")
        (temporary / "metadata.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        checksums = {name: digest(temporary / name) for name in ("state.pt", "metadata.json")}
        (temporary / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
        (temporary / MARKER).write_text(digest(temporary / "checksums.json") + "\n")
        for filename in ("state.pt", "metadata.json", "checksums.json", MARKER):
            with open(temporary / filename, "rb") as stream:
                os.fsync(stream.fileno())
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def verify_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_dir() or not (path / MARKER).is_file():
        raise ValueError(f"Not a complete checkpoint: {path}")
    if (path / MARKER).read_text().strip() != digest(path / "checksums.json"):
        raise ValueError("Checkpoint checksum manifest was modified.")
    checksums = json.loads((path / "checksums.json").read_text())
    if set(checksums) != {"state.pt", "metadata.json"}:
        raise ValueError("Unexpected checkpoint file manifest.")
    for filename, expected in checksums.items():
        if digest(path / filename) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {filename}")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("schema_version") != 1 or not metadata.get("optimizer_boundary"):
        raise ValueError("Unsupported or partial checkpoint state.")
    if metadata.get("outstanding_rollout_buffer") is not None:
        raise ValueError("This runner does not resume retained rollout buffers.")
    return metadata


def load_checkpoint(path: str | Path, policy: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None, *,
                    generator: torch.Generator | None = None,
                    numpy_generator: np.random.Generator | None = None,
                    restore_random_state: bool = True,
                    expected_contract: str | None = None) -> dict:
    metadata = verify_checkpoint(path)
    if metadata["parameter_names"] != trainable_names(policy):
        raise ValueError("Trainable parameter names/order differ from checkpoint.")
    if metadata["policy_metadata"] != getattr(policy, "metadata", {}):
        raise ValueError("Policy/base/tokenizer identity differs from checkpoint.")
    if metadata["software"] != software_identity():
        raise ValueError("Installed training software differs from checkpoint.")
    if expected_contract is not None and metadata.get("resume_contract") != expected_contract:
        raise ValueError("Resume contract changed: phase/data/objective/planned steps must match.")
    if metadata["inference_only"] and (optimizer is not None or restore_random_state):
        raise ValueError("Inference snapshots cannot resume training.")
    payload = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
    restore_policy(policy, payload["policy"])
    if optimizer is not None:
        # Adam.load_state_dict also restores LR/decay/betas. A continuation
        # versus reset ablation must not silently change these with the moments.
        settings = ("lr", "betas", "eps", "weight_decay", "amsgrad")
        current_groups = optimizer.state_dict()["param_groups"]
        saved_groups = payload["optimizer"]["param_groups"]
        if len(current_groups) != len(saved_groups):
            raise ValueError("Optimizer parameter-group count changed.")
        for current, saved in zip(current_groups, saved_groups):
            if any(current.get(key) != saved.get(key) for key in settings):
                raise ValueError("Optimizer hyperparameters differ from saved state; "
                                 "use identical LR/betas/epsilon/decay for recovery arms.")
        optimizer.load_state_dict(payload["optimizer"])
    if restore_random_state:
        if generator is None:
            raise ValueError("RNG restoration requires a sampling generator.")
        restore_rng(payload["rng"], generator, numpy_generator)
    return metadata
