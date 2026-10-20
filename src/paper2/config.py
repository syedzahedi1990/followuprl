"""Read resolved experiment settings without silently truncating task facts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    config = copy.deepcopy(config)
    if config.get("project") != "A":
        raise ValueError("Only Project A is active")
    if config.get("task", {}).get("locations", 6) != 6:
        raise ValueError("The registered diagnostic requires exactly six locations")
    actions = config.get("task", {}).get("action_strings", list("012345?!"))
    if actions != list("012345?!"):
        raise ValueError("Action order must be 0,1,2,3,4,5,?,!")
    training = config.get("training", {})
    if training.get("updates_per_rollout_batch", 1) != 1:
        raise ValueError("This pilot implements exactly one update per rollout batch")
    if training.get("schedule", "constant") != "constant":
        raise ValueError("Pilot comparisons require a constant learning rate")
    if training.get("kl_beta", 0.0) != 0.0:
        raise ValueError("Nonzero KL requires a separately validated common clean reference")
    if training.get("temperature", 1.0) != 1.0:
        raise ValueError("The primary finite-action protocol fixes temperature to 1")
    if config.get("adapter", {}).get("dropout", 0.0) != 0.0:
        raise ValueError("Diagnostic adapter dropout must be disabled")
    return config


def config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

