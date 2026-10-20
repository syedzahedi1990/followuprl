"""Deterministic symbolic splits with explicit final-test access accounting.

Sizes count distinct belief episodes. Additional protected task rows produce
the declared task-family mixture. Every rendering/query variant and future
counterfactual sibling must stay in its canonical ancestor's split.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any

from paper2.environments.belief import Episode, Move, Task
from paper2.evaluation.independent_oracle import gold_answer

DATA_VERSION = "project-a-symbolic-v2"
SPLITS = ("train", "development", "final_test", "structural_test")
FINAL_SPLITS = {"final_test", "structural_test"}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(seed: int, split: str) -> int:
    return int(_hash([seed, split])[:16], 16)


def is_structural_holdout(skeleton: tuple[int, ...]) -> bool:
    """A fixed approximately 20% pattern holdout independent of label/location."""
    return int(_hash(["observation-pattern-v1", skeleton])[:16], 16) % 5 == 0


def _normalise_config(config: dict[str, Any]) -> dict[str, Any]:
    task = config.get("task", {})
    data = config.get("data", {})
    sizes = {
        "train": 6000, "development": 600,
        "final_test": 1200, "structural_test": 1200,
    }
    unknown = set(data.get("sizes", {})) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown split sizes: {sorted(unknown)}")
    sizes.update(data.get("sizes", {}))
    if any(type(x) is not int or x < 0 for x in sizes.values()):
        raise ValueError("Split sizes must be nonnegative integers.")
    horizon = task.get("moves", 5)
    if type(horizon) is not int or not 1 <= horizon <= 8:
        raise ValueError("Use the predeclared 1–8 event generator; larger horizons need a version.")
    if task.get("locations", 6) != 6 or task.get("observers", 2) != 2:
        raise ValueError("This protocol fixes six locations and two observers.")
    fraction = task.get("protected_fraction", 0.20)
    if not isinstance(fraction, (int, float)) or not 0 <= fraction <= 0.5:
        raise ValueError("protected_fraction must be between 0 and 0.5.")
    families = data.get("protected_families", ["world", "arithmetic"])
    if not families or len(set(families)) != len(families) or set(families) - {"world", "arithmetic"}:
        raise ValueError("Protected families must be world and/or arithmetic, without duplicates.")
    if task.get("output_mode", "finite_action") == "trace" and "arithmetic" in families:
        raise ValueError("Trace datasets require data.protected_families: [world].")
    package = Path(__file__).resolve().parents[1]
    source_hashes = {
        relative: _file_hash(package / relative)
        for relative in (
            "experiments/data.py", "environments/belief.py",
            "environments/rendering.py", "evaluation/independent_oracle.py",
        )
    }
    return {
        "version": DATA_VERSION,
        "seed": int(data.get("seed", 20261020)),
        "moves": horizon,
        "sizes": sizes,
        "protected_fraction": float(fraction),
        "protected_families": list(families),
        "source_hashes": source_hashes,
    }


def _generate_belief(
    count: int, horizon: int, rng: random.Random, structural: bool,
    seen: set[str],
) -> list[Task]:
    # Balancing label × queried observer × final world location prevents the
    # observer name or final destination alone from being a useful label.
    # Each block of twelve balances all labels, world destinations, and queried
    # observers exactly. Across six blocks it covers every observer/label/world
    # triple; incomplete final blocks have their achieved counts recorded.
    offsets = list(range(6))
    rng.shuffle(offsets)
    strata = []
    for offset in offsets:
        block = [
            (query, label, (label + offset) % 6)
            for query, label in itertools.product(("A", "B"), range(6))
        ]
        rng.shuffle(block)
        strata.extend(block)
    tasks = []
    for index in range(count):
        query, label, world = strata[index % len(strata)]
        for _ in range(20_000):
            masks = tuple(rng.randrange(4) for _ in range(horizon))
            if is_structural_holdout(masks) != structural:
                continue
            observed_bit = 1 if query == "A" else 2
            observed_indices = [i for i, mask in enumerate(masks) if mask & observed_bit]
            if observed_indices and observed_indices[-1] == horizon - 1 and label != world:
                continue
            initial = rng.randrange(6)
            destinations = [rng.randrange(6) for _ in range(horizon)]
            destinations[-1] = world
            if observed_indices:
                destinations[observed_indices[-1]] = label
            else:
                initial = label
            episode = Episode(initial, tuple(
                Move(destination, bool(mask & 1), bool(mask & 2))
                for destination, mask in zip(destinations, masks)
            ))
            if episode.episode_id in seen:
                continue
            seen.add(episode.episode_id)
            task = Task(episode=episode, query=query)
            if gold_answer(task) != label:
                raise AssertionError("Generator disagrees with independent labels.")
            tasks.append(task)
            break
        else:
            raise ValueError(
                "Could not generate unique balanced episodes. Reduce data.sizes "
                "for this horizon; the generator never duplicates episodes."
            )
    return tasks


def _add_protected(
    belief: list[Task], settings: dict[str, Any], rng: random.Random,
    seen_arithmetic: set[str],
) -> list[Task]:
    fraction = settings["protected_fraction"]
    count = round(len(belief) * fraction / (1 - fraction))
    families = settings["protected_families"]
    counts = {
        family: count // len(families) + int(i < count % len(families))
        for i, family in enumerate(families)
    }
    tasks = list(belief)
    if counts.get("world", 0):
        candidates = defaultdict(list)
        for task in belief:
            candidates[task.episode.events[-1].destination].append(task)
        for values in candidates.values():
            rng.shuffle(values)
        for index in range(counts["world"]):
            label = index % 6
            if not candidates[label]:
                # Small debug datasets may have no example for one label.
                available = next((x for x in candidates.values() if x), None)
                if available is None:
                    raise ValueError("Not enough distinct episodes for protected world tasks.")
            else:
                available = candidates[label]
            selected = available.pop()
            tasks.append(Task(selected.episode, query="world", family="world"))
    if counts.get("arithmetic", 0):
        for index in range(counts["arithmetic"]):
            label = index % 6
            for _ in range(20_000):
                a, b = rng.randrange(96), rng.randrange(96)
                if (a + b) % 6 != label:
                    continue
                task = Task(family="arithmetic", operands=(a, b))
                if task.episode_id not in seen_arithmetic:
                    seen_arithmetic.add(task.episode_id)
                    tasks.append(task)
                    break
            else:
                raise ValueError("Arithmetic pool exhausted; reduce dataset size.")
    rng.shuffle(tasks)
    return tasks


def _statistics(tasks: list[Task]) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for family in ("belief", "world", "arithmetic"):
        rows = [task for task in tasks if task.family == family]
        by_family[family] = {
            "count": len(rows),
            "gold_counts": dict(sorted(Counter(str(gold_answer(task)) for task in rows).items())),
        }
    belief = [task for task in tasks if task.family == "belief"]
    return {
        "task_count": len(tasks),
        "distinct_belief_episodes": len({task.episode_id for task in belief}),
        "family_counts": {key: value["count"] for key, value in by_family.items()},
        "by_family": by_family,
        "queried_observers": dict(sorted(Counter(task.query for task in belief).items())),
        "belief_equals_world_count": sum(
            gold_answer(task) == task.episode.events[-1].destination for task in belief
        ),
        "belief_differs_from_world_count": sum(
            gold_answer(task) != task.episode.events[-1].destination for task in belief
        ),
        "belief_label_world_counts": dict(sorted(Counter(
            f"{gold_answer(task)}:{task.episode.events[-1].destination}" for task in belief
        ).items())),
    }


def validate_splits(splits: dict[str, list[Task]]) -> dict[str, int]:
    """Check episode, sibling, variant, and structural partition integrity."""
    ownership: dict[str, str] = {}
    group_ownership: dict[str, str] = {}
    skeleton_splits: dict[tuple[int, ...], set[str]] = defaultdict(set)
    task_ids: set[str] = set()
    for split, tasks in splits.items():
        for task in tasks:
            prior = ownership.setdefault(task.episode_id, split)
            if prior != split:
                raise ValueError(f"Episode/query/rendering leakage: {prior} versus {split}.")
            prior_group = group_ownership.setdefault(task.group_id, split)
            if prior_group != split:
                raise ValueError(f"Counterfactual sibling leakage: {prior_group} versus {split}.")
            if task.task_id in task_ids:
                raise ValueError("Duplicate task identity.")
            task_ids.add(task.task_id)
            if task.episode is not None:
                skeleton_splits[task.episode.skeleton].add(split)
    for skeleton, names in skeleton_splits.items():
        if "structural_test" in names and len(names) > 1:
            raise ValueError("Structural holdout shares an observation pattern.")
        expected = is_structural_holdout(skeleton)
        if expected != (names == {"structural_test"}):
            raise ValueError("Observation pattern assigned to the wrong structural partition.")
    return {
        "unique_episode_ids": len(ownership),
        "unique_sibling_groups": len(group_ownership),
        "unique_task_ids": len(task_ids),
        "unique_skeletons": len(skeleton_splits),
    }


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Write symbolic data plus a checksum manifest; never render final prompts.

    Existing data are immutable: matching settings/checksums are reused, while
    changed settings require a new directory rather than silent replacement.
    """
    settings = _normalise_config(config)
    directory = Path(config.get("data", {}).get("path", "data/project_a"))
    manifest_path = directory / "manifest.json"
    fingerprint = _hash(settings)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["generator_fingerprint"] != fingerprint:
            raise ValueError("Dataset settings changed: choose a new data.path.")
        for entry in manifest["splits"].values():
            if _file_hash(directory / entry["file"]) != entry["sha256"]:
                raise ValueError("Existing dataset checksum failed.")
        return manifest
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty unmanifested dataset directory.")
    directory.mkdir(parents=True, exist_ok=True)
    all_skeletons = itertools.product(range(4), repeat=settings["moves"])
    structural_count = sum(is_structural_holdout(tuple(x)) for x in all_skeletons)
    total_skeletons = 4 ** settings["moves"]
    for structural, pattern_count in (
        (False, total_skeletons - structural_count), (True, structural_count)
    ):
        requested = sum(
            n for split, n in settings["sizes"].items()
            if (split == "structural_test") == structural
        )
        if requested > pattern_count * 6 ** (settings["moves"] + 1):
            raise ValueError("Requested more distinct episodes than this horizon/partition permits.")
    seen: set[str] = set()
    arithmetic_seen: set[str] = set()
    splits = {}
    for split in SPLITS:
        rng = random.Random(_seed(settings["seed"], split))
        belief = _generate_belief(
            settings["sizes"][split], settings["moves"], rng,
            split == "structural_test", seen,
        )
        splits[split] = _add_protected(belief, settings, rng, arithmetic_seen)
    validation = validate_splits(splits)
    manifest = {
        "schema_version": 1,
        "generator_fingerprint": fingerprint,
        "generator_settings": settings,
        "split_unit": "symbolic_episode_and_counterfactual_ancestor",
        "structural_unit": "complete_observation_pattern",
        "counterfactual_siblings_generated": False,
        "rendered_final_prompts": False,
        "final_test_status": "unopened",
        "validation": validation,
        "splits": {},
    }
    for split, tasks in splits.items():
        filename = f"{split}.jsonl"
        target = directory / filename
        with target.open("x") as stream:
            for task in tasks:
                stream.write(_json(task.to_dict()) + "\n")
        manifest["splits"][split] = {
            "file": filename, "sha256": _file_hash(target),
            "statistics": _statistics(tasks),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def freeze_final_protocol(
    path: str | Path, protocol_id: str, definition: dict[str, Any]
) -> Path:
    """Bind final access to an immutable prespecified evaluation plan.

    A valid definition must identify checkpoint nodes, metrics, and a practical
    effect threshold. This is scientific discipline, not an OS security wall.
    """
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("A nonempty frozen protocol identifier is required.")
    for field in ("evaluation_checkpoints", "metric_definitions", "practical_effect_threshold"):
        if field not in definition or definition[field] is None:
            raise ValueError(f"Frozen protocol requires {field}.")
    if not definition["evaluation_checkpoints"] or not definition["metric_definitions"]:
        raise ValueError("Frozen checkpoints and metrics must be nonempty.")
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    value = {
        "protocol_id": protocol_id,
        "definition": definition,
        "dataset_fingerprint": manifest["generator_fingerprint"],
        "split_hashes": {split: manifest["splits"][split]["sha256"] for split in FINAL_SPLITS},
    }
    value["sha256"] = _hash(value)
    target = directory / "frozen_protocol.json"
    if target.exists():
        if json.loads(target.read_text()) != value:
            raise ValueError("Frozen protocol already exists with different contents.")
    else:
        with target.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return target


def _log_access(directory: Path, split: str, protocol_id: str | None,
                reason: str | None, status: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": split, "protocol_id": protocol_id,
        "reason": reason, "status": status,
    }
    with (directory / "final_test_access.jsonl").open("a") as stream:
        stream.write(_json(record) + "\n")


def load_tasks(
    path: str | Path, split: str, *, protocol_id: str | None = None,
    access_reason: str | None = None, evaluation_checkpoint: str | Path | None = None,
) -> list[Task]:
    directory = Path(path)
    if split not in SPLITS:
        raise ValueError("Unknown data split.")
    manifest = json.loads((directory / "manifest.json").read_text())
    entry = manifest["splits"][split]
    if split in FINAL_SPLITS:
        target = directory / "frozen_protocol.json"
        approved = False
        if target.exists() and protocol_id and access_reason and evaluation_checkpoint is not None:
            frozen = json.loads(target.read_text())
            stored_hash = frozen.pop("sha256")
            specified_checkpoints = frozen["definition"]["evaluation_checkpoints"]
            if not all(isinstance(item, str) for item in specified_checkpoints):
                raise ValueError("Frozen evaluation_checkpoints must be a list of path strings.")
            requested_checkpoint = Path(evaluation_checkpoint).resolve()
            checkpoint_allowed = any(
                requested_checkpoint == Path(item).resolve() for item in specified_checkpoints
            )
            approved = (
                frozen["protocol_id"] == protocol_id
                and frozen["dataset_fingerprint"] == manifest["generator_fingerprint"]
                and frozen["split_hashes"][split] == entry["sha256"]
                and _hash(frozen) == stored_hash
                and checkpoint_allowed
            )
        if not approved:
            _log_access(directory, split, protocol_id, access_reason, "denied")
            raise PermissionError(
                "Final evaluation requires a matching frozen protocol, a listed checkpoint, "
                "and an access reason."
            )
    if _file_hash(directory / entry["file"]) != entry["sha256"]:
        if split in FINAL_SPLITS:
            _log_access(directory, split, protocol_id, access_reason, "checksum_failed")
        raise ValueError("Dataset checksum mismatch.")
    if split in FINAL_SPLITS:
        _log_access(directory, split, protocol_id, access_reason, "opened")
    return [
        Task.from_dict(json.loads(line))
        for line in (directory / entry["file"]).read_text().splitlines()
    ]
