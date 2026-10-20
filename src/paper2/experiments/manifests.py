"""Run provenance, atomic manifest writes and verified durable export bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4


RUN_STATUSES = {
    "created", "running", "success", "budget_stop", "preemption",
    "numerical_failure", "scientific_stop", "error", "blocked",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


_SENSITIVE = re.compile(r"(^|_)(password|secret|credential|api_key|access_token|auth_token|hf_token|github_token)($|_)", re.I)


def redact_config(value, key: str = ""):
    """Do not persist supplied authentication fields or URL credentials.

    This is defense in depth, not permission to put credentials in configuration.
    Model token-count settings are scientific parameters and are retained.
    """
    if _SENSITIVE.search(key) or key.lower() in {"token", "authorization"}:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_config(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_config(v, key) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and "://" in value:
        parts = urlsplit(value)
        if parts.username or parts.password or parts.query or parts.fragment:
            return urlunsplit((parts.scheme, parts.hostname or "", parts.path, "", ""))
    return value


def runtime_inventory() -> dict:
    """Record installed package versions and hardware, without reading env vars."""
    names = [
        "torch", "transformers", "peft", "trl", "accelerate", "datasets",
        "tokenizers", "safetensors", "numpy", "scipy", "matplotlib", "pytest",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_logical_count": os.cpu_count(), "packages": versions,
        "gpu": [], "cuda_available": False,
    }
    if versions["torch"] is not None:
        import torch
        result["cuda_runtime"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["gpu"] = [
                {"index": i, "name": torch.cuda.get_device_name(i),
                 "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
                for i in range(torch.cuda.device_count())
            ]
            result["bf16_supported"] = torch.cuda.is_bf16_supported()
    return result


def create_manifest(
    config: dict,
    *,
    run_id: str,
    arm: str,
    seed: int,
    parent: dict | str | None = None,
    root: str | Path = ".",
    path: str | Path | None = None,
) -> dict:
    root = Path(root).resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    clean = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True, text=True, check=False,
    )
    source_hashes = {}
    for item in sorted((root / "src" / "paper2").rglob("*.py")):
        source_hashes[str(item.relative_to(root))] = sha256_file(item)
    lock_hashes = {
        item.name: sha256_file(item) for item in root.glob("*lock*") if item.is_file()
    }
    safe_config = redact_config(config)
    manifest = {
        "schema_version": 1, "project": config.get("project", "A"),
        "protocol_id": config.get("protocol_version"),
        "run_id": run_id, "arm": arm, "seed": seed, "parent": parent,
        "created_at_utc": utcnow(), "status": "created",
        "status_history": [{"status": "created", "timestamp_utc": utcnow()}],
        "config": safe_config, "config_sha256": canonical_hash(safe_config),
        "code_commit": result.stdout.strip() if result.returncode == 0 else None,
        "code_dirty": bool(clean.stdout.strip()) if clean.returncode == 0 else None,
        "source_sha256": source_hashes, "environment_lock_sha256": lock_hashes,
        "runtime": runtime_inventory(),
        "checkpoints": [], "evaluations": [], "deviations": [],
        "started_at_utc": None, "ended_at_utc": None,
        "processed_tokens": 0, "generated_tokens": 0, "gpu_seconds": None,
        "cash_usd": None, "scheduled_evaluation_points": [],
        "final_test_accessed": False,
    }
    if path is not None:
        if Path(path).exists():
            raise FileExistsError(f"Manifest already exists: {path}")
        atomic_json(path, manifest)
    return manifest


def update_manifest(path: str | Path, **fields) -> dict:
    path = Path(path)
    manifest = json.loads(path.read_text())
    if "run_id" in fields and fields["run_id"] != manifest["run_id"]:
        raise ValueError("Cannot rewrite a run identity")
    if "status" in fields:
        if fields["status"] not in RUN_STATUSES:
            raise ValueError(f"Unknown run status: {fields['status']}")
        manifest.setdefault("status_history", []).append(
            {"status": fields["status"], "timestamp_utc": utcnow()}
        )
    manifest.update(redact_config(fields))
    atomic_json(path, manifest)
    return manifest


def _check_relative(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or any(p in {"..", "."} for p in relative.parts):
        raise ValueError("Unsafe path in checksum index")
    return relative


def verify_snapshot(directory: str | Path) -> dict:
    """Verify complete training snapshot without loading serialized model code."""
    directory = Path(directory).resolve()
    marker = directory / "CHECKPOINT_COMPLETE"
    index = directory / "checksums.json"
    if not marker.is_file() or not index.is_file():
        raise ValueError(f"Incomplete checkpoint: {directory}")
    if marker.read_text().strip() != sha256_file(index):
        raise ValueError(f"Checkpoint checksum index/marker mismatch: {directory}")
    checksums = json.loads(index.read_text())
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Checkpoint checksum index must be a nonempty mapping")
    for name, expected in checksums.items():
        relative = _check_relative(name)
        file = directory / relative
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(directory):
            raise ValueError("Missing, symlinked, or escaping checkpoint file")
        if sha256_file(file) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {relative}")
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if actual != set(checksums) | {"checksums.json", "CHECKPOINT_COMPLETE"}:
        raise ValueError("Checkpoint contains files not covered by checksum index")
    return checksums


def _resolve_artifact(name: str, manifest_path: Path) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    # Existing runner paths are rooted at the repository's working directory.
    # A copied manifest can instead use paths relative to itself.
    if path.exists():
        return path.resolve()
    return (manifest_path.parent / path).resolve()


def _copy_plain(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError("Export refuses symlinks")
    if source.is_dir():
        for item in source.rglob("*"):
            if item.is_symlink():
                raise ValueError(f"Export refuses a symlink inside {source}")
        shutil.copytree(source, destination)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        raise FileNotFoundError(source)


def export_manifest(manifest_path: str | Path, destination: str | Path) -> dict:
    """Export an atomic, verified bundle to a user-configured durable directory.

    This verifies a filesystem copy, not the provider's retention policy. It does
    not upload to an unspecified cloud account or infer storage credentials.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    destination = Path(destination).resolve()
    if not destination.is_dir():
        raise ValueError("Export destination must be an existing configured directory")
    sources = []
    for i, item in enumerate(manifest.get("checkpoints", [])):
        name = item if isinstance(item, str) else item.get("path", item.get("checkpoint_path"))
        if not name:
            raise ValueError("Checkpoint entry lacks path")
        source = _resolve_artifact(name, manifest_path)
        verify_snapshot(source)
        sources.append((f"checkpoints/{i:04d}-{source.name}", source))
    for key in ("rollouts_path", "metrics_path", "evaluations_path", "budget_ledger_path", "budget_ledger"):
        if manifest.get(key):
            source = _resolve_artifact(manifest[key], manifest_path)
            if source not in [item[1] for item in sources]:
                sources.append((f"artifacts/{key}/{source.name}", source))
    if manifest.get("run_directory"):
        run_directory = _resolve_artifact(manifest["run_directory"], manifest_path)
        for name in ("rollouts.jsonl", "steps.jsonl", "evaluations.jsonl"):
            source = run_directory / name
            if source.is_file() and source not in [item[1] for item in sources]:
                sources.append((f"artifacts/run/{name}", source))
    dataset_path = manifest.get("config", {}).get("data", {}).get("path")
    if dataset_path:
        source = _resolve_artifact(dataset_path, manifest_path)
        if not (source / "manifest.json").is_file():
            raise ValueError("Configured dataset has no manifest; export would not support reproducible restoration")
        sources.append(("dataset", source))
    for source_name in manifest.get("artifacts", []):
        source = _resolve_artifact(source_name if isinstance(source_name, str) else source_name["path"], manifest_path)
        sources.append((f"artifacts/extra-{len(sources):04d}/{source.name}", source))
    for _, source in sources:
        if destination == source or destination.is_relative_to(source):
            raise ValueError("Export destination must not be inside a source directory")
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(manifest.get("run_id", "experiment")))
    label = f"{run_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    target = destination / label
    temporary = Path(tempfile.mkdtemp(prefix=".paper2-export-", dir=destination))
    try:
        _copy_plain(manifest_path, temporary / "manifest.json")
        export_sources = {}
        for relative, source in sources:
            target_item = temporary / relative
            target_item.parent.mkdir(parents=True, exist_ok=True)
            _copy_plain(source, target_item)
            if relative.startswith("checkpoints/"):
                verify_snapshot(target_item)
            export_sources[relative] = str(source)
        atomic_json(temporary / "export_metadata.json", {
            "exported_at_utc": utcnow(), "source_manifest_sha256": sha256_file(manifest_path),
            "source_paths": export_sources,
            "durability": "Filesystem copy verified; external retention must be configured by operator.",
        })
        checksums = {
            str(item.relative_to(temporary)): sha256_file(item)
            for item in sorted(temporary.rglob("*")) if item.is_file()
        }
        atomic_json(temporary / "SHA256SUMS.json", checksums)
        for name, digest in checksums.items():
            if sha256_file(temporary / name) != digest:
                raise ValueError("Export copy failed verification")
        (temporary / "EXPORT_COMPLETE").write_text(sha256_file(temporary / "SHA256SUMS.json") + "\n")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return {
        "export_path": str(target), "verified_files": len(checksums),
        "snapshot_count": len(manifest.get("checkpoints", [])), "status": "verified",
        "source_manifest_sha256": sha256_file(manifest_path),
    }


def verify_export(directory: str | Path) -> dict:
    directory = Path(directory).resolve()
    index = directory / "SHA256SUMS.json"
    marker = directory / "EXPORT_COMPLETE"
    if not index.is_file() or not marker.is_file() or marker.read_text().strip() != sha256_file(index):
        raise ValueError("Incomplete or corrupted export index")
    checksums = json.loads(index.read_text())
    for name, digest in checksums.items():
        relative = _check_relative(name)
        file = directory / relative
        if file.is_symlink() or not file.resolve().is_relative_to(directory):
            raise ValueError("Unsafe exported file")
        if not file.is_file() or sha256_file(file) != digest:
            raise ValueError(f"Exported file checksum mismatch: {name}")
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if actual != set(checksums) | {"SHA256SUMS.json", "EXPORT_COMPLETE"}:
        raise ValueError("Export has unexpected unchecksummed files")
    return {"status": "verified", "files": len(checksums), "path": str(directory)}


def restore_export(
    bundle: str | Path,
    destination: str | Path,
    *,
    ledger_path: str | Path | None = None,
) -> dict:
    """Restore a verified bundle to a new explicit location without editing anchors.

    ``destination`` must not exist. The returned JSON configuration names the
    relocated run directory, dataset and cumulative ledger. An explicitly named
    existing ledger must contain the exported ledger as an exact verified prefix;
    newer accounting is preserved. Runtime authorization/rate/durability for a
    paid or Colab machine must be configured again for the actual new runtime.
    """
    from paper2.experiments.budget import BudgetLedger

    bundle = Path(bundle).resolve()
    verify_export(bundle)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Restore requires a new destination; existing runs are never overwritten")
    if destination.is_relative_to(bundle) or bundle.is_relative_to(destination):
        raise ValueError("Restore destination and immutable export must be separate")
    original = json.loads((bundle / "manifest.json").read_text())
    exported = json.loads((bundle / "export_metadata.json").read_text())
    source_paths = exported.get("source_paths", {})
    checkpoint_sources = [name for name in source_paths if name.startswith("checkpoints/")]
    if len(checkpoint_sources) != len(original.get("checkpoints", [])):
        raise ValueError("Export checkpoint list does not match the original manifest")
    for name in checkpoint_sources:
        verify_snapshot(bundle / _check_relative(name))

    # Only a complete original cumulative ledger can reestablish budget offsets.
    ledger_candidates = [
        bundle / name for name in source_paths
        if name.startswith("artifacts/budget_ledger/") or name.startswith("artifacts/budget_ledger_path/")
    ]
    if len(ledger_candidates) != 1:
        raise ValueError("Restoration requires exactly one exported cumulative budget ledger")
    exported_ledger = ledger_candidates[0]
    # Validate without creating a lock file inside the immutable export.
    inspector = object.__new__(BudgetLedger)
    inspector.path = exported_ledger
    exported_entries = inspector._read()
    exported_bytes = exported_ledger.read_bytes()
    existing_ledger = Path(ledger_path).resolve() if ledger_path is not None else None
    if existing_ledger is not None:
        if not existing_ledger.is_file():
            raise ValueError("An explicit ledger_path must name an existing verified ledger")
        with BudgetLedger(existing_ledger)._locked() as current_entries:
            if len(current_entries) < len(exported_entries) or any(
                a["entry_sha256"] != b["entry_sha256"]
                for a, b in zip(exported_entries, current_entries)
            ):
                raise ValueError("Existing ledger does not extend the exported accounting history; reconcile it explicitly")
        if existing_ledger.is_relative_to(destination):
            raise ValueError("Existing ledger cannot be inside a not-yet-created restoration directory")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".paper2-restore-", dir=destination.parent))
    final_run = destination / "run"
    temporary_run = temporary / "run"
    temporary_run.mkdir()
    mapping = {}
    restored = json.loads(json.dumps(original))
    restored["checkpoints"] = []
    try:
        for name, entry in zip(checkpoint_sources, original.get("checkpoints", [])):
            source = bundle / name
            metadata = json.loads((source / "metadata.json").read_text())
            category = "curve" if metadata.get("inference_only") else "checkpoints"
            step = metadata.get("phase_step")
            leaf = f"step-{int(step):06d}" if step is not None else source.name
            relative = Path(category) / leaf
            target = temporary_run / relative
            if target.exists():
                raise ValueError("Export contains conflicting checkpoint locations")
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_plain(source, target)
            verify_snapshot(target)
            new_path = str(final_run / relative)
            old_path = entry if isinstance(entry, str) else entry.get("path", entry.get("checkpoint_path"))
            mapping[old_path] = new_path
            mapping[source_paths[name]] = new_path
            updated = dict(entry) if isinstance(entry, dict) else {}
            updated["path"] = new_path
            updated["original_path"] = old_path
            updated["checksums_sha256"] = sha256_file(target / "checksums.json")
            restored["checkpoints"].append(updated)

        for name in ("rollouts.jsonl", "steps.jsonl", "evaluations.jsonl"):
            artifact = bundle / "artifacts" / "run" / name
            if artifact.is_file():
                _copy_plain(artifact, temporary_run / name)
        # Older compatible exports may have named paths instead of run_directory.
        for key, filename in (
            ("rollouts_path", "rollouts.jsonl"), ("metrics_path", "steps.jsonl"),
            ("evaluations_path", "evaluations.jsonl"),
        ):
            candidates = [bundle / name for name in source_paths if name.startswith(f"artifacts/{key}/")]
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous exported {key}")
            if candidates and not (temporary_run / filename).exists():
                _copy_plain(candidates[0], temporary_run / filename)
            if key in restored and (temporary_run / filename).exists():
                restored[key] = str(final_run / filename)

        if existing_ledger is None:
            accounting = temporary / "accounting"
            accounting.mkdir()
            (accounting / "budget.jsonl").write_bytes(exported_bytes)
            final_ledger = destination / "accounting" / "budget.jsonl"
        else:
            final_ledger = existing_ledger

        config = json.loads(json.dumps(original.get("config", {})))
        if not config:
            raise ValueError("Exported manifest lacks the resolved run configuration")
        if config.get("data", {}).get("path"):
            if not (bundle / "dataset" / "manifest.json").is_file():
                raise ValueError("Dataset absent from this older export; prepare a complete new export")
            _copy_plain(bundle / "dataset", temporary / "dataset")
            config["data"]["path"] = str(destination / "dataset")
        config.setdefault("run", {})["restore_directory"] = str(final_run)
        config["output_dir"] = str(destination / "subsequent-runs")
        config.setdefault("budget", {})["ledger_path"] = str(final_ledger)
        tiny = config.get("backend") in {"tiny", "toy"} or original.get("test_only")
        if not tiny:
            config["budget"].update({
                "runtime_authorized": False, "hourly_rate_usd": None,
                "durability_confirmed": False, "durable_destination": None,
            })
        for key in ("final_checkpoint", "checkpoint"):
            if restored.get(key):
                if restored[key] not in mapping:
                    raise ValueError("Manifest final checkpoint is not included in the verified export")
                restored[key] = mapping[restored[key]]
        for row in restored.get("evaluations", []):
            if row.get("checkpoint") in mapping:
                row["original_checkpoint"] = row["checkpoint"]
                row["checkpoint"] = mapping[row["checkpoint"]]
        restored["config"] = config
        restored["config_sha256"] = canonical_hash(config)
        restored["run_directory"] = str(final_run)
        restored["manifest_path"] = str(final_run / "manifest.json")
        restored["budget_ledger"] = str(final_ledger)
        if "budget_ledger_path" in restored:
            restored["budget_ledger_path"] = str(final_ledger)
        restore_metadata = {
            "restored_at_utc": utcnow(),
            "source_bundle": str(bundle),
            "source_export_index_sha256": sha256_file(bundle / "SHA256SUMS.json"),
            "source_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "original_run_directory": original.get("run_directory"),
            "run_id": original.get("run_id"),
            "checkpoint_path_mapping": mapping,
            "checkpoint_files_modified": False,
            "raw_historical_logs_modified": False,
            "ledger_source": str(existing_ledger) if existing_ledger else "byte-for-byte exported history",
            "exported_ledger_entries": len(exported_entries),
            "accounting_limit": "Reconcile any provider/other-run charges incurred after this export before paid continuation.",
        }
        restored["restore_metadata"] = restore_metadata
        restored.setdefault("deviations", []).append({
            "kind": "explicit_storage_relocation", "details": restore_metadata,
        })
        _copy_plain(bundle / "manifest.json", temporary / "original_manifest.json")
        atomic_json(temporary_run / "manifest.json", restored)
        atomic_json(temporary / "restore.json", restore_metadata)
        atomic_json(temporary / "restore-config.json", config)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return {
        "status": "restored",
        "run_directory": str(final_run),
        "manifest_path": str(final_run / "manifest.json"),
        "config_path": str(destination / "restore-config.json"),
        "ledger_path": str(final_ledger),
        "checkpoint_paths": [entry["path"] for entry in restored["checkpoints"]],
        "final_checkpoint": restored.get("final_checkpoint"),
        "snapshot_count": len(restored["checkpoints"]),
        "requires_runtime_reauthorization": not tiny,
    }
