"""CLI shared by cloud, Vast and Colab. No command rents or publishes compute."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

from .config import load_config


def emit(result: object) -> None:
    print(json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False))


def _write_new(path: str | Path, contents: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(contents, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _config_argument(parser: argparse.ArgumentParser, default=None) -> None:
    parser.add_argument("--config", required=default is None, default=default)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version="paper2 0.1.0")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "preflight", "resume-check", "train"):
        command = sub.add_parser(name)
        _config_argument(command)
        if name == "validate":
            command.add_argument("--cpu-only", action="store_true")
        elif name == "resume-check":
            command.add_argument("--output-dir")
            command.add_argument("--walltime-minutes", type=float)
        elif name == "preflight":
            command.add_argument("--walltime-minutes", type=float)
        elif name == "train":
            command.add_argument("--phase", required=True, choices=["clean", "vulnerable", "control", "recovery"])
            command.add_argument("--seed", type=int)
            command.add_argument("--resume")
            command.add_argument("--parent")
            command.add_argument("--arm")
            command.add_argument("--output-dir")
            command.add_argument("--walltime-minutes", type=float)

    command = sub.add_parser("resolve-model")
    _config_argument(command)
    command.add_argument("--output", required=True)

    command = sub.add_parser("freeze-protocol")
    _config_argument(command)
    command.add_argument("--protocol-id", required=True)
    command.add_argument("--definition", required=True)

    for name in ("branch", "run-plan"):
        command = sub.add_parser(name)
        command.add_argument("--plan", required=True)
        command.add_argument("--seed", type=int)

    command = sub.add_parser("evaluate")
    command.add_argument("--manifest", required=True)
    command.add_argument("--suite", default="development", choices=["development", "final_test", "structural_test"])
    command.add_argument("--config")
    command.add_argument("--checkpoint")
    command.add_argument("--protocol-id")
    command.add_argument("--access-reason")
    command.add_argument("--output")
    command = sub.add_parser("analyse")
    command.add_argument("--manifest", required=True)
    command.add_argument("--output-dir")
    command = sub.add_parser("export")
    command.add_argument("--manifest", required=True)
    command.add_argument("--destination", required=True)
    command = sub.add_parser("restore")
    command.add_argument("--bundle", required=True)
    command.add_argument("--destination", required=True)
    command.add_argument("--ledger-path")
    return p


def _resolve_model(args: argparse.Namespace) -> dict:
    import re
    import yaml
    from huggingface_hub import HfApi

    config = load_config(args.config)
    if Path(args.output).exists():
        raise ValueError("Refusing to overwrite an existing resolved configuration")
    model = config["model"]
    api = HfApi()
    model_info = api.model_info(model["id"], revision=model.get("revision") or "main")
    tokenizer_info = api.model_info(model["id"], revision=model.get("tokenizer_revision") or model_info.sha)
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha or "") for sha in [model_info.sha, tokenizer_info.sha]):
        raise ValueError("Hub did not return exact model and tokenizer commit SHAs")
    model["revision"] = model_info.sha
    model["tokenizer_revision"] = tokenizer_info.sha
    card = model_info.card_data
    model["license"] = card.get("license") if card else None
    if not model["license"]:
        raise ValueError("Model license metadata missing; inspect the model card before a paid run")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return {"status": "resolved", "config": str(path), "model": model["id"],
            "revision": model_info.sha, "tokenizer_revision": tokenizer_info.sha,
            "license": model["license"], "weights_downloaded": False}


def _bounded_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    cap = getattr(args, "walltime_minutes", None)
    if cap is not None:
        if cap <= 0:
            raise ValueError("Walltime cap must be positive")
        configured = config.setdefault("budget", {}).get("job_walltime_cap_minutes")
        if configured and cap > configured:
            raise ValueError("CLI walltime can shorten, but cannot increase, the configured cap")
        config["budget"]["job_walltime_cap_minutes"] = cap
    return config


def _branch_plan(args: argparse.Namespace, run: bool) -> dict:
    from .training.runner import create_branch, train_phase
    from .training.checkpoints import verify_checkpoint

    plan = json.loads(Path(args.plan).read_text())
    config = load_config(plan["config"])
    engineering = config.get("backend") in ("tiny", "toy")
    if bool(plan.get("engineering_only")) != engineering:
        raise ValueError("Plan engineering_only must match its configured backend")
    if not engineering:
        evidence = plan.get("learned_exploitation_evidence")
        if not evidence or not Path(evidence).is_file():
            raise ValueError("Recovery runs require recorded learned-exploitation evidence against a clean control")
        gate = json.loads(Path(evidence).read_text())
        if gate.get("status") != "passed" or not gate.get("matched_control_manifest"):
            raise ValueError("Learned-exploitation gate has not passed with a matched clean control")
    seed = args.seed if args.seed is not None else int(plan.get("seed", config.get("seed", 101)))
    output = Path(plan["output_dir"])
    result = {"seed": seed, "engineering_only": engineering, "branches": []}
    for arm in plan["arms"]:
        branch_path = output / f"seed-{seed}" / arm / "initial"
        if arm == "clean_reference":
            parent = plan.get("control_checkpoint")
            if not parent:
                raise ValueError("clean_reference requires the matched correct-verifier control_checkpoint")
        elif arm == "rollback":
            parent = plan["clean_checkpoint"]
        else:
            parent = plan["affected_checkpoint"]
        if branch_path.exists():
            metadata = verify_checkpoint(branch_path)
            if (metadata.get("seed") != seed or metadata.get("arm") != arm
                    or Path(metadata.get("parent", "")).resolve() != Path(parent).resolve()
                    or metadata.get("config") != config):
                raise ValueError(f"Existing branch disagrees with this plan: {branch_path}")
            actual = str(branch_path)
        else:
            actual = create_branch(config, parent, str(branch_path), arm, seed,
                                   incident_parent=plan["affected_checkpoint"])
        entry = {"arm": arm, "checkpoint": str(actual)}
        if run:
            arm_config = copy.deepcopy(config)
            arm_config["output_dir"] = str(output / f"seed-{seed}" / arm / "run")
            if list(Path(arm_config["output_dir"]).glob("*/manifest.json")):
                raise ValueError(f"Arm {arm} already has a run; inspect its manifest and resume explicitly")
            entry["run"] = train_phase(arm_config, "recovery", seed, parent=str(actual), arm=arm)
        result["branches"].append(entry)
    return result


def _evaluate(args: argparse.Namespace) -> dict:
    from .experiments.data import load_tasks
    from .training.runner import evaluate_checkpoint

    manifest = json.loads(Path(args.manifest).read_text())
    config = load_config(args.config) if args.config else manifest.get("resolved_config", manifest.get("config"))
    if not isinstance(config, dict):
        raise ValueError("Manifest needs a resolved config, or supply --config")
    checkpoint = args.checkpoint or manifest.get("final_checkpoint") or manifest.get("checkpoint")
    if not checkpoint:
        raise ValueError("Specify an immutable --checkpoint to evaluate")
    tasks = load_tasks(config["data"]["path"], args.suite, protocol_id=args.protocol_id,
                       access_reason=args.access_reason, evaluation_checkpoint=checkpoint)
    result = evaluate_checkpoint(config, checkpoint, tasks)
    result.update({"suite": args.suite, "checkpoint": str(checkpoint),
                   "protocol_id": args.protocol_id, "manifest": args.manifest})
    if args.output:
        _write_new(args.output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            from .experiments.data import prepare_dataset
            result = prepare_dataset(load_config(args.config))
        elif args.command == "validate":
            from .evaluation.suites import validate_cpu
            from .experiments.data import prepare_dataset
            config = load_config(args.config)
            dataset = prepare_dataset(config)
            result = {"status": "passed", "oracles_and_verifiers": validate_cpu(), "dataset": dataset}
        elif args.command == "preflight":
            from .training.runner import preflight
            result = preflight(_bounded_config(args))
        elif args.command == "resume-check":
            from .training.runner import resume_check
            result = resume_check(_bounded_config(args), output_dir=args.output_dir)
        elif args.command == "train":
            from .training.runner import train_phase
            config = _bounded_config(args)
            if args.output_dir:
                config["output_dir"] = args.output_dir
            result = train_phase(config, args.phase,
                                 args.seed if args.seed is not None else config.get("seed", 101),
                                 resume=args.resume, parent=args.parent, arm=args.arm)
        elif args.command == "resolve-model":
            result = _resolve_model(args)
        elif args.command == "freeze-protocol":
            from .experiments.data import freeze_final_protocol
            config = load_config(args.config)
            definition = json.loads(Path(args.definition).read_text())
            result = freeze_final_protocol(config["data"]["path"], args.protocol_id, definition)
        elif args.command in ("branch", "run-plan"):
            result = _branch_plan(args, run=args.command == "run-plan")
        elif args.command == "evaluate":
            result = _evaluate(args)
        elif args.command == "analyse":
            from .analysis.report import analyse_manifest
            result = analyse_manifest(args.manifest, output_dir=args.output_dir)
        elif args.command == "export":
            from .experiments.manifests import export_manifest
            result = export_manifest(args.manifest, args.destination)
        elif args.command == "restore":
            from .experiments.manifests import restore_export
            result = restore_export(args.bundle, args.destination, ledger_path=args.ledger_path)
        else:
            raise ValueError(f"Unknown command {args.command}")
        emit(result)
        if isinstance(result, dict) and result.get("status") in ("failed", "numerical_failure", "blocked"):
            return 1
        return 0
    except KeyboardInterrupt:
        emit({"status": "interrupted", "error": "Interrupted; inspect run manifest for completed checkpoints"})
        return 130
    except Exception as exc:
        emit({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
        if os.environ.get("PAPER2_DEBUG") == "1":
            raise
        return 2


if __name__ == "__main__":
    sys.exit(main())
