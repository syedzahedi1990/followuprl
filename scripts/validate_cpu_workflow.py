#!/usr/bin/env python3
"""Exercise the shared CLI end to end using a synthetic CPU engineering policy.

No result from this script is evidence about a pretrained language model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    root = Path("runs") / f"cpu-workflow-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True)
    outputs = []
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}

    def invoke(*args):
        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-m", "paper2", *map(str, args)],
            text=True, capture_output=True, env=environment, timeout=180,
        )
        index = len(outputs)
        (root / f"command-{index:02d}.stdout.json").write_text(proc.stdout)
        (root / f"command-{index:02d}.stderr.log").write_text(proc.stderr)
        outputs.append({"command": list(map(str, args)), "returncode": proc.returncode,
                        "elapsed_seconds": time.monotonic() - start})
        if proc.returncode:
            raise RuntimeError(f"{args}: {proc.stdout}\n{proc.stderr}")
        return json.loads(proc.stdout)

    config = "configs/cpu_smoke.yaml"
    invoke("prepare", "--config", config)
    validation = invoke("validate", "--config", config, "--cpu-only")
    preflight = invoke("preflight", "--config", config)
    resume = invoke("resume-check", "--config", "configs/resume_smoke.yaml",
                    "--output-dir", root / "resume")
    clean = invoke("train", "--config", config, "--phase", "clean",
                   "--seed", 101, "--output-dir", root / "clean")
    affected = invoke("train", "--config", config, "--phase", "vulnerable",
                      "--seed", 101, "--parent", clean["final_checkpoint"],
                      "--output-dir", root / "vulnerable")
    control = invoke("train", "--config", config, "--phase", "control",
                     "--seed", 101, "--parent", clean["final_checkpoint"],
                     "--output-dir", root / "control")
    plan = {
        "config": config, "seed": 101, "engineering_only": True,
        "clean_checkpoint": clean["final_checkpoint"],
        "affected_checkpoint": affected["final_checkpoint"],
        "control_checkpoint": control["final_checkpoint"],
        "output_dir": str(root / "recovery"),
        "arms": ["continue", "reset_optimizer", "rollback",
                 "continue_exploration", "clean_reference"],
    }
    plan_path = root / "recovery_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    recoveries = invoke("run-plan", "--plan", plan_path)
    manifests = [clean, affected, control] + [item["run"] for item in recoveries["branches"]]
    evaluated = []
    for manifest in manifests:
        result = invoke("evaluate", "--manifest", manifest["manifest_path"], "--suite", "development")
        evaluated.append({"run_id": manifest["run_id"], "status": manifest["status"],
                          "phase": manifest["phase"], "arm": manifest["arm"],
                          "metrics": result})
    analysis = invoke("analyse", "--manifest", manifests[3]["manifest_path"])
    (root / "local-export-check").mkdir()
    exported = invoke("export", "--manifest", manifests[3]["manifest_path"],
                      "--destination", root / "local-export-check")
    # Export to this same machine checks integrity only, not persistence after VM loss.
    report = {
        "status": "passed", "test_only": True, "scientific_results": False,
        "runtime": "CPU synthetic text-hash policy",
        "source_manifest_directory": str(root), "commands": outputs,
        "oracles_and_verifiers": validation["oracles_and_verifiers"],
        "preflight": preflight, "resume": resume,
        "runs": evaluated, "analysis": analysis, "local_export": exported,
        "durability_verified": False,
        "final_test_opened": False,
        "limitations": [
            "No pretrained weights/tokenizer or CUDA execution were available.",
            "Synthetic-policy curves do not establish learned LM exploitation or recovery.",
            "Local export verification is not proof of external durable storage.",
            "The final evaluation sets remain unopened.",
        ],
    }
    (root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/cpu_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "runs": len(manifests),
                      "report": "reports/cpu_validation.json", "artifacts": str(root)}, indent=2))


if __name__ == "__main__":
    main()
