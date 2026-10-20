"""Evidence-bound analysis reports. Missing comparisons remain unresolved."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import re

from paper2.analysis.aggregate import paired_seed_differences, utility_auc
from paper2.experiments.manifests import atomic_json


def analyse_manifest(manifest_path: str | Path, output_dir: str | Path | None = None) -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    output = Path(output_dir) if output_dir else manifest_path.parent / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    records = list(manifest.get("evaluations", []))
    if manifest.get("evaluations_path"):
        path = Path(manifest["evaluations_path"])
        if not path.exists():
            path = manifest_path.parent / path
        extra = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # Inline and JSONL copies are commonly the same observations. Exact
        # duplicates do not create additional nodes or statistical replication.
        identities = {json.dumps(row, sort_keys=True) for row in records}
        records.extend(row for row in extra if json.dumps(row, sort_keys=True) not in identities)
    notes = []
    if not records:
        notes.append("No observed evaluation records are present. No recovery or scientific effect is established.")
    groups = defaultdict(list)
    hardware_fallback = json.dumps(manifest.get("hardware"), sort_keys=True) if manifest.get("hardware") else "unspecified"
    backend_fallback = manifest.get("config", {}).get("backend", "unspecified")
    for row in records:
        # Mixing hardware changes the estimand. Stratify by hardware/backend.
        key = (
            row.get("hardware_id", manifest.get("hardware_id", hardware_fallback)),
            row.get("backend_id", manifest.get("backend_id", backend_fallback)),
            str(row.get("seed", manifest.get("seed", "unknown"))),
            row.get("arm", manifest.get("arm", "unknown")),
            row.get("suite", "development"),
            row.get("phase", "recovery"),
        )
        groups[key].append(row)
    budget = manifest.get("analysis", {}).get("auc_budget_seconds")
    auc_records = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        hardware, backend, seed, arm, suite, phase = key
        if budget is None:
            break
        points = [
            (r["additional_compute_seconds"], r["metrics"]["overall"]["p_correct"])
            for r in rows if r.get("additional_compute_seconds") is not None
            and r.get("metrics", {}).get("overall", {}).get("p_correct") is not None
        ]
        try:
            result = utility_auc(points, budget)
            auc_records.append(dict(result, hardware_id=hardware, backend_id=backend,
                                    seed=seed, arm=arm, suite=suite, phase=phase))
        except ValueError as exc:
            notes.append(f"AUC unavailable for {arm}, seed {seed}, {suite}, {phase}: {exc}")
    if records and budget is None:
        notes.append("AUC not computed: declare a common analysis.auc_budget_seconds before comparison.")
    comparison_groups = defaultdict(list)
    for row in auc_records:
        comparison_groups[(row["hardware_id"], row["backend_id"], row["suite"], row["phase"])].append(
            {"seed": row["seed"], "arm": row["arm"], "value": row["auc"]}
        )
    paired = {
        "|".join(map(str, key)): paired_seed_differences(value, manifest.get("analysis", {}).get("reference_arm", "continue"))
        for key, value in comparison_groups.items()
    }
    figures = []
    if records:
        # Never draw CPU engineering smoke measurements as GPU research curves.
        if manifest.get("test_only") or manifest.get("config", {}).get("model", {}).get("id") in {"toy", "tiny"}:
            notes.append("CPU/test-only measurements retained; publication research plots suppressed.")
        else:
            from paper2.analysis.plots import plot_recovery_curves
            figure_groups = defaultdict(list)
            for key, rows in groups.items():
                hardware, backend, _, _, suite, phase = key
                figure_groups[(hardware, backend, suite, phase)].extend(rows)
            for index, (stratum, rows) in enumerate(sorted(figure_groups.items(), key=lambda item: str(item[0]))):
                suite_phase = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{stratum[-2]}-{stratum[-1]}")
                figures.extend(plot_recovery_curves(rows, output / "figures" / f"{index:03d}-{suite_phase}"))
    summary = {
        "manifest": str(manifest_path), "run_status": manifest.get("status"),
        "evaluation_records": len(records), "auc_records": auc_records,
        "paired_comparisons": paired, "notes": notes, "figures": figures,
        "claim_ledger": [
            {"claim": "Vulnerable RL increases accepted errors", "status": "unresolved",
             "required": "Matched correct-verifier control and pre/post independently trained measurements."},
            {"claim": "Carrying optimizer state changes recovery", "status": "unresolved",
             "required": "Paired identical-policy branches with isolated optimizer treatment and uncertainty."},
            {"claim": "New recovery method improves on simple controls", "status": "unresolved",
             "required": "Information- and compute-matched exploration, reset, rollback and generic correction."},
            {"claim": "Result extends beyond the eight-action diagnostic", "status": "unresolved",
             "required": "Unrestricted-generation confirmation or materially different verifier defect."},
        ],
    }
    atomic_json(output / "summary.json", summary)
    report = [
        "# Project A evidence report", "",
        f"Run status: `{manifest.get('status', 'unknown')}`. Observed evaluation records: {len(records)}.",
        "",
        "This report describes available measurements. Engineering smoke tests do not establish learned exploitation or a recovery benefit.",
        "",
        "## Analysis limitations", "",
        *[f"- {note}" for note in notes],
        "",
        "## Claims awaiting scientific comparison", "",
        "| Claim | Status | Required evidence |", "|---|---|---|",
        *[f"| {r['claim']} | {r['status']} | {r['required']} |" for r in summary["claim_ledger"]],
        "",
        "Raw AUC points, interpolation choices and seed-paired differences are in `summary.json`.",
        "Missing/failed runs remain in the manifest. Costs must include provider idle/storage charges after reconciliation.",
        "",
    ]
    (output / "report.md").write_text("\n".join(report))
    return dict(summary, summary_path=str(output / "summary.json"), report_path=str(output / "report.md"))
