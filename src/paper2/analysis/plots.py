"""Publication export of actual, unsmoothed recovery measurements."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def plot_recovery_curves(records: list[dict], output_dir: str | Path) -> list[str]:
    if not records:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in records:
        if row.get("additional_compute_seconds") is not None:
            groups[(row.get("arm", "unknown"), row.get("seed", "unknown"))].append(row)
    if not groups:
        return []
    definitions = [
        ("p_correct", "True utility", "true_utility"),
        ("p_invalid", "Invalid-output probability", "invalid_probability"),
        ("conditional_valid_correct", "Correctness conditional on valid output", "conditional_valid"),
    ]
    paths = []
    for metric, ylabel, basename in definitions:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        plotted = False
        for (arm, seed), rows in sorted(groups.items(), key=lambda item: str(item[0])):
            points = sorted(
                (r["additional_compute_seconds"], r["metrics"]["overall"][metric])
                for r in rows if r.get("metrics", {}).get("overall", {}).get(metric) is not None
            )
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                        linewidth=1.2, markersize=3, label=f"{arm}; seed {seed}")
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        suites = {r.get("suite", "development") for r in records}
        ax.set_title(f"Project A — {', '.join(sorted(suites))}")
        ax.set_xlabel("Additional measured compute time (seconds)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
        fig.tight_layout()
        for extension in ("pdf", "svg", "png"):
            path = output_dir / f"{basename}.{extension}"
            fig.savefig(path, dpi=160 if extension == "png" else None)
            paths.append(str(path))
        plt.close(fig)
    return paths
