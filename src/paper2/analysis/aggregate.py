"""Exact categorical metrics and paired recovery-curve summaries."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Iterable, Sequence


def _group_metrics(rows: list[tuple[list[float], int]], mass_tolerance: float) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "p_correct": None, "p_invalid": None, "p_valid": None,
                "conditional_valid_correct": None, "valid_only_ranking_accuracy": None,
                "mean_policy_entropy": None, "conditional_valid_denominator": 0.0,
                "conditional_valid_prompt_count": 0}
    correct = [p[y] for p, y in rows]
    valid = [sum(p[:6]) for p, _ in rows]
    invalid = [sum(p[6:]) for p, _ in rows]
    stable = [(p[y] / v) for (p, y), v in zip(rows, valid) if v > mass_tolerance]
    mass = sum(valid)
    # A task-uniform deployment followed by conditioning on a valid response
    # weights prompts by their valid mass. Report task-uniform conditionals too.
    return {
        "n": n, "p_correct": mean(correct), "p_invalid": mean(invalid),
        "p_valid": mean(valid),
        "conditional_valid_correct": sum(correct) / mass if mass > n * mass_tolerance else None,
        "conditional_valid_denominator": mass,
        "conditional_valid_mean_per_prompt": mean(stable) if stable else None,
        "conditional_valid_prompt_count": len(stable),
        "conditional_valid_negligible_prompts": n - len(stable),
        "valid_only_ranking_accuracy": mean(float(max(range(6), key=lambda i: p[i]) == y) for p, y in rows),
        "valid_only_tie_rule": "smallest digit (argmax first)",
        "mean_policy_entropy": mean(-sum(value * math.log(value) for value in p if value > 0) for p, _ in rows),
        "mass_tolerance": mass_tolerance,
    }


def finite_metrics(
    probabilities: Iterable[Sequence[float]],
    gold: Iterable[int],
    families: Iterable[str] | None = None,
    *,
    mass_tolerance: float = 1e-12,
) -> dict:
    """Metrics of the underlying 8-action softmax at the common evaluation rule.

    Pass q separately to report exploration behavior. This function never
    silently renormalizes probabilities or drops invalid actions.
    """
    ps = [list(map(float, row)) for row in probabilities]
    ys = list(gold)
    fs = list(families) if families is not None else ["belief"] * len(ps)
    if len(ps) != len(ys) or len(fs) != len(ps):
        raise ValueError("Probability, label, and family lengths must agree")
    if not math.isfinite(mass_tolerance) or mass_tolerance <= 0:
        raise ValueError("mass_tolerance must be positive and finite")
    groups: dict[str, list] = defaultdict(list)
    all_rows = []
    for p, y, family in zip(ps, ys, fs):
        if len(p) != 8 or any(not math.isfinite(x) or x < 0 for x in p):
            raise ValueError("Each finite policy row must contain eight finite nonnegative probabilities")
        if not math.isclose(sum(p), 1.0, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("Categorical probabilities must sum to one")
        if isinstance(y, bool) or not isinstance(y, int) or y not in range(6):
            raise ValueError("Gold answers must be integer digits 0 through 5")
        all_rows.append((p, y))
        groups[str(family)].append((p, y))
    return {
        "overall": _group_metrics(all_rows, mass_tolerance),
        "by_family": {key: _group_metrics(rows, mass_tolerance) for key, rows in sorted(groups.items())},
        "evaluation_policy": "caller-supplied; primary must be underlying p at temperature 1",
    }


def utility_auc(points: Iterable[Sequence[float]], budget: float) -> dict:
    """Normalized trapezoidal AUC on [0,B], interpolating but never extrapolating.

    At least one observed point must cover B. Equal compute coordinates are
    rejected rather than assigning an arbitrary duration to duplicate nodes.
    """
    budget = float(budget)
    data = sorted((float(x), float(y)) for x, y in points)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("AUC budget must be finite and positive")
    if len(data) < 2 or any(not math.isfinite(x) or not math.isfinite(y) for x, y in data):
        raise ValueError("AUC needs at least two finite measured points")
    if any(x < 0 or y < 0 or y > 1 for x, y in data):
        raise ValueError("Compute must be nonnegative and utility must lie in [0,1]")
    if data[0][0] != 0:
        raise ValueError("Recovery AUC requires each arm's measured initial state at c=0")
    if data[-1][0] < budget:
        raise ValueError("Observed curve does not cover B; no extrapolation is allowed")
    if any(data[i][0] <= data[i - 1][0] for i in range(1, len(data))):
        raise ValueError("Compute coordinates must be strictly increasing")
    selected = [data[0]]
    for left, right in zip(data, data[1:]):
        if right[0] < budget:
            selected.append(right)
        else:
            fraction = (budget - left[0]) / (right[0] - left[0])
            selected.append((budget, left[1] + fraction * (right[1] - left[1])))
            break
    area = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(selected, selected[1:]))
    return {
        "auc": area / budget,
        "budget": budget,
        "integration": "linear interpolation with trapezoids; no extrapolation",
        "raw_points": [list(p) for p in data],
        "integration_points": [list(p) for p in selected],
    }


def paired_seed_differences(values: Iterable[dict], reference_arm: str) -> dict:
    """Retain independent training seeds as the comparison unit, never episodes."""
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for record in values:
        seed, arm = str(record["seed"]), str(record["arm"])
        if arm in grouped[seed]:
            raise ValueError(f"Duplicate seed/arm summary: {seed}/{arm}")
        value = float(record["value"])
        if not math.isfinite(value):
            raise ValueError("Paired summaries must be finite")
        grouped[seed][arm] = value
    arms = sorted({arm for data in grouped.values() for arm in data if arm != reference_arm})
    result = {}
    for arm in arms:
        paired = [
            {"seed": seed, "arm_value": vals[arm], "reference_value": vals[reference_arm],
             "difference": vals[arm] - vals[reference_arm]}
            for seed, vals in sorted(grouped.items()) if arm in vals and reference_arm in vals
        ]
        result[arm] = {
            "paired_seeds": paired,
            "n_independent_training_seeds": len(paired),
            "mean_paired_difference": mean(row["difference"] for row in paired) if paired else None,
            "unpaired_seeds": sorted(seed for seed, vals in grouped.items() if (arm in vals) != (reference_arm in vals)),
            "population_interval": None,
            "uncertainty_note": "Raw seed differences shown; no episode-level pseudoreplication or significance claim.",
        }
    return {"reference_arm": reference_arm, "arms": result}
