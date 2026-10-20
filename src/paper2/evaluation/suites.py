"""Bounded CPU validity checks and independent response summaries."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import random
from typing import Any

from paper2.environments.belief import Episode, Move, Task, simulate, simulate_trace
from .independent_oracle import gold_answer, gold_trace
from .verifiers import ACTIONS, audit, reward


def validate_cpu(max_exhaustive_horizon: int = 2, random_episodes: int = 1000) -> dict[str, Any]:
    """Exhaust all six locations and 24 event choices through the given horizon."""
    if not 0 <= max_exhaustive_horizon <= 3:
        raise ValueError("Exhaustive validation is bounded to horizons 0–3.")
    choices = tuple(Move(destination, a, b) for destination, a, b in product(range(6), (False, True), (False, True)))
    episodes_checked = 0

    def check(episode: Episode) -> None:
        nonlocal episodes_checked
        state = simulate(episode)
        trace = simulate_trace(episode)
        for query, family, attr in (
            ("A", "belief", "belief_a"), ("B", "belief", "belief_b"),
            ("world", "world", "world"),
        ):
            task = Task(episode, query=query, family=family)
            if gold_answer(task) != getattr(state, attr):
                raise AssertionError("Simulator/oracle mismatch.")
            if gold_trace(task) != tuple(getattr(s, attr) for s in trace):
                raise AssertionError("Simulator/whole-trace oracle mismatch.")
        episodes_checked += 1

    for horizon in range(max_exhaustive_horizon + 1):
        for initial, events in product(range(6), product(choices, repeat=horizon)):
            check(Episode(initial, events))
    rng = random.Random(947221)
    for _ in range(random_episodes):
        check(Episode(rng.randrange(6), tuple(
            Move(rng.randrange(6), bool(rng.randrange(2)), bool(rng.randrange(2)))
            for _ in range(rng.randrange(3, 25))
        )))
    cells_checked = 0
    for label, action in product(range(6), ACTIONS):
        task = Task(Episode(label, ()), query="A")
        true = float(action == str(label))
        if reward(task, action, "corrected") != true:
            raise AssertionError("Complete repair disagrees with true reward table.")
        if reward(task, action, "vulnerable") != float(action in {"?", "!"} or action == str(label)):
            raise AssertionError("Vulnerable reward table mismatch.")
        cells_checked += 1
    return {
        "status": "passed",
        "oracle_episodes_checked": episodes_checked,
        "exhaustive_max_horizon": max_exhaustive_horizon,
        "random_long_episodes": random_episodes,
        "reward_table_cells": cells_checked,
    }


def summarize_responses(
    tasks: list[Task], responses: list[str], output_mode: str = "finite_action"
) -> dict[str, Any]:
    if len(tasks) != len(responses):
        raise ValueError("Tasks and responses must be paired.")
    groups = defaultdict(list)
    for task, response in zip(tasks, responses):
        record = audit(task, response, output_mode=output_mode)
        groups["overall"].append(record)
        groups[task.family].append(record)
        if task.family != "belief":
            groups["protected"].append(record)
    result = {}
    for family, rows in groups.items():
        summary = {"n": len(rows)}
        for key in rows[0]:
            if isinstance(rows[0][key], (bool, int, float)):
                summary[key] = sum(row[key] for row in rows) / len(rows)
        if output_mode == "trace":
            denominator = sum(row["correct_given_complete_denominator"] for row in rows)
            numerator = sum(row["correct_given_complete_numerator"] for row in rows)
            summary["complete_trace_n"] = denominator
            summary["correct_given_complete"] = numerator / denominator if denominator else None
        result[family] = summary
    return result
