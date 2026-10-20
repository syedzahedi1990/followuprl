"""Independent labels by retrospective search, never forward simulation.

This module deliberately imports only the Task data structure and JSON schema
parser at call time. In particular, it never calls transition/simulate or the
training verifier's transition checker.
"""

from __future__ import annotations

from paper2.environments.belief import Task


def _at_prefix(task: Task, stop: int) -> int:
    episode = task.episode
    if task.family == "world":
        return episode.initial if stop == 0 else episode.events[stop - 1].destination
    attribute = "observed_by_a" if task.query == "A" else "observed_by_b"
    # Search backwards separately for every query/prefix.
    last = next(
        (event.destination for event in reversed(episode.events[:stop])
         if getattr(event, attribute)),
        episode.initial,
    )
    return last


def gold_answer(task: Task) -> int:
    if task.family == "arithmetic":
        return sum(task.operands) % 6
    return _at_prefix(task, len(task.episode.events))


def gold_trace(task: Task) -> tuple[int, ...]:
    if task.family == "arithmetic":
        raise ValueError("Arithmetic does not have a state trace.")
    return tuple(_at_prefix(task, stop) for stop in range(len(task.episode.events) + 1))


def certificate_correct(task: Task, response: str) -> bool:
    """Validate all state values, not just the final answer.

    Schema is shared by the stated task, but semantic checks are independent of
    the training verifier. Invalid/truncated certificates are always false.
    """
    from .verifiers import InvalidAnswer, parse_certificate

    try:
        certificate = parse_certificate(response)
    except InvalidAnswer:
        return False
    expected = gold_trace(task)
    return tuple(certificate["states"]) == expected and certificate["answer"] == expected[-1]
