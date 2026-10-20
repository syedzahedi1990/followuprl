"""Finite, fully specified state tracking; no model or evaluator dependency."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

LOCATIONS = tuple(range(6))


def _location(value: int) -> None:
    if type(value) is not int or value not in LOCATIONS:
        raise ValueError("A location must be an integer from 0 through 5.")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class State:
    world: int
    belief_a: int
    belief_b: int

    def __post_init__(self) -> None:
        for value in (self.world, self.belief_a, self.belief_b):
            _location(value)


@dataclass(frozen=True)
class Move:
    destination: int
    observed_by_a: bool
    observed_by_b: bool

    def __post_init__(self) -> None:
        _location(self.destination)
        if type(self.observed_by_a) is not bool or type(self.observed_by_b) is not bool:
            raise ValueError("Observation flags must be booleans.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination": self.destination,
            "observed_by_a": self.observed_by_a,
            "observed_by_b": self.observed_by_b,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Move":
        return cls(**value)


@dataclass(frozen=True)
class Episode:
    initial: int
    events: tuple[Move, ...]

    def __post_init__(self) -> None:
        _location(self.initial)
        object.__setattr__(self, "events", tuple(self.events))
        if not all(isinstance(event, Move) for event in self.events):
            raise ValueError("Events must be Move objects.")

    @property
    def episode_id(self) -> str:
        """Identity ignores rendering, question, split, and generator seed."""
        return _digest(self.to_dict())

    @property
    def skeleton(self) -> tuple[int, ...]:
        """Observation/dependency structure, without destination values."""
        return tuple(
            int(event.observed_by_a) + 2 * int(event.observed_by_b)
            for event in self.events
        )

    def to_dict(self) -> dict[str, Any]:
        return {"initial": self.initial, "events": [event.to_dict() for event in self.events]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Episode":
        return cls(value["initial"], tuple(Move.from_dict(x) for x in value["events"]))


@dataclass(frozen=True)
class Task:
    """A model-visible question and separate provenance.

    ``sibling_group`` is the canonical ancestor identity for a generated
    counterfactual sibling. Existing generator emits no counterfactual siblings;
    future generators must preserve this field across such expansions.
    ``variant`` affects wording only and never split assignment.
    """

    episode: Episode | None = None
    query: str = "A"
    family: str = "belief"
    operands: tuple[int, int] | None = None
    variant: int = 0
    sibling_group: str | None = None

    def __post_init__(self) -> None:
        if self.family not in {"belief", "world", "arithmetic"}:
            raise ValueError("Unknown task family.")
        if self.family == "arithmetic":
            if self.episode is not None or self.operands is None or len(self.operands) != 2:
                raise ValueError("Arithmetic needs two operands and no episode.")
            object.__setattr__(self, "operands", tuple(self.operands))
            if any(type(x) is not int or x < 0 for x in self.operands):
                raise ValueError("Arithmetic operands must be nonnegative integers.")
        else:
            if self.episode is None or self.operands is not None:
                raise ValueError("State-tracking tasks require an episode, without operands.")
            if self.query not in {"A", "B", "world"}:
                raise ValueError("Query must be A, B, or world.")
            if self.family == "belief" and self.query not in {"A", "B"}:
                raise ValueError("Belief queries must identify an observer.")
            if self.family == "world" and self.query != "world":
                raise ValueError("World tasks must use query='world'.")
        if type(self.variant) is not int or self.variant not in {0, 1}:
            raise ValueError("Rendering variant must be 0 or 1.")
        if self.sibling_group is not None and not self.sibling_group:
            raise ValueError("Sibling group cannot be empty.")

    @property
    def episode_id(self) -> str:
        if self.episode is not None:
            return self.episode.episode_id
        # Canonicalise commutative addition: a+b and b+a are siblings.
        return _digest({"operation": "addition_mod6", "operands": sorted(self.operands)})

    @property
    def group_id(self) -> str:
        return self.sibling_group or self.episode_id

    @property
    def task_id(self) -> str:
        return _digest(
            {"episode_id": self.episode_id, "query": self.query,
             "family": self.family, "variant": self.variant}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode.to_dict() if self.episode is not None else None,
            "query": self.query,
            "family": self.family,
            "operands": list(self.operands) if self.operands is not None else None,
            "variant": self.variant,
            "sibling_group": self.sibling_group,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Task":
        value = dict(value)
        if value.get("episode") is not None:
            value["episode"] = Episode.from_dict(value["episode"])
        if value.get("operands") is not None:
            value["operands"] = tuple(value["operands"])
        return cls(**value)


def transition(state: State, event: Move) -> State:
    return State(
        event.destination,
        event.destination if event.observed_by_a else state.belief_a,
        event.destination if event.observed_by_b else state.belief_b,
    )


def simulate(episode: Episode) -> State:
    state = State(episode.initial, episode.initial, episode.initial)
    for event in episode.events:
        state = transition(state, event)
    return state


def simulate_trace(episode: Episode) -> tuple[State, ...]:
    states = [State(episode.initial, episode.initial, episode.initial)]
    for event in episode.events:
        states.append(transition(states[-1], event))
    return tuple(states)
