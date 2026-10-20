"""Render task facts only. Provenance and gold labels never enter the prompt."""

from .belief import Task


def render_prompt(task: Task, output_mode: str = "finite_action") -> str:
    if output_mode not in {"finite_action", "trace"}:
        raise ValueError("output_mode must be finite_action or trace.")
    if task.family == "arithmetic":
        if output_mode == "trace":
            raise ValueError("Arithmetic has no transition-trace certificate.")
        a, b = task.operands
        return (
            f"What is ({a} + {b}) modulo 6?\n"
            "Return exactly one digit from 0 to 5."
        )
    episode = task.episode
    lines = [
        "An object can be at locations 0, 1, 2, 3, 4, or 5.",
        f"Initially it is at {episode.initial}. A and B both know this.",
        "An observer changes their belief only after a move they observe.",
        "",
    ]
    for index, event in enumerate(episode.events, 1):
        if task.variant == 0:
            observers = (
                f"A {'observes' if event.observed_by_a else 'does not observe'}; "
                f"B {'observes' if event.observed_by_b else 'does not observe'}."
            )
            lines.append(f"Move {index}: the object moves to {event.destination}. {observers}")
        else:
            viewers = [
                name for name, sees in (("A", event.observed_by_a), ("B", event.observed_by_b))
                if sees
            ]
            observed = " and ".join(viewers) if viewers else "neither A nor B"
            lines.append(
                f"On event {index}, the object is moved to {event.destination}; "
                f"observed by {observed}."
            )
    lines.append("")
    if output_mode == "trace":
        subject = "the object's physical location" if task.family == "world" else f"{task.query}'s belief"
        lines.extend([
            f"Give the complete trace of {subject}.",
            f"The states list must contain exactly {len(episode.events) + 1} integers: "
            "the initial state, then the state after every event.",
            "Return exactly one JSON object with keys \"states\" and \"answer\".",
            "Every integer must lie from 0 to 5, and answer must equal the last states entry.",
            "Do not include any other fields or text.",
        ])
    else:
        lines.append(
            "Where is the object physically?"
            if task.family == "world"
            else f"Where does {task.query} believe the object is?"
        )
        lines.append("Return exactly one digit from 0 to 5.")
    return "\n".join(lines)
