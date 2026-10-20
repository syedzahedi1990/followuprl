"""Private fresh-process CPU validation worker; no language-model claims."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from .checkpoints import load_checkpoint, save_checkpoint
from .finite_grpo import training_step
from .runner import PromptSchedule, _build_policy, _optimizer, _seed, is_tiny
from .toy import TinyPolicy


def main() -> None:
    config_path, mode, output_path, *rest = sys.argv[1:]
    config = json.loads(Path(config_path).read_text())
    torch.set_num_threads(1)
    seed = int(config.get("seed", 101))
    _seed(seed)
    policy = _build_policy(config, seed)
    optimizer = _optimizer(policy, config)
    generator = torch.Generator().manual_seed(171)
    if is_tiny(config):
        all_prompts = [f"debug prompt {i}" for i in range(23)]
        gold = [i % 6 for i in range(23)]
    else:
        from paper2.environments.rendering import render_prompt
        from paper2.evaluation.independent_oracle import gold_answer
        from paper2.experiments.data import load_tasks
        tasks = load_tasks(config.get("data", {}).get("path", "data/project_a"), "train")
        all_prompts = [render_prompt(t) for t in tasks[:32]]
        gold = [gold_answer(t) for t in tasks[:32]]
        policy.validate_prompt(all_prompts[0])
    schedule = PromptSchedule(len(all_prompts), 161)
    cursor = 0
    if mode == "resume":
        metadata = load_checkpoint(rest[0], policy, optimizer, generator=generator,
                                   numpy_generator=schedule.generator,
                                   expected_contract="tiny-debug-v1")
        schedule.restore(metadata["sampler"])
        cursor = metadata["phase_step"]
    stop = int(config["resume_check"]["interrupt_after" if mode == "partial" else "updates"])
    records = []
    while cursor < stop:
        indices = schedule.next(min(4, len(all_prompts)))
        prompts = [all_prompts[i] for i in indices]
        result = training_step(policy, prompts, lambda i, text: float(text == str(gold[indices[i]])),
                               optimizer, generator=generator,
                               microbatch_size=int(config.get("training", {}).get("microbatch_size", 1)))
        records.append({"prompt_ids": indices, "step": cursor + 1,
                        "actions": [r["actions"] for r in result.records],
                        "loss": result.metrics["loss"], "moments": result.metrics["adam_first_moment_norm"],
                        "lr": optimizer.param_groups[0]["lr"]})
        cursor += 1
    output = Path(output_path)
    output.mkdir(parents=True)
    save_checkpoint(output / "checkpoint", policy, optimizer,
                    metadata={"phase_step": cursor, "sampler": schedule.state(),
                              "resume_contract": "tiny-debug-v1", "seed": seed},
                    generator=generator, numpy_generator=schedule.generator)
    (output / "records.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
