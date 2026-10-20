import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from paper2.training.checkpoints import (
    digest, load_checkpoint, save_checkpoint, verify_checkpoint,
)
from paper2.training.finite_grpo import training_step
from paper2.training.runner import PromptSchedule, create_branch, resume_check, train_phase
from paper2.training.toy import TinyPolicy


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def initialized():
    policy = TinyPolicy(2)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0)
    generator = torch.Generator().manual_seed(99)
    training_step(policy, ["one", "two"], lambda i, a: float(a == "0"),
                  optimizer, generator=generator, group_size=64)
    return policy, optimizer, generator


def test_atomic_checkpoint_and_rng_roundtrip(tmp_path):
    policy, optimizer, generator = initialized()
    private_numpy = np.random.default_rng(52)
    checkpoint = save_checkpoint(tmp_path / "anchor", policy, optimizer,
                                 metadata={"phase_step": 1, "resume_contract": "fixed"},
                                 generator=generator, numpy_generator=private_numpy)
    expected_actions = torch.randint(0, 1000, (10,), generator=generator)
    expected_np = private_numpy.integers(0, 1000, 10)
    restored_policy, restored_opt, restored_generator = initialized()
    restored_np = np.random.default_rng(0)
    load_checkpoint(checkpoint, restored_policy, restored_opt,
                    generator=restored_generator, numpy_generator=restored_np,
                    expected_contract="fixed")
    assert torch.equal(expected_actions, torch.randint(0, 1000, (10,), generator=restored_generator))
    assert np.array_equal(expected_np, restored_np.integers(0, 1000, 10))
    assert all(torch.equal(x, y) for x, y in zip(policy.parameters(), restored_policy.parameters()))
    assert all(s["step"] == 1 for s in restored_opt.state.values())
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, policy, optimizer, metadata={}, generator=generator)


def test_reject_incomplete_tampered_and_incompatible_checkpoint(tmp_path):
    policy, optimizer, generator = initialized()
    with pytest.raises(ValueError, match="complete checkpoint"):
        verify_checkpoint(tmp_path / "missing")
    path = save_checkpoint(tmp_path / "state", policy, optimizer,
                           metadata={"resume_contract": "one"}, generator=generator)
    with pytest.raises(ValueError, match="Resume contract"):
        load_checkpoint(path, policy, optimizer, generator=generator, expected_contract="two")
    with (path / "state.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checkpoint(path)


def test_inference_snapshot_cannot_resume_optimizer(tmp_path):
    policy, optimizer, generator = initialized()
    path = save_checkpoint(tmp_path / "inference", policy, None, metadata={}, inference_only=True)
    with pytest.raises(ValueError, match="Inference snapshots"):
        load_checkpoint(path, policy, optimizer, generator=generator)
    load_checkpoint(path, policy, restore_random_state=False)


def test_prompt_schedule_resume_is_exact_across_epoch():
    schedule = PromptSchedule(7, 2)
    schedule.next(6)
    state = copy.deepcopy(schedule.state())
    rng = copy.deepcopy(schedule.generator.bit_generator.state)
    expected = [schedule.next(5), schedule.next(5)]
    other = PromptSchedule(7, 9)
    other.restore(state)
    other.generator.bit_generator.state = rng
    actual = [other.next(5), other.next(5)]
    assert actual == expected
    assert all(len(set(batch)) == 5 for batch in actual)


def test_fresh_process_resume_matches_uninterrupted(tmp_path):
    result = resume_check({"output_dir": str(tmp_path), "training": {"learning_rate": 0.001},
                           "seed": 101})
    assert result["status"] == "passed"
    assert result["fresh_process"] is True
    assert result["max_parameter_difference"] == 0
    assert result["optimizer_state_equal"]


def runner_config(tmp_path):
    return {
        "project": "A", "protocol_version": "runner-test", "backend": "tiny", "seed": 101,
        "model": {"id": "tiny-test-policy", "hidden_size": 16, "precision": "fp32"},
        "task": {"moves": 3, "protected_fraction": 0.2},
        "data": {"path": str(tmp_path / "data"), "seed": 5,
                 "sizes": {"train": 72, "development": 36, "final_test": 36, "structural_test": 36}},
        "output_dir": str(tmp_path / "runs"),
        "training": {"phase_update_cap": 4, "unique_prompts_per_update": 4,
                     "microbatch_size": 2, "samples_per_prompt": 8, "learning_rate": 0.001,
                     "evaluation_every_updates": 2, "schedule": "constant"},
        "checkpoint": {"save_every_updates": 1},
        "budget": {"mode": "cpu_test", "job_walltime_cap_minutes": 2,
                   "ledger_path": str(tmp_path / "budget.jsonl")},
    }


def test_runner_interruption_resume_and_stale_resume_guard(tmp_path, monkeypatch):
    import signal
    import paper2.training.runner as runner
    from paper2.experiments.data import prepare_dataset
    config = runner_config(tmp_path)
    prepare_dataset(config)
    full = train_phase(config, "clean", 101)
    actual_step = runner.training_step
    count = 0

    def stop_after_two(*args, **kwargs):
        nonlocal count
        result = actual_step(*args, **kwargs)
        count += 1
        if count == 2:
            signal.raise_signal(signal.SIGTERM)
        return result

    monkeypatch.setattr(runner, "training_step", stop_after_two)
    interrupted = train_phase(config, "clean", 101)
    assert interrupted["status"] == "preemption"
    assert interrupted["phase_step"] == 2
    monkeypatch.setattr(runner, "training_step", actual_step)
    resumed = train_phase(config, "clean", 101, resume=interrupted["final_checkpoint"])
    assert resumed["status"] == "success" and resumed["final_checkpoint"]
    a = torch.load(Path(full["final_checkpoint"]) / "state.pt", weights_only=False)
    b = torch.load(Path(resumed["final_checkpoint"]) / "state.pt", weights_only=False)
    assert all(torch.equal(a["policy"][k], b["policy"][k]) for k in a["policy"])
    completed = train_phase(config, "clean", 101, resume=resumed["final_checkpoint"])
    assert completed["final_checkpoint"] == resumed["final_checkpoint"]
    with pytest.raises(ValueError, match="predates retained run logs"):
        train_phase(config, "clean", 101, resume=interrupted["final_checkpoint"])


def test_branch_optimizer_isolation_history_and_parent_immutability(tmp_path):
    from paper2.experiments.data import prepare_dataset
    config = runner_config(tmp_path)
    prepare_dataset(config)
    clean = train_phase(config, "clean", 101)
    affected = train_phase(config, "vulnerable", 101, parent=clean["final_checkpoint"])
    parent = Path(affected["final_checkpoint"])
    before = {p.name: digest(p) for p in parent.iterdir()}
    carry = create_branch(config, parent, tmp_path / "carry", "continue", 101)
    reset = create_branch(config, parent, tmp_path / "reset", "reset_optimizer", 101)
    rollback = create_branch(config, clean["final_checkpoint"], tmp_path / "rollback", "rollback", 101,
                             incident_parent=parent)
    a = torch.load(Path(carry) / "state.pt", weights_only=False)
    b = torch.load(Path(reset) / "state.pt", weights_only=False)
    assert all(torch.equal(a["policy"][k], b["policy"][k]) for k in a["policy"])
    assert a["optimizer"]["state"] and not b["optimizer"]["state"]
    assert verify_checkpoint(rollback)["total_updates"] == verify_checkpoint(parent)["total_updates"]
    assert before == {p.name: digest(p) for p in parent.iterdir()}
    wrong = copy.deepcopy(config)
    wrong["training"]["learning_rate"] = 1e-5
    with pytest.raises(ValueError, match="Optimizer hyperparameters"):
        create_branch(wrong, parent, tmp_path / "wrong", "continue", 101)
    with pytest.raises(ValueError, match="Rollback must use"):
        train_phase(config, "recovery", 101, parent=clean["final_checkpoint"], arm="rollback")
