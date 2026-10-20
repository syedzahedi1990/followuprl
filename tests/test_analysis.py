import hashlib
import json
from pathlib import Path

import pytest

from paper2.analysis.aggregate import finite_metrics, paired_seed_differences, utility_auc
from paper2.analysis.report import analyse_manifest
from paper2.experiments.manifests import (
    export_manifest, redact_config, restore_export, verify_export, verify_snapshot,
)


def test_exact_probability_metrics_protect_deployment_estimand():
    ps = [[0.3, 0.1, 0, 0, 0, 0, 0.6, 0], [0, 0.8, 0, 0, 0, 0, 0, 0.2]]
    result = finite_metrics(ps, [0, 1], ["belief", "world"])
    overall = result["overall"]
    assert overall["p_correct"] == pytest.approx(0.55)
    assert overall["p_invalid"] == pytest.approx(0.4)
    assert overall["conditional_valid_correct"] == pytest.approx(1.1 / 1.2)
    assert overall["conditional_valid_denominator"] == pytest.approx(1.2)
    assert overall["conditional_valid_mean_per_prompt"] == pytest.approx((0.75 + 1) / 2)
    assert overall["valid_only_ranking_accuracy"] == 1
    assert result["by_family"]["belief"]["p_correct"] == 0.3
    assert result["by_family"]["world"]["p_correct"] == 0.8


def test_negligible_valid_mass_has_no_fabricated_conditional_accuracy():
    row = [1e-20] * 6 + [1 - 6e-20, 0]
    metric = finite_metrics([row], [0])["overall"]
    assert metric["conditional_valid_correct"] is None
    assert metric["conditional_valid_prompt_count"] == 0
    assert metric["conditional_valid_denominator"] == pytest.approx(6e-20)


def test_metrics_reject_missing_invalid_mass_and_invalid_labels():
    with pytest.raises(ValueError, match="sum"):
        finite_metrics([[0.1] * 8], [0])
    with pytest.raises(ValueError, match="Gold"):
        finite_metrics([[0.125] * 8], [True])
    with pytest.raises(ValueError, match="length"):
        finite_metrics([[0.125] * 8], [])


def test_auc_trapezoids_and_partial_last_interval():
    result = utility_auc([(0, 0.2), (2, 0.8), (4, 1)], 3)
    assert result["integration_points"][-1] == pytest.approx([3, 0.9])
    assert result["auc"] == pytest.approx((1.0 + 0.85) / 3)
    assert len(result["raw_points"]) == 3


@pytest.mark.parametrize("points,budget", [
    ([(1, 0.1), (2, 0.9)], 2),
    ([(0, 0.1), (2, 0.9)], 3),
    ([(0, 0.1), (0, 0.3), (2, 0.9)], 2),
    ([(0, 0.1), (2, 1.1)], 2),
])
def test_auc_refuses_invalid_estimands(points, budget):
    with pytest.raises(ValueError):
        utility_auc(points, budget)


def test_paired_uncertainty_uses_training_seed_not_episode_count():
    result = paired_seed_differences([
        {"seed": 101, "arm": "continue", "value": 0.2},
        {"seed": 101, "arm": "reset", "value": 0.4},
        {"seed": 102, "arm": "continue", "value": 0.5},
        {"seed": 102, "arm": "reset", "value": 0.4},
        {"seed": 103, "arm": "reset", "value": 0.9},
    ], "continue")
    reset = result["arms"]["reset"]
    assert reset["n_independent_training_seeds"] == 2
    assert reset["mean_paired_difference"] == pytest.approx(0.05)
    assert reset["unpaired_seeds"] == ["103"]
    assert reset["population_interval"] is None


def _snapshot(path):
    path.mkdir()
    (path / "state.pt").write_bytes(b"opaque-model-state-for-copy-test")
    (path / "metadata.json").write_text('{"inference_only": false}')
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.iterdir()
    }
    index = path / "checksums.json"
    index.write_text(json.dumps(checksums))
    (path / "CHECKPOINT_COMPLETE").write_text(hashlib.sha256(index.read_bytes()).hexdigest() + "\n")
    return path


def test_export_verifies_source_and_destination_and_detects_corruption(tmp_path):
    snapshot = _snapshot(tmp_path / "checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"run_id": "example", "checkpoints": [{"path": str(snapshot)}]}))
    durable = tmp_path / "durable"
    durable.mkdir()
    result = export_manifest(manifest, durable)
    exported = Path(result["export_path"])
    assert verify_export(exported)["status"] == "verified"
    assert result["snapshot_count"] == 1
    copied_state = next(exported.rglob("state.pt"))
    copied_state.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        verify_export(exported)


def test_incomplete_or_unchecksummed_checkpoint_cannot_be_exported(tmp_path):
    snapshot = _snapshot(tmp_path / "checkpoint")
    (snapshot / "CHECKPOINT_COMPLETE").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        verify_snapshot(snapshot)
    snapshot = _snapshot(tmp_path / "other")
    (snapshot / "untracked-secret").write_text("should never be copied")
    with pytest.raises(ValueError, match="not covered"):
        verify_snapshot(snapshot)


def test_snapshot_refuses_escaping_checksum_paths(tmp_path):
    snapshot = _snapshot(tmp_path / "checkpoint")
    outside = tmp_path / "outside"
    outside.write_text("out of scope")
    index = snapshot / "checksums.json"
    index.write_text(json.dumps({"../outside": hashlib.sha256(outside.read_bytes()).hexdigest()}))
    (snapshot / "CHECKPOINT_COMPLETE").write_text(hashlib.sha256(index.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="Unsafe"):
        verify_snapshot(snapshot)


def test_manifest_redaction_preserves_scientific_token_counts():
    config = redact_config({"model": {"max_prompt_tokens": 512, "tokenizer_revision": "abcdef",
                                     "hf_token": "should-not-persist"},
                            "url": "https://user:password@example.test/path?secret=value"})
    assert config["model"]["max_prompt_tokens"] == 512
    assert config["model"]["hf_token"] == "[REDACTED]"
    assert config["url"] == "https://example.test/path"


def test_empty_analysis_keeps_research_claims_unresolved(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"run_id": "no-data", "status": "blocked"}))
    result = analyse_manifest(manifest)
    assert result["evaluation_records"] == 0
    assert result["auc_records"] == []
    assert all(row["status"] == "unresolved" for row in result["claim_ledger"])
    assert Path(result["report_path"]).exists()


def _restorable_bundle(tmp_path):
    from paper2.experiments.budget import BudgetLedger
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = _snapshot(source / "checkpoint")
    ledger = BudgetLedger(source / "budget.jsonl")
    ledger.record_usage(job_id="old", stage="setup", seconds=3, rate_usd_per_hour=0,
                        mode="cpu_test", event_id="old-segment")
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({
        "run_id": "restore-example", "test_only": True,
        "config": {"backend": "tiny", "budget": {"ledger_path": str(ledger.path)}},
        "checkpoints": [{"path": str(checkpoint)}], "budget_ledger": str(ledger.path),
        "final_checkpoint": str(checkpoint), "run_directory": str(source),
    }))
    durable = tmp_path / "durable"
    durable.mkdir()
    bundle = export_manifest(manifest, durable)["export_path"]
    return Path(bundle), ledger


def test_restore_preserves_newer_ledger_and_immutable_anchors(tmp_path):
    bundle, ledger = _restorable_bundle(tmp_path)
    ledger.record_usage(job_id="later", stage="setup", seconds=4, rate_usd_per_hour=0,
                        mode="cpu_test", event_id="later-segment")
    ledger_before = ledger.path.read_bytes()
    checksums_before = (bundle / "SHA256SUMS.json").read_bytes()
    result = restore_export(bundle, tmp_path / "restored", ledger_path=ledger.path)
    assert result["status"] == "restored"
    assert ledger.path.read_bytes() == ledger_before
    assert (bundle / "SHA256SUMS.json").read_bytes() == checksums_before
    verify_export(bundle)
    restored = json.loads(Path(result["manifest_path"]).read_text())
    assert restored["restore_metadata"]["checkpoint_files_modified"] is False
    assert restored["budget_ledger"] == str(ledger.path)
    assert verify_snapshot(result["final_checkpoint"])
    with pytest.raises(FileExistsError):
        restore_export(bundle, tmp_path / "restored")


def test_restore_refuses_forked_accounting_history(tmp_path):
    from paper2.experiments.budget import BudgetLedger
    bundle, _ = _restorable_bundle(tmp_path)
    unrelated = BudgetLedger(tmp_path / "forked.jsonl")
    unrelated.record_usage(job_id="other", stage="setup", seconds=2, rate_usd_per_hour=0,
                           mode="cpu_test", event_id="unrelated")
    with pytest.raises(ValueError, match="does not extend"):
        restore_export(bundle, tmp_path / "restored", ledger_path=unrelated.path)
    assert not (tmp_path / "restored").exists()


def test_export_restore_after_original_runtime_loss_resumes_exactly(tmp_path, monkeypatch):
    """Full runner—not only tensor serialization—survives storage relocation."""
    import copy
    import os
    import shutil
    import signal
    import torch
    import yaml

    from paper2.experiments.budget import BudgetLedger
    from paper2.experiments.data import prepare_dataset
    from paper2.training import runner

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = yaml.safe_load((Path(__file__).parents[1] / "configs/cpu_smoke.yaml").read_text())
        original = tmp_path / "disposable-runtime"
        original.mkdir()
        config["data"]["path"] = str(original / "dataset")
        config["output_dir"] = str(original / "runs")
        config["budget"]["ledger_path"] = str(original / "budget.jsonl")
        prepare_dataset(config)
        uninterrupted = runner.train_phase(config, "clean", 101)
        full_state = torch.load(Path(uninterrupted["final_checkpoint"]) / "state.pt", weights_only=False)

        partial_config = copy.deepcopy(config)
        partial_config["output_dir"] = str(original / "interrupted")
        original_step = runner.training_step
        calls = 0

        def stop_after_two(*args, **kwargs):
            nonlocal calls
            result = original_step(*args, **kwargs)
            calls += 1
            if calls == 2:
                os.kill(os.getpid(), signal.SIGTERM)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(runner, "training_step", stop_after_two)
            interrupted = runner.train_phase(partial_config, "clean", 101)
        assert interrupted["status"] == "preemption"
        assert interrupted["phase_step"] == 2
        original_accounting = BudgetLedger(config["budget"]["ledger_path"]).snapshot()
        durable = tmp_path / "durable"
        durable.mkdir()
        export = export_manifest(interrupted["manifest_path"], durable)
        bundle = Path(export["export_path"])
        # Destroy all original paths to ensure resume cannot accidentally depend
        # on the previous instance's checkpoints, data, logs or cost ledger.
        shutil.rmtree(original)
        restored = restore_export(bundle, tmp_path / "restored")
        restored_config = json.loads(Path(restored["config_path"]).read_text())
        continued = runner.train_phase(restored_config, "clean", 101,
                                       resume=restored["final_checkpoint"], arm=interrupted["arm"])
        assert continued["status"] == "success"
        assert continued["phase_step"] == 4
        actual = torch.load(Path(continued["final_checkpoint"]) / "state.pt", weights_only=False)
        assert full_state["policy"].keys() == actual["policy"].keys()
        assert all(torch.equal(full_state["policy"][name], actual["policy"][name])
                   for name in full_state["policy"])
        for index, state in full_state["optimizer"]["state"].items():
            assert all(torch.equal(value, actual["optimizer"]["state"][index][name])
                       if isinstance(value, torch.Tensor) else value == actual["optimizer"]["state"][index][name]
                       for name, value in state.items())
        accounting = BudgetLedger(restored["ledger_path"]).snapshot()
        assert accounting["entries"] > original_accounting["entries"]
        assert accounting["total_runtime_seconds"] > original_accounting["total_runtime_seconds"]
        assert accounting["total_usd"] == original_accounting["total_usd"]
        assert not (Path(restored_config["data"]["path"]) / "final_test_access.jsonl").exists()
        verify_export(bundle)
    finally:
        torch.set_num_threads(previous_threads)
