import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from paper2.experiments.budget import BudgetError, BudgetLedger


def budget_config(tmp_path, **overrides):
    durable = tmp_path / "external"
    durable.mkdir(exist_ok=True)
    return {
        "runtime_authorized": True,
        "durability_confirmed": True,
        "durable_destination": str(durable),
        "authorized_spend_usd": 500,
        "project_ceiling_usd": 500,
        "hourly_rate_usd": 2,
        "stage_caps_usd": {"setup": 50, "recovery": 200},
        "job_walltime_cap_minutes": 60,
        "contingency_multiplier": 1.25,
        **overrides,
    }


@pytest.mark.parametrize("field", [
    "runtime_authorized", "durability_confirmed", "authorized_spend_usd",
    "hourly_rate_usd", "project_ceiling_usd", "durable_destination",
])
def test_paid_run_requires_each_prerequisite(tmp_path, field):
    config = budget_config(tmp_path)
    config.pop(field)
    ledger = BudgetLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(BudgetError):
        ledger.authorize_job(config, job_id="not-authorized", stage="setup", estimated_seconds=60)
    assert ledger.snapshot()["entries"] == 0


def test_caps_include_reservations_and_contingency(tmp_path):
    config = budget_config(tmp_path, authorized_spend_usd=3)
    ledger = BudgetLedger(tmp_path / "ledger.jsonl")
    first = ledger.authorize_job(config, job_id="one", stage="setup", estimated_seconds=3600)
    assert first["estimated_usd"] == 2.5
    with pytest.raises(BudgetError, match="cumulative"):
        ledger.authorize_job(config, job_id="two", stage="setup", estimated_seconds=1800)
    ledger.record_usage(job_id="one", stage="setup", seconds=900,
                        rate_usd_per_hour=2, mode="paid", event_id="one-segment")
    # reservation shrinks by actual incurred charge; it is not counted twice.
    assert ledger.snapshot()["total_usd"] == 0.5
    assert ledger.snapshot()["reserved_usd"] == 2
    ledger.finish_job("one", "success")
    assert ledger.snapshot()["reserved_usd"] == 0
    ledger.authorize_job(config, job_id="two", stage="setup", estimated_seconds=1800)


def test_stage_limit_cannot_be_replaced_with_project_limit(tmp_path):
    config = budget_config(tmp_path, stage_caps_usd={"setup": 0.1})
    with pytest.raises(BudgetError, match="stage"):
        BudgetLedger(tmp_path / "ledger.jsonl").authorize_job(
            config, job_id="one", stage="setup", estimated_seconds=3600
        )


def test_ledger_monotonic_rollback_resume_and_duplicate_event(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.jsonl")
    ledger.authorize_job(budget_config(tmp_path), job_id="parent", stage="setup", estimated_seconds=3600)
    kwargs = dict(job_id="parent", stage="setup", seconds=1800,
                  rate_usd_per_hour=2, mode="paid", event_id="parent:one")
    ledger.record_usage(**kwargs)
    checkpoint_offset = ledger.snapshot()["entries"]
    ledger.record_usage(**kwargs)  # idempotent retry after interrupted acknowledgment
    assert ledger.snapshot()["entries"] == checkpoint_offset
    with pytest.raises(BudgetError, match="Duplicate"):
        ledger.record_usage(**dict(kwargs, seconds=1801))
    ledger.finish_job("parent", "success")
    before = ledger.snapshot()["total_usd"]
    ledger.assert_checkpoint_offset(checkpoint_offset)
    ledger.authorize_job(budget_config(tmp_path), job_id="rollback", stage="recovery", estimated_seconds=3600)
    ledger.record_usage(job_id="rollback", stage="recovery", seconds=1800,
                        rate_usd_per_hour=2, mode="paid", event_id="rollback:one")
    assert ledger.snapshot()["total_usd"] == before + 1
    with pytest.raises(BudgetError, match="offset"):
        ledger.assert_checkpoint_offset(100)


def test_cpu_time_and_colab_consumption_are_not_silently_free(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.jsonl")
    ledger.authorize_job({"job_walltime_cap_minutes": 1}, job_id="cpu", stage="setup",
                         estimated_seconds=60, mode="cpu_test")
    ledger.record_usage(job_id="cpu", stage="setup", seconds=60, rate_usd_per_hour=0,
                        mode="cpu_test", event_id="cpu")
    ledger.finish_job("cpu", "success")
    config = budget_config(tmp_path, hourly_rate_usd=0, authorized_spend_usd=0)
    ledger.authorize_job(config, job_id="colab", stage="setup", estimated_seconds=3600, mode="colab")
    ledger.record_usage(job_id="colab", stage="setup", seconds=1800, rate_usd_per_hour=0,
                        mode="colab", event_id="colab-unknown")
    ledger.record_usage(job_id="colab", stage="setup", seconds=1800, rate_usd_per_hour=0,
                        mode="colab", compute_units=2.2, event_id="colab-metered")
    state = ledger.snapshot()
    assert state["total_usd"] == 0
    assert state["cpu_seconds"] == 60
    assert state["colab_seconds"] == 3600
    assert state["unmetered_colab_seconds"] == 1800
    assert state["compute_units"] == 2.2


def test_atomic_admission_blocks_concurrent_overspend(tmp_path):
    config = budget_config(tmp_path, authorized_spend_usd=2.5)
    path = tmp_path / "ledger.jsonl"

    def admit(job_id):
        try:
            BudgetLedger(path).authorize_job(config, job_id=job_id, stage="setup", estimated_seconds=3600)
            return True
        except BudgetError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sum(executor.map(admit, ["one", "two"])) == 1
    assert BudgetLedger(path).snapshot()["reserved_usd"] == 2.5


def test_ledger_corruption_fails_closed(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = BudgetLedger(path)
    ledger.record_usage(job_id="external-invoice", stage="setup", seconds=0,
                        rate_usd_per_hour=0, mode="paid", event_id="invoice", extra_usd=3)
    row = json.loads(path.read_text())
    row["charge_usd"] = 0
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(BudgetError, match="checksum"):
        ledger.snapshot()


def test_real_overrun_is_recorded_then_blocks_future_admission(tmp_path):
    config = budget_config(tmp_path, authorized_spend_usd=3)
    ledger = BudgetLedger(tmp_path / "ledger.jsonl")
    ledger.record_usage(job_id="provider-invoice", stage="setup", seconds=0,
                        rate_usd_per_hour=0, mode="paid", event_id="invoice", extra_usd=4)
    assert ledger.snapshot()["total_usd"] == 4
    with pytest.raises(BudgetError):
        ledger.authorize_job(config, job_id="new", stage="setup", estimated_seconds=1)
