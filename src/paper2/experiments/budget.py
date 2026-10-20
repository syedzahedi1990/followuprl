"""Append-only project accounting, independent of model/checkpoint rollback.

The ledger accounts for process time and explicit extra charges. It cannot stop
provider billing, observe idle instance time, or prevent a cloud-side crash from
incurring charges. Reconcile provider invoices using ``record_usage`` with
``extra_usd``; never rewrite historical entries.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class BudgetError(RuntimeError):
    """A proposed job lacks authorization, information, or available budget."""


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise BudgetError(f"{name} must be an explicit nonnegative number")
    try:
        value = float(value)
    except (ValueError, TypeError) as exc:
        raise BudgetError(f"{name} must be an explicit nonnegative number") from exc
    if not math.isfinite(value) or value < 0:
        raise BudgetError(f"{name} must be finite and nonnegative")
    return value


def _canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BudgetLedger:
    """Serialized reservations and immutable usage entries for one whole project.

    ``authorize_job`` reserves the conservative cost estimate. ``record_usage``
    adds actual, incremental consumption; a repeated ``event_id`` is idempotent
    only if the full usage payload matches. ``finish_job`` releases unused
    reservation. Actual expenditure is always recorded even if it exceeds a cap:
    concealing incurred costs would be worse than reporting an overrun.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def _locked(self) -> Iterator[list[dict]]:
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield self._read()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise BudgetError("Incomplete budget ledger entry; reconcile it before continuing")
        entries = []
        previous = None
        for sequence, line in enumerate(raw.splitlines(), start=1):
            try:
                entry = json.loads(line)
                digest = entry.pop("entry_sha256")
            except (ValueError, KeyError, TypeError) as exc:
                raise BudgetError(f"Corrupt budget ledger entry {sequence}") from exc
            if entry.get("sequence") != sequence or entry.get("previous_sha256") != previous:
                raise BudgetError("Budget ledger sequence/hash chain mismatch")
            if hashlib.sha256(_canonical(entry)).hexdigest() != digest:
                raise BudgetError("Budget ledger checksum mismatch")
            entry["entry_sha256"] = digest
            entries.append(entry)
            previous = digest
        return entries

    def _append(self, entries: list[dict], payload: dict) -> dict:
        entry = dict(
            payload,
            sequence=len(entries) + 1,
            timestamp_utc=_utcnow(),
            previous_sha256=entries[-1]["entry_sha256"] if entries else None,
        )
        entry["entry_sha256"] = hashlib.sha256(_canonical(entry)).hexdigest()
        with self.path.open("ab") as handle:
            handle.write(_canonical(entry) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        entries.append(entry)
        return entry

    @staticmethod
    def _summarize(entries: list[dict]) -> dict:
        stage_totals: dict[str, float] = {}
        job_totals: dict[str, float] = {}
        active: dict[str, dict] = {}
        runtime = {"cpu_test": 0.0, "paid": 0.0, "colab": 0.0}
        units = 0.0
        units_measured = False
        unmetered_colab_seconds = 0.0
        for item in entries:
            kind = item["kind"]
            if kind == "admit":
                active[item["job_id"]] = item
            elif kind == "finish":
                active.pop(item["job_id"], None)
            elif kind == "usage":
                cost = item["charge_usd"]
                stage_totals[item["stage"]] = stage_totals.get(item["stage"], 0.0) + cost
                job_totals[item["job_id"]] = job_totals.get(item["job_id"], 0.0) + cost
                runtime[item["mode"]] += item["seconds"]
                if item.get("compute_units") is not None:
                    units += item["compute_units"]
                    units_measured = True
                elif item["mode"] == "colab":
                    unmetered_colab_seconds += item["seconds"]
        reserved_by_stage: dict[str, float] = {}
        reservations: dict[str, float] = {}
        for job_id, admission in active.items():
            # Job IDs are never reused for a fresh job, so all charges belong
            # to this single reservation (including interruption segments).
            remaining = max(0.0, admission["estimated_usd"] - job_totals.get(job_id, 0.0))
            reservations[job_id] = remaining
            stage = admission["stage"]
            reserved_by_stage[stage] = reserved_by_stage.get(stage, 0.0) + remaining
        return {
            "total_usd": sum(stage_totals.values()),
            "by_stage_usd": stage_totals,
            "by_job_usd": job_totals,
            "reserved_usd": sum(reservations.values()),
            "reserved_by_stage_usd": reserved_by_stage,
            "reservations_usd": reservations,
            "active_jobs": sorted(active),
            "total_runtime_seconds": sum(runtime.values()),
            "cpu_seconds": runtime["cpu_test"],
            "paid_process_seconds": runtime["paid"],
            "colab_seconds": runtime["colab"],
            "compute_units": units if units_measured else None,
            "unmetered_colab_seconds": unmetered_colab_seconds,
            "entries": len(entries),
            "last_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
            "accounting_scope": "Process runtime and explicitly reconciled extras; provider idle/storage charges unknown unless recorded.",
        }

    def snapshot(self) -> dict:
        with self._locked() as entries:
            return self._summarize(entries)

    def authorize_job(
        self,
        config: dict,
        *,
        job_id: str,
        stage: str,
        estimated_seconds: float,
        mode: str = "paid",
        estimated_extra_usd: float = 0.0,
    ) -> dict:
        """Validate prerequisites, then reserve estimated total job cost.

        Include setup, training, evaluation, checkpointing and export time in
        ``estimated_seconds``. Passing the complete walltime cap is conservative.
        Repeating an identical active admission is safe after interruption.
        """
        budget = config.get("budget", config)
        if mode not in {"cpu_test", "paid", "colab"}:
            raise BudgetError(f"Unknown accounting mode: {mode}")
        if not job_id or not stage:
            raise BudgetError("job_id and stage are required")
        seconds = _nonnegative(estimated_seconds, "estimated_seconds")
        extras = _nonnegative(estimated_extra_usd, "estimated_extra_usd")
        multiplier = _nonnegative(budget.get("contingency_multiplier", 1.25), "contingency_multiplier")
        if multiplier < 1:
            raise BudgetError("contingency_multiplier must be at least 1")
        wall_minutes = _nonnegative(budget.get("job_walltime_cap_minutes"), "job_walltime_cap_minutes")
        if wall_minutes <= 0 or seconds > wall_minutes * 60 + 1e-9:
            raise BudgetError("Job duration must fit the positive declared walltime cap")
        if mode == "cpu_test":
            rate = 0.0
            if extras != 0:
                raise BudgetError("cpu_test cannot authorize paid extras")
            project_cap = float("inf")
            stage_cap = float("inf")
        else:
            if budget.get("runtime_authorized") is not True:
                raise BudgetError("This runtime has not been explicitly authorized")
            rate = _nonnegative(budget.get("hourly_rate_usd"), "hourly_rate_usd")
            authorized = _nonnegative(budget.get("authorized_spend_usd"), "authorized_spend_usd")
            if mode == "paid" and (rate <= 0 or authorized <= 0):
                raise BudgetError("Paid mode requires a positive actual rate and authorized spending ceiling")
            project_cap = min(
                _nonnegative(budget.get("project_ceiling_usd"), "project_ceiling_usd"),
                authorized,
            )
            caps = budget.get("stage_caps_usd", {})
            stage_cap = _nonnegative(caps.get(stage, budget.get("stage_ceiling_usd")), f"stage cap for {stage}")
            durable = budget.get("durable_destination")
            if not durable or not Path(durable).is_dir():
                raise BudgetError("Configure an existing durable_destination before a runtime job")
            if budget.get("durability_confirmed") is not True:
                raise BudgetError("Confirm durable_destination is outside the disposable runtime with durability_confirmed=true")
        estimate = (seconds / 3600 * rate + extras) * multiplier
        payload = {
            "kind": "admit", "job_id": str(job_id), "stage": str(stage), "mode": mode,
            "estimated_seconds": seconds, "estimated_extra_usd": extras,
            "hourly_rate_usd": rate, "contingency_multiplier": multiplier,
            "estimated_usd": estimate,
            "authorized_project_ceiling_usd": None if math.isinf(project_cap) else project_cap,
            "authorized_stage_ceiling_usd": None if math.isinf(stage_cap) else stage_cap,
        }
        with self._locked() as entries:
            existing = [e for e in entries if e.get("job_id") == job_id and e["kind"] == "admit"]
            summary = self._summarize(entries)
            if existing:
                if job_id not in summary["active_jobs"]:
                    raise BudgetError("Completed job_id cannot be reused; assign a new run/segment ID")
                old_payload = {key: existing[-1][key] for key in payload}
                if old_payload != payload:
                    raise BudgetError("Admission changed for an active job; reconcile and close it first")
                return existing[-1]
            if summary["total_usd"] + summary["reserved_usd"] + estimate > project_cap + 1e-9:
                raise BudgetError("Projected job exceeds cumulative authorized project budget")
            stage_used = summary["by_stage_usd"].get(stage, 0.0)
            stage_reserved = summary["reserved_by_stage_usd"].get(stage, 0.0)
            if stage_used + stage_reserved + estimate > stage_cap + 1e-9:
                raise BudgetError(f"Projected job exceeds stage budget: {stage}")
            return self._append(entries, payload)

    def record_usage(
        self,
        *,
        job_id: str,
        stage: str,
        seconds: float,
        rate_usd_per_hour: float,
        mode: str,
        event_id: str,
        extra_usd: float = 0.0,
        compute_units: float | None = None,
        note: str | None = None,
    ) -> dict:
        """Append incurred usage, even if actual expenses exceed an estimate."""
        seconds = _nonnegative(seconds, "seconds")
        rate = _nonnegative(rate_usd_per_hour, "rate_usd_per_hour")
        extras = _nonnegative(extra_usd, "extra_usd")
        if mode not in {"cpu_test", "paid", "colab"}:
            raise BudgetError("Unknown accounting mode")
        if not job_id or not stage or not event_id:
            raise BudgetError("job_id, stage and event_id are required")
        if mode == "cpu_test" and (rate != 0 or extras != 0):
            raise BudgetError("CPU test entries cannot contain paid charges")
        units = None if compute_units is None else _nonnegative(compute_units, "compute_units")
        payload = {
            "kind": "usage", "job_id": str(job_id), "stage": str(stage),
            "seconds": seconds, "hourly_rate_usd": rate, "extra_usd": extras,
            "charge_usd": seconds / 3600 * rate + extras,
            "mode": mode, "event_id": str(event_id), "compute_units": units,
            "note": note,
        }
        with self._locked() as entries:
            duplicate = next((e for e in entries if e.get("event_id") == event_id), None)
            if duplicate is not None:
                if {key: duplicate[key] for key in payload} != payload:
                    raise BudgetError("Duplicate event_id has different usage; refusing to overwrite history")
                return duplicate
            return self._append(entries, payload)

    def finish_job(self, job_id: str, status: str) -> dict:
        with self._locked() as entries:
            old = next((e for e in reversed(entries) if e["kind"] == "finish" and e["job_id"] == job_id), None)
            if old:
                if old["status"] != status:
                    raise BudgetError("Completed job status cannot be overwritten")
                return old
            if job_id not in self._summarize(entries)["active_jobs"]:
                raise BudgetError("Cannot finish a job without an active admission")
            return self._append(entries, {"kind": "finish", "job_id": str(job_id), "status": str(status)})

    def assert_checkpoint_offset(self, checkpoint_offset: int) -> None:
        """An older model checkpoint may reference an earlier ledger, never a future one."""
        if checkpoint_offset < 0 or checkpoint_offset > self.snapshot()["entries"]:
            raise BudgetError("Checkpoint ledger offset exceeds current durable ledger; reconcile missing accounting")
