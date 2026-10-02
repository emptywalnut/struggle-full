from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import scripts.continue_p30_search_scaling_evaluation as continuation


def test_anomaly_failure_records_auditable_cross_mode_skip(tmp_path, monkeypatch) -> None:
    output = tmp_path / "cross-mode" / "summary.json"
    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    monkeypatch.setattr(continuation, "CROSS_MODE", output)

    continuation.record_skipped_cross_mode(
        {"checkpoint": "/checkpoint/u10", "update": 10, "passed": False}
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["checkpoint"] == "/checkpoint/u10"
    assert payload["confirmatory_scaling_supported"] is False
    assert payload["skipped_reason"] == "frozen_anomaly_gate_failed"


def test_run_once_accepts_nonzero_only_when_requested_and_summary_exists(
    tmp_path, monkeypatch
) -> None:
    done = tmp_path / "summary.json"
    log = tmp_path / "run.log"
    monkeypatch.setattr(continuation, "ROOT", tmp_path)

    def fake_run(*_args, **_kwargs):
        done.write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(continuation.subprocess, "run", fake_run)
    continuation.run_once(
        ["false"],
        done_path=done,
        log_path=log,
        stage="expected_gate_failure",
        accept_nonzero_with_done=True,
    )

    done.unlink()
    with pytest.raises(RuntimeError, match="failed"):
        continuation.run_once(
            ["false"],
            done_path=done,
            log_path=log,
            stage="unexpected_gate_failure",
        )
