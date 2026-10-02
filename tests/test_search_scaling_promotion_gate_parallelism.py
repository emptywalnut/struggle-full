from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import scripts.run_search_scaling_promotion_gates as gates


def test_run_gate_jobs_uses_bounded_parallelism(monkeypatch) -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_run_gate(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"update": kwargs["update"], "mode": kwargs["mode"]}

    monkeypatch.setattr(gates, "run_gate", fake_run_gate)
    jobs = [
        (
            f"u{update}:{mode}",
            {"update": update, "mode": mode, "seed_pairs": 1, "workers": 1},
        )
        for update, mode in ((5, "raw"), (5, "d2"), (10, "raw"), (10, "d2"))
    ]

    results = gates.run_gate_jobs(jobs, max_parallel=2)

    assert set(results) == {key for key, _ in jobs}
    assert peak == 2


def test_run_gate_jobs_serial_mode(monkeypatch) -> None:
    calls = []

    def fake_run_gate(**kwargs):
        calls.append((kwargs["update"], kwargs["mode"]))
        return {"score": 0.5}

    monkeypatch.setattr(gates, "run_gate", fake_run_gate)
    jobs = [
        ("u5:raw", {"update": 5, "mode": "raw", "seed_pairs": 1, "workers": 1}),
        ("u5:d2", {"update": 5, "mode": "d2", "seed_pairs": 1, "workers": 1}),
    ]

    assert set(gates.run_gate_jobs(jobs, max_parallel=1)) == {"u5:raw", "u5:d2"}
    assert calls == [(5, "raw"), (5, "d2")]


def promotion_result(tmp_path: Path) -> tuple[dict, Path, Path]:
    candidate = tmp_path / "candidate"
    opponent = tmp_path / "opponent"
    payload = {
        "config": {
            "candidate_checkpoint": str(candidate),
            "opponent_checkpoint": str(opponent),
            "candidate_label": "candidate-raw",
            "opponent_label": "parent-raw",
            "seed_pairs": 3,
            "seed_base": 1234,
        },
        "summary": {"games": 6},
    }
    return payload, candidate, opponent


def test_promotion_result_provenance_accepts_exact_side_swapped_block(tmp_path) -> None:
    payload, candidate, opponent = promotion_result(tmp_path)

    gates.validate_result_provenance(
        payload,
        candidate_checkpoint=candidate,
        opponent_checkpoint=opponent,
        candidate_label="candidate-raw",
        opponent_label="parent-raw",
        seed_pairs=3,
        seed_base=1234,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    (("candidate_checkpoint", "/stale/checkpoint"), ("seed_base", 9999)),
)
def test_promotion_result_provenance_rejects_stale_results(
    tmp_path, key, value
) -> None:
    payload, candidate, opponent = promotion_result(tmp_path)
    payload["config"][key] = value

    with pytest.raises(RuntimeError, match="mismatch"):
        gates.validate_result_provenance(
            payload,
            candidate_checkpoint=candidate,
            opponent_checkpoint=opponent,
            candidate_label="candidate-raw",
            opponent_label="parent-raw",
            seed_pairs=3,
            seed_base=1234,
        )


def test_promotion_result_provenance_rejects_partial_seed_block(tmp_path) -> None:
    payload, candidate, opponent = promotion_result(tmp_path)
    payload["summary"]["games"] = 5

    with pytest.raises(RuntimeError, match="complete side-swapped"):
        gates.validate_result_provenance(
            payload,
            candidate_checkpoint=candidate,
            opponent_checkpoint=opponent,
            candidate_label="candidate-raw",
            opponent_label="parent-raw",
            seed_pairs=3,
            seed_base=1234,
        )
