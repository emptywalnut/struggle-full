import threading
import time
from pathlib import Path

import pytest
import scripts.run_search_scaling_cross_mode_gates as gates
from scripts.run_search_scaling_cross_mode_gates import (
    CONFIRMATORY_SEED_BASE,
    CONFIRMATORY_SIDE_FLOORS,
    REQUIRED_SCALING_LABELS,
    comparison_args,
    confirmatory_comparison_jobs,
    cumulative_mode_elo,
    model_search_args,
    run_comparison_jobs,
)


def test_model_search_args_keep_budget_and_depth_independent() -> None:
    args = model_search_args("candidate", 4, 512)
    assert args[args.index("--candidate-simulations") + 1] == "512"
    assert args[args.index("--candidate-meaningful-depth") + 1] == "4"
    assert args[args.index("--candidate-max-expanded-nodes") + 1] == "1048576"


def test_search_scaling_node_limits_do_not_truncate_belief_search() -> None:
    d2 = model_search_args("candidate", 2, 32)
    d4 = model_search_args("candidate", 4, 128)
    d6 = model_search_args("candidate", 6, 512)
    assert d2[d2.index("--candidate-max-expanded-nodes") + 1] == "32768"
    assert d4[d4.index("--candidate-max-expanded-nodes") + 1] == "262144"
    assert d6[d6.index("--candidate-max-expanded-nodes") + 1] == "2097152"


def test_cross_mode_comparison_uses_distinct_agent_configs() -> None:
    args, candidate, opponent = comparison_args("d4_128_vs_d2_32")
    assert candidate == "D4/128"
    assert opponent == "D2/32"
    assert args[args.index("--candidate-meaningful-depth") + 1] == "4"
    assert args[args.index("--opponent-meaningful-depth") + 1] == "2"


def test_d4_512_is_the_required_deeper_gate_and_d6_is_optional() -> None:
    assert REQUIRED_SCALING_LABELS == (
        "d2_vs_raw",
        "d4_128_vs_d2_32",
        "d4_512_vs_d4_128",
    )


def test_cross_mode_jobs_use_bounded_parallelism(monkeypatch) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_run_comparison(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"label": kwargs["label"]}

    monkeypatch.setattr(gates, "run_comparison", fake_run_comparison)
    jobs = [
        (
            label,
            {
                "checkpoint": None,
                "label": label,
                "seed_pairs": 1,
                "workers": 1,
            },
        )
        for label in REQUIRED_SCALING_LABELS
    ]
    results = run_comparison_jobs(jobs, max_parallel=2)

    assert set(results) == set(REQUIRED_SCALING_LABELS)
    assert peak == 2


def test_cross_mode_jobs_can_run_serially(monkeypatch) -> None:
    calls = []

    def fake_run_comparison(**kwargs):
        calls.append(kwargs["label"])
        return {"label": kwargs["label"]}

    monkeypatch.setattr(gates, "run_comparison", fake_run_comparison)
    jobs = [
        (
            label,
            {
                "checkpoint": None,
                "label": label,
                "seed_pairs": 1,
                "workers": 1,
            },
        )
        for label in REQUIRED_SCALING_LABELS[:2]
    ]
    results = run_comparison_jobs(jobs, max_parallel=1)

    assert calls == list(REQUIRED_SCALING_LABELS[:2])
    assert set(results) == set(REQUIRED_SCALING_LABELS[:2])


def test_confirmatory_scaling_edges_share_matched_seed_block() -> None:
    jobs = confirmatory_comparison_jobs(
        checkpoint=None,
        plan=[
            ("d2_vs_raw", 25),
            ("d4_128_vs_d2_32", 25),
            ("d4_512_vs_d4_128", 8),
        ],
    )
    assert {kwargs["seed_base"] for _label, kwargs in jobs} == {
        CONFIRMATORY_SEED_BASE
    }
    assert [kwargs["seed_pairs"] for _label, kwargs in jobs] == [25, 25, 8]


def test_confirmatory_side_floors_cover_every_required_scaling_edge() -> None:
    assert set(CONFIRMATORY_SIDE_FLOORS) == set(REQUIRED_SCALING_LABELS)
    assert CONFIRMATORY_SIDE_FLOORS["d2_vs_raw"] == 0.40
    assert CONFIRMATORY_SIDE_FLOORS["d4_128_vs_d2_32"] == 0.40
    assert CONFIRMATORY_SIDE_FLOORS["d4_512_vs_d4_128"] == 0.25


def test_cumulative_mode_elo_builds_raw_relative_search_ladder() -> None:
    records = {
        "d2_vs_raw": {"elo_delta": 20.0},
        "d4_128_vs_d2_32": {"elo_delta": 15.0},
        "d4_512_vs_d4_128": {"elo_delta": -2.0},
        "d6_512_vs_d4_512": {"elo_delta": 5.0},
    }

    assert cumulative_mode_elo(records) == {
        "raw": 0.0,
        "d2_32": 20.0,
        "d4_128": 35.0,
        "d4_512": 33.0,
        "d6_512": 38.0,
    }


def test_cumulative_mode_elo_stops_at_missing_edge() -> None:
    records = {
        "d2_vs_raw": {"elo_delta": 20.0},
        "d4_512_vs_d4_128": {"elo_delta": 50.0},
    }

    assert cumulative_mode_elo(records) == {"raw": 0.0, "d2_32": 20.0}


def cross_mode_result(tmp_path: Path) -> tuple[dict, Path]:
    checkpoint = tmp_path / "checkpoint"
    payload = {
        "config": {
            "candidate_checkpoint": str(checkpoint),
            "opponent_checkpoint": str(checkpoint),
            "candidate_label": "D4/128",
            "opponent_label": "D2/32",
            "seed_pairs": 4,
            "seed_base": 9876,
        },
        "summary": {"games": 8},
    }
    return payload, checkpoint


def test_cross_mode_result_provenance_accepts_exact_matched_block(tmp_path) -> None:
    payload, checkpoint = cross_mode_result(tmp_path)

    gates.validate_result_provenance(
        payload,
        checkpoint=checkpoint,
        candidate_label="D4/128",
        opponent_label="D2/32",
        seed_pairs=4,
        seed_base=9876,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    (("opponent_checkpoint", "/stale/checkpoint"), ("seed_pairs", 3)),
)
def test_cross_mode_result_provenance_rejects_stale_results(
    tmp_path, key, value
) -> None:
    payload, checkpoint = cross_mode_result(tmp_path)
    payload["config"][key] = value

    with pytest.raises(RuntimeError, match="mismatch"):
        gates.validate_result_provenance(
            payload,
            checkpoint=checkpoint,
            candidate_label="D4/128",
            opponent_label="D2/32",
            seed_pairs=4,
            seed_base=9876,
        )


def test_cross_mode_result_provenance_rejects_partial_seed_block(tmp_path) -> None:
    payload, checkpoint = cross_mode_result(tmp_path)
    payload["summary"]["games"] = 7

    with pytest.raises(RuntimeError, match="complete side-swapped"):
        gates.validate_result_provenance(
            payload,
            checkpoint=checkpoint,
            candidate_label="D4/128",
            opponent_label="D2/32",
            seed_pairs=4,
            seed_base=9876,
        )
