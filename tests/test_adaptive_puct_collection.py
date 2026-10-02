from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.collect_puct_selfplay_parallel import (
    _decision_comparison,
    _fails_configured_search_evidence_gate,
    _requires_deep_adjudication,
    _reseed_planner_for_root,
    _should_adapt,
    _sync_continuation,
    adaptive_root_search_seed,
    stratified_deep_candidates,
)


def args(**overrides):
    values = {
        "adaptive_min_rank_correlation": 0.7,
        "adaptive_meaningful_q_gap": 0.01,
        "adaptive_always_compare": False,
        "adaptive_on_flip": True,
        "adaptive_escalation_q_gap": 0.03,
        "adaptive_deep_always_trigger": [],
        "adaptive_fail_closed_trigger": [],
        "adaptive_fail_closed_min_depth": 0,
        "adaptive_fail_closed_require_terminal": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def decision(q, *, gap=0.1, flipped=False, triggers=()):
    return SimpleNamespace(
        visits=np.asarray([8.0, 16.0, 32.0]),
        q_root=np.asarray(q, dtype=np.float64),
        root_q_gap=gap,
        action_flipped=flipped,
        triggers=triggers,
    )


def test_adaptive_comparison_uses_q_ranking() -> None:
    result = _decision_comparison(
        decision([0.0, 0.2, 0.5]),
        decision([-0.1, 0.3, 0.6]),
        args(),
    )
    assert result["stable"] is True
    assert result["right_top"] == 2


def test_adaptive_comparison_rejects_cutoff_evidence() -> None:
    left = decision([0.0, 0.2, 0.5])
    right = decision([-0.1, 0.3, 0.6])
    right.node_budget_cutoffs = 1
    result = _decision_comparison(left, right, args())
    assert result["stable"] is False
    assert result["reason"] == "incomplete_search_evidence"


def test_adaptive_escalation_ignores_branching_without_uncertainty_or_stakes() -> None:
    ordinary = decision([0.0, 0.2, 0.5])
    assert _should_adapt(ordinary, 30, args()) is False
    assert _should_adapt(ordinary, 3, args()) is False
    strategic = decision([0.0, 0.2, 0.5], triggers=("scoring_card",))
    assert _should_adapt(strategic, 3, args()) is True


def test_adaptive_escalation_can_reserve_deep_search_despite_actor_flip() -> None:
    flipped = decision([0.0, 0.2, 0.5], flipped=True)
    assert _should_adapt(flipped, 10, args()) is True
    assert _should_adapt(flipped, 10, args(adaptive_on_flip=False)) is False


def test_scoring_pressure_can_require_deeper_horizon_despite_budget_agreement() -> None:
    pressure = decision([0.0, 0.2, 0.5], triggers=("scoring_pressure",))
    assert _requires_deep_adjudication(
        pressure,
        args(adaptive_deep_always_trigger=["scoring_pressure"]),
    )
    ordinary = decision([0.0, 0.2, 0.5], triggers=("scoring_card",))
    assert not _requires_deep_adjudication(
        ordinary,
        args(adaptive_deep_always_trigger=["scoring_pressure"]),
    )


def test_scoring_pressure_falls_back_when_search_lacks_terminal_depth() -> None:
    pressure = decision([0.0, 0.2, 0.5], triggers=("scoring_pressure",))
    pressure.meaningful_depth = 6
    pressure.terminal_evaluations = 0
    gate_args = args(
        adaptive_fail_closed_trigger=["scoring_pressure"],
        adaptive_fail_closed_min_depth=8,
        adaptive_fail_closed_require_terminal=True,
    )
    assert _fails_configured_search_evidence_gate(pressure, True, gate_args)
    pressure.meaningful_depth = 8
    pressure.terminal_evaluations = 3
    assert not _fails_configured_search_evidence_gate(pressure, True, gate_args)


def test_selected_hierarchical_continuation_is_synchronized() -> None:
    selected = SimpleNamespace(
        pending_continuation=[{"side": "us", "action": "France"}],
        pending_continuation_seed=42,
    )
    planners = {
        name: {
            "us": SimpleNamespace(pending_continuation=[], pending_continuation_seed=None),
            "ussr": SimpleNamespace(pending_continuation=[], pending_continuation_seed=None),
        }
        for name in ("base", "budget", "deep")
    }
    _sync_continuation(planners, selected, "us")
    assert all(group["us"].pending_continuation == selected.pending_continuation for group in planners.values())
    assert all(group["us"].pending_continuation_seed == 42 for group in planners.values())
    assert all(group["ussr"].pending_continuation == [] for group in planners.values())


def test_adaptive_root_seed_pairs_configs_without_clearing_continuation() -> None:
    seed = adaptive_root_search_seed(82_100_000, 97_200_003, 314)
    planners = [
        SimpleNamespace(
            seed=index,
            rng=np.random.default_rng(index),
            search_index=9,
            pending_continuation=[{"side": "us", "action": "France"}],
        )
        for index in range(3)
    ]
    for planner in planners:
        _reseed_planner_for_root(planner, seed)

    draws = [planner.rng.random(8).tolist() for planner in planners]
    assert draws[0] == draws[1] == draws[2]
    assert all(planner.seed == seed and planner.search_index == 0 for planner in planners)
    assert all(planner.pending_continuation == [{"side": "us", "action": "France"}] for planner in planners)


def test_deep_candidate_cap_is_stratified_across_side_era_and_game() -> None:
    records = []
    for game_index in range(4):
        candidates = []
        for side in ("us", "ussr"):
            for turn in (2, 5, 9):
                for offset in range(3):
                    candidates.append(
                        {
                            "source_game": f"game-{game_index}",
                            "side": side,
                            "turn": turn,
                            "action_index": 100 * game_index + 10 * turn + offset,
                            "convergence_q_gap": 0.03 + offset * 0.01,
                            "latency_seconds": 1.0,
                        }
                    )
        records.append({"source_game": f"game-{game_index}", "deep_candidates": candidates})

    selected = stratified_deep_candidates(records, 24)
    assert len(selected) == 24
    assert {(row["side"], "early" if row["turn"] <= 3 else "mid" if row["turn"] <= 7 else "late") for row in selected} == {
        (side, era) for side in ("us", "ussr") for era in ("early", "mid", "late")
    }
    assert {row["source_game"] for row in selected} == {f"game-{index}" for index in range(4)}
