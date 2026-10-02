from __future__ import annotations

from dataclasses import replace

import numpy as np

from struggle_ai.mcts import (
    ConvergedAdaptivePuctConfig,
    ConvergedAdaptivePuctPlanner,
    SelectiveSearchDecision,
)


def decision(
    q: list[float],
    *,
    depth: int,
    simulations: int,
    visits: list[float] | None = None,
    cutoffs: int = 0,
) -> SelectiveSearchDecision:
    legal = len(q)
    visit_array = np.asarray(visits or [simulations / legal] * legal, dtype=np.float64)
    action = int(np.argmax(q))
    return SelectiveSearchDecision(
        action=action,
        raw_action=0,
        raw_policy=np.full(legal, 1 / legal),
        search_policy=visit_array / visit_array.sum(),
        visits=visit_array,
        q_root=np.asarray(q, dtype=np.float64),
        searched=True,
        triggers=("opponent_event",),
        simulations=simulations,
        budgets={},
        root_q_gap=0.1,
        action_flipped=action != 0,
        latency_seconds=1.0,
        expanded_nodes=simulations,
        exact_chance_nodes=0,
        belief_particles=32,
        meaningful_depth=depth,
        hierarchical_path=(),
        teacher_action=action,
        node_budget_cutoffs=cutoffs,
    )


class FakePlanner:
    def __init__(self, result: SelectiveSearchDecision) -> None:
        self.result = result
        self.pending_continuation: list[dict[str, str]] = []
        self.pending_continuation_seed = None
        self.seed = 0
        self.search_index = 0
        self.rng = np.random.default_rng(0)
        self.calls = 0

    def plan(self, _snapshot, _obs, *, force_search=False):
        self.calls += 1
        return self.result


def planner(base, budget, deep) -> ConvergedAdaptivePuctPlanner:
    return ConvergedAdaptivePuctPlanner(
        FakePlanner(base),
        FakePlanner(budget),
        FakePlanner(deep),
        ConvergedAdaptivePuctConfig(always_compare=True),
        seed=7,
    )


def test_adaptive_planner_stops_at_converged_same_depth_budget() -> None:
    adaptive = planner(
        decision([0.6, 0.3, 0.1], depth=2, simulations=32),
        decision([0.6, 0.3, 0.1], depth=2, simulations=128),
        decision([0.1, 0.3, 0.7], depth=4, simulations=128),
    )
    # Same-depth evidence already converges, so no unnecessary deep search occurs.
    result = adaptive.plan({"episode_seed": 1, "episode_step": 2}, {})
    assert result.convergence_status == "BUDGET_CONVERGENCE"
    assert result.meaningful_depth == 2
    assert adaptive.deep.calls == 0
    assert result.simulations == 160


def test_adaptive_planner_accepts_converged_deeper_ranking() -> None:
    adaptive = planner(
        decision([0.6, 0.3, 0.1], depth=2, simulations=32),
        decision([0.1, 0.7, 0.2], depth=2, simulations=128),
        decision([0.0, 0.8, 0.1], depth=4, simulations=128),
    )
    result = adaptive.plan({"episode_seed": 1, "episode_step": 2}, {})
    assert result.convergence_status == "HORIZON_CONVERGENCE"
    assert result.adaptive_fallback is False
    assert result.action == 1
    assert result.meaningful_depth == 4
    assert result.simulations == 288


def test_adaptive_planner_fails_to_budget_when_deep_ranking_is_unresolved() -> None:
    adaptive = planner(
        decision([0.6, 0.3, 0.1], depth=2, simulations=32),
        decision([0.1, 0.7, 0.2], depth=2, simulations=128),
        decision([0.1, 0.2, 0.8], depth=4, simulations=128),
    )
    result = adaptive.plan({"episode_seed": 1, "episode_step": 2}, {})
    assert result.convergence_status == "UNRESOLVED"
    assert result.adaptive_fallback is True
    assert result.action == 1
    assert result.meaningful_depth == 2
    assert result.simulations == 288


def test_adaptive_planner_rejects_cutoff_deep_evidence() -> None:
    base = decision([0.6, 0.3, 0.1], depth=2, simulations=32)
    budget = decision([0.1, 0.7, 0.2], depth=2, simulations=128)
    deep = replace(
        decision([0.1, 0.2, 0.8], depth=4, simulations=128),
        node_budget_cutoffs=1,
    )
    result = planner(base, budget, deep).plan(
        {"episode_seed": 1, "episode_step": 2}, {}
    )
    assert result.convergence_status == "UNRESOLVED"
    assert result.adaptive_fallback is True
    assert result.action == budget.action
    assert result.node_budget_cutoffs == 1
