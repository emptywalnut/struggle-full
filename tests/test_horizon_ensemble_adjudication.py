from __future__ import annotations

from scripts.adjudicate_horizon_ensemble import adjudicate_horizon_root
from tests.test_search_convergence import example


def adjudicate(d6, deep):
    return adjudicate_horizon_root(
        d6,
        deep,
        min_replicates=3,
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
        q_temperature=0.05,
        min_action_coverage=2 / 3,
    )


def test_stable_deep_budget_confirms_horizon_teacher() -> None:
    d6 = example([0.0, 0.2, 0.6], depth=6, budget=1536)
    deep = [
        example(values, depth=6, budget=2048)
        for values in ([0.0, 0.25, 0.65], [0.01, 0.24, 0.64], [-0.01, 0.26, 0.66])
    ]
    status, teacher, ensemble, _evidence = adjudicate(d6, deep)
    assert status == "HORIZON_CONVERGENCE"
    assert teacher is not None
    assert ensemble is not None
    assert teacher.metadata.convergence_status == "HORIZON_CONVERGENCE"


def test_changed_deep_ranking_remains_unresolved() -> None:
    d6 = example([0.0, 0.2, 0.6], depth=6, budget=1536)
    deep = [
        example(values, depth=6, budget=2048)
        for values in ([0.7, 0.2, 0.0], [0.69, 0.21, 0.01], [0.71, 0.19, -0.01])
    ]
    status, teacher, _ensemble, _evidence = adjudicate(d6, deep)
    assert status == "UNRESOLVED"
    assert teacher is None
