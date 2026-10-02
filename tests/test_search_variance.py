from __future__ import annotations

from dataclasses import replace

from scripts.analyze_search_variance import (
    classify_ensemble_root,
    jackknife_ensemble_stability,
    meaningful_comparison,
)
from tests.test_search_convergence import example


def replicas(rows: list[list[float]], *, depth: int) -> list:
    return [example(q, depth=depth, budget=512) for q in rows]


def test_ensemble_convergence_accepts_stable_replicates_and_depths() -> None:
    d4 = replicas([[0.0, 0.2, 0.5], [0.02, 0.18, 0.48], [-0.01, 0.21, 0.51]], depth=4)
    d6 = replicas([[0.0, 0.25, 0.6], [0.01, 0.24, 0.59], [-0.02, 0.26, 0.61]], depth=6)
    status, teacher, evidence = classify_ensemble_root(
        d4,
        d6,
        min_replicates=3,
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
        q_temperature=0.05,
        min_action_coverage=2 / 3,
    )
    assert status == "ENSEMBLE_CONVERGENCE"
    assert teacher is not None
    assert teacher.metadata.search_action == 2
    assert evidence["cross_depth"]["stable"] is True


def test_particle_variance_rejects_inconsistent_replicates() -> None:
    d4 = replicas([[0.0, 0.2, 0.5], [0.6, 0.2, 0.0], [0.1, 0.5, 0.2]], depth=4)
    d6 = replicas([[0.0, 0.25, 0.6], [0.01, 0.24, 0.59], [-0.02, 0.26, 0.61]], depth=6)
    status, teacher, _evidence = classify_ensemble_root(
        d4,
        d6,
        min_replicates=3,
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
        q_temperature=0.05,
        min_action_coverage=2 / 3,
    )
    assert status == "PARTICLE_VARIANCE"
    assert teacher is None


def test_horizon_disagreement_requires_stable_within_depth() -> None:
    d4 = replicas([[0.6, 0.2, 0.0], [0.59, 0.21, 0.01], [0.61, 0.19, -0.01]], depth=4)
    d6 = replicas([[0.0, 0.25, 0.6], [0.01, 0.24, 0.59], [-0.02, 0.26, 0.61]], depth=6)
    status, teacher, _evidence = classify_ensemble_root(
        d4,
        d6,
        min_replicates=3,
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
        q_temperature=0.05,
        min_action_coverage=2 / 3,
    )
    assert status == "HORIZON_DISAGREEMENT"
    assert teacher is None


def test_cutoff_replicate_fails_closed() -> None:
    d4 = replicas([[0.0, 0.2, 0.5]] * 3, depth=4)
    d4[1] = replace(d4[1], metadata=replace(d4[1].metadata, node_budget_cutoffs=1))
    d6 = replicas([[0.0, 0.25, 0.6]] * 3, depth=6)
    status, teacher, _evidence = classify_ensemble_root(
        d4,
        d6,
        min_replicates=3,
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
        q_temperature=0.05,
        min_action_coverage=2 / 3,
    )
    assert status == "INSUFFICIENT"
    assert teacher is None


def test_meaningful_comparison_ignores_small_lower_rank_swaps() -> None:
    left = example([0.60, 0.20, 0.201, 0.202], depth=4, budget=512)
    right = example([0.61, 0.202, 0.201, 0.20], depth=4, budget=512)
    result = meaningful_comparison(
        left,
        right,
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
    )
    assert result["stable"] is True
    assert result["top_consistent"] is True
    assert result["meaningful_pair_agreement"] == 1.0


def test_jackknife_ensemble_stability_reports_leave_one_out_evidence() -> None:
    values = [
        [0.60, 0.20, 0.19],
        [0.62, 0.19, 0.20],
        [0.61, 0.21, 0.18],
    ]
    result = jackknife_ensemble_stability(
        replicas(values, depth=4),
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
        q_temperature=0.05,
        min_action_coverage=2 / 3,
    )
    assert result["stable"] is True
    assert len(result["leave_one_out"]) == 3
