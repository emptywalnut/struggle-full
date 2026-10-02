from __future__ import annotations

from dataclasses import replace

import numpy as np

from scripts.analyze_search_depth_compatibility import adaptive_policy_summary, pair_summary
from struggle_ai.search_replay import SearchReplayExample, SearchReplayMetadata


def example(q: list[float], visits: list[float], *, side: str = "us") -> SearchReplayExample:
    legal = len(q)
    metadata = SearchReplayMetadata(
        side=side,
        trigger_categories=(),
        legal_actions=tuple(
            {"type": "choice", "value": str(index), "label": str(index)}
            for index in range(legal)
        ),
        hierarchical_path=(),
        raw_action=0,
        search_action=int(np.argmax(q)),
        search_depth=2,
        simulation_budget=int(sum(visits)),
        root_q_gap=0.1,
        action_flipped=True,
        source_game="seed-1-occurrence-0",
        source_step=2,
    )
    visit_array = np.asarray(visits, dtype=np.float32)
    return SearchReplayExample(
        observation={"action_mask": np.ones(legal, dtype=np.float32)},
        raw_policy=np.full(legal, 1 / legal, dtype=np.float32),
        visit_counts=visit_array,
        search_policy=visit_array / visit_array.sum(),
        q_values=np.asarray(q, dtype=np.float32),
        metadata=metadata,
    )


def test_pair_summary_reports_stable_q_ranking_and_scale() -> None:
    left = example([0.1, 0.3, 0.6], [1, 4, 10])
    right = replace(left, q_values=np.asarray([0.2, 0.5, 1.1], dtype=np.float32))
    summary = pair_summary(
        {(1, 2): left},
        {(1, 2): right},
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
    )
    assert summary["counts"]["comparable"] == 1
    assert summary["stable_rate"] == 1.0
    assert summary["same_top_rate"] == 1.0
    assert summary["q_scale_ratio_right_over_left"]["median"] > 1.0


def test_pair_summary_excludes_cutoff_roots() -> None:
    left = example([0.1, 0.3], [4, 4])
    right = replace(
        left,
        metadata=replace(left.metadata, node_budget_cutoffs=1),
    )
    summary = pair_summary(
        {(1, 2): left},
        {(1, 2): right},
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
    )
    assert summary["counts"]["cutoff_or_rejected"] == 1
    assert summary["counts"].get("comparable", 0) == 0


def test_adaptive_policy_falls_shallow_when_deep_ranking_is_unresolved() -> None:
    key = (1, 2)
    maps = {
        "d2_32": {key: example([0.7, 0.2, 0.1], [20, 8, 4])},
        "d2_128": {key: example([0.1, 0.7, 0.2], [10, 90, 28])},
        "d4_128": {key: example([0.1, 0.2, 0.8], [10, 20, 98])},
        "d6_512": {key: example([0.0, 0.8, 0.1], [20, 450, 42])},
    }
    summary = adaptive_policy_summary(
        maps,
        base_label="d2_32",
        budget_label="d2_128",
        deep_label="d4_128",
        benchmark_label="d6_512",
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
    )
    assert summary["counts"]["status:UNRESOLVED"] == 1
    assert summary["fallback_rate"] == 1.0
    assert summary["benchmark_top_agreement_rate"] == 1.0
