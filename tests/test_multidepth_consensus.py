from dataclasses import replace

import numpy as np

from scripts.build_multidepth_consensus_replay import consensus_example, rank_correlation
from struggle_ai.search_replay import SearchReplayExample, SearchReplayMetadata


def example(q, visits):
    legal = ({"type": "choice", "value": "a", "label": "A"}, {"type": "choice", "value": "b", "label": "B"})
    metadata = SearchReplayMetadata(
        side="us", trigger_categories=(), legal_actions=legal, hierarchical_path=(),
        raw_action=1, search_action=0, search_depth=2, simulation_budget=32,
        root_q_gap=0.1, action_flipped=True, complete_game=True,
    )
    return SearchReplayExample(
        observation={"action_mask": np.asarray([1, 1], dtype=np.float32)},
        raw_policy=np.asarray([0.4, 0.6], dtype=np.float32),
        visit_counts=np.asarray(visits, dtype=np.float32),
        search_policy=np.asarray([0.5, 0.5], dtype=np.float32),
        q_values=np.asarray(q, dtype=np.float32), metadata=metadata,
    )


def test_consensus_uses_agreed_q_ranking():
    d2 = example([0.4, 0.1], [20, 12])
    d4 = replace(d2, q_values=np.asarray([0.5, 0.0], dtype=np.float32), metadata=replace(d2.metadata, search_depth=4, simulation_budget=128))
    result, evidence = consensus_example(d2, d4, None, q_temperature=0.02, min_rank_correlation=0.5, min_q_gap=0.002)
    assert result is not None
    assert result.metadata.search_action == 0
    assert evidence["top_agreement"]


def test_consensus_rejects_structural_depth_disagreement():
    d2 = example([0.4, 0.1], [20, 12])
    d4 = replace(d2, q_values=np.asarray([0.0, 0.5], dtype=np.float32), metadata=replace(d2.metadata, search_depth=4, simulation_budget=128))
    result, evidence = consensus_example(d2, d4, None, q_temperature=0.02, min_rank_correlation=0.5, min_q_gap=0.002)
    assert result is None
    assert evidence["reason"] == "depth_disagreement"


def test_consensus_rejects_d6_that_supports_neither_depth():
    legal = (
        {"type": "choice", "value": "a", "label": "A"},
        {"type": "choice", "value": "b", "label": "B"},
        {"type": "choice", "value": "c", "label": "C"},
    )
    base = example([0.5, 0.2], [20, 12])
    base = replace(
        base,
        observation={"action_mask": np.asarray([1, 1, 1], dtype=np.float32)},
        raw_policy=np.asarray([0.4, 0.3, 0.3], dtype=np.float32),
        visit_counts=np.asarray([12, 10, 10], dtype=np.float32),
        search_policy=np.asarray([0.4, 0.3, 0.3], dtype=np.float32),
        q_values=np.asarray([0.5, 0.2, 0.1], dtype=np.float32),
        metadata=replace(base.metadata, legal_actions=legal),
    )
    d4 = replace(base, q_values=np.asarray([0.1, 0.5, 0.2], dtype=np.float32))
    d6 = replace(base, q_values=np.asarray([0.1, 0.2, 0.6], dtype=np.float32))
    result, evidence = consensus_example(
        base, d4, d6,
        q_temperature=0.02,
        min_rank_correlation=0.5,
        min_q_gap=0.002,
    )
    assert result is None
    assert evidence["reason"] == "unresolved_with_d6"


def test_consensus_uses_d6_to_adjudicate_for_d4():
    legal = (
        {"type": "choice", "value": "a", "label": "A"},
        {"type": "choice", "value": "b", "label": "B"},
        {"type": "choice", "value": "c", "label": "C"},
    )
    base = example([0.6, 0.2], [20, 12])
    base = replace(
        base,
        observation={"action_mask": np.asarray([1, 1, 1], dtype=np.float32)},
        raw_policy=np.asarray([0.4, 0.3, 0.3], dtype=np.float32),
        visit_counts=np.asarray([12, 10, 10], dtype=np.float32),
        search_policy=np.asarray([0.4, 0.3, 0.3], dtype=np.float32),
        q_values=np.asarray([0.6, 0.2, 0.1], dtype=np.float32),
        metadata=replace(base.metadata, legal_actions=legal),
    )
    d4 = replace(base, q_values=np.asarray([0.1, 0.6, 0.2], dtype=np.float32))
    d6 = replace(base, q_values=np.asarray([0.0, 0.7, 0.3], dtype=np.float32))
    result, evidence = consensus_example(
        base, d4, d6,
        q_temperature=0.02,
        min_rank_correlation=0.5,
        min_q_gap=0.002,
    )
    assert result is not None
    assert result.metadata.search_action == 1
    assert evidence["adjudicated_depth"] == 4


def test_rank_correlation_is_one_for_matching_order():
    assert rank_correlation(np.asarray([1.0, 2.0, 3.0]), np.asarray([0.0, 5.0, 8.0]), np.asarray([True, True, True])) == 1.0
