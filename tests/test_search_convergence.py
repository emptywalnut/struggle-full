from __future__ import annotations

from dataclasses import replace

import numpy as np

from scripts.classify_search_convergence import adjudication_priority, classify_root, comparison
from struggle_ai.search_replay import SearchReplayExample, SearchReplayMetadata


def example(q: list[float], *, depth: int, budget: int, visits: list[float] | None = None) -> SearchReplayExample:
    legal = len(q)
    visits_array = np.asarray(visits or [budget / legal] * legal, dtype=np.float32)
    policy = visits_array / visits_array.sum()
    metadata = SearchReplayMetadata(
        side="us",
        trigger_categories=(),
        legal_actions=tuple({"type": "choice", "value": str(index), "label": str(index)} for index in range(legal)),
        hierarchical_path=(),
        raw_action=0,
        search_action=int(np.argmax(q)),
        search_depth=depth,
        simulation_budget=budget,
        root_q_gap=0.1,
        action_flipped=True,
    )
    return SearchReplayExample(
        observation={"action_mask": np.ones(legal, dtype=np.float32)},
        raw_policy=np.full(legal, 1 / legal, dtype=np.float32),
        visit_counts=visits_array,
        search_policy=policy,
        q_values=np.asarray(q, dtype=np.float32),
        metadata=metadata,
    )


def test_comparison_detects_stable_ranking() -> None:
    result = comparison(
        example([0.1, 0.2, 0.5], depth=4, budget=128),
        example([0.0, 0.3, 0.6], depth=4, budget=512),
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
    )
    assert result["stable"] is True
    assert result["left_top"] == result["right_top"] == 2


def test_comparison_rejects_reversed_ranking() -> None:
    result = comparison(
        example([0.1, 0.2, 0.5], depth=4, budget=128),
        example([0.6, 0.3, 0.0], depth=4, budget=512),
        min_rank_correlation=0.7,
        meaningful_q_gap=0.01,
    )
    assert result["stable"] is False


def test_classify_budget_convergence() -> None:
    examples = {
        "d2_32": example([0.1, 0.2, 0.5], depth=2, budget=32),
        "d2_128": example([0.0, 0.3, 0.6], depth=2, budget=128),
        "d4_128": example([-0.1, 0.25, 0.55], depth=4, budget=128),
        "d4_512": example([-0.2, 0.2, 0.5], depth=4, budget=512),
        "d6_512": example([-0.1, 0.15, 0.45], depth=6, budget=512),
    }
    configs = {name: tuple(map(int, name[1:].split("_"))) for name in examples}
    status, teacher, _evidence = classify_root(
        examples, configs, min_rank_correlation=0.7, meaningful_q_gap=0.01
    )
    assert status == "BUDGET_CONVERGENCE"
    assert teacher == "d6_512"


def test_classify_horizon_convergence() -> None:
    examples = {
        "d2_128": example([0.6, 0.3, 0.0], depth=2, budget=128),
        "d4_512": example([0.0, 0.3, 0.6], depth=4, budget=512),
        "d6_2048": example([-0.1, 0.2, 0.5], depth=6, budget=2048),
    }
    configs = {name: tuple(map(int, name[1:].split("_"))) for name in examples}
    status, teacher, _evidence = classify_root(
        examples, configs, min_rank_correlation=0.7, meaningful_q_gap=0.01
    )
    assert status == "HORIZON_CONVERGENCE"
    assert teacher == "d6_2048"


def test_classify_horizon_convergence_after_same_depth_budget_stabilizes() -> None:
    examples = {
        "d4_512": example([0.6, 0.3, 0.0], depth=4, budget=512),
        "d6_512": example([0.0, 0.3, 0.6], depth=6, budget=512),
        "d6_2048": example([-0.1, 0.2, 0.5], depth=6, budget=2048),
    }
    configs = {name: tuple(map(int, name[1:].split("_"))) for name in examples}
    status, teacher, _evidence = classify_root(
        examples, configs, min_rank_correlation=0.7, meaningful_q_gap=0.01
    )
    assert status == "HORIZON_CONVERGENCE"
    assert teacher == "d6_2048"


def test_cutoff_search_is_not_a_teacher() -> None:
    good = example([0.1, 0.2, 0.5], depth=4, budget=512)
    bad = replace(good, metadata=replace(good.metadata, search_depth=6, simulation_budget=2048, node_budget_cutoffs=1))
    examples = {"d4_512": good, "d6_2048": bad}
    configs = {"d4_512": (4, 512), "d6_2048": (6, 2048)}
    status, teacher, _evidence = classify_root(
        examples, configs, min_rank_correlation=0.7, meaningful_q_gap=0.01
    )
    assert status == "UNRESOLVED"
    assert teacher == "d4_512"


def test_adjudication_priority_prefers_strategic_large_gap_before_branching() -> None:
    scoring = {
        "seed": 2,
        "action_index": 4,
        "category": "scoring",
        "strongest_q_gap": 0.03,
        "legal_count": 8,
    }
    high_branch = {
        "seed": 1,
        "action_index": 3,
        "category": "influence",
        "strongest_q_gap": 0.20,
        "legal_count": 72,
    }
    assert sorted([high_branch, scoring], key=adjudication_priority)[0] is scoring
