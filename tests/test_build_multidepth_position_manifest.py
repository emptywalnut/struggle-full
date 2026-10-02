from types import SimpleNamespace

import numpy as np

from scripts.build_multidepth_position_manifest import root_category, select_roots


def example(
    source: str,
    step: int,
    *,
    category: str,
    legal_count: int,
    q_gap: float,
    turn: int = 2,
    side: str = "us",
) -> SimpleNamespace:
    if category == "scoring_card":
        decisions = ["list"] * legal_count
        prompts = ["US pick a card: "] * legal_count
        triggers = ("scoring_card",)
    else:
        decisions = ["country_click"] * legal_count
        prompts = ["Place influence"] * legal_count
        triggers = ()
    metadata = SimpleNamespace(
        hierarchy_index=0,
        complete_game=True,
        atomic_horizon_cutoffs=0,
        node_budget_cutoffs=0,
        legal_actions=[
            {"decision": decision, "prompt": prompt}
            for decision, prompt in zip(decisions, prompts)
        ],
        trigger_categories=triggers,
        root_q_gap=q_gap,
        source_game=source,
        source_step=step,
        side=side,
    )
    global_features = np.zeros(8, dtype=np.float32)
    global_features[2] = float(turn) / 10.0
    return SimpleNamespace(
        metadata=metadata,
        legal_count=legal_count,
        observation={"global": global_features},
    )


def test_select_roots_preserves_held_scoring_root_before_per_game_cap() -> None:
    source = "puct-selfplay-000002-seed-98440002"
    allocation = example(
        source,
        117,
        category="influence_allocation",
        legal_count=72,
        q_gap=0.2,
    )
    scoring = example(
        source,
        208,
        category="scoring_card",
        legal_count=3,
        q_gap=0.02,
    )

    selected = select_roots(
        [allocation, scoring],
        max_roots=1,
        max_per_game=1,
        held_scoring_sources={source},
    )

    assert selected == [scoring]


def test_select_roots_reserves_late_era_before_game_quota_is_filled() -> None:
    source = "puct-selfplay-000008-seed-98400008"
    early = [
        example(
            source,
            step,
            category="influence_allocation",
            legal_count=72 - step,
            q_gap=0.2,
            turn=2,
        )
        for step in range(8)
    ]
    late_us = example(
        source,
        500,
        category="scoring_card",
        legal_count=4,
        q_gap=0.03,
        turn=8,
        side="us",
    )
    late_ussr = example(
        source,
        501,
        category="scoring_card",
        legal_count=3,
        q_gap=0.02,
        turn=8,
        side="ussr",
    )

    selected = select_roots(
        [*early, late_us, late_ussr],
        max_roots=8,
        max_per_game=8,
    )

    assert late_us in selected
    assert late_ussr in selected


def test_root_category_recognizes_saito_country_mouseup_allocation() -> None:
    root = example(
        "puct-selfplay-000001-seed-1",
        10,
        category="other",
        legal_count=3,
        q_gap=0.01,
    )
    root.metadata.legal_actions = [
        {"decision": "country_mouseup", "prompt": "Place 2 influence"}
        for _ in range(3)
    ]

    assert root_category(root) == "influence_allocation"
