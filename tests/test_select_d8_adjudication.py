from scripts.select_d8_adjudication import select_positions


def test_select_positions_requires_d4_disagreement_or_ambiguity_and_balances_sides():
    positions = [
        {"seed": 1, "action_index": 10, "game_occurrence": 0, "side": "us", "teacher_action": 2},
        {"seed": 2, "action_index": 20, "game_occurrence": 0, "side": "ussr", "teacher_action": 3},
        {"seed": 3, "action_index": 30, "game_occurrence": 0, "side": "us", "teacher_action": 4},
    ]
    d4 = {
        (1, 10, 0): {
            "search_action": 8,
            "raw_action": 2,
            "root_q_gap": 0.3,
            "q_advantage_over_raw": 0.4,
            "teacher_visit_share": 0.8,
        },
        (2, 20, 0): {
            "search_action": 3,
            "raw_action": 1,
            "root_q_gap": 0.01,
            "q_advantage_over_raw": 0.1,
            "teacher_visit_share": 0.6,
        },
        (3, 30, 0): {
            "search_action": 4,
            "raw_action": 4,
            "root_q_gap": 0.2,
            "q_advantage_over_raw": 0.0,
            "teacher_visit_share": 0.9,
        },
    }

    selected = select_positions(positions, d4, q_gap_threshold=0.02, limit=4)

    assert [item["side"] for item in selected] == ["us", "ussr"]
    assert selected[0]["adjudication_reasons"] == ["d2_d4_action_disagreement"]
    assert selected[1]["adjudication_reasons"] == ["d4_small_q_gap"]


def test_completion_aware_selection_rejects_wide_or_cutoff_d4_roots():
    positions = [
        {"seed": 1, "action_index": 10, "game_occurrence": 0, "side": "us", "teacher_action": 2},
        {"seed": 2, "action_index": 20, "game_occurrence": 0, "side": "ussr", "teacher_action": 3},
    ]
    base = {
        "search_action": 8,
        "raw_action": 2,
        "root_q_gap": 0.01,
        "q_advantage_over_raw": 0.4,
        "teacher_visit_share": 0.6,
        "atomic_horizon_cutoffs": 0,
        "node_budget_cutoffs": 0,
    }
    d4 = {
        (1, 10, 0): {**base, "legal_actions": [{"value": str(i)} for i in range(8)]},
        (2, 20, 0): {
            **base,
            "legal_actions": [{"value": "a"}, {"value": "b"}],
            "node_budget_cutoffs": 1,
        },
    }
    assert select_positions(
        positions,
        d4,
        q_gap_threshold=0.02,
        limit=4,
        completion_aware=True,
    ) == []

    d4[(2, 20, 0)]["node_budget_cutoffs"] = 0
    selected = select_positions(
        positions,
        d4,
        q_gap_threshold=0.02,
        limit=4,
        completion_aware=True,
    )
    assert [item["seed"] for item in selected] == [2]
    assert selected[0]["d4"]["likely_complete"] is True
