from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.evaluate_search_replay_checkpoint import aggregate, algorithm_args, grouped


def test_algorithm_args_forwards_value_state_tower(tmp_path) -> None:
    args = SimpleNamespace(
        checkpoint=tmp_path,
        num_gpus=1.0,
        hidden=256,
        value_state_tower=True,
        history_layers=2,
        card_history_layers=2,
        history_attention_heads=4,
        history_dropout=0.05,
        graph_layers=2,
        graph_neighbor_hops=2,
        policy_temperature=1.0,
    )

    assert algorithm_args(args).value_state_tower is True


def test_aggregate_reports_weighted_teacher_transfer() -> None:
    records = [
        {
            "sample_weight": 1.0,
            "teacher_probability": 0.25,
            "teacher_top1": False,
            "q_advantage_over_raw": 0.01,
            "side": "us",
            "hierarchy_level": "card",
            "triggers": ["opponent_event"],
        },
        {
            "sample_weight": 3.0,
            "teacher_probability": 0.75,
            "teacher_top1": True,
            "q_advantage_over_raw": 0.09,
            "side": "ussr",
            "hierarchy_level": "influence_allocation",
            "triggers": ["allocation", "scoring_card"],
        },
    ]

    result = aggregate(records)

    assert result["examples"] == 2
    assert result["teacher_top1_accuracy"] == 0.5
    assert result["mean_teacher_probability"] == 0.5
    assert result["q_weighted_teacher_top1_accuracy"] == 0.75
    assert result["q_weighted_teacher_probability"] == pytest.approx(0.625)
    assert result["mean_q_advantage"] == pytest.approx(0.05)


def test_grouped_expands_multilabel_trigger_categories() -> None:
    records = [
        {
            "sample_weight": 1.0,
            "teacher_probability": 0.5,
            "teacher_top1": True,
            "q_advantage_over_raw": 0.03,
            "side": "us",
            "hierarchy_level": "card",
            "triggers": ["opponent_event", "scoring_card"],
        }
    ]

    result = grouped(records, "triggers")

    assert set(result) == {"opponent_event", "scoring_card"}
    assert result["opponent_event"]["examples"] == 1
