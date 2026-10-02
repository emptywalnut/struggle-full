from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.collect_puct_selfplay_parallel import _load_completed_records
from scripts.run_puct_multiday_campaign import (
    apply_compatibility_adaptation,
    campaign_should_continue,
    distillation_checkpoints,
    eval_command,
)
from scripts.postprocess_puct_cycle import _required_recovery_stages


def test_collector_resume_refuses_identity_drift(tmp_path) -> None:
    identity = {
        "checkpoint": "/tmp/a",
        "games": 4,
        "seed_base": 100,
        "meaningful_depth": 2,
        "belief_particles": 32,
        "initial_simulations": 32,
        "medium_simulations": 32,
        "max_simulations": 32,
        "search_temperature": 1.0,
        "behavior_temperature": 0.85,
    }
    assert _load_completed_records(tmp_path, identity) == []
    (tmp_path / "completed_games").mkdir()
    (tmp_path / "completed_games" / "game-000001.json").write_text(
        json.dumps({"game_index": 1, "seed": 101}) + "\n",
        encoding="utf-8",
    )
    assert _load_completed_records(tmp_path, identity)[0]["seed"] == 101

    changed = {**identity, "seed_base": 200}
    with pytest.raises(SystemExit, match="mismatched identity"):
        _load_completed_records(tmp_path, changed)


def test_campaign_eval_is_matched_side_swapped_dual_gpu(tmp_path) -> None:
    args = SimpleNamespace(search_workers=32, gpus="4,5")
    command = eval_command(
        args,
        tmp_path / "candidate",
        tmp_path / "parent",
        tmp_path / "eval",
        search=True,
        seed_pairs=10,
        seed_base=123,
    )
    assert command[command.index("--workers") + 1] == "20"
    assert command[command.index("--gpus") + 1] == "4,5"
    assert command[command.index("--seed-pairs") + 1] == "10"
    assert command[-1] == "--search"


def test_distillation_checkpoint_labels_and_compatibility_adaptation(tmp_path) -> None:
    report = {
        "updates": 200,
        "checkpoint": str(tmp_path / "distilled"),
        "intermediate_checkpoints": [
            {"update": 50, "checkpoint": str(tmp_path / "u50")},
            {"update": 100, "checkpoint": str(tmp_path / "u100")},
        ],
    }
    assert [label for label, _ in distillation_checkpoints(report)] == ["u050", "u100", "u200"]

    state = {}
    apply_compatibility_adaptation(state, {"diagnosis": "value_degradation"})
    assert state["distill_lambda_value_override"] == 0.125
    assert state["distill_lambda_value_anchor_override"] == 0.20


def test_recovery_requires_large_pair_when_large_evaluation_exists(tmp_path) -> None:
    assert _required_recovery_stages(tmp_path) == [
        "collect", "distill", "ppo", "eval_raw", "eval_d2"
    ]
    (tmp_path / "evaluation" / "large_raw").mkdir(parents=True)
    (tmp_path / "evaluation" / "large_raw" / "results.json").write_text("{}\n")
    assert _required_recovery_stages(tmp_path)[-2:] == ["eval_large_raw", "eval_large_d2"]


def test_campaign_finishes_started_cycle_after_deadline_without_starting_another(tmp_path) -> None:
    state = {"cycle": 4, "deadline": 100.0}
    assert not campaign_should_continue(state, tmp_path, 1000, now=101.0)
    cycle5 = tmp_path / "cycles" / "cycle-0005"
    cycle5.mkdir(parents=True)
    (cycle5 / "cycle_config.json").write_text("{}\n", encoding="utf-8")
    assert campaign_should_continue(state, tmp_path, 1000, now=101.0)
    (cycle5 / "cycle_report.json").write_text("{}\n", encoding="utf-8")
    assert not campaign_should_continue(state, tmp_path, 1000, now=101.0)
