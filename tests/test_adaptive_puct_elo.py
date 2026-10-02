from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_adaptive_puct_elo import (
    Agent,
    aggregate_agent,
    contender_ids,
    mode_args,
    update_existing_elo,
)


def result(agent: str, mode: str, score: float, low: float, high: float, us: float, ussr: float, games: int = 50) -> dict:
    return {
        "agent_id": agent,
        "mode": mode,
        "games": games,
        "score": score,
        "paired_bootstrap_ci95_low": low,
        "paired_bootstrap_ci95_high": high,
        "by_candidate_side": {"us": {"score": us}, "ussr": {"score": ussr}},
    }


def test_mode_args_use_asymmetric_search_against_raw_anchor() -> None:
    assert "--no-candidate-search" in mode_args("raw")
    assert "--candidate-search" in mode_args("d2")
    assert "--no-opponent-search" in mode_args("d2")
    assert mode_args("d4")[mode_args("d4").index("--meaningful-depth") + 1] == "4"


def test_contenders_keep_leader_and_overlapping_runner_up_but_drop_dominated() -> None:
    results = [
        result("leader", "d2", 0.64, 0.52, 0.76, 0.68, 0.60),
        result("overlap", "d2", 0.58, 0.45, 0.70, 0.56, 0.60),
        result("dominated", "d2", 0.38, 0.25, 0.50, 0.40, 0.36),
    ]
    assert contender_ids(results, 50) == {"leader", "overlap"}


def test_informative_100_game_leader_does_not_force_200_games() -> None:
    results = [
        result("leader", "raw", 0.68, 0.58, 0.77, 0.70, 0.66, games=100),
        result("runner-up", "raw", 0.46, 0.35, 0.54, 0.48, 0.44, games=100),
    ]
    assert contender_ids(results, 100) == set()


def test_stage_aggregation_ignores_later_completed_shards(
    tmp_path: Path, monkeypatch
) -> None:
    args = SimpleNamespace(
        output_dir=tmp_path,
        screen_games=50,
        confirm_games=100,
        max_games=200,
        bootstrap_samples=10,
        seed_base=123,
    )
    agent = Agent("candidate", tmp_path / "checkpoint", "d2")
    for target in (50, 100):
        output = tmp_path / "evaluations" / "candidate" / "d2" / f"to-{target:03d}"
        output.mkdir(parents=True)
        (output / "results.json").write_text(
            json.dumps({"records": [{"seed": target, "orientation": "candidate_us"}]}),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        "scripts.run_adaptive_puct_elo.summary",
        lambda _args, records: {"games": len(records)},
    )

    stage = aggregate_agent(args, agent, max_target=50)
    full = aggregate_agent(args, agent)

    assert stage is not None and stage["games"] == 1
    assert stage["completed_targets"] == [50]
    assert full is not None and full["games"] == 2
    assert full["completed_targets"] == [50, 100]


def test_existing_elo_is_updated_in_place_with_mode_identity(tmp_path: Path) -> None:
    elo_path = tmp_path / "elo_unified.json"
    elo_path.write_text(json.dumps({
        "version": 1,
        "k_factor": 32,
        "leaderboards": {
            side: {"ratings": {"historical": 1600.0}, "games": {"historical": 10}}
            for side in ("us", "ussr", "bot")
        },
        "matches": [{"historical": True}],
    }), encoding="utf-8")
    args = SimpleNamespace(
        elo_path=elo_path,
        output_dir=tmp_path / "adaptive-run",
        reference_label="parent",
        reference_checkpoint=tmp_path / "parent",
        confirm_games=100,
    )
    evaluated = {
        "agent_id": "adaptive:student/d2",
        "games": 50,
        "elo_delta": 70.0,
        "by_candidate_side": {
            "us": {"games": 25, "score": 0.6},
            "ussr": {"games": 25, "score": 0.56},
        },
    }
    update_existing_elo(args, {"agents": []}, [evaluated], {"adaptive:student/d2"})
    state = json.loads(elo_path.read_text(encoding="utf-8"))
    assert state["matches"] == [{"historical": True}]
    assert state["leaderboards"]["bot"]["ratings"]["historical"] == 1600.0
    assert state["leaderboards"]["bot"]["ratings"]["adaptive:student/d2:shared_policy"] == 1570.0
    entry = state["adaptive_evaluations"]["campaigns"]["adaptive-run"]["entries"]["adaptive:student/d2"]
    assert entry["status"] == "preliminary"
