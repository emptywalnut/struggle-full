from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_puct_teacher_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_puct_teacher_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_collects_balanced_zero_cutoff_roots(tmp_path: Path) -> None:
    shard = tmp_path / "shard-000"
    (shard / "game_logs").mkdir(parents=True)
    game_log = shard / "game_logs" / "games.jsonl"
    game_log.write_text("{}\n{}\n", encoding="utf-8")
    records = []
    for orientation, side, advantage in (
        ("candidate_us", "us", 0.12),
        ("candidate_ussr", "ussr", 0.08),
    ):
        records.append(
            {
                "seed": 7,
                "orientation": orientation,
                "search_events": [
                    {
                        "episode_step": 42,
                        "side": side,
                        "action_flipped": True,
                        "q_advantage_over_raw": advantage,
                        "root_q_gap": advantage,
                        "triggers": ["influence_allocation"],
                        "simulations": 128,
                        "max_meaningful_depth_reached": 4,
                        "max_turn_delta": 1,
                        "cross_turn_leaf_evaluations": 3,
                        "atomic_horizon_cutoffs": 0,
                        "node_budget_cutoffs": 0,
                    },
                    {
                        "episode_step": 43,
                        "side": side,
                        "action_flipped": True,
                        "q_advantage_over_raw": 0.5,
                        "triggers": ["opponent_event"],
                        "atomic_horizon_cutoffs": 1,
                    },
                ],
            }
        )
    (shard / "results.json").write_text(json.dumps({"records": records}), encoding="utf-8")

    positions = module.collect_positions(
        tmp_path,
        min_q_advantage=0.03,
        require_flip=True,
        require_cross_turn=True,
    )
    selected = module.balanced_top_with_cross_turn(positions, 2, 2)

    assert [position["side"] for position in selected] == ["us", "ussr"]
    assert [position["game_occurrence"] for position in selected] == [0, 1]
    assert all(position["action_index"] == 42 for position in selected)
    assert all(position["game_log"] == str(game_log.resolve()) for position in selected)
