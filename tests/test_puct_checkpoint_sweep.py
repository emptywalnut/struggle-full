from pathlib import Path

from scripts.sweep_puct_checkpoints import choose, checkpoint_update
from scripts.run_policy_value_compatibility import summarize


def summary(score, us, ussr):
    return {
        "score": score,
        "by_candidate_side": {
            "us": {"score": us},
            "ussr": {"score": ussr},
        },
    }


def test_intermediate_requires_clear_strength_margin_over_final():
    records = {
        "u100": {
            "raw": summary(0.625, 0.5, 0.75),
            "d2": summary(0.625, 0.5, 0.75),
        },
        "ppo_final": {
            "raw": summary(0.5, 0.5, 0.5),
            "d2": summary(0.5, 0.5, 0.5),
        },
    }
    selected, details = choose(records, "ppo_final", 0.125)
    assert selected == "u100"
    assert details["intermediate_clearly_dominates_final"] is True

    records["u100"]["d2"] = summary(0.625, 0.25, 1.0)
    selected, details = choose(records, "ppo_final", 0.125)
    assert selected == "ppo_final"
    assert details["intermediate_clearly_dominates_final"] is False


def test_checkpoint_update_supports_intermediate_and_distilled_final(tmp_path):
    intermediate = tmp_path / "updates-000150"
    final = tmp_path / "distilled"
    assert checkpoint_update("u150", intermediate, None) == 150
    assert checkpoint_update(
        "u200",
        final,
        {"checkpoint": str(final), "updates": 200},
    ) == 200


def test_compatibility_summary_identifies_value_degradation(tmp_path):
    positions = ["root"]
    rows = {
        "parent_policy_parent_value": 1,
        "child_policy_parent_value": 1,
        "parent_policy_child_value": 2,
        "child_policy_child_value": 2,
    }
    for combination, action_index in rows.items():
        path = tmp_path / combination
        path.mkdir()
        (path / "root.json").write_text(
            __import__("json").dumps({
                "raw": {
                    "top10": [{"action": "raw", "action_index": 0}],
                    "raw_critic": 0.0,
                    "calibrated_critic": 0.0,
                },
                "root": {"information_mask": {"opponent_hand_masked": True}},
                "puct_depth2": {
                    "budgets": {
                        "32": [{
                            "action": str(action_index),
                            "action_index": action_index,
                            "N": 32,
                            "Q": 0.1,
                            "P": 0.5,
                        }]
                    }
                },
            }),
            encoding="utf-8",
        )
    report = summarize(tmp_path, positions, 32)
    assert report["diagnosis"] == "value_degradation"
