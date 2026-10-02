import json

from scripts.build_search_scaling_manifest import annotate, select_positions


def position(index: int, *, side: str, category: str, reason: str = "") -> dict:
    return {
        "seed": 10_000 + index,
        "action_index": index,
        "game_occurrence": 0,
        "source_game": f"game-{index}",
        "side": side,
        "era": "early",
        "category": category,
        "branch_bucket": "low",
        "legal_count": 4,
        "terminal_reason": reason,
        "d4_q_gap": float(index) / 100.0,
    }


def test_select_positions_reserves_held_scoring_horizon_root() -> None:
    rows = [
        position(index, side="us" if index < 4 else "ussr", category="other")
        for index in range(8)
    ]
    held = position(
        20,
        side="us",
        category="scoring_pressure",
        reason="scoring card held",
    )
    rows.append(held)

    selected = select_positions(rows, count=6, max_per_game=1)

    assert held in selected
    assert sum(row["side"] == "us" for row in selected) == 3
    assert sum(row["side"] == "ussr" for row in selected) == 3


def test_select_positions_does_not_force_non_scoring_terminal_failure() -> None:
    rows = [
        position(index, side="us" if index < 3 else "ussr", category="other")
        for index in range(6)
    ]
    unrelated = position(
        30,
        side="us",
        category="other",
        reason="scoring card held",
    )
    rows.append(unrelated)

    selected = select_positions(rows, count=6, max_per_game=1)

    assert len(selected) == 6


def test_annotate_reads_nested_terminal_reason(tmp_path) -> None:
    game_log = tmp_path / "game.jsonl"
    game_log.write_text(
        json.dumps(
            {
                "kind": "terminal",
                "seed": 123,
                "info": {"terminal_reason": "scoring card held"},
                "actions": [
                    {
                        "before": {
                            "turn": 3,
                            "action_round": 6,
                            "phase": "action",
                            "prompt": "US pick a card",
                        }
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = position(1, side="us", category="scoring_card")
    row.update({"seed": 123, "action_index": 0, "game_log": str(game_log)})

    annotated = annotate(row, {})

    assert annotated["terminal_reason"] == "scoring card held"


def test_select_positions_preserves_era_coverage_with_single_late_game() -> None:
    rows = []
    for index in range(8):
        row = position(index, side="us" if index % 2 == 0 else "ussr", category="other")
        row["era"] = "early" if index < 4 else "mid"
        rows.append(row)
    for index, side in ((20, "us"), (21, "ussr")):
        row = position(index, side=side, category="scoring_card")
        row.update({"era": "late", "source_game": "only-late-game"})
        rows.append(row)

    selected = select_positions(rows, count=6, max_per_game=1)

    assert sum(row["era"] == "late" for row in selected) == 2
    assert sum(row["side"] == "us" for row in selected) == 3
    assert sum(row["side"] == "ussr" for row in selected) == 3


def test_select_positions_balances_eras_when_distinct_games_are_available() -> None:
    rows = []
    for era_index, era in enumerate(("early", "mid", "late")):
        for local_index in range(12):
            index = era_index * 100 + local_index
            row = position(
                index,
                side="us" if local_index % 2 == 0 else "ussr",
                category=("influence_allocation", "coup_target", "realignment_target")[
                    local_index % 3
                ],
            )
            row.update({"era": era, "source_game": f"{era}-game-{local_index}"})
            rows.append(row)

    selected = select_positions(rows, count=24, max_per_game=1)

    assert {era: sum(row["era"] == era for row in selected) for era in ("early", "mid", "late")} == {
        "early": 8,
        "mid": 8,
        "late": 8,
    }
    assert sum(row["side"] == "us" for row in selected) == 12
    assert sum(row["side"] == "ussr" for row in selected) == 12
    assert {
        (side, era): sum(
            row["side"] == side and row["era"] == era for row in selected
        )
        for side in ("us", "ussr")
        for era in ("early", "mid", "late")
    } == {
        (side, era): 4
        for side in ("us", "ussr")
        for era in ("early", "mid", "late")
    }
    assert len({row["source_game"] for row in selected}) == 24
