from __future__ import annotations

from scripts.build_collected_search_audit_panel import select_audit_panel


def candidate(tmp_path, *, side: str, turn: int, index: int, category: str = "scoring_card"):
    log = tmp_path / f"game-{index}.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    return {
        "seed": 1000 + index,
        "action_index": index,
        "side": side,
        "turn": turn,
        "source_game": f"game-{index}",
        "game_log": str(log),
        "teacher_eligible": True,
        "convergence_status": "BUDGET_CONVERGENCE",
        "convergence_q_gap": 0.03,
        "latency_seconds": 1.0,
        "triggers": [category],
    }


def test_panel_balances_side_and_era(tmp_path):
    rows = []
    index = 0
    for side in ("us", "ussr"):
        for turn in (2, 5, 9):
            rows.append(candidate(tmp_path, side=side, turn=turn, index=index))
            index += 1

    selected = select_audit_panel(
        rows,
        per_side_era=1,
        min_q_gap=0.02,
        max_base_latency=10.0,
        max_per_game=1,
    )

    assert len(selected) == 6
    assert {(row["side"], row["era"]) for row in selected} == {
        (side, phase)
        for side in ("us", "ussr")
        for phase in ("early", "mid", "late")
    }


def test_panel_rejects_unvalidated_or_small_gap(tmp_path):
    rows = [candidate(tmp_path, side="us", turn=2, index=0)]
    rows[0]["convergence_status"] = "UNVALIDATED"

    try:
        select_audit_panel(
            rows,
            per_side_era=1,
            min_q_gap=0.02,
            max_base_latency=10.0,
            max_per_game=1,
        )
    except ValueError as exc:
        assert "insufficient audit candidates" in str(exc)
    else:
        raise AssertionError("unvalidated candidate entered audit panel")


def test_panel_reserves_games_for_scarce_late_buckets(tmp_path):
    rows = []
    index = 0
    for side in ("us", "ussr"):
        for turn in (2, 5):
            for game in range(8):
                row = candidate(tmp_path, side=side, turn=turn, index=index)
                row["source_game"] = f"game-{game}"
                rows.append(row)
                index += 1
        for game in range(6, 8):
            row = candidate(tmp_path, side=side, turn=9, index=index)
            row["source_game"] = f"game-{game}"
            rows.append(row)
            index += 1

    selected = select_audit_panel(
        rows,
        per_side_era=1,
        min_q_gap=0.02,
        max_base_latency=10.0,
        max_per_game=1,
    )
    assert len(selected) == 6
    assert len({row["source_game"] for row in selected}) == 6
