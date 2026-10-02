from scripts.select_d4_adjudication import select_positions


def test_select_d4_positions_prefers_flips_balances_sides_and_spreads_games():
    positions = [
        {
            "seed": 1,
            "action_index": 10 + index,
            "game_occurrence": 0,
            "game_log": "/game-a.jsonl",
            "side": "us",
            "action_flipped": True,
            "adjudication_reasons": ["raw_search_disagreement"],
        }
        for index in range(3)
    ]
    positions += [
        {
            "seed": 2,
            "action_index": 20,
            "game_occurrence": 0,
            "game_log": "/game-b.jsonl",
            "side": "ussr",
            "action_flipped": False,
            "adjudication_reasons": ["scoring_card"],
        },
        {
            "seed": 3,
            "action_index": 30,
            "game_occurrence": 0,
            "game_log": "/game-c.jsonl",
            "side": "us",
            "action_flipped": False,
            "adjudication_reasons": ["small_q_gap"],
        },
        {
            "seed": 4,
            "action_index": 40,
            "game_occurrence": 0,
            "game_log": "/game-d.jsonl",
            "side": "ussr",
            "action_flipped": True,
            "adjudication_reasons": ["raw_search_disagreement"],
        },
    ]

    selected = select_positions(positions, limit=4, max_per_game=1)

    assert [position["side"] for position in selected] == ["us", "ussr", "us", "ussr"]
    assert len({position["game_log"] for position in selected}) == 4
    assert selected[0]["action_flipped"] is True
    assert selected[1]["action_flipped"] is True

