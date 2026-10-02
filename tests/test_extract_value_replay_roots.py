from scripts.extract_value_replay_roots import game_fold


def test_game_fold_is_stable_and_bounded() -> None:
    values = [game_fold(f"campaign:game-{index}", 5) for index in range(100)]
    assert values == [game_fold(f"campaign:game-{index}", 5) for index in range(100)]
    assert set(values) <= set(range(5))
    assert len(set(values)) == 5
