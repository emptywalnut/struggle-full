from __future__ import annotations

from types import SimpleNamespace

from scripts.split_search_replay_by_game import split_games


def test_split_games_is_disjoint_and_complete() -> None:
    examples = []
    for index in range(12):
        winner = "us" if index % 2 == 0 else "ussr"
        for _ in range(index + 1):
            examples.append(
                SimpleNamespace(
                    metadata=SimpleNamespace(
                        source_game=f"game-{index}",
                        complete_game=True,
                        game_winner=winner,
                    )
                )
            )
    training, validation = split_games(examples, folds=6, validation_folds=2)
    assert not training & validation
    assert training | validation == {f"game-{index}" for index in range(12)}
    assert len(validation) == 4
