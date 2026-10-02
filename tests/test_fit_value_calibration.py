from __future__ import annotations

import numpy as np

from scripts.fit_value_calibration import evaluate, fit_side, game_folds, grouped_weights


def records() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for game_index in range(8):
        winner = "us" if game_index % 2 == 0 else "ussr"
        for side in ("us", "ussr"):
            target = 1.0 if side == winner else -1.0
            raw = 0.25 * target + (0.15 if side == "ussr" else -0.1)
            for _ in range(game_index + 1):
                result.append(
                    {
                        "source_game": f"game-{game_index}",
                        "side": side,
                        "game_winner": winner,
                        "prediction": raw,
                        "target": target,
                    }
                )
    return result


def test_grouped_weights_equalize_game_side_groups_within_outcome() -> None:
    data = records()
    weights = grouped_weights(data)
    totals: dict[tuple[str, str], float] = {}
    for record, weight in zip(data, weights, strict=True):
        key = (str(record["source_game"]), str(record["side"]))
        totals[key] = totals.get(key, 0.0) + float(weight)
    assert np.allclose(list(totals.values()), list(totals.values())[0])


def test_fit_improves_biased_value_predictions() -> None:
    data = records()
    parameters = {
        side: fit_side([record for record in data if record["side"] == side], 0.001)
        for side in ("us", "ussr")
    }
    report = evaluate(data, parameters)
    assert report["calibrated_weighted_mse"] < report["baseline_weighted_mse"]


def test_game_folds_never_split_a_game() -> None:
    folds = game_folds(records(), 4)
    flattened = [game for fold in folds for game in fold]
    assert len(flattened) == len(set(flattened)) == 8
