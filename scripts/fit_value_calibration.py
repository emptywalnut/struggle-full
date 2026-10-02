#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a game-grouped side-conditioned affine-tanh value calibration."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--regularization", default="0,0.001,0.01,0.1,1")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    missing = [index for index, record in enumerate(records) if not record.get("source_game")]
    if missing:
        raise ValueError(f"value records missing source_game at rows {missing[:8]}")
    return records


def grouped_weights(records: list[dict[str, Any]]) -> np.ndarray:
    game_side_counts = Counter(
        (str(record["source_game"]), str(record["side"])) for record in records
    )
    stratum_game_sides: Counter[tuple[str, str]] = Counter()
    for game, side in game_side_counts:
        example = next(
            record
            for record in records
            if str(record["source_game"]) == game and str(record["side"]) == side
        )
        outcome = (
            "win"
            if float(example["target"]) > 0.0
            else "loss"
            if float(example["target"]) < 0.0
            else "draw"
        )
        stratum_game_sides[(side, outcome)] += 1
    weights = []
    for record in records:
        game_side = (str(record["source_game"]), str(record["side"]))
        outcome = (
            "win"
            if float(record["target"]) > 0.0
            else "loss"
            if float(record["target"]) < 0.0
            else "draw"
        )
        stratum = (str(record["side"]), outcome)
        weights.append(
            1.0 / game_side_counts[game_side] / max(1, stratum_game_sides[stratum])
        )
    weights = np.asarray(weights, dtype=np.float64)
    return weights / max(float(weights.sum()), 1e-12)


def prediction(raw: np.ndarray, scale: float, bias: float) -> np.ndarray:
    return np.tanh(float(scale) * raw + float(bias))


def fit_side(records: list[dict[str, Any]], regularization: float) -> dict[str, float]:
    raw = np.asarray([record["prediction"] for record in records], dtype=np.float64)
    target = np.asarray([record["target"] for record in records], dtype=np.float64)
    weights = grouped_weights(records)

    def objective(parameters: np.ndarray) -> float:
        scale = float(np.exp(parameters[0]))
        bias = float(parameters[1])
        residual = prediction(raw, scale, bias) - target
        anchor = (scale - 1.0) ** 2 + bias**2
        return float(np.sum(weights * residual**2) + float(regularization) * anchor)

    result = minimize(
        objective,
        np.asarray([0.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
    )
    if not result.success:
        raise RuntimeError(f"value calibration optimization failed: {result.message}")
    return {
        "scale": float(np.exp(result.x[0])),
        "bias": float(result.x[1]),
        "objective": float(result.fun),
    }


def evaluate(records: list[dict[str, Any]], parameters: dict[str, dict[str, float]]) -> dict[str, Any]:
    if not records:
        return {"examples": 0, "games": 0}
    target = np.asarray([record["target"] for record in records], dtype=np.float64)
    raw = np.asarray([record["prediction"] for record in records], dtype=np.float64)
    weights = grouped_weights(records)
    baseline = np.tanh(raw)
    calibrated = np.asarray(
        [
            prediction(
                np.asarray([record["prediction"]], dtype=np.float64),
                parameters[str(record["side"])]["scale"],
                parameters[str(record["side"])]["bias"],
            )[0]
            for record in records
        ],
        dtype=np.float64,
    )
    return {
        "examples": len(records),
        "games": len({str(record["source_game"]) for record in records}),
        "baseline_weighted_mse": float(np.sum(weights * (baseline - target) ** 2)),
        "calibrated_weighted_mse": float(np.sum(weights * (calibrated - target) ** 2)),
        "baseline_sign_accuracy": float(np.sum(weights * (np.sign(baseline) == np.sign(target)))),
        "calibrated_sign_accuracy": float(
            np.sum(weights * (np.sign(calibrated) == np.sign(target)))
        ),
    }


def game_folds(records: list[dict[str, Any]], folds: int) -> list[set[str]]:
    games: dict[str, str] = {}
    for record in records:
        game = str(record["source_game"])
        games.setdefault(game, str(record.get("game_winner") or "unknown"))
    buckets = [set() for _ in range(max(2, min(int(folds), len(games))))]
    by_winner: dict[str, list[str]] = {}
    for game, winner in games.items():
        by_winner.setdefault(winner, []).append(game)
    offset = 0
    for winner in sorted(by_winner):
        ordered = sorted(
            by_winner[winner],
            key=lambda game: hashlib.sha256(game.encode("utf-8")).hexdigest(),
        )
        for index, game in enumerate(ordered):
            buckets[(offset + index) % len(buckets)].add(game)
        offset += len(ordered)
    return buckets


def main() -> None:
    args = parse_args()
    records = load_records(args.records.expanduser().resolve())
    folds = game_folds(records, args.folds)
    regularizations = [float(value) for value in str(args.regularization).split(",")]
    candidates: list[dict[str, Any]] = []
    all_games = {str(record["source_game"]) for record in records}
    for regularization in regularizations:
        fold_reports: list[dict[str, Any]] = []
        for fold_index, heldout_games in enumerate(folds):
            train = [record for record in records if str(record["source_game"]) not in heldout_games]
            heldout = [record for record in records if str(record["source_game"]) in heldout_games]
            parameters = {
                side: fit_side(
                    [record for record in train if str(record["side"]) == side],
                    regularization,
                )
                for side in ("us", "ussr")
            }
            fold_reports.append(
                {
                    "fold": fold_index,
                    "heldout_games": sorted(heldout_games),
                    "parameters": parameters,
                    "heldout": evaluate(heldout, parameters),
                }
            )
        mean_baseline = float(
            np.mean([fold["heldout"]["baseline_weighted_mse"] for fold in fold_reports])
        )
        mean_calibrated = float(
            np.mean([fold["heldout"]["calibrated_weighted_mse"] for fold in fold_reports])
        )
        candidates.append(
            {
                "regularization": regularization,
                "mean_heldout_baseline_mse": mean_baseline,
                "mean_heldout_calibrated_mse": mean_calibrated,
                "mean_heldout_mse_improvement": mean_baseline - mean_calibrated,
                "folds": fold_reports,
            }
        )
    selected = min(candidates, key=lambda candidate: candidate["mean_heldout_calibrated_mse"])
    parameters = {
        side: fit_side(
            [record for record in records if str(record["side"]) == side],
            float(selected["regularization"]),
        )
        for side in ("us", "ussr")
    }
    payload = {
        "version": 1,
        "method": "side_conditioned_affine_tanh",
        "records": str(args.records.expanduser().resolve()),
        "games": len(all_games),
        "examples": len(records),
        "folds": len(folds),
        "selected_regularization": float(selected["regularization"]),
        "parameters": parameters,
        "cross_validation": candidates,
        "all_data": evaluate(records, parameters),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
