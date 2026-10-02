#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare critic audits with game-block bootstrap.")
    parser.add_argument("--parent-records", type=Path, required=True)
    parser.add_argument("--child-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=27)
    return parser.parse_args()


def load_records(path: Path) -> dict[tuple[str, int | None, str], dict]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["source_game"]), row.get("source_step"), str(row["side"]))
            if key in records:
                raise ValueError(f"duplicate value-audit key in {path}: {key}")
            records[key] = row
    return records


def summarize_deltas(deltas: list[float]) -> dict[str, float | int]:
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "examples": int(array.size),
        "mean_child_minus_parent_mse": float(array.mean()),
        "median_child_minus_parent_mse": float(np.median(array)),
        "child_better_fraction": float(np.mean(array < 0.0)),
    }


def main() -> None:
    args = parse_args()
    parent = load_records(args.parent_records.expanduser().resolve())
    child = load_records(args.child_records.expanduser().resolve())
    if parent.keys() != child.keys():
        raise SystemExit(
            f"record keys differ: parent_only={len(parent.keys()-child.keys())}, "
            f"child_only={len(child.keys()-parent.keys())}"
        )

    by_game: dict[str, list[float]] = defaultdict(list)
    by_stratum: dict[str, list[float]] = defaultdict(list)
    probability_deltas = []
    for key, parent_row in parent.items():
        child_row = child[key]
        target = float(parent_row["target"])
        if target != float(child_row["target"]):
            raise ValueError(f"target mismatch at {key}")
        delta = (float(child_row["calibrated"]) - target) ** 2 - (
            float(parent_row["calibrated"]) - target
        ) ** 2
        by_game[key[0]].append(delta)
        outcome = "win" if target > 0.0 else "loss" if target < 0.0 else "draw"
        by_stratum[f"{key[2]}_{outcome}"].append(delta)
        probability_deltas.append(
            abs(float(child_row["teacher_probability"]) - float(parent_row["teacher_probability"]))
        )

    game_means = {game: float(np.mean(values)) for game, values in by_game.items()}
    games = sorted(game_means)
    rng = random.Random(args.seed)
    bootstrap = []
    for _ in range(max(1, int(args.bootstrap_samples))):
        sampled = [game_means[rng.choice(games)] for _ in games]
        bootstrap.append(float(np.mean(sampled)))
    low, high = np.quantile(np.asarray(bootstrap), [0.025, 0.975])
    all_deltas = [delta for values in by_game.values() for delta in values]
    result = {
        "parent_records": str(args.parent_records.expanduser().resolve()),
        "child_records": str(args.child_records.expanduser().resolve()),
        "games": len(games),
        "overall": summarize_deltas(all_deltas),
        "game_macro_mean_child_minus_parent_mse": float(np.mean(list(game_means.values()))),
        "game_block_bootstrap_95_ci": [float(low), float(high)],
        "games_improved_fraction": float(np.mean(np.asarray(list(game_means.values())) < 0.0)),
        "by_stratum": {
            stratum: summarize_deltas(values) for stratum, values in sorted(by_stratum.items())
        },
        "max_teacher_probability_abs_delta": float(max(probability_deltas, default=0.0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
