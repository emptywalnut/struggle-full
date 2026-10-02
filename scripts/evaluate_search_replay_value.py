#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.batched_inference import evaluate_observation_batch, load_checkpoint_model
from struggle_ai.search_replay import SearchReplayExample, load_search_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a frozen critic against complete-game PUCT replay outcomes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--search-replay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--records-output",
        type=Path,
        default=None,
        help="Optional JSONL output with per-root predictions for game-grouped calibration.",
    )
    parser.add_argument("--max-examples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-load-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def balanced_sample(
    examples: list[SearchReplayExample], max_examples: int, seed: int
) -> list[SearchReplayExample]:
    by_side: dict[str, list[SearchReplayExample]] = defaultdict(list)
    for example in examples:
        by_side[str(example.metadata.side)].append(example)
    rng = random.Random(seed)
    selected: list[SearchReplayExample] = []
    sides = [side for side in ("us", "ussr") if by_side[side]]
    quota = max(1, int(max_examples) // max(1, len(sides)))
    for side in sides:
        pool = by_side[side]
        selected.extend(rng.sample(pool, min(quota, len(pool))))
    rng.shuffle(selected)
    return selected[: max(1, int(max_examples))]


def metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"examples": 0}
    predictions = np.asarray([record["prediction"] for record in records], dtype=np.float64)
    calibrated = np.asarray([record["calibrated"] for record in records], dtype=np.float64)
    targets = np.asarray([record["target"] for record in records], dtype=np.float64)
    teacher_probabilities = np.asarray(
        [record["teacher_probability"] for record in records], dtype=np.float64
    )
    teacher_top1 = np.asarray([record["teacher_top1"] for record in records], dtype=np.float64)
    search_cross_entropy = np.asarray(
        [record["search_cross_entropy"] for record in records], dtype=np.float64
    )
    return {
        "examples": len(records),
        "raw_mse": float(np.mean(np.square(predictions - targets))),
        "calibrated_mse": float(np.mean(np.square(calibrated - targets))),
        "sign_accuracy": float(np.mean(np.sign(predictions) == np.sign(targets))),
        "teacher_top1_accuracy": float(teacher_top1.mean()),
        "mean_teacher_probability": float(teacher_probabilities.mean()),
        "mean_search_cross_entropy": float(search_cross_entropy.mean()),
        "raw_mean": float(predictions.mean()),
        "raw_std": float(predictions.std()),
        "calibrated_mean": float(calibrated.mean()),
        "target_mean": float(targets.mean()),
        "saturated_fraction_abs_raw_gt_2": float(np.mean(np.abs(predictions) > 2.0)),
    }


def main() -> None:
    args = parse_args()
    discovered = load_search_replay(
        [path.expanduser().resolve() for path in args.search_replay],
        workers=int(args.replay_load_workers),
    )
    complete_roots = [
        example
        for example in discovered
        if example.metadata.complete_game
        and example.metadata.hierarchy_index == 0
        and example.metadata.root_value_target is not None
    ]
    examples = balanced_sample(complete_roots, args.max_examples, args.seed)
    model = load_checkpoint_model(args.checkpoint, args.device)
    records: list[dict[str, Any]] = []
    for start in range(0, len(examples), max(1, int(args.batch_size))):
        batch = examples[start : start + int(args.batch_size)]
        results = evaluate_observation_batch(
            model,
            [example.observation for example in batch],
            [example.legal_count for example in batch],
            device=args.device,
        )
        for example, result in zip(batch, results, strict=True):
            probabilities = np.asarray(result["probabilities"], dtype=np.float64)
            teacher = int(example.metadata.search_action)
            turn = int(round(float(example.observation["global"][2]) * 10.0))
            target_policy = np.asarray(
                example.search_policy[: example.legal_count], dtype=np.float64
            )
            records.append(
                {
                    "side": example.metadata.side,
                    "phase": "early" if turn <= 3 else "mid" if turn <= 7 else "late",
                    "source_game": example.metadata.source_game,
                    "source_step": example.metadata.source_step,
                    "game_winner": example.metadata.game_winner,
                    "terminal_reason": example.metadata.terminal_reason,
                    "target": float(example.metadata.root_value_target),
                    "prediction": float(result["raw_value"]),
                    "calibrated": float(result["calibrated_value"]),
                    "teacher_top1": int(np.argmax(probabilities)) == teacher,
                    "teacher_probability": float(probabilities[teacher]),
                    "search_cross_entropy": float(
                        -(target_policy * np.log(np.maximum(probabilities, 1e-12))).sum()
                    ),
                }
            )
    payload = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "search_replay": [str(path.expanduser().resolve()) for path in args.search_replay],
        "discovered_examples": len(discovered),
        "complete_root_examples": len(complete_roots),
        "sampled_examples": len(examples),
        "overall": metrics(records),
        "by_side": {
            side: metrics([record for record in records if record["side"] == side])
            for side in ("us", "ussr")
        },
        "by_phase": {
            phase: metrics([record for record in records if record["phase"] == phase])
            for phase in ("early", "mid", "late")
        },
        "by_side_target": {
            side: {
                target_name: metrics(
                    [
                        record
                        for record in records
                        if record["side"] == side
                        and (
                            record["target"] > 0.0
                            if target_name == "win"
                            else record["target"] < 0.0
                            if target_name == "loss"
                            else record["target"] == 0.0
                        )
                    ]
                )
                for target_name in ("win", "loss", "draw")
            }
            for side in ("us", "ussr")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.records_output is not None:
        records_output = args.records_output.expanduser().resolve()
        records_output.parent.mkdir(parents=True, exist_ok=True)
        records_output.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
