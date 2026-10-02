#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fit_value_calibration import game_folds
from struggle_ai.search_replay import SearchReplayWriter, load_search_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split search replay without crossing game boundaries.")
    parser.add_argument("--search-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--validation-folds", type=int, default=2)
    parser.add_argument("--load-workers", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=256)
    return parser.parse_args()


def split_games(examples, folds: int, validation_folds: int) -> tuple[set[str], set[str]]:
    game_records = {}
    for example in examples:
        game = example.metadata.source_game
        if not game or not example.metadata.complete_game:
            continue
        game_records.setdefault(
            str(game),
            {
                "source_game": str(game),
                "game_winner": str(example.metadata.game_winner or "unknown"),
            },
        )
    partitions = game_folds(list(game_records.values()), folds)
    count = max(1, min(int(validation_folds), len(partitions) - 1))
    validation = set().union(*partitions[:count])
    training = set(game_records) - validation
    return training, validation


def main() -> None:
    args = parse_args()
    source = args.search_replay.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    examples = load_search_replay((source,), workers=int(args.load_workers))
    training_games, validation_games = split_games(examples, args.folds, args.validation_folds)
    if not training_games or not validation_games:
        raise SystemExit("game split requires non-empty training and validation sets")
    shutil.rmtree(output, ignore_errors=True)
    train_writer = SearchReplayWriter(output / "train", shard_size=args.shard_size)
    validation_writer = SearchReplayWriter(output / "validation", shard_size=args.shard_size)
    counts = {"train": 0, "validation": 0, "excluded": 0}
    for example in examples:
        game = str(example.metadata.source_game or "")
        if game in training_games:
            train_writer.add(example)
            counts["train"] += 1
        elif game in validation_games:
            validation_writer.add(example)
            counts["validation"] += 1
        else:
            counts["excluded"] += 1
    train_manifest = train_writer.close()
    validation_manifest = validation_writer.close()
    summary = {
        "source": str(source),
        "training_games": sorted(training_games),
        "validation_games": sorted(validation_games),
        "game_overlap": sorted(training_games & validation_games),
        "counts": counts,
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
