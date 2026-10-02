#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from struggle_ai.search_replay import SearchReplayWriter, _load_search_replay_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream complete hierarchy roots into game-disjoint value replay shards."
    )
    parser.add_argument("--search-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--game-namespace", required=True)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--part-count", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--output-shard-size", type=int, default=128)
    return parser.parse_args()


def game_fold(game: str, folds: int) -> int:
    digest = hashlib.blake2b(game.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % int(folds)


def main() -> None:
    args = parse_args()
    if args.part_count <= 0 or not 0 <= args.part_index < args.part_count:
        raise SystemExit("part index must be in [0, part count)")
    if args.folds < 2 or not 0 <= args.validation_fold < args.folds:
        raise SystemExit("validation fold must be in [0, folds), with folds >= 2")

    manifest = args.search_replay.expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    selected_shards = list(payload.get("shards", []))[args.part_index :: args.part_count]
    part_dir = args.output_dir.expanduser().resolve() / f"part-{args.part_index:03d}"
    train_writer = SearchReplayWriter(part_dir / "train", shard_size=args.output_shard_size)
    validation_writer = SearchReplayWriter(
        part_dir / "validation", shard_size=args.output_shard_size
    )
    counts: Counter[str] = Counter()
    games: dict[str, str] = {}
    for shard in selected_shards:
        shard_path = manifest.parent / str(shard["path"])
        for example in _load_search_replay_shard(shard_path):
            metadata = example.metadata
            if (
                not metadata.complete_game
                or metadata.root_value_target is None
                or int(metadata.hierarchy_index) != 0
                or not metadata.source_game
            ):
                continue
            namespaced_game = f"{args.game_namespace}:{metadata.source_game}"
            split = (
                "validation"
                if game_fold(namespaced_game, args.folds) == args.validation_fold
                else "train"
            )
            example = replace(
                example,
                metadata=replace(metadata, source_game=namespaced_game),
            )
            (validation_writer if split == "validation" else train_writer).add(example)
            winner = str(metadata.game_winner or "unknown")
            counts[f"{split}_examples"] += 1
            counts[f"{split}_{metadata.side}_{'win' if metadata.root_value_target > 0 else 'loss'}"] += 1
            games[namespaced_game] = winner

    train_manifest = train_writer.close()
    validation_manifest = validation_writer.close()
    train_games = sorted(
        game for game in games if game_fold(game, args.folds) != args.validation_fold
    )
    validation_games = sorted(
        game for game in games if game_fold(game, args.folds) == args.validation_fold
    )
    summary = {
        "source_manifest": str(manifest),
        "game_namespace": args.game_namespace,
        "part_index": args.part_index,
        "part_count": args.part_count,
        "source_shards": len(selected_shards),
        "counts": dict(sorted(counts.items())),
        "train_games": train_games,
        "validation_games": validation_games,
        "game_overlap": sorted(set(train_games) & set(validation_games)),
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
    }
    (part_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
