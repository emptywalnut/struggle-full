#!/usr/bin/env python3
"""Audit newly committed corrected search games while collection continues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from struggle_ai.search_semantics_audit import audit_search_reroots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", action="append", type=Path, required=True)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def collections_with_completed_games(collections: list[Path]) -> list[Path]:
    """Exclude only shards that cannot yet have a durable replay manifest."""

    return [
        collection
        for collection in collections
        if any((collection / "completed_games").glob("game-*.json"))
    ]


def transient_inventory_mismatch(exc: Exception) -> bool:
    """Recognize the atomic commit window between a game record and replay manifest."""

    return isinstance(exc, RuntimeError) and str(exc).startswith(
        "corrected-game reroot audit inventory mismatch:"
    )


def main() -> None:
    args = parse_args()
    migration = json.loads(args.migration.read_text(encoding="utf-8"))
    legacy = set(map(str, migration["legacy_completed_games"]))
    last_corrected = -1
    while True:
        records = []
        for collection in args.collection:
            for path in (collection / "completed_games").glob("game-*.json"):
                records.append(json.loads(path.read_text(encoding="utf-8")))
        corrected = sum(str(record["source_game"]) not in legacy for record in records)
        if corrected != last_corrected:
            if corrected == 0:
                summary = {
                    "games": 0,
                    "legacy_games": 0,
                    "corrected_games": 0,
                    "required_card_reroots": 0,
                    "corrected_missing_required_card_reroots": [],
                    "legacy_missing_required_card_reroots": [],
                    "passed": True,
                    "game_reports": [],
                }
                summary["updated_at"] = time.time()
                atomic_json(args.output, summary)
                last_corrected = 0
                time.sleep(max(1.0, float(args.poll_seconds)))
                continue
            try:
                committed_collections = collections_with_completed_games(args.collection)
                summary = audit_search_reroots(
                    committed_collections,
                    legacy_source_games=legacy,
                    expected_corrected_games=corrected,
                )
            except Exception as exc:
                if transient_inventory_mismatch(exc):
                    print(
                        json.dumps(
                            {
                                "event": "search_reroot_audit_retry",
                                "corrected_games": corrected,
                                "reason": str(exc),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    time.sleep(max(1.0, float(args.poll_seconds)))
                    continue
                atomic_json(
                    args.output,
                    {
                        "passed": False,
                        "corrected_games": corrected,
                        "error": repr(exc),
                        "updated_at": time.time(),
                    },
                )
                raise
            summary["updated_at"] = time.time()
            atomic_json(args.output, summary)
            print(
                json.dumps(
                    {
                        "event": "search_reroot_audit_updated",
                        "corrected_games": corrected,
                        "passed": summary["passed"],
                    }
                ),
                flush=True,
            )
            last_corrected = corrected
        if all((collection / "collection_summary.json").is_file() for collection in args.collection):
            return
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
