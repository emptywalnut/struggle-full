#!/usr/bin/env python3
"""Audit complete search games for independent card-decision re-rooting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from struggle_ai.search_semantics_audit import audit_search_reroots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", action="append", type=Path, required=True)
    parser.add_argument("--legacy-source-game", action="append", default=[])
    parser.add_argument("--expected-corrected-games", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_search_reroots(
        args.collection,
        legacy_source_games=args.legacy_source_game,
        expected_corrected_games=args.expected_corrected_games,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "game_reports"}))


if __name__ == "__main__":
    main()
