#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank and side-balance high-value D2 roots for practical D4 adjudication."
    )
    parser.add_argument("--d2-position-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=16)
    parser.add_argument("--max-per-game", type=int, default=1)
    return parser.parse_args()


def priority(position: dict[str, Any]) -> tuple[float, float, float]:
    reasons = set(position.get("adjudication_reasons") or [])
    score = 0.0
    score += 12.0 * bool(position.get("action_flipped"))
    score += 8.0 * ("scoring_card" in reasons)
    score += 6.0 * ("forced_response" in reasons)
    score += 5.0 * ("small_q_gap" in reasons)
    score += 3.0 * ("chance_heavy" in reasons)
    score += 2.0 * ("opponent_event" in reasons)
    score += min(2.0, 5.0 * abs(float(position.get("q_advantage_over_raw") or 0.0)))
    return (
        score,
        abs(float(position.get("q_advantage_over_raw") or 0.0)),
        -abs(float(position.get("root_q_gap") or 0.0)),
    )


def select_positions(
    positions: list[dict[str, Any]], *, limit: int, max_per_game: int
) -> list[dict[str, Any]]:
    valid = [
        position
        for position in positions
        if position.get("game_log")
        and position.get("side") in {"us", "ussr"}
        and position.get("action_index") is not None
    ]
    by_side = {
        side: sorted(
            (position for position in valid if position["side"] == side),
            key=priority,
            reverse=True,
        )
        for side in ("us", "ussr")
    }
    selected: list[dict[str, Any]] = []
    game_counts: dict[str, int] = {}
    seen_roots: set[tuple[int, int, int]] = set()
    while len(selected) < max(0, int(limit)) and any(by_side.values()):
        made_progress = False
        for side in ("us", "ussr"):
            candidates = by_side[side]
            while candidates and len(selected) < int(limit):
                position = candidates.pop(0)
                root = (
                    int(position["seed"]),
                    int(position["action_index"]),
                    int(position.get("game_occurrence", 0)),
                )
                game = str(position.get("source_game") or f"{position['game_log']}:{position['seed']}")
                if root in seen_roots or game_counts.get(game, 0) >= int(max_per_game):
                    continue
                selected.append(position)
                seen_roots.add(root)
                game_counts[game] = game_counts.get(game, 0) + 1
                made_progress = True
                break
        if not made_progress:
            break
    return selected


def main() -> None:
    args = parse_args()
    source = json.loads(
        args.d2_position_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    positions = list(source.get("positions") or [])
    selected = select_positions(
        positions,
        limit=args.max_positions,
        max_per_game=args.max_per_game,
    )
    payload = {
        "version": 1,
        "description": "Ranked, side-balanced D2 roots selected for parallel D4 adjudication.",
        "source_d2_manifest": str(args.d2_position_manifest.expanduser().resolve()),
        "selection": {
            "input_positions": len(positions),
            "positions_with_game_logs": sum(bool(position.get("game_log")) for position in positions),
            "selected_positions": len(selected),
            "selected_us": sum(position.get("side") == "us" for position in selected),
            "selected_ussr": sum(position.get("side") == "ussr" for position in selected),
            "max_per_game": args.max_per_game,
        },
        "positions": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "d4_adjudication_selected", **payload["selection"]}), flush=True)


if __name__ == "__main__":
    main()
