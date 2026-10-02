#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select balanced, high-regret PUCT roots for deeper teacher regeneration."
    )
    parser.add_argument("--tournament-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-q-advantage", type=float, default=0.03)
    parser.add_argument("--max-positions", type=int, default=24)
    parser.add_argument("--min-cross-turn-positions", type=int, default=8)
    parser.add_argument("--require-flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-cross-turn", action="store_true")
    return parser.parse_args()


def _game_log(shard: Path) -> Path:
    logs = sorted((shard / "game_logs").glob("*.jsonl"))
    if len(logs) != 1:
        raise RuntimeError(f"expected one game log under {shard}, found {len(logs)}")
    return logs[0].resolve()


def _rank(event: dict[str, Any]) -> tuple[float, float, int, int]:
    triggers = set(event.get("triggers") or [])
    cross_turn = int(event.get("max_turn_delta") or 0) > 0
    leverage = 0.0
    if cross_turn:
        leverage += 0.05
    if "scoring_card" in triggers:
        leverage += 0.03
    if "opponent_event" in triggers or "forced_response" in triggers:
        leverage += 0.02
    if "influence_allocation" in triggers:
        leverage += 0.01
    return (
        float(event.get("q_advantage_over_raw") or 0.0) + leverage,
        float(event.get("root_q_gap") or 0.0),
        int(event.get("cross_turn_leaf_evaluations") or 0),
        int(event.get("simulations") or 0),
    )


def collect_positions(
    tournament_root: Path,
    *,
    min_q_advantage: float,
    require_flip: bool,
    require_cross_turn: bool,
) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, int, int], dict[str, Any]] = {}
    for result_path in sorted(tournament_root.glob("shard-*/results.json")):
        shard = result_path.parent
        game_log = _game_log(shard)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for occurrence, record in enumerate(payload.get("records") or []):
            for event in record.get("search_events") or []:
                if require_flip and not bool(event.get("action_flipped")):
                    continue
                q_advantage = float(event.get("q_advantage_over_raw") or 0.0)
                if q_advantage < min_q_advantage:
                    continue
                if int(event.get("atomic_horizon_cutoffs") or 0) > 0:
                    continue
                if int(event.get("node_budget_cutoffs") or 0) > 0:
                    continue
                if require_cross_turn and int(event.get("max_turn_delta") or 0) <= 0:
                    continue
                action_index = int(event["episode_step"])
                key = (str(game_log), occurrence, action_index)
                position = {
                    "id": f"seed-{record['seed']}-{record['orientation']}-step-{action_index}",
                    "game_log": str(game_log),
                    "seed": int(record["seed"]),
                    "game_occurrence": occurrence,
                    "action_index": action_index,
                    "side": str(event["side"]),
                    "categories": sorted(set(event.get("triggers") or [])),
                    "selection": {
                        "source_search_depth": int(event.get("max_meaningful_depth_reached") or 0),
                        "source_simulations": int(event.get("simulations") or 0),
                        "q_advantage_over_raw": q_advantage,
                        "root_q_gap": float(event.get("root_q_gap") or 0.0),
                        "max_turn_delta": int(event.get("max_turn_delta") or 0),
                        "cross_turn_leaf_evaluations": int(
                            event.get("cross_turn_leaf_evaluations") or 0
                        ),
                    },
                    "_rank": _rank(event),
                }
                previous = candidates.get(key)
                if previous is None or position["_rank"] > previous["_rank"]:
                    candidates[key] = position
    return sorted(candidates.values(), key=lambda item: item["_rank"], reverse=True)


def balanced_top(positions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_side = {
        side: [position for position in positions if position["side"] == side]
        for side in ("us", "ussr")
    }
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(by_side.values()):
        for side in ("us", "ussr"):
            if by_side[side] and len(selected) < limit:
                selected.append(by_side[side].pop(0))
    return selected


def balanced_top_with_cross_turn(
    positions: list[dict[str, Any]], limit: int, min_cross_turn: int
) -> list[dict[str, Any]]:
    cross_turn = [
        position
        for position in positions
        if int(position["selection"].get("max_turn_delta") or 0) > 0
    ]
    selected = balanced_top(cross_turn, min(limit, min_cross_turn))
    selected_ids = {position["id"] for position in selected}
    remaining = [position for position in positions if position["id"] not in selected_ids]
    by_side = {
        side: [position for position in remaining if position["side"] == side]
        for side in ("us", "ussr")
    }
    side_counts = {
        side: sum(position["side"] == side for position in selected)
        for side in ("us", "ussr")
    }
    while len(selected) < limit and any(by_side.values()):
        side = min(("us", "ussr"), key=lambda item: (side_counts[item], item))
        if not by_side[side]:
            side = "ussr" if side == "us" else "us"
        selected.append(by_side[side].pop(0))
        side_counts[side] += 1
    return selected


def main() -> None:
    args = parse_args()
    if args.max_positions <= 0:
        raise SystemExit("--max-positions must be positive")
    if args.min_cross_turn_positions < 0:
        raise SystemExit("--min-cross-turn-positions must be non-negative")
    positions = collect_positions(
        args.tournament_root.expanduser().resolve(),
        min_q_advantage=args.min_q_advantage,
        require_flip=args.require_flip,
        require_cross_turn=args.require_cross_turn,
    )
    selected = balanced_top_with_cross_turn(
        positions,
        args.max_positions,
        args.min_cross_turn_positions,
    )
    for position in selected:
        position.pop("_rank", None)
    payload = {
        "version": 1,
        "description": (
            "Balanced high-regret roots selected from paired PUCT games for deeper "
            "information-safe teacher regeneration. No preferred action is encoded."
        ),
        "source_tournament": str(args.tournament_root.expanduser().resolve()),
        "selection": {
            "min_q_advantage": args.min_q_advantage,
            "require_flip": args.require_flip,
            "require_cross_turn": args.require_cross_turn,
            "available_positions": len(positions),
            "selected_positions": len(selected),
            "selected_us": sum(position["side"] == "us" for position in selected),
            "selected_ussr": sum(position["side"] == "ussr" for position in selected),
            "selected_cross_turn": sum(
                int(position["selection"].get("max_turn_delta") or 0) > 0
                for position in selected
            ),
        },
        "positions": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "teacher_manifest_written", **payload["selection"]}, sort_keys=True))


if __name__ == "__main__":
    main()
