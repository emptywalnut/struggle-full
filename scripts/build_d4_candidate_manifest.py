#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild D4 candidate roots from a complete merged search replay."
    )
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q-gap-threshold", type=float, default=0.03)
    return parser.parse_args()


def game_records(collection_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with (collection_dir / "games.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            game_log = record.get("game_log")
            if not game_log:
                candidates = sorted(
                    (
                        collection_dir
                        / "workers"
                        / f"worker-{int(record['worker_id']):03d}"
                        / "game_logs"
                    ).glob("*.jsonl")
                )
                game_log = str(candidates[-1].resolve()) if candidates else None
            records[str(record["source_game"])] = {**record, "game_log": game_log}
    return records


def reasons(metadata: dict[str, Any], q_gap_threshold: float) -> list[str]:
    result: list[str] = []
    triggers = set(metadata.get("trigger_categories") or [])
    if abs(float(metadata.get("root_q_gap") or 0.0)) <= q_gap_threshold:
        result.append("small_q_gap")
    if metadata.get("action_flipped"):
        result.append("raw_search_disagreement")
    for trigger in (
        "scoring_card",
        "scoring_pressure",
        "forced_response",
        "opponent_event",
        "chance_heavy",
    ):
        if trigger in triggers:
            result.append(trigger)
    return result


def build_positions(collection_dir: Path, q_gap_threshold: float) -> list[dict[str, Any]]:
    games = game_records(collection_dir)
    replay = collection_dir / "search_replay"
    manifest = json.loads((replay / "manifest.json").read_text(encoding="utf-8"))
    positions: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for shard in manifest.get("shards", []):
        with np.load(replay / str(shard["path"]), allow_pickle=False) as arrays:
            for raw in arrays["metadata_json"]:
                metadata = json.loads(str(raw))
                if int(metadata.get("hierarchy_index") or 0) != 0:
                    continue
                source_game = str(metadata.get("source_game") or "")
                source_step = metadata.get("source_step")
                record = games.get(source_game)
                if record is None or source_step is None or not record.get("game_log"):
                    continue
                key = (source_game, int(source_step))
                if key in seen:
                    continue
                selected_reasons = reasons(metadata, q_gap_threshold)
                if not selected_reasons:
                    continue
                seen.add(key)
                positions.append(
                    {
                        "source_game": source_game,
                        "seed": int(record["seed"]),
                        "action_index": int(source_step),
                        "game_occurrence": 0,
                        "game_log": str(record["game_log"]),
                        "side": str(metadata["side"]),
                        "triggers": list(metadata.get("trigger_categories") or []),
                        "simulations": int(metadata.get("simulation_budget") or 0),
                        "meaningful_depth": int(metadata.get("search_depth") or 0),
                        "raw_action": int(metadata["raw_action"]),
                        "teacher_action": int(metadata["search_action"]),
                        "behavior_action": int(metadata.get("behavior_action", metadata["search_action"])),
                        "action_flipped": bool(metadata.get("action_flipped")),
                        "root_q_gap": float(metadata.get("root_q_gap") or 0.0),
                        "q_advantage_over_raw": float(metadata.get("q_advantage_over_raw") or 0.0),
                        "adjudication_reasons": selected_reasons,
                    }
                )
    return positions


def main() -> None:
    args = parse_args()
    collection_dir = args.collection_dir.expanduser().resolve()
    positions = build_positions(collection_dir, args.q_gap_threshold)
    payload = {
        "version": 1,
        "description": "D4 candidate roots rebuilt from all complete D2 replay roots.",
        "collection_dir": str(collection_dir),
        "q_gap_threshold": args.q_gap_threshold,
        "positions": positions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "d4_candidate_manifest_built",
                "positions": len(positions),
                "source_games": len({position["source_game"] for position in positions}),
                "us": sum(position["side"] == "us" for position in positions),
                "ussr": sum(position["side"] == "ussr" for position in positions),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
