#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_RE = re.compile(r"^seed-(?P<seed>\d+)-occurrence-(?P<occurrence>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select rare D8 roots where D2 and D4 disagree or D4 remains ambiguous."
    )
    parser.add_argument("--d2-position-manifest", type=Path, required=True)
    parser.add_argument("--d4-search-replay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--d4-q-gap-threshold", type=float, default=0.02)
    parser.add_argument("--max-positions", type=int, default=4)
    parser.add_argument("--completion-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-d4-legal-actions", type=int, default=4)
    parser.add_argument("--min-d4-teacher-visit-share", type=float, default=0.45)
    parser.add_argument("--min-d4-q-advantage", type=float, default=0.05)
    parser.add_argument("--max-estimated-branch-load", type=float, default=10.0)
    return parser.parse_args()


def root_metadata(replay_paths: list[Path]) -> dict[tuple[int, int, int], dict[str, Any]]:
    roots: dict[tuple[int, int, int], dict[str, Any]] = {}
    for raw_path in replay_paths:
        manifest = raw_path / "manifest.json" if raw_path.is_dir() else raw_path
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for shard in payload.get("shards", []):
            with np.load(manifest.parent / str(shard["path"]), allow_pickle=False) as arrays:
                for raw in arrays["metadata_json"]:
                    metadata = json.loads(str(raw))
                    if int(metadata.get("hierarchy_index") or 0) != 0:
                        continue
                    match = SOURCE_RE.match(str(metadata.get("source_game") or ""))
                    if match is None or metadata.get("source_step") is None:
                        continue
                    key = (
                        int(match.group("seed")),
                        int(metadata["source_step"]),
                        int(match.group("occurrence")),
                    )
                    roots[key] = metadata
    return roots


def select_positions(
    d2_positions: list[dict[str, Any]],
    d4_roots: dict[tuple[int, int, int], dict[str, Any]],
    *,
    q_gap_threshold: float,
    limit: int,
    completion_aware: bool = False,
    max_legal_actions: int = 4,
    min_teacher_visit_share: float = 0.45,
    min_q_advantage: float = 0.05,
    max_estimated_branch_load: float = 10.0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for position in d2_positions:
        key = (
            int(position["seed"]),
            int(position["action_index"]),
            int(position.get("game_occurrence", 0)),
        )
        d4 = d4_roots.get(key)
        if d4 is None:
            continue
        legal_count = len(d4.get("legal_actions") or [])
        teacher_share = float(d4.get("teacher_visit_share") or 0.0)
        estimated_branch_load = legal_count / max(1e-6, teacher_share)
        cutoff_free = (
            int(d4.get("atomic_horizon_cutoffs") or 0) == 0
            and int(d4.get("node_budget_cutoffs") or 0) == 0
        )
        likely_complete = (
            cutoff_free
            and 0 < legal_count <= int(max_legal_actions)
            and teacher_share >= float(min_teacher_visit_share)
            and float(d4.get("q_advantage_over_raw") or 0.0) >= float(min_q_advantage)
            and estimated_branch_load <= float(max_estimated_branch_load)
        )
        if completion_aware and not likely_complete:
            continue
        d2_action = position.get("teacher_action")
        if d2_action is None:
            d2_action = position.get("behavior_action", position.get("raw_action"))
        disagrees = int(d2_action) != int(d4["search_action"])
        ambiguous = abs(float(d4.get("root_q_gap") or 0.0)) <= float(q_gap_threshold)
        if not (disagrees or ambiguous):
            continue
        reasons = []
        if disagrees:
            reasons.append("d2_d4_action_disagreement")
        if ambiguous:
            reasons.append("d4_small_q_gap")
        selected.append(
            {
                **position,
                "adjudication_reasons": reasons,
                "d4": {
                    "search_action": int(d4["search_action"]),
                    "raw_action": int(d4["raw_action"]),
                    "root_q_gap": float(d4.get("root_q_gap") or 0.0),
                    "q_advantage_over_raw": float(d4.get("q_advantage_over_raw") or 0.0),
                    "teacher_visit_share": float(d4.get("teacher_visit_share") or 0.0),
                    "legal_count": legal_count,
                    "atomic_horizon_cutoffs": int(d4.get("atomic_horizon_cutoffs") or 0),
                    "node_budget_cutoffs": int(d4.get("node_budget_cutoffs") or 0),
                    "estimated_branch_load": estimated_branch_load,
                    "likely_complete": likely_complete,
                },
            }
        )
    selected.sort(
        key=lambda position: (
            "d2_d4_action_disagreement" in position["adjudication_reasons"],
            -abs(float(position["d4"]["root_q_gap"])),
            float(position["d4"]["q_advantage_over_raw"]),
        ),
        reverse=True,
    )
    balanced: list[dict[str, Any]] = []
    by_side = {
        side: [position for position in selected if position.get("side") == side]
        for side in ("us", "ussr")
    }
    while len(balanced) < max(0, int(limit)) and any(by_side.values()):
        for side in ("us", "ussr"):
            if by_side[side] and len(balanced) < int(limit):
                balanced.append(by_side[side].pop(0))
    return balanced


def main() -> None:
    args = parse_args()
    source = json.loads(args.d2_position_manifest.expanduser().resolve().read_text(encoding="utf-8"))
    roots = root_metadata([path.expanduser().resolve() for path in args.d4_search_replay])
    selected = select_positions(
        list(source.get("positions") or []),
        roots,
        q_gap_threshold=args.d4_q_gap_threshold,
        limit=args.max_positions,
        completion_aware=args.completion_aware,
        max_legal_actions=args.max_d4_legal_actions,
        min_teacher_visit_share=args.min_d4_teacher_visit_share,
        min_q_advantage=args.min_d4_q_advantage,
        max_estimated_branch_load=args.max_estimated_branch_load,
    )
    payload = {
        "version": 1,
        "description": "Rare D8 adjudication roots selected only after practical D4 search.",
        "source_d2_manifest": str(args.d2_position_manifest.expanduser().resolve()),
        "d4_replay": [str(path.expanduser().resolve()) for path in args.d4_search_replay],
        "selection": {
            "d4_q_gap_threshold": args.d4_q_gap_threshold,
            "d4_roots": len(roots),
            "selected_positions": len(selected),
            "selected_us": sum(position.get("side") == "us" for position in selected),
            "selected_ussr": sum(position.get("side") == "ussr" for position in selected),
            "completion_aware": args.completion_aware,
            "max_d4_legal_actions": args.max_d4_legal_actions,
            "min_d4_teacher_visit_share": args.min_d4_teacher_visit_share,
            "min_d4_q_advantage": args.min_d4_q_advantage,
            "max_estimated_branch_load": args.max_estimated_branch_load,
        },
        "positions": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "d8_adjudication_selected", **payload["selection"]}), flush=True)


if __name__ == "__main__":
    main()
