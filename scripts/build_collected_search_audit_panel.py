#!/usr/bin/env python3
"""Select a balanced particle-variance audit from complete-game search candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {"BUDGET_CONVERGENCE", "HORIZON_CONVERGENCE"}
TRIGGER_PRIORITY = (
    "scoring_card",
    "opponent_event",
    "forced_response",
    "chance_heavy",
    "allocation",
    "low_confidence",
)


def era(turn: int) -> str:
    if turn <= 3:
        return "early"
    if turn <= 7:
        return "mid"
    return "late"


def category(row: dict[str, Any]) -> str:
    triggers = set(map(str, row.get("triggers") or ()))
    return next((name for name in TRIGGER_PRIORITY if name in triggers), "other")


def stable_key(row: dict[str, Any]) -> str:
    value = f"{row['seed']}:{row['action_index']}:{row.get('source_game', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_audit_panel(
    rows: list[dict[str, Any]],
    *,
    per_side_era: int,
    min_q_gap: float,
    max_base_latency: float,
    max_per_game: int,
) -> list[dict[str, Any]]:
    deduplicated: dict[tuple[int, int], dict[str, Any]] = {}
    for original in rows:
        row = dict(original)
        key = int(row["seed"]), int(row["action_index"])
        if not bool(row.get("teacher_eligible")):
            continue
        if str(row.get("convergence_status")) not in ALLOWED_STATUSES:
            continue
        if abs(float(row.get("convergence_q_gap") or row.get("root_q_gap") or 0.0)) < min_q_gap:
            continue
        if float(row.get("latency_seconds") or 0.0) > max_base_latency:
            continue
        if not Path(str(row.get("game_log") or "")).is_file():
            continue
        row["turn"] = int(row.get("turn") or 0)
        row["era"] = era(row["turn"])
        row["category"] = category(row)
        row["game_occurrence"] = int(row.get("game_occurrence") or 0)
        current = deduplicated.get(key)
        if current is None or abs(float(row.get("convergence_q_gap") or 0.0)) > abs(
            float(current.get("convergence_q_gap") or 0.0)
        ):
            deduplicated[key] = row

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in deduplicated.values():
        buckets[(str(row["side"]), str(row["era"]))].append(row)
    selected: list[dict[str, Any]] = []
    game_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    bucket_order = sorted(
        ((side, phase) for side in ("us", "ussr") for phase in ("early", "mid", "late")),
        key=lambda key: (
            len({str(row.get("source_game") or row["seed"]) for row in buckets[key]}),
            key,
        ),
    )
    for side, phase in bucket_order:
            candidates = sorted(
                buckets[(side, phase)],
                key=lambda row: (
                    category_counts[row["category"]],
                    -abs(float(row.get("convergence_q_gap") or 0.0)),
                    stable_key(row),
                ),
            )
            chosen = 0
            while candidates and chosen < per_side_era:
                candidates.sort(
                    key=lambda row: (
                        category_counts[row["category"]],
                        -abs(float(row.get("convergence_q_gap") or 0.0)),
                        stable_key(row),
                    )
                )
                row = candidates.pop(0)
                source_game = str(row.get("source_game") or row["seed"])
                if game_counts[source_game] >= max_per_game:
                    continue
                selected.append(row)
                chosen += 1
                game_counts[source_game] += 1
                category_counts[row["category"]] += 1
            if chosen != per_side_era:
                raise ValueError(
                    f"insufficient audit candidates for {side}/{phase}: {chosen}/{per_side_era}"
                )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", action="append", type=Path, default=[])
    parser.add_argument("--completed-games-dir", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-side-era", type=int, default=3)
    parser.add_argument("--min-q-gap", type=float, default=0.02)
    parser.add_argument("--max-base-latency", type=float, default=300.0)
    parser.add_argument("--max-per-game", type=int, default=1)
    args = parser.parse_args()
    if not args.source_manifest and not args.completed_games_dir:
        parser.error("at least one --source-manifest or --completed-games-dir is required")
    rows: list[dict[str, Any]] = []
    for path in args.source_manifest:
        rows.extend(json.loads(path.read_text(encoding="utf-8")).get("positions") or [])
    completed_records = []
    for directory in args.completed_games_dir:
        for path in sorted(directory.glob("game-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            completed_records.append(str(path.resolve()))
            rows.extend(payload.get("deep_candidates") or [])
    selected = select_audit_panel(
        rows,
        per_side_era=max(1, int(args.per_side_era)),
        min_q_gap=float(args.min_q_gap),
        max_base_latency=float(args.max_base_latency),
        max_per_game=max(1, int(args.max_per_game)),
    )
    summary = {
        "positions": len(selected),
        "side": dict(Counter(str(row["side"]) for row in selected)),
        "era": dict(Counter(str(row["era"]) for row in selected)),
        "category": dict(Counter(str(row["category"]) for row in selected)),
        "source_games": len({str(row["source_game"]) for row in selected}),
        "min_q_gap": float(args.min_q_gap),
        "max_base_latency": float(args.max_base_latency),
    }
    payload = {
        "source_manifests": [str(path.resolve()) for path in args.source_manifest],
        "completed_game_records": completed_records,
        "selection": summary,
        "positions": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
