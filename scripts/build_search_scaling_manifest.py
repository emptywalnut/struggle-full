#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def branch_bucket(legal_count: int) -> str:
    if legal_count <= 8:
        return "low"
    if legal_count <= 24:
        return "medium"
    return "high"


def era_for_turn(turn: int) -> str:
    if turn <= 3:
        return "early"
    if turn <= 7:
        return "mid"
    return "late"


def stable_key(position: dict[str, Any]) -> str:
    source = f"{position['seed']}:{position['action_index']}:{position.get('game_occurrence', 0)}"
    return hashlib.sha256(source.encode("ascii")).hexdigest()


def terminal_record(position: dict[str, Any], cache: dict[Path, list[dict[str, Any]]]) -> dict[str, Any]:
    path = Path(str(position["game_log"])).expanduser().resolve()
    if path not in cache:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("kind") == "terminal":
                    rows.append(payload)
        cache[path] = rows
    seed = int(position["seed"])
    matching = [row for row in cache[path] if int(row.get("seed", -1)) == seed]
    occurrence = int(position.get("game_occurrence", 0))
    if occurrence >= len(matching):
        raise ValueError(f"missing complete game seed={seed} occurrence={occurrence} in {path}")
    return matching[occurrence]


def annotate(position: dict[str, Any], cache: dict[Path, list[dict[str, Any]]]) -> dict[str, Any]:
    row = dict(position)
    game = terminal_record(row, cache)
    action_index = int(row["action_index"])
    actions = list(game.get("actions") or [])
    if not 0 <= action_index < len(actions):
        raise ValueError(f"action index {action_index} outside complete game seed={row['seed']}")
    action = actions[action_index]
    before = dict(action.get("before") or {})
    turn = int(before.get("turn") or action.get("turn") or 0)
    terminal_info = dict(game.get("info") or {})
    row.update(
        {
            "turn": turn,
            "action_round": int(before.get("action_round") or action.get("action_round") or 0),
            "phase": str(before.get("phase") or ""),
            "prompt": str(before.get("prompt") or ""),
            "era": era_for_turn(turn),
            "branch_bucket": branch_bucket(int(row["legal_count"])),
            "terminal_reason": str(
                game.get("terminal_reason") or terminal_info.get("terminal_reason") or ""
            ),
        }
    )
    return row


def select_positions(
    positions: list[dict[str, Any]], *, count: int, max_per_game: int
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in positions:
        buckets[(row["side"], row["era"], row["category"], row["branch_bucket"])].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: (stable_key(row), -abs(float(row.get("d4_q_gap", 0.0)))))

    selected: list[dict[str, Any]] = []
    game_counts: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    sides = sorted({str(row["side"]) for row in positions})
    side_targets = {
        side: count // len(sides) + int(index < count % len(sides))
        for index, side in enumerate(sides)
    }

    selected_ids: set[int] = set()

    def reserve(candidate: dict[str, Any]) -> None:
        source_game = str(candidate.get("source_game") or candidate["seed"])
        selected.append(candidate)
        selected_ids.add(id(candidate))
        game_counts[source_game] += 1
        side_counts[str(candidate["side"])] += 1
        bucket = (
            str(candidate["side"]),
            str(candidate["era"]),
            str(candidate["category"]),
            str(candidate["branch_bucket"]),
        )
        buckets[bucket] = [row for row in buckets[bucket] if row is not candidate]

    # Reserve one high-information scoring-horizon failure when available. This
    # is a diagnostic root only; search must still determine the action ranking.
    held_scoring_roots = [
        row
        for row in positions
        if row["category"] in {"scoring_card", "scoring_pressure"}
        and row.get("terminal_reason") == "scoring card held"
    ]
    held_scoring_roots.sort(
        key=lambda row: (-abs(float(row.get("d4_q_gap", 0.0))), stable_key(row))
    )
    if held_scoring_roots:
        reserve(held_scoring_roots[0])

    # Reserve two roots per era before ordinary strata fill the panel. Prefer
    # distinct source games; if only one game reached an era, admit a second
    # root from it rather than silently dropping that era from diagnostics.
    era_order = sorted(
        ("early", "mid", "late"),
        key=lambda era: len(
            {
                str(row.get("source_game") or row["seed"])
                for row in positions
                if str(row["era"]) == era
            }
        ),
    )
    for target_era in era_order:
        while (
            sum(str(row["era"]) == target_era for row in selected) < 2
            and len(selected) < count
        ):
            candidates = [
                row
                for row in positions
                if id(row) not in selected_ids
                and str(row["era"]) == target_era
                and side_counts[str(row["side"])] < side_targets[str(row["side"])]
            ]
            if not candidates:
                break
            under_cap = [
                row
                for row in candidates
                if game_counts[str(row.get("source_game") or row["seed"])] < max_per_game
            ]
            pool = under_cap or candidates
            era_side_counts = Counter(
                str(row["side"])
                for row in selected
                if str(row["era"]) == target_era
            )
            pool.sort(
                key=lambda row: (
                    era_side_counts[str(row["side"])],
                    game_counts[str(row.get("source_game") or row["seed"])],
                    -abs(float(row.get("d4_q_gap", 0.0))),
                    stable_key(row),
                )
            )
            reserve(pool[0])

    # Expand the minimum coverage toward an even era split when distinct source
    # games exist. The minimum pass above may intentionally reuse a scarce game;
    # quota expansion is stricter so a nominally balanced panel cannot be built
    # by repeating several roots from the same late-game trajectory.
    eras = ("early", "mid", "late")
    era_targets = {
        era: count // len(eras) + int(index < count % len(eras))
        for index, era in enumerate(eras)
    }
    for target_era in era_order:
        while (
            sum(str(row["era"]) == target_era for row in selected)
            < era_targets[target_era]
            and len(selected) < count
        ):
            candidates = [
                row
                for row in positions
                if id(row) not in selected_ids
                and str(row["era"]) == target_era
                and side_counts[str(row["side"])] < side_targets[str(row["side"])]
                and game_counts[str(row.get("source_game") or row["seed"])]
                < max_per_game
            ]
            if not candidates:
                break
            era_side_counts = Counter(
                str(row["side"])
                for row in selected
                if str(row["era"]) == target_era
            )
            category_counts = Counter(str(row["category"]) for row in selected)
            branch_counts = Counter(str(row["branch_bucket"]) for row in selected)
            candidates.sort(
                key=lambda row: (
                    era_side_counts[str(row["side"])],
                    category_counts[str(row["category"])],
                    branch_counts[str(row["branch_bucket"])],
                    -abs(float(row.get("d4_q_gap", 0.0))),
                    stable_key(row),
                )
            )
            reserve(candidates[0])

    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            side = key[0]
            if side_counts[side] >= side_targets[side]:
                continue
            rows = buckets[key]
            while rows:
                candidate = rows.pop(0)
                source_game = str(candidate.get("source_game") or candidate["seed"])
                if game_counts[source_game] >= max_per_game:
                    continue
                selected.append(candidate)
                game_counts[source_game] += 1
                side_counts[side] += 1
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed, stratified search-scaling root panel.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=96)
    parser.add_argument("--max-per-game", type=int, default=8)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.expanduser().resolve().read_text(encoding="utf-8"))
    cache: dict[Path, list[dict[str, Any]]] = {}
    annotated = [annotate(position, cache) for position in source.get("positions", [])]
    selected = select_positions(
        annotated,
        count=max(1, int(args.roots)),
        max_per_game=max(1, int(args.max_per_game)),
    )
    if len(selected) < int(args.roots):
        raise SystemExit(f"could select only {len(selected)} of requested {args.roots} roots")

    strata = Counter(
        (row["side"], row["era"], row["category"], row["branch_bucket"])
        for row in selected
    )
    summary = {
        "selected_roots": len(selected),
        "by_side": dict(Counter(row["side"] for row in selected)),
        "by_era": dict(Counter(row["era"] for row in selected)),
        "by_side_era": {
            "|".join(key): value
            for key, value in sorted(
                Counter((row["side"], row["era"]) for row in selected).items()
            )
        },
        "by_category": dict(Counter(row["category"] for row in selected)),
        "by_branch_bucket": dict(Counter(row["branch_bucket"] for row in selected)),
        "source_games": len({str(row.get("source_game") or row["seed"]) for row in selected}),
        "max_roots_per_source_game": max(
            Counter(str(row.get("source_game") or row["seed"]) for row in selected).values(),
            default=0,
        ),
        "by_stratum": {"|".join(key): value for key, value in sorted(strata.items())},
    }
    payload = {
        "source_manifest": str(args.source_manifest.expanduser().resolve()),
        "positions": selected,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
