#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def position_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["seed"]), int(row["action_index"])


def select_panel(
    positions: list[dict[str, Any]],
    evidence: dict[tuple[int, int], dict[str, Any]],
    *,
    per_side_era: int,
    max_legal_count: int,
    excluded: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in positions
        if evidence.get(position_key(row), {}).get("status") == "UNRESOLVED"
        and str(row.get("branch_bucket")) != "high"
        and int(row.get("legal_count", 0)) <= max_legal_count
        and position_key(row) not in excluded
    ]
    selected: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        buckets[(str(row["side"]), str(row["era"]))].append(row)
    for side in ("us", "ussr"):
        for era in ("early", "mid", "late"):
            candidates = buckets[(side, era)]
            for _ in range(per_side_era):
                if not candidates:
                    raise ValueError(f"insufficient unresolved roots for {side}/{era}")
                candidates.sort(
                    key=lambda row: (
                        category_counts[str(row.get("category"))],
                        int(row.get("legal_count", 0)),
                        position_key(row),
                    )
                )
                chosen = candidates.pop(0)
                selected.append(chosen)
                category_counts[str(chosen.get("category"))] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced, cost-bounded unresolved-root panel.")
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-side-era", type=int, default=3)
    parser.add_argument("--max-legal-count", type=int, default=24)
    parser.add_argument("--exclude", action="append", default=[], help="SEED:ACTION_INDEX")
    args = parser.parse_args()
    payload = json.loads(args.positions.read_text(encoding="utf-8"))
    evidence_rows = [json.loads(line) for line in args.evidence.open(encoding="utf-8") if line.strip()]
    evidence = {(int(row["seed"]), int(row["action_index"])): row for row in evidence_rows}
    excluded = {
        tuple(map(int, value.split(":", 1)))
        for value in args.exclude
    }
    selected = select_panel(
        list(payload.get("positions") or []),
        evidence,
        per_side_era=int(args.per_side_era),
        max_legal_count=int(args.max_legal_count),
        excluded=excluded,
    )
    output = {
        "source_positions": str(args.positions.resolve()),
        "source_evidence": str(args.evidence.resolve()),
        "selection": {
            "status": "UNRESOLVED",
            "per_side_era": int(args.per_side_era),
            "max_legal_count": int(args.max_legal_count),
            "excluded": [list(key) for key in sorted(excluded)],
        },
        "positions": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "positions": len(selected),
        "side": dict(Counter(str(row["side"]) for row in selected)),
        "era": dict(Counter(str(row["era"]) for row in selected)),
        "category": dict(Counter(str(row["category"]) for row in selected)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
