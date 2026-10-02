#!/usr/bin/env python3
"""Combine compatible paired-evaluation shards without averaging percentages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_puct_agent_pair_parallel import summary


IGNORED_IDENTITY_KEYS = {"seed_base", "seed_pairs", "log_games"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument(
        "--complete-pairs-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def progress_path(path: Path) -> Path:
    return path / "progress.json" if path.is_dir() else path


def compatible_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in identity.items() if key not in IGNORED_IDENTITY_KEYS}


def aggregate_payloads(
    payloads: list[dict[str, Any]],
    *,
    complete_pairs_only: bool,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("no evaluation payloads")
    expected_identity = compatible_identity(dict(payloads[0]["identity"]))
    records: dict[tuple[int, str], dict[str, Any]] = {}
    source_counts: list[int] = []
    for payload in payloads:
        actual_identity = compatible_identity(dict(payload["identity"]))
        if actual_identity != expected_identity:
            raise ValueError("evaluation shard identity mismatch")
        shard_records = list(payload.get("records", []))
        source_counts.append(len(shard_records))
        for record in shard_records:
            key = (int(record["seed"]), str(record["orientation"]))
            if key in records:
                raise ValueError(f"duplicate evaluation record: {key}")
            records[key] = record

    selected = list(records.values())
    excluded_unpaired = 0
    if complete_pairs_only:
        orientations: dict[int, set[str]] = {}
        for record in selected:
            orientations.setdefault(int(record["seed"]), set()).add(str(record["orientation"]))
        complete_seeds = {
            seed
            for seed, values in orientations.items()
            if values == {"candidate_us", "candidate_ussr"}
        }
        filtered = [record for record in selected if int(record["seed"]) in complete_seeds]
        excluded_unpaired = len(selected) - len(filtered)
        selected = filtered
    if not selected:
        raise ValueError("no complete evaluation records to aggregate")

    seed_pairs = len({int(record["seed"]) for record in selected})
    stats = summary(
        SimpleNamespace(
            bootstrap_samples=int(bootstrap_samples),
            seed_base=91_700_000,
            seed_pairs=seed_pairs,
        ),
        selected,
    )
    return {
        "status": "complete_pairs" if complete_pairs_only else "all_completed_games",
        "source_record_counts": source_counts,
        "excluded_unpaired_records": excluded_unpaired,
        "identity": expected_identity,
        "summary": stats,
        "records": sorted(selected, key=lambda row: (int(row["seed"]), str(row["orientation"]))),
    }


def main() -> None:
    args = parse_args()
    paths = [progress_path(path.expanduser().resolve()) for path in args.input]
    available = [path for path in paths if path.is_file()]
    pending = [path for path in paths if not path.is_file()]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in available]
    result = aggregate_payloads(
        payloads,
        complete_pairs_only=bool(args.complete_pairs_only),
        bootstrap_samples=int(args.bootstrap_samples),
    )
    result["sources"] = [str(path) for path in available]
    result["pending_sources"] = [str(path) for path in pending]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
