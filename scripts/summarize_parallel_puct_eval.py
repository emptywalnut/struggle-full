#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine disjoint parallel PUCT evaluation shards.")
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def elo_delta(score: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, score))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def bootstrap_pair_ci(
    records: list[dict[str, object]], samples: int, seed: int
) -> tuple[float, float]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in records:
        grouped[(str(record["source_run"]), int(record["seed"]))].append(
            float(record["candidate_score"])
        )
    pair_scores = np.asarray(
        [np.mean(scores) for scores in grouped.values()], dtype=np.float64
    )
    if not len(pair_scores):
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        pair_scores, size=(int(samples), len(pair_scores)), replace=True
    ).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    args = parse_args()
    records: list[dict[str, object]] = []
    sources: list[str] = []
    expected_identity = None
    for path in args.run:
        results_path = path / "results.json" if path.is_dir() else path
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        identity = (
            payload.get("candidate", {}).get("policy_checkpoint"),
            payload.get("candidate", {}).get("value_checkpoint"),
            payload.get("opponent", {}).get("policy_checkpoint"),
            payload.get("opponent", {}).get("value_checkpoint"),
        )
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError(f"agent identity mismatch in {results_path}: {identity}")
        source = str(results_path.resolve())
        sources.append(source)
        for record in payload.get("records", []):
            records.append({**record, "source_run": source})
    if not records:
        raise SystemExit("no complete evaluation records found")
    scores = np.asarray([record["candidate_score"] for record in records], dtype=np.float64)
    score = float(scores.mean())
    ci_low, ci_high = bootstrap_pair_ci(records, args.bootstrap_samples, args.seed)
    by_side = {}
    for side in ("us", "ussr"):
        side_records = [record for record in records if record["candidate_side"] == side]
        by_side[side] = {
            "games": len(side_records),
            "wins": sum(float(record["candidate_score"]) == 1.0 for record in side_records),
            "score": float(np.mean([record["candidate_score"] for record in side_records])),
        }
    searched = sum(int(record["searched_decisions"]) for record in records)
    simulations = sum(int(record["simulations"]) for record in records)
    payload = {
        "sources": sources,
        "games": len(records),
        "seed_pairs": len(records) // 2,
        "score": score,
        "paired_bootstrap_ci95_low": ci_low,
        "paired_bootstrap_ci95_high": ci_high,
        "elo_delta": elo_delta(score),
        "elo_ci95_low": elo_delta(ci_low),
        "elo_ci95_high": elo_delta(ci_high),
        "by_candidate_side": by_side,
        "terminal_reason_distribution": dict(
            Counter(str(record["terminal_reason"]) for record in records)
        ),
        "average_searches_per_game": searched / len(records),
        "average_simulations_per_game": simulations / len(records),
        "search_flip_rate": sum(int(record["action_flips"]) for record in records)
        / max(1, searched),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
