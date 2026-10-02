#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate offline PUCT depth/budget sweeps.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def config_key(result: dict[str, Any]) -> tuple[int, int]:
    return int(result["config"]["depth"]), int(result["config"]["simulations"])


def q_map(result: dict[str, Any]) -> dict[str, float]:
    return {str(row["action"]): float(row["Q"]) for row in result.get("root_table", [])}


def main() -> None:
    args = parse_args()
    positions: dict[tuple[int, int, int], dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    sources: list[str] = []
    for path in args.input:
        resolved = path.expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        sources.append(str(resolved))
        for position in payload.get("positions", []):
            position_key = (
                int(position["seed"]),
                int(position.get("game_occurrence", 0)),
                int(position["action_index"]),
            )
            for result in position.get("results", []):
                positions[position_key][config_key(result)] = result

    by_config: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for results in positions.values():
        for key, result in results.items():
            by_config[key].append(result)
    config_summary = []
    for (depth, simulations), results in sorted(by_config.items()):
        searched = [result for result in results if result.get("searched", True)]
        config_summary.append(
            {
                "depth": depth,
                "simulations": simulations,
                "positions": len(results),
                "searched_positions": len(searched),
                "action_flip_rate": float(np.mean([bool(result["action_flipped"]) for result in searched])) if searched else 0.0,
                "average_seconds": float(np.mean([float(result["seconds"]) for result in searched])) if searched else 0.0,
                "simulations_per_second": float(np.mean([float(result["simulations_per_second"]) for result in searched])) if searched else 0.0,
                "cross_turn_fraction": float(np.mean([int(result.get("max_turn_delta", 0)) > 0 for result in searched])) if searched else 0.0,
                "average_cross_turn_leaf_evaluations": float(
                    np.mean([int(result.get("cross_turn_leaf_evaluations", 0)) for result in searched])
                ) if searched else 0.0,
                "average_expanded_nodes": float(np.mean([int(result.get("expanded_nodes", 0)) for result in searched])) if searched else 0.0,
                "node_budget_cutoffs": int(sum(int(result.get("node_budget_cutoffs", 0)) for result in searched)),
                "atomic_horizon_cutoffs": int(sum(int(result.get("atomic_horizon_cutoffs", 0)) for result in searched)),
            }
        )

    pair_values: dict[tuple[tuple[int, int], tuple[int, int]], list[dict[str, Any]]] = defaultdict(list)
    for position_key, results in positions.items():
        keys = sorted(results)
        for left_index, left_key in enumerate(keys):
            for right_key in keys[left_index + 1 :]:
                left = results[left_key]
                right = results[right_key]
                left_q = q_map(left)
                right_q = q_map(right)
                shared = sorted(set(left_q) & set(right_q))
                q_linf = None
                if shared:
                    q_linf = max(abs(left_q[action] - right_q[action]) for action in shared)
                pair_values[(left_key, right_key)].append(
                    {
                        "position": position_key,
                        "action_agreement": left["search_action_text"] == right["search_action_text"],
                        "q_linf": q_linf,
                    }
                )
    pair_summary = []
    for (left, right), values in sorted(pair_values.items()):
        q_values = [float(value["q_linf"]) for value in values if value["q_linf"] is not None]
        pair_summary.append(
            {
                "left": {"depth": left[0], "simulations": left[1]},
                "right": {"depth": right[0], "simulations": right[1]},
                "positions": len(values),
                "action_agreement_rate": float(np.mean([value["action_agreement"] for value in values])),
                "average_q_linf": float(np.mean(q_values)) if q_values else None,
                "max_q_linf": max(q_values) if q_values else None,
                "disagreements": [value["position"] for value in values if not value["action_agreement"]],
            }
        )

    payload = {
        "sources": sources,
        "positions": len(positions),
        "configs": config_summary,
        "pairs": pair_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
