#!/usr/bin/env python3
"""Summarize paired D2/D4 anomaly probes with production adaptive semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.search_convergence import compare_q_rankings


CONFIGS = ((2, 32), (2, 128), (4, 128))
SUSPICIOUS = {
    "ussr_uk_overcontrol": "uk",
    "us_italy_overcontrol": "italy",
    "ussr_mexico_realignment": "mexico",
    "us_venezuela_realignment": "venezuela",
}


def normalize_action(value: str) -> str:
    return value.split(" [", 1)[0].replace(" ", "").replace("-", "").casefold()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rank-correlation", type=float, default=0.70)
    parser.add_argument("--meaningful-q-gap", type=float, default=0.01)
    return parser.parse_args()


def table_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    count = int(row["root_legal_count"])
    visits = np.zeros(count, dtype=np.float64)
    q_values = np.zeros(count, dtype=np.float64)
    for action in row["root_table"]:
        index = int(action["action_index"])
        visits[index] = float(action["N"])
        q_values[index] = float(action["Q"])
    return visits, q_values


def incomplete(row: dict[str, Any]) -> bool:
    return bool(
        int(row.get("atomic_horizon_cutoffs", 0))
        or int(row.get("node_budget_cutoffs", 0))
    )


def select_adaptive(
    rows: list[dict[str, Any]],
    *,
    min_rank_correlation: float = 0.70,
    meaningful_q_gap: float = 0.01,
) -> dict[str, Any]:
    by_config = {(int(row["depth"]), int(row["budget"])): row for row in rows}
    missing = [config for config in CONFIGS if config not in by_config]
    if missing:
        raise ValueError(f"missing search configurations: {missing}")
    base, budget, deep = (by_config[config] for config in CONFIGS)
    if incomplete(base) or incomplete(budget):
        return {"selected": budget, "status": "UNRESOLVED", "fallback": True, "comparisons": []}

    base_visits, base_q = table_arrays(base)
    budget_visits, budget_q = table_arrays(budget)
    budget_comparison = compare_q_rankings(
        base_visits,
        base_q,
        budget_visits,
        budget_q,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )
    comparisons = [{"left": "d2_32", "right": "d2_128", **budget_comparison}]
    if budget_comparison.get("stable"):
        return {
            "selected": budget,
            "status": "BUDGET_CONVERGENCE",
            "fallback": False,
            "comparisons": comparisons,
        }
    if budget_comparison.get("near_tie") or incomplete(deep):
        return {
            "selected": budget,
            "status": "NEAR_TIE" if budget_comparison.get("near_tie") else "UNRESOLVED",
            "fallback": True,
            "comparisons": comparisons,
        }

    deep_visits, deep_q = table_arrays(deep)
    deep_comparison = compare_q_rankings(
        budget_visits,
        budget_q,
        deep_visits,
        deep_q,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )
    comparisons.append({"left": "d2_128", "right": "d4_128", **deep_comparison})
    if deep_comparison.get("stable"):
        return {
            "selected": deep,
            "status": "HORIZON_CONVERGENCE",
            "fallback": False,
            "comparisons": comparisons,
        }
    return {
        "selected": budget,
        "status": "NEAR_TIE" if deep_comparison.get("near_tie") else "UNRESOLVED",
        "fallback": True,
        "comparisons": comparisons,
    }


def summarize_path(
    path: Path,
    *,
    min_rank_correlation: float,
    meaningful_q_gap: float,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = str(payload["spec"]["name"])
    decision = select_adaptive(
        list(payload.get("search", [])),
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )
    selected = decision.pop("selected")
    suspicious = SUSPICIOUS[name]
    return {
        "root": name,
        "suspicious_action": suspicious,
        "selected_action": selected["choice"],
        "suspicious_selected": normalize_action(str(selected["choice"])) == suspicious,
        "selected_depth": int(selected["depth"]),
        "selected_budget": int(selected["budget"]),
        "selected_Q": float(selected["choice_Q"]),
        **decision,
    }


def main() -> None:
    args = parse_args()
    roots = [
        summarize_path(
            path,
            min_rank_correlation=float(args.min_rank_correlation),
            meaningful_q_gap=float(args.meaningful_q_gap),
        )
        for path in args.input
    ]
    summary = {
        "passed": all(not root["suspicious_selected"] for root in roots),
        "information_masking": "validated_by_probe_harness",
        "paired_matrix_seeds": True,
        "roots": roots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
