#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.search_convergence import compare_q_rankings, visited_q_top
from struggle_ai.search_replay import SearchReplayExample, load_search_replay


CONFIG_RE = re.compile(r"^d(?P<depth>\d+)_(?P<budget>\d+)$")
SEED_RE = re.compile(r"seed-(\d+)")


def parse_replay(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or CONFIG_RE.fullmatch(label) is None:
        raise argparse.ArgumentTypeError("replay must be dDEPTH_BUDGET=MANIFEST")
    return label, Path(raw_path)


def source_key(example: SearchReplayExample) -> tuple[int, int]:
    match = SEED_RE.search(str(example.metadata.source_game or ""))
    if match is None or example.metadata.source_step is None:
        raise ValueError(f"cannot identify root: {example.metadata.source_game!r}")
    return int(match.group(1)), int(example.metadata.source_step)


def action_signature(example: SearchReplayExample) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            str(action.get("type") or ""),
            str(action.get("value") or ""),
            str(action.get("label") or ""),
        )
        for action in example.metadata.legal_actions
    )


def complete(example: SearchReplayExample) -> bool:
    return not (
        example.metadata.atomic_horizon_cutoffs
        or example.metadata.node_budget_cutoffs
        or example.metadata.evidence_gate_rejected
    )


def root_map(path: Path) -> dict[tuple[int, int], SearchReplayExample]:
    examples = load_search_replay((path,), workers=8)
    return {
        source_key(example): example
        for example in examples
        if example.metadata.hierarchy_index == 0
    }


def percentile(values: list[float], quantile: float) -> float | None:
    return None if not values else float(np.quantile(np.asarray(values), quantile))


def pair_summary(
    left: dict[tuple[int, int], SearchReplayExample],
    right: dict[tuple[int, int], SearchReplayExample],
    *,
    min_rank_correlation: float,
    meaningful_q_gap: float,
) -> dict[str, Any]:
    counts: defaultdict[str, int] = defaultdict(int)
    correlations: list[float] = []
    q_linf: list[float] = []
    q_scale_ratios: list[float] = []
    q_mean_shifts: list[float] = []
    top_cross_regrets: list[float] = []
    side: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    records: list[dict[str, Any]] = []
    for key in sorted(set(left) & set(right)):
        lhs, rhs = left[key], right[key]
        counts["shared_roots"] += 1
        if not complete(lhs) or not complete(rhs):
            counts["cutoff_or_rejected"] += 1
            continue
        if action_signature(lhs) != action_signature(rhs):
            counts["action_signature_mismatch"] += 1
            continue
        comparison = compare_q_rankings(
            lhs.visit_counts[: lhs.legal_count],
            lhs.q_values[: lhs.legal_count],
            rhs.visit_counts[: rhs.legal_count],
            rhs.q_values[: rhs.legal_count],
            min_rank_correlation=min_rank_correlation,
            meaningful_q_gap=meaningful_q_gap,
        )
        if not comparison.get("comparable"):
            counts["insufficient_shared_actions"] += 1
            continue
        counts["comparable"] += 1
        side_name = str(lhs.metadata.side)
        side[side_name]["comparable"] += 1
        stable = bool(comparison["stable"])
        near_tie = bool(comparison["near_tie"])
        same_top = int(comparison["left_top"]) == int(comparison["right_top"])
        counts[f"stable:{stable}"] += 1
        counts[f"near_tie:{near_tie}"] += 1
        counts[f"same_top:{same_top}"] += 1
        side[side_name][f"stable:{stable}"] += 1
        side[side_name][f"same_top:{same_top}"] += 1
        correlations.append(float(comparison["rank_correlation"]))
        q_linf.append(float(comparison["q_linf"]))

        common = (
            (np.asarray(lhs.visit_counts[: lhs.legal_count]) > 0.0)
            & (np.asarray(rhs.visit_counts[: rhs.legal_count]) > 0.0)
        )
        q_left = np.asarray(lhs.q_values[: lhs.legal_count], dtype=np.float64)[common]
        q_right = np.asarray(rhs.q_values[: rhs.legal_count], dtype=np.float64)[common]
        left_std = float(np.std(q_left))
        right_std = float(np.std(q_right))
        if left_std > 1e-8:
            q_scale_ratios.append(right_std / left_std)
        q_mean_shifts.append(float(np.mean(q_right) - np.mean(q_left)))
        left_top = int(comparison["left_top"])
        right_top = int(comparison["right_top"])
        top_cross_regrets.append(float(rhs.q_values[right_top] - rhs.q_values[left_top]))
        records.append(
            {
                "seed": key[0],
                "action_index": key[1],
                "side": side_name,
                "legal_count": lhs.legal_count,
                **comparison,
            }
        )

    comparable = max(1, counts["comparable"])
    by_side = {}
    for side_name, values in sorted(side.items()):
        denominator = max(1, values["comparable"])
        by_side[side_name] = {
            **dict(values),
            "stable_rate": values["stable:True"] / denominator,
            "same_top_rate": values["same_top:True"] / denominator,
        }
    return {
        "counts": dict(counts),
        "stable_rate": counts["stable:True"] / comparable,
        "same_top_rate": counts["same_top:True"] / comparable,
        "rank_correlation": {
            "mean": None if not correlations else float(np.mean(correlations)),
            "median": percentile(correlations, 0.5),
            "p10": percentile(correlations, 0.1),
        },
        "q_linf": {
            "mean": None if not q_linf else float(np.mean(q_linf)),
            "median": percentile(q_linf, 0.5),
            "p90": percentile(q_linf, 0.9),
        },
        "q_scale_ratio_right_over_left": {
            "mean": None if not q_scale_ratios else float(np.mean(q_scale_ratios)),
            "median": percentile(q_scale_ratios, 0.5),
            "p10": percentile(q_scale_ratios, 0.1),
            "p90": percentile(q_scale_ratios, 0.9),
        },
        "q_mean_shift_right_minus_left": {
            "mean": None if not q_mean_shifts else float(np.mean(q_mean_shifts)),
            "median": percentile(q_mean_shifts, 0.5),
        },
        "right_q_regret_of_left_top": {
            "mean": None if not top_cross_regrets else float(np.mean(top_cross_regrets)),
            "median": percentile(top_cross_regrets, 0.5),
            "p90": percentile(top_cross_regrets, 0.9),
        },
        "by_side": by_side,
        "records": records,
    }


def checkpoint_delta(reports: list[Path]) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    if not payloads:
        return {}
    parent = payloads[0]
    parent_records = {
        (record["source_game"], record["source_step"], record["hierarchy_index"]): record
        for record in parent["records"]
    }
    result: dict[str, Any] = {}
    for payload in payloads:
        deltas: list[float] = []
        raw_deltas: list[float] = []
        side_deltas: defaultdict[str, list[float]] = defaultdict(list)
        hierarchy_deltas: defaultdict[str, list[float]] = defaultdict(list)
        top1_changed_to_teacher = 0
        top1_changed_from_teacher = 0
        shared = 0
        for record in payload["records"]:
            key = (record["source_game"], record["source_step"], record["hierarchy_index"])
            baseline = parent_records.get(key)
            if baseline is None:
                continue
            shared += 1
            delta = float(record["teacher_probability"] - baseline["teacher_probability"])
            raw_delta = float(record["raw_action_probability"] - baseline["raw_action_probability"])
            deltas.append(delta)
            raw_deltas.append(raw_delta)
            side_deltas[str(record["side"])].append(delta)
            hierarchy_deltas[str(record["hierarchy_level"])].append(delta)
            top1_changed_to_teacher += int(record["teacher_top1"] and not baseline["teacher_top1"])
            top1_changed_from_teacher += int(baseline["teacher_top1"] and not record["teacher_top1"])
        label = Path(str(payload["checkpoint"])).name
        result[label] = {
            "checkpoint": payload["checkpoint"],
            "shared_examples": shared,
            "overall": payload["overall"],
            "mean_teacher_probability_delta_vs_parent": float(np.mean(deltas)) if deltas else 0.0,
            "median_teacher_probability_delta_vs_parent": percentile(deltas, 0.5),
            "mean_raw_action_probability_delta_vs_parent": float(np.mean(raw_deltas)) if raw_deltas else 0.0,
            "top1_changed_to_teacher": top1_changed_to_teacher,
            "top1_changed_from_teacher": top1_changed_from_teacher,
            "teacher_probability_delta_by_side": {
                key: float(np.mean(values)) for key, values in sorted(side_deltas.items())
            },
            "teacher_probability_delta_by_hierarchy": {
                key: float(np.mean(values)) for key, values in sorted(hierarchy_deltas.items())
            },
        }
    return result


def adaptive_policy_summary(
    maps: dict[str, dict[tuple[int, int], SearchReplayExample]],
    *,
    base_label: str,
    budget_label: str,
    deep_label: str,
    benchmark_label: str,
    min_rank_correlation: float,
    meaningful_q_gap: float,
) -> dict[str, Any]:
    labels = (base_label, budget_label, deep_label, benchmark_label)
    if any(label not in maps for label in labels):
        return {"available": False, "missing": [label for label in labels if label not in maps]}
    keys = set(maps[base_label])
    for label in labels[1:]:
        keys &= set(maps[label])
    counts: defaultdict[str, int] = defaultdict(int)
    regrets: list[float] = []
    side_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    records: list[dict[str, Any]] = []
    for key in sorted(keys):
        base, budget, deep, benchmark = (maps[label][key] for label in labels)
        if not all(complete(example) for example in (base, budget, deep, benchmark)):
            counts["incomplete"] += 1
            continue
        signatures = {action_signature(example) for example in (base, budget, deep, benchmark)}
        if len(signatures) != 1:
            counts["action_signature_mismatch"] += 1
            continue
        budget_comparison = compare_q_rankings(
            base.visit_counts[: base.legal_count],
            base.q_values[: base.legal_count],
            budget.visit_counts[: budget.legal_count],
            budget.q_values[: budget.legal_count],
            min_rank_correlation=min_rank_correlation,
            meaningful_q_gap=meaningful_q_gap,
        )
        selected = budget
        status = "BUDGET_CONVERGENCE"
        fallback = False
        deep_comparison = None
        if not budget_comparison.get("stable"):
            if budget_comparison.get("near_tie"):
                status = "NEAR_TIE"
                fallback = True
            else:
                deep_comparison = compare_q_rankings(
                    budget.visit_counts[: budget.legal_count],
                    budget.q_values[: budget.legal_count],
                    deep.visit_counts[: deep.legal_count],
                    deep.q_values[: deep.legal_count],
                    min_rank_correlation=min_rank_correlation,
                    meaningful_q_gap=meaningful_q_gap,
                )
                if deep_comparison.get("stable"):
                    selected = deep
                    status = "HORIZON_CONVERGENCE"
                else:
                    status = "NEAR_TIE" if deep_comparison.get("near_tie") else "UNRESOLVED"
                    fallback = True
        selected_top, _selected_gap = visited_q_top(
            selected.visit_counts[: selected.legal_count],
            selected.q_values[: selected.legal_count],
        )
        benchmark_top, _benchmark_gap = visited_q_top(
            benchmark.visit_counts[: benchmark.legal_count],
            benchmark.q_values[: benchmark.legal_count],
        )
        regret = float(benchmark.q_values[benchmark_top] - benchmark.q_values[selected_top])
        side = str(base.metadata.side)
        counts["roots"] += 1
        counts[f"status:{status}"] += 1
        counts[f"fallback:{fallback}"] += 1
        counts[f"benchmark_top_agreement:{selected_top == benchmark_top}"] += 1
        side_counts[side]["roots"] += 1
        side_counts[side][f"status:{status}"] += 1
        side_counts[side][f"benchmark_top_agreement:{selected_top == benchmark_top}"] += 1
        regrets.append(regret)
        records.append(
            {
                "seed": key[0],
                "action_index": key[1],
                "side": side,
                "status": status,
                "fallback": fallback,
                "selected_config": deep_label if selected is deep else budget_label,
                "selected_top": selected_top,
                "benchmark_top": benchmark_top,
                "benchmark_q_regret": regret,
                "budget_comparison": budget_comparison,
                "deep_comparison": deep_comparison,
            }
        )
    roots = max(1, counts["roots"])
    return {
        "available": True,
        "configs": {
            "base": base_label,
            "budget": budget_label,
            "deep": deep_label,
            "benchmark": benchmark_label,
        },
        "counts": dict(counts),
        "deep_accept_rate": counts["status:HORIZON_CONVERGENCE"] / roots,
        "fallback_rate": counts["fallback:True"] / roots,
        "benchmark_top_agreement_rate": counts["benchmark_top_agreement:True"] / roots,
        "benchmark_q_regret": {
            "mean": None if not regrets else float(np.mean(regrets)),
            "median": percentile(regrets, 0.5),
            "p90": percentile(regrets, 0.9),
        },
        "by_side": {side: dict(values) for side, values in sorted(side_counts.items())},
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze search depth/Q and actor-teacher compatibility.")
    parser.add_argument("--replay", action="append", type=parse_replay, required=True)
    parser.add_argument("--checkpoint-report", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rank-correlation", type=float, default=0.70)
    parser.add_argument("--meaningful-q-gap", type=float, default=0.01)
    args = parser.parse_args()

    maps = {label: root_map(path) for label, path in args.replay}
    configs = {
        label: tuple(int(value) for value in CONFIG_RE.fullmatch(label).groups())
        for label in maps
    }
    ordered = sorted(maps, key=lambda label: configs[label])
    pairs: dict[str, Any] = {}
    same_depth: dict[int, list[str]] = defaultdict(list)
    for label in ordered:
        same_depth[configs[label][0]].append(label)
    requested_pairs: list[tuple[str, str]] = []
    for labels in same_depth.values():
        requested_pairs.extend(zip(labels, labels[1:]))
    for budget in sorted({budget for _depth, budget in configs.values()}):
        labels = sorted(
            (label for label in ordered if configs[label][1] == budget),
            key=lambda label: configs[label][0],
        )
        requested_pairs.extend(zip(labels, labels[1:]))
    strongest_by_depth = {
        depth: max(labels, key=lambda label: configs[label][1])
        for depth, labels in same_depth.items()
    }
    depth_labels = [strongest_by_depth[depth] for depth in sorted(strongest_by_depth)]
    requested_pairs.extend(zip(depth_labels, depth_labels[1:]))
    for left, right in (
        ("d2_32", "d4_128"),
        ("d4_128", "d6_512"),
        ("d2_128", "d6_512"),
    ):
        if left in maps and right in maps:
            requested_pairs.append((left, right))
    seen: set[tuple[str, str]] = set()
    for left, right in requested_pairs:
        if (left, right) in seen:
            continue
        seen.add((left, right))
        pairs[f"{left}:{right}"] = pair_summary(
            maps[left],
            maps[right],
            min_rank_correlation=float(args.min_rank_correlation),
            meaningful_q_gap=float(args.meaningful_q_gap),
        )

    payload = {
        "configs": {label: {"depth": configs[label][0], "budget": configs[label][1]} for label in ordered},
        "root_counts": {label: len(maps[label]) for label in ordered},
        "thresholds": {
            "min_rank_correlation": float(args.min_rank_correlation),
            "meaningful_q_gap": float(args.meaningful_q_gap),
        },
        "pair_comparisons": pairs,
        "checkpoint_teacher_absorption": checkpoint_delta(args.checkpoint_report),
        "adaptive_policy": adaptive_policy_summary(
            maps,
            base_label="d2_32",
            budget_label="d2_128",
            deep_label="d4_128",
            benchmark_label="d6_512",
            min_rank_correlation=float(args.min_rank_correlation),
            meaningful_q_gap=float(args.meaningful_q_gap),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "root_counts": payload["root_counts"],
        "pairs": {
            name: {
                "comparable": row["counts"].get("comparable", 0),
                "stable_rate": row["stable_rate"],
                "same_top_rate": row["same_top_rate"],
                "rank_correlation_mean": row["rank_correlation"]["mean"],
                "q_scale_ratio_median": row["q_scale_ratio_right_over_left"]["median"],
            }
            for name, row in pairs.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
