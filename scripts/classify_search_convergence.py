#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.search_replay import SearchReplayExample, SearchReplayWriter, load_search_replay
from struggle_ai.search_convergence import compare_q_rankings, visited_q_top


SEED_RE = re.compile(r"seed-(\d+)")
CONFIG_RE = re.compile(r"d(\d+)[_/-](\d+)", re.IGNORECASE)


def parse_replay(value: str) -> tuple[str, Path, int, int]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("replay must be LABEL=MANIFEST") from exc
    match = CONFIG_RE.fullmatch(label)
    if match is None:
        raise argparse.ArgumentTypeError(f"replay label must look like d4_512, got {label!r}")
    return label.lower(), Path(raw_path), int(match.group(1)), int(match.group(2))


def source_key(example: SearchReplayExample) -> tuple[int, int]:
    match = SEED_RE.search(str(example.metadata.source_game or ""))
    if match is None or example.metadata.source_step is None:
        raise ValueError(f"unkeyed replay example: {example.metadata.source_game}")
    return int(match.group(1)), int(example.metadata.source_step)


def action_signature(example: SearchReplayExample) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (str(action.get("type") or ""), str(action.get("value") or ""), str(action.get("label") or ""))
        for action in example.metadata.legal_actions
    )


def root_map(path: Path) -> dict[tuple[int, int], SearchReplayExample]:
    examples = load_search_replay(
        (path,), workers=1, metadata_predicate=lambda metadata: metadata.hierarchy_index == 0
    )
    return {source_key(example): example for example in examples}


def visited(example: SearchReplayExample) -> np.ndarray:
    return np.asarray(example.visit_counts[: example.legal_count], dtype=np.float64) > 0.0


def q_top(example: SearchReplayExample) -> tuple[int, float]:
    return visited_q_top(
        example.visit_counts[: example.legal_count],
        example.q_values[: example.legal_count],
    )


def comparison(
    left: SearchReplayExample,
    right: SearchReplayExample,
    *,
    min_rank_correlation: float,
    meaningful_q_gap: float,
) -> dict[str, Any]:
    if action_signature(left) != action_signature(right):
        return {"comparable": False, "stable": False, "reason": "action_signature_mismatch"}
    return compare_q_rankings(
        left.visit_counts[: left.legal_count],
        left.q_values[: left.legal_count],
        right.visit_counts[: right.legal_count],
        right.q_values[: right.legal_count],
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )


def complete(example: SearchReplayExample | None) -> bool:
    return bool(
        example is not None
        and example.metadata.atomic_horizon_cutoffs == 0
        and example.metadata.node_budget_cutoffs == 0
    )


def soft_q_teacher(example: SearchReplayExample, temperature: float, status: str) -> SearchReplayExample:
    legal = example.legal_count
    valid = visited(example)
    indices = np.flatnonzero(valid)
    q = np.asarray(example.q_values[:legal], dtype=np.float64)
    logits = q[indices] / max(temperature, 1e-6)
    logits -= np.max(logits)
    weights = np.exp(logits)
    policy = np.zeros_like(example.search_policy, dtype=np.float32)
    policy[indices] = (weights / weights.sum()).astype(np.float32)
    best = int(indices[np.argmax(policy[indices])])
    visits = np.zeros_like(example.visit_counts, dtype=np.float32)
    visits[:legal] = policy[:legal] * max(float(example.visit_counts.sum()), 1.0)
    metadata = replace(
        example.metadata,
        search_action=best,
        action_flipped=best != int(example.metadata.raw_action),
        q_advantage_over_raw=float(q[best] - q[int(example.metadata.raw_action)]),
        teacher_action_visits=float(visits[best]),
        total_visits=float(visits.sum()),
        teacher_visit_share=float(policy[best]),
        behavior_policy=f"converged_search:{status.lower()}",
    )
    return SearchReplayExample(
        observation=example.observation,
        raw_policy=example.raw_policy,
        visit_counts=visits,
        search_policy=policy,
        q_values=example.q_values,
        metadata=metadata,
    )


def classify_root(
    examples: dict[str, SearchReplayExample],
    configs: dict[str, tuple[int, int]],
    *,
    min_rank_correlation: float,
    meaningful_q_gap: float,
) -> tuple[str, str | None, dict[str, Any]]:
    ordered = sorted(
        (label for label, example in examples.items() if complete(example)),
        key=lambda label: configs[label],
    )
    if not ordered:
        return "UNRESOLVED", None, {"reason": "no_cutoff_free_search"}
    strongest = max(ordered, key=lambda label: (configs[label][0], configs[label][1]))
    strongest_top, strongest_gap = q_top(examples[strongest])
    comparisons: dict[str, Any] = {}

    def compare(left: str, right: str) -> dict[str, Any] | None:
        if left not in examples or right not in examples or not complete(examples[left]) or not complete(examples[right]):
            return None
        key = f"{left}:{right}"
        comparisons[key] = comparison(
            examples[left], examples[right],
            min_rank_correlation=min_rank_correlation,
            meaningful_q_gap=meaningful_q_gap,
        )
        return comparisons[key]

    budget_pairs = [
        ("d2_32", "d2_128"),
        ("d4_128", "d4_512"),
        ("d6_512", "d6_2048"),
    ]
    budget_evidence = {f"{a}:{b}": compare(a, b) for a, b in budget_pairs}
    high_by_depth = {
        depth: max(
            (label for label in ordered if configs[label][0] == depth),
            key=lambda label: configs[label][1],
            default=None,
        )
        for depth in (2, 4, 6, 8)
    }
    depth_labels = [high_by_depth[depth] for depth in (2, 4, 6, 8) if high_by_depth[depth]]
    depth_evidence = {}
    for left, right in zip(depth_labels, depth_labels[1:]):
        depth_evidence[f"{left}:{right}"] = compare(left, right)

    evidence = {
        "strongest": strongest,
        "strongest_top": strongest_top,
        "strongest_q_gap": strongest_gap,
        "budget_evidence": budget_evidence,
        "depth_evidence": depth_evidence,
        "comparisons": comparisons,
    }
    if strongest_gap < meaningful_q_gap:
        return "NEAR_TIE", strongest, evidence

    usable_depth = [label for label in depth_labels if label]
    adjacent = [depth_evidence.get(f"{a}:{b}") for a, b in zip(usable_depth, usable_depth[1:])]
    stable_adjacent = [item for item in adjacent if item is not None]
    if stable_adjacent and all(bool(item.get("stable")) for item in stable_adjacent):
        return "BUDGET_CONVERGENCE", strongest, evidence

    if len(usable_depth) >= 2:
        deepest_pair = depth_evidence.get(f"{usable_depth[-2]}:{usable_depth[-1]}")
        earlier = adjacent[:-1]
        if deepest_pair and deepest_pair.get("stable") and any(
            item is not None and not item.get("stable") and not item.get("near_tie") for item in earlier
        ):
            return "HORIZON_CONVERGENCE", strongest, evidence

    strongest_depth = configs[strongest][0]
    same_depth_pair = next(
        (
            item for name, item in budget_evidence.items()
            if item is not None and name.startswith(f"d{strongest_depth}_")
        ),
        None,
    )
    if same_depth_pair and same_depth_pair.get("stable") and len(usable_depth) >= 2:
        prior_depth_pair = depth_evidence.get(
            f"{usable_depth[-2]}:{usable_depth[-1]}"
        )
        if (
            prior_depth_pair is not None
            and not prior_depth_pair.get("stable")
            and not prior_depth_pair.get("near_tie")
        ):
            return "HORIZON_CONVERGENCE", strongest, evidence
    if same_depth_pair and same_depth_pair.get("stable") and len(usable_depth) == 1:
        return "BUDGET_CONVERGENCE", strongest, evidence
    return "UNRESOLVED", strongest, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify PUCT budget/depth convergence and build safe teachers.")
    parser.add_argument("--position-manifest", type=Path, required=True)
    parser.add_argument("--replay", action="append", type=parse_replay, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-rank-correlation", type=float, default=0.70)
    parser.add_argument("--meaningful-q-gap", type=float, default=0.01)
    parser.add_argument("--q-temperature", type=float, default=0.05)
    parser.add_argument("--max-d6-adjudications", type=int, default=48)
    parser.add_argument("--max-d8-adjudications", type=int, default=8)
    args = parser.parse_args()

    configs = {label: (depth, budget) for label, _path, depth, budget in args.replay}
    maps = {label: root_map(path) for label, path, _depth, _budget in args.replay}
    manifest = json.loads(args.position_manifest.read_text(encoding="utf-8"))
    positions = list(manifest.get("positions") or [])
    position_by_key = {(int(row["seed"]), int(row["action_index"])): row for row in positions}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "convergence_evidence.jsonl"
    status_counts: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    d6_candidates: list[dict[str, Any]] = []
    d8_candidates: list[dict[str, Any]] = []

    with SearchReplayWriter(args.output_dir / "teacher_replay", shard_size=128) as writer, evidence_path.open("w", encoding="utf-8") as handle:
        for key in sorted(position_by_key):
            available = {label: mapping[key] for label, mapping in maps.items() if key in mapping}
            status, teacher_label, evidence = classify_root(
                available,
                configs,
                min_rank_correlation=float(args.min_rank_correlation),
                meaningful_q_gap=float(args.meaningful_q_gap),
            )
            position = position_by_key[key]
            status_counts[status] += 1
            side_counts[f"{status}:{position['side']}"] += 1
            payload = {
                "seed": key[0],
                "action_index": key[1],
                "side": position["side"],
                "era": position.get("era"),
                "category": position.get("category"),
                "branch_bucket": position.get("branch_bucket"),
                "legal_count": position.get("legal_count"),
                "status": status,
                "teacher_config": teacher_label,
                **evidence,
                "search": {
                    label: {
                        "P": example.raw_policy[: example.legal_count].tolist(),
                        "Q": example.q_values[: example.legal_count].tolist(),
                        "N": example.visit_counts[: example.legal_count].tolist(),
                        "search_action": int(example.metadata.search_action),
                        "cutoffs": int(example.metadata.atomic_horizon_cutoffs + example.metadata.node_budget_cutoffs),
                    }
                    for label, example in available.items()
                },
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            if status in {"BUDGET_CONVERGENCE", "HORIZON_CONVERGENCE"} and teacher_label:
                writer.add(soft_q_teacher(available[teacher_label], float(args.q_temperature), status))
                teacher_counts[teacher_label] += 1
            elif status == "UNRESOLVED":
                candidate = dict(position)
                candidate["convergence_status"] = status
                candidate["strongest_config"] = teacher_label
                candidate["strongest_q_gap"] = float(evidence.get("strongest_q_gap", 0.0))
                if "d6_2048" not in available:
                    d6_candidates.append(candidate)
                elif "d8_2048" not in available:
                    d8_candidates.append(candidate)

    d6_candidates.sort(key=adjudication_priority)
    d8_candidates.sort(key=adjudication_priority)
    d6_candidates = d6_candidates[: max(0, int(args.max_d6_adjudications))]
    d8_candidates = d8_candidates[: max(0, int(args.max_d8_adjudications))]
    (args.output_dir / "d6_2048_candidates.json").write_text(
        json.dumps({"positions": d6_candidates}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "d8_2048_candidates.json").write_text(
        json.dumps({"positions": d8_candidates}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "positions": len(positions),
        "status_counts": dict(status_counts),
        "status_by_side": dict(side_counts),
        "teacher_configs": dict(teacher_counts),
        "teacher_examples": sum(teacher_counts.values()),
        "d6_2048_candidates": len(d6_candidates),
        "d8_2048_candidates": len(d8_candidates),
        "min_rank_correlation": float(args.min_rank_correlation),
        "meaningful_q_gap": float(args.meaningful_q_gap),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def stable_position_key(position: dict[str, Any]) -> str:
    return f"{int(position['seed']):012d}:{int(position['action_index']):06d}"


def adjudication_priority(position: dict[str, Any]) -> tuple[int, float, int, str]:
    strategic = str(position.get("category") or "") in {
        "scoring",
        "scoring_card",
        "opponent_event",
        "card_timing",
        "coup",
        "coup_target",
        "realign",
        "realignment",
        "realignment_target",
        "influence_allocation",
    }
    return (
        -int(strategic),
        -abs(float(position.get("strongest_q_gap", 0.0))),
        -int(position.get("legal_count", 0)),
        stable_position_key(position),
    )


if __name__ == "__main__":
    main()
