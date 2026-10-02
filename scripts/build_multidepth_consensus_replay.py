#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.search_replay import SearchReplayExample, SearchReplayWriter, load_search_replay


SEED_RE = re.compile(r"seed-(\d+)")


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


def rank_correlation(q_a: np.ndarray, q_b: np.ndarray, valid: np.ndarray) -> float:
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return 0.0
    rank_a = np.empty(len(indices), dtype=np.float64)
    rank_b = np.empty(len(indices), dtype=np.float64)
    rank_a[np.argsort(q_a[indices])] = np.arange(len(indices), dtype=np.float64)
    rank_b[np.argsort(q_b[indices])] = np.arange(len(indices), dtype=np.float64)
    if np.std(rank_a) == 0.0 or np.std(rank_b) == 0.0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def soft_q_policy(q_values: np.ndarray, valid: np.ndarray, temperature: float) -> np.ndarray:
    result = np.zeros_like(q_values, dtype=np.float32)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return result
    logits = q_values[indices] / max(float(temperature), 1e-6)
    logits -= np.max(logits)
    weights = np.exp(logits)
    result[indices] = (weights / weights.sum()).astype(np.float32)
    return result


def consensus_example(
    d2: SearchReplayExample,
    d4: SearchReplayExample,
    d6: SearchReplayExample | None,
    *,
    q_temperature: float,
    min_rank_correlation: float,
    min_q_gap: float,
) -> tuple[SearchReplayExample | None, dict[str, Any]]:
    if action_signature(d2) != action_signature(d4) or (d6 is not None and action_signature(d2) != action_signature(d6)):
        return None, {"accepted": False, "reason": "action_signature_mismatch"}
    legal = d4.legal_count
    depths = [(2, d2, 0.35), (4, d4, 0.65)]
    valid = np.ones(legal, dtype=bool)
    comparison_depths = depths if d6 is None else [*depths, (6, d6, 0.0)]
    for _depth, example, _weight in comparison_depths:
        valid &= np.asarray(example.visit_counts[:legal]) > 0.0
    if int(valid.sum()) < 2:
        return None, {"accepted": False, "reason": "insufficient_shared_q"}
    q2 = np.asarray(d2.q_values[:legal], dtype=np.float64)
    q4 = np.asarray(d4.q_values[:legal], dtype=np.float64)
    correlation = rank_correlation(q2, q4, valid)
    top_by_depth = {
        str(depth): int(np.flatnonzero(valid)[np.argmax(np.asarray(example.q_values[:legal])[valid])])
        for depth, example, _weight in comparison_depths
    }
    top_agreement = len(set(top_by_depth.values())) == 1
    if d6 is None and not top_agreement and correlation < float(min_rank_correlation):
        return None, {
            "accepted": False,
            "reason": "depth_disagreement",
            "rank_correlation": correlation,
            "top_by_depth": top_by_depth,
        }
    pairwise_correlations = {"2:4": correlation}
    adjudicated_depth = None
    if d6 is not None:
        q6 = np.asarray(d6.q_values[:legal], dtype=np.float64)
        corr_26 = rank_correlation(q2, q6, valid)
        corr_46 = rank_correlation(q4, q6, valid)
        pairwise_correlations.update({"2:6": corr_26, "4:6": corr_46})
        d6_top = top_by_depth["6"]
        supported = []
        if d6_top == top_by_depth["2"] and corr_26 >= float(min_rank_correlation):
            supported.append((2, d2, corr_26))
        if d6_top == top_by_depth["4"] and corr_46 >= float(min_rank_correlation):
            supported.append((4, d4, corr_46))
        if not supported:
            return None, {
                "accepted": False,
                "reason": "unresolved_with_d6",
                "rank_correlation": correlation,
                "pairwise_correlations": pairwise_correlations,
                "top_by_depth": top_by_depth,
            }
        adjudicated_depth, supported_example, _correlation = max(
            supported, key=lambda item: (item[2], item[0])
        )
        depths = [(adjudicated_depth, supported_example, 0.4), (6, d6, 0.6)]
        valid = (
            (np.asarray(supported_example.visit_counts[:legal]) > 0.0)
            & (np.asarray(d6.visit_counts[:legal]) > 0.0)
        )
        if int(valid.sum()) < 2:
            return None, {
                "accepted": False,
                "reason": "insufficient_adjudicated_q",
                "pairwise_correlations": pairwise_correlations,
                "top_by_depth": top_by_depth,
            }
    consensus_q = np.zeros(legal, dtype=np.float64)
    total_weight = 0.0
    for _depth, example, weight in depths:
        consensus_q[valid] += weight * np.asarray(example.q_values[:legal], dtype=np.float64)[valid]
        total_weight += weight
    consensus_q[valid] /= total_weight
    ranked = np.sort(consensus_q[valid])
    gap = float(ranked[-1] - ranked[-2])
    if gap < float(min_q_gap):
        return None, {
            "accepted": False,
            "reason": "low_consensus_q_gap",
            "rank_correlation": correlation,
            "top_by_depth": top_by_depth,
            "consensus_q_gap": gap,
        }
    policy = soft_q_policy(consensus_q, valid, q_temperature)
    best = int(np.argmax(policy))
    output_size = len(d4.search_policy)
    q_padded = np.zeros(output_size, dtype=np.float32)
    q_padded[:legal] = consensus_q.astype(np.float32)
    visits = np.zeros(output_size, dtype=np.float32)
    visits[:legal] = policy[:legal] * float(sum(example.visit_counts.sum() for _, example, _ in depths))
    search_policy = np.zeros(output_size, dtype=np.float32)
    search_policy[:legal] = policy[:legal]
    metadata = replace(
        d4.metadata,
        search_action=best,
        search_depth=max(depth for depth, _example, _weight in depths),
        simulation_budget=sum(int(example.metadata.simulation_budget) for _depth, example, _weight in depths),
        root_q_gap=gap,
        q_advantage_over_raw=float(consensus_q[best] - consensus_q[int(d4.metadata.raw_action)]),
        action_flipped=best != int(d4.metadata.raw_action),
        behavior_policy="multidepth_q_consensus",
        teacher_action_visits=float(visits[best]),
        total_visits=float(visits.sum()),
        teacher_visit_share=float(search_policy[best]),
    )
    evidence = {
        "accepted": True,
        "rank_correlation": correlation,
        "pairwise_correlations": pairwise_correlations,
        "adjudicated_depth": adjudicated_depth,
        "top_agreement": top_agreement,
        "top_by_depth": top_by_depth,
        "consensus_action": best,
        "consensus_q_gap": gap,
    }
    return SearchReplayExample(
        observation=d4.observation,
        raw_policy=d4.raw_policy,
        visit_counts=visits,
        search_policy=search_policy,
        q_values=q_padded,
        metadata=metadata,
    ), evidence


def root_map(path: Path) -> dict[tuple[int, int], SearchReplayExample]:
    examples = load_search_replay(
        (path,),
        workers=1,
        metadata_predicate=lambda metadata: metadata.hierarchy_index == 0,
    )
    return {source_key(example): example for example in examples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Q-aware D2/D4/D6 actor consensus replay.")
    parser.add_argument("--d2-replay", type=Path, required=True)
    parser.add_argument("--d4-replay", type=Path, required=True)
    parser.add_argument("--d6-replay", type=Path)
    parser.add_argument("--position-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q-temperature", type=float, default=0.02)
    parser.add_argument("--min-rank-correlation", type=float, default=0.5)
    parser.add_argument("--min-q-gap", type=float, default=0.002)
    parser.add_argument("--max-d6-conflicts", type=int, default=32)
    args = parser.parse_args()

    d2 = root_map(args.d2_replay)
    d4 = root_map(args.d4_replay)
    d6 = root_map(args.d6_replay) if args.d6_replay else {}
    positions = json.loads(args.position_manifest.read_text(encoding="utf-8"))["positions"]
    position_by_key = {(int(row["seed"]), int(row["action_index"])): row for row in positions}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / "multidepth_evidence.jsonl"
    conflict_positions = []
    accepted = rejected = 0
    correlations = []
    with SearchReplayWriter(args.output_dir / "search_replay", shard_size=128) as writer, evidence_path.open("w", encoding="utf-8") as evidence_file:
        for key in sorted(position_by_key):
            if key not in d2 or key not in d4:
                rejected += 1
                evidence_file.write(json.dumps({"key": key, "accepted": False, "reason": "missing_depth"}) + "\n")
                continue
            example, evidence = consensus_example(
                d2[key], d4[key], d6.get(key),
                q_temperature=args.q_temperature,
                min_rank_correlation=args.min_rank_correlation,
                min_q_gap=args.min_q_gap,
            )
            row = {
                "seed": key[0],
                "action_index": key[1],
                "legal_actions": list(d4[key].metadata.legal_actions),
                **evidence,
                "depths": {
                    str(depth): {
                        "P": source.raw_policy[:source.legal_count].tolist(),
                        "Q": source.q_values[:source.legal_count].tolist(),
                        "N": source.visit_counts[:source.legal_count].tolist(),
                    }
                    for depth, source in ((2, d2[key]), (4, d4[key]), (6, d6.get(key)))
                    if source is not None
                },
            }
            evidence_file.write(json.dumps(row, sort_keys=True) + "\n")
            if example is None:
                rejected += 1
                if evidence.get("reason") == "depth_disagreement" and key not in d6:
                    conflict = dict(position_by_key[key])
                    conflict["d2_d4_rank_correlation"] = float(evidence.get("rank_correlation", 0.0))
                    conflict["d2_top_action"] = int(evidence["top_by_depth"]["2"])
                    conflict["d4_top_action"] = int(evidence["top_by_depth"]["4"])
                    conflict_positions.append(conflict)
                continue
            writer.add(example)
            accepted += 1
            correlations.append(float(evidence["rank_correlation"]))
    conflict_positions.sort(
        key=lambda row: (
            float(row.get("d2_d4_rank_correlation", 0.0)),
            -int(row.get("legal_count", 0)),
            -abs(float(row.get("d4_q_gap", 0.0))),
        )
    )
    conflict_positions = conflict_positions[: max(0, int(args.max_d6_conflicts))]
    conflict_manifest = args.output_dir / "d6_conflicts.json"
    conflict_manifest.write_text(json.dumps({"positions": conflict_positions}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "accepted": accepted,
        "rejected": rejected,
        "d6_conflicts": len(conflict_positions),
        "mean_d2_d4_rank_correlation": float(np.mean(correlations)) if correlations else None,
        "evidence": str(evidence_path.resolve()),
        "conflict_manifest": str(conflict_manifest.resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
