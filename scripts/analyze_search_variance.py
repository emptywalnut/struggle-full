#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.classify_search_convergence import (
        action_signature,
        complete,
        comparison,
        root_map,
    )
except ModuleNotFoundError:
    from classify_search_convergence import (
        action_signature,
        complete,
        comparison,
        root_map,
    )
from struggle_ai.search_convergence import visited_q_top
from struggle_ai.search_replay import SearchReplayExample, SearchReplayWriter


def parse_replay(value: str) -> tuple[str, str, Path]:
    try:
        label, raw_path = value.split("=", 1)
        depth, replicate = label.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("replay must be DEPTH:REPLICATE=MANIFEST") from exc
    if depth not in {"d4", "d6"}:
        raise argparse.ArgumentTypeError(f"depth must be d4 or d6, got {depth!r}")
    return depth, replicate, Path(raw_path)


def aggregate_replicates(
    examples: list[SearchReplayExample],
    *,
    q_temperature: float,
    min_action_coverage: float,
) -> tuple[SearchReplayExample | None, dict[str, Any]]:
    if not examples:
        return None, {"reason": "missing_replicates"}
    if any(not complete(example) for example in examples):
        return None, {"reason": "cutoff_replicate"}
    signature = action_signature(examples[0])
    if any(action_signature(example) != signature for example in examples[1:]):
        return None, {"reason": "action_signature_mismatch"}
    legal = examples[0].legal_count
    if any(example.legal_count != legal for example in examples[1:]):
        return None, {"reason": "legal_count_mismatch"}

    visits_by_rep = np.stack(
        [np.asarray(example.visit_counts[:legal], dtype=np.float64) for example in examples]
    )
    q_by_rep = np.stack(
        [np.asarray(example.q_values[:legal], dtype=np.float64) for example in examples]
    )
    required = max(1, int(math.ceil(len(examples) * float(min_action_coverage))))
    coverage = (visits_by_rep > 0.0).sum(axis=0)
    valid = coverage >= required
    if int(valid.sum()) < 2:
        return None, {
            "reason": "insufficient_ensemble_action_coverage",
            "required_replicates": required,
            "covered_actions": int(valid.sum()),
        }

    q = np.zeros(legal, dtype=np.float64)
    for action in np.flatnonzero(valid):
        present = visits_by_rep[:, action] > 0.0
        q[action] = float(np.mean(q_by_rep[present, action]))
    logits = q[valid] / max(float(q_temperature), 1e-6)
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    policy_legal = np.zeros(legal, dtype=np.float32)
    policy_legal[valid] = (weights / weights.sum()).astype(np.float32)
    top, q_gap = visited_q_top(valid.astype(np.float32), q)

    source = examples[0]
    output_size = len(source.search_policy)
    policy = np.zeros(output_size, dtype=np.float32)
    policy[:legal] = policy_legal
    visits = np.zeros(output_size, dtype=np.float32)
    total_visits = float(sum(np.asarray(example.visit_counts).sum() for example in examples))
    visits[:legal] = policy_legal * max(total_visits, 1.0)
    q_padded = np.zeros(output_size, dtype=np.float32)
    q_padded[:legal] = q.astype(np.float32)
    raw_action = int(source.metadata.raw_action)
    metadata = replace(
        source.metadata,
        search_action=int(top),
        simulation_budget=sum(int(example.metadata.simulation_budget) for example in examples),
        root_q_gap=float(q_gap),
        action_flipped=int(top) != raw_action,
        q_advantage_over_raw=(
            float(q[top] - q[raw_action]) if raw_action < legal and valid[raw_action] else 0.0
        ),
        teacher_action_visits=float(visits[top]),
        total_visits=float(visits.sum()),
        teacher_visit_share=float(policy[top]),
        behavior_policy=f"ensemble_q:{source.metadata.search_depth}",
        atomic_horizon_cutoffs=0,
        node_budget_cutoffs=0,
    )
    return SearchReplayExample(
        observation=source.observation,
        raw_policy=np.mean(np.stack([example.raw_policy for example in examples]), axis=0).astype(np.float32),
        visit_counts=visits,
        search_policy=policy,
        q_values=q_padded,
        metadata=metadata,
    ), {
        "reason": None,
        "valid_actions": int(valid.sum()),
        "required_replicates": required,
        "q_top": int(top),
        "q_gap": float(q_gap),
        "coverage": coverage.tolist(),
    }


def meaningful_comparison(
    left: SearchReplayExample,
    right: SearchReplayExample,
    *,
    min_rank_correlation: float,
    meaningful_q_gap: float,
    min_pair_agreement: float = 0.80,
) -> dict[str, Any]:
    result = comparison(
        left,
        right,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )
    if not result.get("comparable"):
        return result
    legal = left.legal_count
    left_visits = np.asarray(left.visit_counts[:legal], dtype=np.float64)
    right_visits = np.asarray(right.visit_counts[:legal], dtype=np.float64)
    left_q = np.asarray(left.q_values[:legal], dtype=np.float64)
    right_q = np.asarray(right.q_values[:legal], dtype=np.float64)
    indices = np.flatnonzero((left_visits > 0.0) & (right_visits > 0.0))
    agreements = 0
    compared = 0
    for offset, left_action in enumerate(indices):
        for right_action in indices[offset + 1 :]:
            left_delta = float(left_q[left_action] - left_q[right_action])
            right_delta = float(right_q[left_action] - right_q[right_action])
            if min(abs(left_delta), abs(right_delta)) < float(meaningful_q_gap):
                continue
            compared += 1
            agreements += int(np.sign(left_delta) == np.sign(right_delta))
    pair_agreement = agreements / compared if compared else None
    top_consistent = bool(
        result["left_top"] == result["right_top"]
        or float(result["cross_q_gap_at_right"]) < float(meaningful_q_gap)
    )
    result.update(
        {
            "spearman_stable": bool(result.get("stable")),
            "meaningful_pairs": compared,
            "meaningful_pair_agreement": pair_agreement,
            "top_consistent": top_consistent,
            "stable": bool(
                top_consistent
                and compared > 0
                and pair_agreement is not None
                and pair_agreement >= float(min_pair_agreement)
            ),
        }
    )
    return result


def replicate_stability(
    examples: list[SearchReplayExample],
    *,
    min_rank_correlation: float,
    meaningful_q_gap: float,
) -> dict[str, Any]:
    pairs = []
    for left_index, right_index in itertools.combinations(range(len(examples)), 2):
        result = meaningful_comparison(
            examples[left_index],
            examples[right_index],
            min_rank_correlation=min_rank_correlation,
            meaningful_q_gap=meaningful_q_gap,
        )
        pairs.append({"left": left_index, "right": right_index, **result})
    comparable = [item for item in pairs if item.get("comparable")]
    correlations = [float(item["rank_correlation"]) for item in comparable]
    pair_agreements = [
        float(item["meaningful_pair_agreement"])
        for item in comparable
        if item.get("meaningful_pair_agreement") is not None
    ]
    tops = []
    for example in examples:
        top, _gap = visited_q_top(
            example.visit_counts[: example.legal_count], example.q_values[: example.legal_count]
        )
        tops.append(int(top))
    top_counts = Counter(tops)
    top_share = max(top_counts.values(), default=0) / max(len(tops), 1)
    stable_fraction = sum(bool(item.get("stable")) for item in comparable) / max(len(comparable), 1)
    mean_correlation = float(np.mean(correlations)) if correlations else None
    mean_pair_agreement = float(np.mean(pair_agreements)) if pair_agreements else None
    stable = bool(
        len(comparable) == len(pairs)
        and stable_fraction >= (2.0 / 3.0)
        and mean_pair_agreement is not None
        and mean_pair_agreement >= 0.80
        and top_share >= (2.0 / 3.0)
    )
    return {
        "stable": stable,
        "replicates": len(examples),
        "pair_count": len(pairs),
        "stable_pair_fraction": stable_fraction,
        "mean_rank_correlation": mean_correlation,
        "mean_meaningful_pair_agreement": mean_pair_agreement,
        "top_actions": tops,
        "top_action_share": top_share,
        "pairs": pairs,
    }


def jackknife_ensemble_stability(
    examples: list[SearchReplayExample],
    *,
    min_rank_correlation: float,
    meaningful_q_gap: float,
    q_temperature: float,
    min_action_coverage: float,
) -> dict[str, Any]:
    if len(examples) < 3:
        return {"stable": False, "reason": "requires_three_replicates"}
    leave_one_out = []
    omitted = []
    for omitted_index in range(len(examples)):
        ensemble, info = aggregate_replicates(
            [example for index, example in enumerate(examples) if index != omitted_index],
            q_temperature=q_temperature,
            min_action_coverage=min_action_coverage,
        )
        if ensemble is None:
            return {
                "stable": False,
                "reason": "jackknife_ensemble_failed",
                "omitted_index": omitted_index,
                "ensemble": info,
            }
        leave_one_out.append(ensemble)
        omitted.append({"omitted_index": omitted_index, **info})
    stability = replicate_stability(
        leave_one_out,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )
    return {**stability, "leave_one_out": omitted}


def classify_ensemble_root(
    d4_examples: list[SearchReplayExample],
    d6_examples: list[SearchReplayExample],
    *,
    min_replicates: int,
    min_rank_correlation: float,
    meaningful_q_gap: float,
    q_temperature: float,
    min_action_coverage: float,
) -> tuple[str, SearchReplayExample | None, dict[str, Any]]:
    if len(d4_examples) < min_replicates or len(d6_examples) < min_replicates:
        return "INSUFFICIENT", None, {
            "reason": "insufficient_replicates",
            "d4_replicates": len(d4_examples),
            "d6_replicates": len(d6_examples),
        }
    d4_stability = replicate_stability(
        d4_examples,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )
    d6_stability = replicate_stability(
        d6_examples,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )
    d4_jackknife = jackknife_ensemble_stability(
        d4_examples,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
        q_temperature=q_temperature,
        min_action_coverage=min_action_coverage,
    )
    d6_jackknife = jackknife_ensemble_stability(
        d6_examples,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
        q_temperature=q_temperature,
        min_action_coverage=min_action_coverage,
    )
    d4_ensemble, d4_info = aggregate_replicates(
        d4_examples,
        q_temperature=q_temperature,
        min_action_coverage=min_action_coverage,
    )
    d6_ensemble, d6_info = aggregate_replicates(
        d6_examples,
        q_temperature=q_temperature,
        min_action_coverage=min_action_coverage,
    )
    evidence: dict[str, Any] = {
        "d4_stability": d4_stability,
        "d6_stability": d6_stability,
        "d4_jackknife_stability": d4_jackknife,
        "d6_jackknife_stability": d6_jackknife,
        "d4_ensemble": d4_info,
        "d6_ensemble": d6_info,
    }
    if d4_ensemble is None or d6_ensemble is None:
        return "INSUFFICIENT", None, evidence
    cross = meaningful_comparison(
        d4_ensemble,
        d6_ensemble,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )
    evidence["cross_depth"] = cross
    if float(d6_info["q_gap"]) < meaningful_q_gap:
        return "NEAR_TIE", None, evidence
    d4_effective_stable = bool(d4_stability["stable"] or d4_jackknife["stable"])
    d6_effective_stable = bool(d6_stability["stable"] or d6_jackknife["stable"])
    evidence["d4_effective_stable"] = d4_effective_stable
    evidence["d6_effective_stable"] = d6_effective_stable
    if not d4_effective_stable or not d6_effective_stable:
        return "PARTICLE_VARIANCE", None, evidence
    if bool(cross.get("stable")):
        metadata = replace(
            d6_ensemble.metadata,
            convergence_status="ENSEMBLE_CONVERGENCE",
            convergence_configs=("d4_512_ensemble", "d6_512_ensemble"),
            convergence_rank_correlation=float(cross.get("rank_correlation", 0.0)),
            convergence_q_gap=float(d6_info["q_gap"]),
            behavior_policy="ensemble_q:converged_d6",
        )
        return "ENSEMBLE_CONVERGENCE", replace(d6_ensemble, metadata=metadata), evidence
    return "HORIZON_DISAGREEMENT", None, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate PUCT particle variance from horizon disagreement.")
    parser.add_argument("--position-manifest", type=Path, required=True)
    parser.add_argument("--replay", action="append", type=parse_replay, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-replicates", type=int, default=3)
    parser.add_argument("--min-rank-correlation", type=float, default=0.70)
    parser.add_argument("--meaningful-q-gap", type=float, default=0.01)
    parser.add_argument("--q-temperature", type=float, default=0.05)
    parser.add_argument("--min-action-coverage", type=float, default=2.0 / 3.0)
    args = parser.parse_args()

    replay_maps: dict[str, list[dict[tuple[int, int], SearchReplayExample]]] = {"d4": [], "d6": []}
    replay_labels: dict[str, list[str]] = {"d4": [], "d6": []}
    for depth, replicate, path in args.replay:
        replay_maps[depth].append(root_map(path))
        replay_labels[depth].append(replicate)
    positions = json.loads(args.position_manifest.read_text(encoding="utf-8"))["positions"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_counts: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    positions_by_status: dict[str, list[dict[str, Any]]] = {}
    evidence_path = args.output_dir / "variance_evidence.jsonl"
    with (
        SearchReplayWriter(args.output_dir / "teacher_replay", shard_size=128) as writer,
        SearchReplayWriter(args.output_dir / "d4_ensemble_replay", shard_size=128) as d4_writer,
        SearchReplayWriter(args.output_dir / "d6_ensemble_replay", shard_size=128) as d6_writer,
        evidence_path.open("w", encoding="utf-8") as handle,
    ):
        for position in positions:
            key = int(position["seed"]), int(position["action_index"])
            d4_examples = [mapping[key] for mapping in replay_maps["d4"] if key in mapping]
            d6_examples = [mapping[key] for mapping in replay_maps["d6"] if key in mapping]
            d4_ensemble, _d4_info = aggregate_replicates(
                d4_examples,
                q_temperature=float(args.q_temperature),
                min_action_coverage=float(args.min_action_coverage),
            )
            d6_ensemble, _d6_info = aggregate_replicates(
                d6_examples,
                q_temperature=float(args.q_temperature),
                min_action_coverage=float(args.min_action_coverage),
            )
            if d4_ensemble is not None:
                d4_writer.add(d4_ensemble)
            if d6_ensemble is not None:
                d6_writer.add(d6_ensemble)
            status, teacher, evidence = classify_ensemble_root(
                d4_examples,
                d6_examples,
                min_replicates=int(args.min_replicates),
                min_rank_correlation=float(args.min_rank_correlation),
                meaningful_q_gap=float(args.meaningful_q_gap),
                q_temperature=float(args.q_temperature),
                min_action_coverage=float(args.min_action_coverage),
            )
            status_counts[status] += 1
            side_counts[f"{status}:{position['side']}"] += 1
            positions_by_status.setdefault(status, []).append(dict(position))
            handle.write(json.dumps({
                **position,
                "status": status,
                "replicate_labels": replay_labels,
                **evidence,
            }, sort_keys=True) + "\n")
            if teacher is not None:
                writer.add(teacher)
    status_manifests = {}
    for status, selected in positions_by_status.items():
        path = args.output_dir / f"{status.lower()}_positions.json"
        path.write_text(
            json.dumps({"positions": selected}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        status_manifests[status] = str(path.resolve())
    summary = {
        "positions": len(positions),
        "status_counts": dict(status_counts),
        "status_by_side": dict(side_counts),
        "teacher_examples": int(status_counts["ENSEMBLE_CONVERGENCE"]),
        "replicate_labels": replay_labels,
        "min_replicates": int(args.min_replicates),
        "min_rank_correlation": float(args.min_rank_correlation),
        "meaningful_q_gap": float(args.meaningful_q_gap),
        "min_action_coverage": float(args.min_action_coverage),
        "evidence": str(evidence_path.resolve()),
        "status_manifests": status_manifests,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
