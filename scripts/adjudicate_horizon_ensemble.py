#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_search_variance import (
        aggregate_replicates,
        jackknife_ensemble_stability,
        meaningful_comparison,
        replicate_stability,
    )
    from scripts.classify_search_convergence import root_map
except ModuleNotFoundError:
    from analyze_search_variance import (
        aggregate_replicates,
        jackknife_ensemble_stability,
        meaningful_comparison,
        replicate_stability,
    )
    from classify_search_convergence import root_map

from struggle_ai.search_replay import SearchReplayExample, SearchReplayWriter


def adjudicate_horizon_root(
    d6_ensemble: SearchReplayExample | None,
    deep_replicates: list[SearchReplayExample],
    *,
    min_replicates: int,
    min_rank_correlation: float,
    meaningful_q_gap: float,
    q_temperature: float,
    min_action_coverage: float,
) -> tuple[str, SearchReplayExample | None, SearchReplayExample | None, dict[str, Any]]:
    if d6_ensemble is None or len(deep_replicates) < int(min_replicates):
        return "INSUFFICIENT", None, None, {
            "reason": "missing_d6_or_deep_replicates",
            "deep_replicates": len(deep_replicates),
        }
    stability = replicate_stability(
        deep_replicates,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )
    jackknife = jackknife_ensemble_stability(
        deep_replicates,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
        q_temperature=q_temperature,
        min_action_coverage=min_action_coverage,
    )
    deep_ensemble, ensemble_info = aggregate_replicates(
        deep_replicates,
        q_temperature=q_temperature,
        min_action_coverage=min_action_coverage,
    )
    evidence: dict[str, Any] = {
        "deep_stability": stability,
        "deep_jackknife_stability": jackknife,
        "deep_ensemble": ensemble_info,
    }
    if deep_ensemble is None:
        return "INSUFFICIENT", None, None, evidence
    comparison = meaningful_comparison(
        d6_ensemble,
        deep_ensemble,
        min_rank_correlation=min_rank_correlation,
        meaningful_q_gap=meaningful_q_gap,
    )
    evidence["d6_to_deep"] = comparison
    if float(ensemble_info["q_gap"]) < float(meaningful_q_gap):
        return "DEEP_NEAR_TIE", None, deep_ensemble, evidence
    effective_stable = bool(stability["stable"] or jackknife["stable"])
    evidence["deep_effective_stable"] = effective_stable
    if not effective_stable:
        return "DEEP_PARTICLE_VARIANCE", None, deep_ensemble, evidence
    if not comparison.get("stable"):
        return "UNRESOLVED", None, deep_ensemble, evidence
    metadata = replace(
        deep_ensemble.metadata,
        convergence_status="HORIZON_CONVERGENCE",
        convergence_configs=("d4_512_ensemble", "d6_512_ensemble", "d6_2048_ensemble"),
        convergence_rank_correlation=float(comparison.get("rank_correlation", 0.0)),
        convergence_q_gap=float(ensemble_info["q_gap"]),
        behavior_policy="ensemble_q:horizon_converged_d6_2048",
    )
    return "HORIZON_CONVERGENCE", replace(deep_ensemble, metadata=metadata), deep_ensemble, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirm D4->D6 horizon changes at replicated D6/2048.")
    parser.add_argument("--position-manifest", type=Path, required=True)
    parser.add_argument("--d6-ensemble-replay", type=Path, required=True)
    parser.add_argument("--deep-replay", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-replicates", type=int, default=3)
    parser.add_argument("--min-rank-correlation", type=float, default=0.70)
    parser.add_argument("--meaningful-q-gap", type=float, default=0.01)
    parser.add_argument("--q-temperature", type=float, default=0.05)
    parser.add_argument("--min-action-coverage", type=float, default=2.0 / 3.0)
    args = parser.parse_args()

    d6 = root_map(args.d6_ensemble_replay)
    deep_maps = [root_map(path) for path in args.deep_replay]
    positions = json.loads(args.position_manifest.read_text(encoding="utf-8"))["positions"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statuses: Counter[str] = Counter()
    side_statuses: Counter[str] = Counter()
    positions_by_status: dict[str, list[dict[str, Any]]] = {}
    evidence_path = args.output_dir / "horizon_evidence.jsonl"
    with (
        SearchReplayWriter(args.output_dir / "teacher_replay", shard_size=128) as teacher_writer,
        SearchReplayWriter(args.output_dir / "d6_2048_ensemble_replay", shard_size=128) as ensemble_writer,
        evidence_path.open("w", encoding="utf-8") as handle,
    ):
        for position in positions:
            key = int(position["seed"]), int(position["action_index"])
            deep_examples = [mapping[key] for mapping in deep_maps if key in mapping]
            status, teacher, deep_ensemble, evidence = adjudicate_horizon_root(
                d6.get(key),
                deep_examples,
                min_replicates=int(args.min_replicates),
                min_rank_correlation=float(args.min_rank_correlation),
                meaningful_q_gap=float(args.meaningful_q_gap),
                q_temperature=float(args.q_temperature),
                min_action_coverage=float(args.min_action_coverage),
            )
            statuses[status] += 1
            side_statuses[f"{status}:{position['side']}"] += 1
            positions_by_status.setdefault(status, []).append(dict(position))
            handle.write(json.dumps({**position, "status": status, **evidence}, sort_keys=True) + "\n")
            if deep_ensemble is not None:
                ensemble_writer.add(deep_ensemble)
            if teacher is not None:
                teacher_writer.add(teacher)
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
        "status_counts": dict(statuses),
        "status_by_side": dict(side_statuses),
        "teacher_examples": int(statuses["HORIZON_CONVERGENCE"]),
        "deep_replays": [str(path.resolve()) for path in args.deep_replay],
        "evidence": str(evidence_path.resolve()),
        "status_manifests": status_manifests,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
