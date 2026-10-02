#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from struggle_ai.search_replay import SearchReplayExample, load_search_replay
from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import SHARED_POLICY_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate how strongly a frozen checkpoint has internalized PUCT replay targets."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--search-replay", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-q-advantage", type=float, default=0.003)
    parser.add_argument("--min-teacher-visits", type=float, default=4.0)
    parser.add_argument("--min-teacher-visit-share", type=float, default=0.5)
    parser.add_argument("--max-atomic-horizon-cutoffs", type=int, default=0)
    parser.add_argument("--max-node-budget-cutoffs", type=int, default=0)
    parser.add_argument("--require-flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--q-weight-scale", type=float, default=0.05)
    parser.add_argument("--max-q-weight", type=float, default=4.0)
    parser.add_argument("--hierarchy-weight", type=float, default=1.25)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--value-state-tower",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--card-history-layers", type=int, default=2)
    parser.add_argument("--history-attention-heads", type=int, default=4)
    parser.add_argument("--history-dropout", type=float, default=0.05)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-neighbor-hops", type=int, default=2)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    return parser.parse_args()


def algorithm_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        restore_from=args.checkpoint.expanduser().resolve(),
        num_gpus=args.num_gpus,
        hidden=args.hidden,
        value_state_tower=args.value_state_tower,
        history_layers=args.history_layers,
        card_history_layers=args.card_history_layers,
        history_attention_heads=args.history_attention_heads,
        history_dropout=args.history_dropout,
        graph_layers=args.graph_layers,
        graph_neighbor_hops=args.graph_neighbor_hops,
        policy_temperature=args.policy_temperature,
        ray_num_cpus=1,
        ray_object_store_memory=150_000_000,
    )


def sample_weight(example: SearchReplayExample, args: argparse.Namespace) -> float:
    q_weight = 1.0 + min(
        max(0.0, float(args.max_q_weight)),
        max(0.0, float(example.metadata.q_advantage_over_raw))
        / max(1e-6, float(args.q_weight_scale)),
    )
    tactical_levels = {"influence_allocation", "coup_target", "realignment_target"}
    path_levels = {str(item.get("level") or "") for item in example.metadata.hierarchical_path}
    hierarchy = (
        float(args.hierarchy_weight)
        if example.metadata.hierarchy_level in tactical_levels or path_levels & tactical_levels
        else 1.0
    )
    return (2.0 if example.metadata.action_flipped else 1.0) * q_weight * hierarchy


def evaluate(policy: Any, examples: list[SearchReplayExample], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    model = policy.model
    device = next(model.parameters()).device
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(examples), max(1, int(args.inference_batch_size))):
            batch = examples[start : start + max(1, int(args.inference_batch_size))]
            obs = {
                key: torch.as_tensor(
                    np.stack([example.observation[key] for example in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                for key in batch[0].observation
            }
            logits, _state = model({"obs": obs}, [], None)
            for index, example in enumerate(batch):
                probabilities = torch.softmax(
                    logits[index, : example.legal_count], dim=-1
                ).cpu().numpy()
                teacher = int(example.metadata.search_action)
                predicted = int(np.argmax(probabilities))
                turn = int(round(float(example.observation["global"][2]) * 10.0))
                records.append({
                    "source_game": example.metadata.source_game,
                    "source_step": example.metadata.source_step,
                    "side": example.metadata.side,
                    "phase": "early" if turn <= 3 else "mid" if turn <= 7 else "late",
                    "triggers": list(example.metadata.trigger_categories),
                    "hierarchical_path": list(example.metadata.hierarchical_path),
                    "hierarchy_index": example.metadata.hierarchy_index,
                    "hierarchy_level": example.metadata.hierarchy_level,
                    "q_advantage_over_raw": example.metadata.q_advantage_over_raw,
                    "teacher_action_visits": example.metadata.teacher_action_visits,
                    "teacher_visit_share": example.metadata.teacher_visit_share,
                    "raw_action": example.metadata.raw_action,
                    "teacher_action": teacher,
                    "predicted_action": predicted,
                    "teacher_top1": predicted == teacher,
                    "teacher_probability": float(probabilities[teacher]),
                    "policy_entropy": float(
                        -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum()
                    ),
                    "raw_action_probability": float(probabilities[example.metadata.raw_action]),
                    "teacher_action_text": example.metadata.legal_actions[teacher],
                    "predicted_action_text": example.metadata.legal_actions[predicted],
                    "sample_weight": sample_weight(example, args),
                })
    return records


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"examples": 0}
    weights = np.asarray([record["sample_weight"] for record in records], dtype=np.float64)
    probabilities = np.asarray([record["teacher_probability"] for record in records], dtype=np.float64)
    accuracies = np.asarray([record["teacher_top1"] for record in records], dtype=np.float64)
    q_advantages = np.asarray([record["q_advantage_over_raw"] for record in records], dtype=np.float64)
    entropies = np.asarray([record["policy_entropy"] for record in records], dtype=np.float64)
    return {
        "examples": len(records),
        "teacher_top1_accuracy": float(accuracies.mean()),
        "mean_teacher_probability": float(probabilities.mean()),
        "q_weighted_teacher_top1_accuracy": float(np.average(accuracies, weights=weights)),
        "q_weighted_teacher_probability": float(np.average(probabilities, weights=weights)),
        "mean_q_advantage": float(q_advantages.mean()),
        "mean_policy_entropy": float(entropies.mean()),
    }


def grouped(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        items = record[key] if key == "triggers" else [record[key]]
        for item in items:
            values[str(item)].append(record)
    return {name: aggregate(group) for name, group in sorted(values.items())}


def main() -> None:
    args = parse_args()
    replay_paths = sorted({path.expanduser().resolve() for path in args.search_replay})
    if not replay_paths:
        raise SystemExit("provide at least one --search-replay path")
    def qualifies(metadata) -> bool:
        return (
            float(metadata.q_advantage_over_raw) >= args.min_q_advantage
            and float(metadata.teacher_action_visits) >= args.min_teacher_visits
            and float(metadata.teacher_visit_share) >= args.min_teacher_visit_share
            and int(metadata.atomic_horizon_cutoffs) <= args.max_atomic_horizon_cutoffs
            and int(metadata.node_budget_cutoffs) <= args.max_node_budget_cutoffs
            and (not args.require_flip or metadata.action_flipped)
        )

    discovered_examples = 0
    for path in replay_paths:
        manifest = path / "manifest.json" if path.is_dir() else path
        discovered_examples += int(json.loads(manifest.read_text(encoding="utf-8")).get("examples", 0))
    examples = load_search_replay(replay_paths, metadata_predicate=qualifies)
    if not examples:
        raise SystemExit("search replay contains no examples after filtering")
    algo = build_algorithm(algorithm_args(args))
    try:
        records = evaluate(algo.get_policy(SHARED_POLICY_ID), examples, args)
    finally:
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass
    payload = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "search_replay": [str(path) for path in replay_paths],
        "discovered_examples": discovered_examples,
        "qualified_examples": len(examples),
        "filters": {
            "min_q_advantage": args.min_q_advantage,
            "min_teacher_visits": args.min_teacher_visits,
            "min_teacher_visit_share": args.min_teacher_visit_share,
            "max_atomic_horizon_cutoffs": args.max_atomic_horizon_cutoffs,
            "max_node_budget_cutoffs": args.max_node_budget_cutoffs,
            "require_flip": args.require_flip,
        },
        "overall": aggregate(records),
        "by_side": grouped(records, "side"),
        "by_phase": grouped(records, "phase"),
        "by_hierarchy_level": grouped(records, "hierarchy_level"),
        "by_trigger": grouped(records, "triggers"),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("checkpoint", "qualified_examples", "overall")}, indent=2))


if __name__ == "__main__":
    main()
