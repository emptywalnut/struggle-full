#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from struggle_ai.search_distill import (
    GameOutcomeBalancedValueSampler,
    SearchReplaySampler,
    build_search_optimizer,
    distill_search_batch,
    limit_examples_per_source_root,
    restore_search_optimizer_state,
    save_search_optimizer_state,
)
from struggle_ai.batched_inference import load_checkpoint_model
from struggle_ai.search_replay import SearchReplayExample, load_search_replay
from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import SHARED_POLICY_ID, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy/value distillation from exact-PUCT replay shards.")
    parser.add_argument("--restore-from", type=Path, required=True)
    parser.add_argument("--search-replay", action="append", default=[])
    parser.add_argument(
        "--search-replay-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Consume search replay manifests from either iterative "
            "shard-*/manifest.json or tournament shard-*/search_replay/manifest.json layouts."
        ),
    )
    parser.add_argument(
        "--anchor-replay",
        action="append",
        default=[],
        help="Replay manifest used only for parent-policy anchoring, never as a search teacher.",
    )
    parser.add_argument(
        "--anchor-replay-root",
        action="append",
        type=Path,
        default=[],
        help="Discover anchor-only replay manifests below this directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=0,
        help="Save auditable intermediate policy/value checkpoints at this update interval.",
    )
    parser.add_argument(
        "--restore-optimizer-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore search_distill_optimizers.pt from --restore-from when compatible.",
    )
    parser.add_argument("--report-examples", type=int, default=32)
    parser.add_argument(
        "--replay-load-workers",
        type=int,
        default=16,
        help="Parallel compressed-shard decoders used before training starts.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--min-value-examples-per-batch",
        type=int,
        default=8,
        help="Guarantee terminal-value roots in joint policy/value minibatches, alternating sides where possible.",
    )
    parser.add_argument(
        "--separate-value-batch-size",
        type=int,
        default=0,
        help=(
            "Sample this many complete-game root targets independently of policy-label filters. "
            "This permits flip-only actor supervision without starving critic training."
        ),
    )
    parser.add_argument("--value-us-win-weight", type=float, default=1.0)
    parser.add_argument("--value-us-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-ussr-win-weight", type=float, default=1.0)
    parser.add_argument("--value-ussr-loss-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda-search", type=float, default=1.0)
    parser.add_argument("--lambda-pairwise", type=float, default=0.0)
    parser.add_argument("--pairwise-min-q-gap", type=float, default=0.002)
    parser.add_argument("--pairwise-q-scale", type=float, default=0.02)
    parser.add_argument("--pairwise-max-pairs", type=int, default=8)
    parser.add_argument(
        "--search-policy-mix",
        type=float,
        default=1.0,
        help="Convex weight on PUCT visits versus the local parent raw policy target.",
    )
    parser.add_argument("--lambda-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-value", type=float, default=0.0)
    parser.add_argument(
        "--value-loss-calibrated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train terminal targets against tanh(raw critic), matching PUCT leaf semantics.",
    )
    parser.add_argument("--lambda-search-q", type=float, default=0.0)
    parser.add_argument(
        "--lambda-value-anchor",
        type=float,
        default=0.0,
        help="Keep critic outputs close to the frozen source checkpoint on replay states.",
    )
    parser.add_argument("--confidence-weight-floor", type=float, default=1.0)
    parser.add_argument(
        "--anchor-from-all-safe-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--anchor-replay-batch-size", type=int, default=64)
    parser.add_argument(
        "--anchor-sample-fraction",
        type=float,
        default=1.0,
        help="Stable metadata-hash fraction used to bound broad anchor replay memory.",
    )
    parser.add_argument("--balance-sides", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--us-policy-weight", type=float, default=1.0)
    parser.add_argument("--ussr-policy-weight", type=float, default=1.0)
    parser.add_argument(
        "--parameter-scope",
        choices=(
            "actor",
            "head",
            "action_head",
            "actor_value",
            "actor_value_heads",
            "policy_value_heads",
            "value_adapter",
            "value_head",
            "value_tower",
            "value_tower_head",
        ),
        default="actor",
    )
    parser.add_argument(
        "--require-complete-game",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--root-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train only hierarchy_index=0 replay roots.",
    )
    parser.add_argument("--min-q-advantage", type=float, default=0.0)
    parser.add_argument("--min-teacher-visits", type=float, default=0.0)
    parser.add_argument("--min-teacher-visit-share", type=float, default=0.0)
    parser.add_argument("--max-atomic-horizon-cutoffs", type=int, default=0)
    parser.add_argument("--max-node-budget-cutoffs", type=int, default=0)
    parser.add_argument(
        "--max-examples-per-source-root",
        type=int,
        default=None,
        help="Optional post-quality-filter cap that prevents one searched root from dominating training.",
    )
    parser.add_argument(
        "--require-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--q-weight-scale", type=float, default=0.05)
    parser.add_argument("--max-q-weight", type=float, default=4.0)
    parser.add_argument("--hierarchy-weight", type=float, default=1.25)
    parser.add_argument("--diversity-max-per-game", type=int, default=0)
    parser.add_argument("--diversity-max-per-turn", type=int, default=0)
    parser.add_argument("--diversity-max-per-card", type=int, default=0)
    parser.add_argument("--diversity-max-per-region", type=int, default=0)
    parser.add_argument("--diversity-max-per-trigger", type=int, default=0)
    parser.add_argument(
        "--min-search-examples-per-era",
        type=int,
        default=0,
        help="Minimum early/mid/late search examples in every policy minibatch.",
    )
    parser.add_argument(
        "--min-search-examples-per-side-era",
        type=int,
        default=0,
        help="Minimum examples for each US/USSR by early/mid/late cell per minibatch.",
    )
    parser.add_argument("--seed", type=int, default=8_110_101)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--value-adapter-hidden",
        type=int,
        default=0,
        help=(
            "Enable a zero-initialized critic-only residual adapter of this width. "
            "The default 0 preserves the checkpoint architecture."
        ),
    )
    parser.add_argument(
        "--value-state-tower",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a critic-only clone of the shared state encoder.",
    )
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--card-history-layers", type=int, default=2)
    parser.add_argument("--history-attention-heads", type=int, default=4)
    parser.add_argument("--history-dropout", type=float, default=0.05)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-neighbor-hops", type=int, default=2)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    return parser.parse_args()


def discover_search_replay_manifests(
    explicit_paths: list[str | Path],
    roots: list[Path],
) -> list[Path]:
    manifests = [Path(path).expanduser().resolve() for path in explicit_paths]
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.name == "manifest.json" and resolved.is_file():
            manifests.append(resolved)
            continue
        direct_manifest = resolved / "manifest.json"
        if direct_manifest.is_file():
            manifests.append(direct_manifest)
        manifests.extend(resolved.glob("shard-*/manifest.json"))
        manifests.extend(resolved.glob("shard-*/search_replay/manifest.json"))
    return sorted(set(manifests))


def algorithm_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        restore_from=args.restore_from.expanduser().resolve(),
        num_gpus=args.num_gpus,
        hidden=args.hidden,
        value_adapter_hidden=args.value_adapter_hidden,
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


def predictions(policy: Any, examples: list[SearchReplayExample]) -> list[dict[str, Any]]:
    import torch

    model = policy.model
    device = next(model.parameters()).device
    model.eval()
    result: list[dict[str, Any]] = []
    with torch.no_grad():
        for example in examples:
            obs = {
                key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
                for key, value in example.observation.items()
            }
            logits, _state = model({"obs": obs}, [], None)
            legal_count = example.legal_count
            probabilities = torch.softmax(logits[0, :legal_count], dim=-1).cpu().numpy()
            result.append(
                {
                    "source_game": example.metadata.source_game,
                    "source_step": example.metadata.source_step,
                    "side": example.metadata.side,
                    "hierarchy_index": example.metadata.hierarchy_index,
                    "hierarchy_level": example.metadata.hierarchy_level,
                    "q_advantage_over_raw": example.metadata.q_advantage_over_raw,
                    "teacher_action_visits": example.metadata.teacher_action_visits,
                    "teacher_visit_share": example.metadata.teacher_visit_share,
                    "raw_action": example.metadata.raw_action,
                    "teacher_action": example.metadata.search_action,
                    "predicted_action": int(np.argmax(probabilities)),
                    "probabilities": probabilities.tolist(),
                    "legal_actions": list(example.metadata.legal_actions),
                }
            )
    return result


def parameter_hashes(model: Any, names: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if name not in names:
            continue
        value = np.ascontiguousarray(parameter.detach().cpu().numpy())
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
        result[name] = digest.hexdigest()
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    replay_paths = discover_search_replay_manifests(
        args.search_replay,
        args.search_replay_root,
    )
    if not replay_paths:
        raise SystemExit("no search replay manifests were provided or discovered")
    discovered_examples = load_search_replay(
        replay_paths,
        workers=args.replay_load_workers,
    )
    anchor_replay_paths = discover_search_replay_manifests(
        args.anchor_replay,
        args.anchor_replay_root,
    )
    anchor_fraction = float(args.anchor_sample_fraction)
    if not 0.0 < anchor_fraction <= 1.0:
        raise SystemExit("--anchor-sample-fraction must be in (0, 1]")

    def retain_anchor(metadata: Any) -> bool:
        payload = "|".join(
            (
                str(metadata.source_game or ""),
                str(metadata.source_step if metadata.source_step is not None else ""),
                str(metadata.side),
                str(metadata.hierarchy_index),
            )
        ).encode("utf-8")
        bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2**64
        return bucket < anchor_fraction

    anchor_discovered_examples = (
        load_search_replay(
            anchor_replay_paths,
            workers=(args.replay_load_workers if anchor_fraction >= 1.0 else 0),
            metadata_predicate=(None if anchor_fraction >= 1.0 else retain_anchor),
        )
        if anchor_replay_paths
        else discovered_examples
    )
    quality_filtered_examples = [
        example
        for example in discovered_examples
        if float(example.metadata.q_advantage_over_raw) >= args.min_q_advantage
        and float(example.metadata.teacher_action_visits) >= args.min_teacher_visits
        and float(example.metadata.teacher_visit_share) >= args.min_teacher_visit_share
        and int(example.metadata.atomic_horizon_cutoffs) <= args.max_atomic_horizon_cutoffs
        and int(example.metadata.node_budget_cutoffs) <= args.max_node_budget_cutoffs
        and (not args.root_only or int(example.metadata.hierarchy_index) == 0)
        and (not args.require_flip or example.metadata.action_flipped)
        and (not args.require_complete_game or example.metadata.complete_game)
    ]
    examples = limit_examples_per_source_root(
        quality_filtered_examples,
        args.max_examples_per_source_root,
    )
    if not examples:
        raise SystemExit("search replay contains no examples after filtering")
    safe_anchor_examples = [
        example
        for example in anchor_discovered_examples
        if int(example.metadata.atomic_horizon_cutoffs) <= args.max_atomic_horizon_cutoffs
        and int(example.metadata.node_budget_cutoffs) <= args.max_node_budget_cutoffs
    ]
    if not safe_anchor_examples:
        raise SystemExit("search replay contains no cutoff-free anchor examples")
    algo = build_algorithm(algorithm_args(args))
    policy = algo.get_policy(SHARED_POLICY_ID)
    value_anchor_model = None
    if args.lambda_value_anchor > 0.0:
        value_anchor_model = load_checkpoint_model(
            args.restore_from,
            str(next(policy.model.parameters()).device),
        )
        value_anchor_model.eval()
        for parameter in value_anchor_model.parameters():
            parameter.requires_grad_(False)
    optimizer = build_search_optimizer(
        policy,
        lr=args.lr,
        parameter_scope=args.parameter_scope,
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_parameter_names = [
        name
        for name, parameter in policy.model.named_parameters()
        if id(parameter) in optimizer_parameter_ids
    ]
    trainable_parameter_count = sum(
        int(parameter.numel())
        for parameter in policy.model.parameters()
        if id(parameter) in optimizer_parameter_ids
    )
    frozen_parameter_names = {
        name
        for name, parameter in policy.model.named_parameters()
        if id(parameter) not in optimizer_parameter_ids
    }
    frozen_parameter_hashes_before = parameter_hashes(
        policy.model, frozen_parameter_names
    )
    optimizer_restored = bool(
        args.restore_optimizer_state
        and restore_search_optimizer_state(
            args.restore_from.expanduser().resolve(),
            {SHARED_POLICY_ID: optimizer},
        )
    )
    report_examples = examples[: max(0, int(args.report_examples))]
    before = predictions(policy, report_examples)
    history: list[dict[str, float]] = []
    intermediate_checkpoints: list[dict[str, Any]] = []
    sampler = SearchReplaySampler(examples, seed=args.seed)
    value_population = [
        example
        for example in safe_anchor_examples
        if example.metadata.root_value_target is not None
        and example.metadata.complete_game
        and int(example.metadata.hierarchy_index) == 0
    ]
    value_sampler = (
        GameOutcomeBalancedValueSampler(
            value_population,
            seed=args.seed + 29,
            stratum_weights={
                ("us", "win"): args.value_us_win_weight,
                ("us", "loss"): args.value_us_loss_weight,
                ("ussr", "win"): args.value_ussr_win_weight,
                ("ussr", "loss"): args.value_ussr_loss_weight,
            },
        )
        if args.separate_value_batch_size > 0 and value_population
        else None
    )
    anchor_rng = random.Random(args.seed + 17)
    diversity_caps = {
        "game": int(args.diversity_max_per_game),
        "turn": int(args.diversity_max_per_turn),
        "card": int(args.diversity_max_per_card),
        "region": int(args.diversity_max_per_region),
        "trigger": int(args.diversity_max_per_trigger),
    }
    diversity_caps = {key: value for key, value in diversity_caps.items() if value > 0}
    try:
        for update in range(1, args.updates + 1):
            batch = sampler.sample(
                args.batch_size,
                diversity_caps=diversity_caps,
                min_examples_per_era=args.min_search_examples_per_era,
                min_examples_per_side_era=args.min_search_examples_per_side_era,
                min_value_examples=(
                    0
                    if value_sampler is not None
                    else
                    args.min_value_examples_per_batch
                    if args.parameter_scope
                    in {
                        "actor_value",
                        "actor_value_heads",
                        "policy_value_heads",
                        "value_adapter",
                        "value_head",
                        "value_tower",
                        "value_tower_head",
                    }
                    and args.lambda_value > 0.0
                    else 0
                ),
            )
            value_batch = None
            if value_sampler is not None:
                requested = min(int(args.separate_value_batch_size), len(value_population))
                value_batch = value_sampler.sample(requested)
                if not value_batch:
                    raise RuntimeError("separate value replay produced an empty batch")
            anchor_batch = None
            if args.anchor_from_all_safe_replay and args.lambda_anchor > 0.0:
                anchor_size = min(
                    max(1, int(args.anchor_replay_batch_size)),
                    len(safe_anchor_examples),
                )
                anchor_batch = anchor_rng.sample(safe_anchor_examples, anchor_size)
            metrics = distill_search_batch(
                policy,
                optimizer,
                batch,
                anchor_examples=anchor_batch,
                value_examples=value_batch,
                lambda_search=args.lambda_search,
                lambda_pairwise=args.lambda_pairwise,
                lambda_anchor=args.lambda_anchor,
                max_grad_norm=args.max_grad_norm,
                parameter_scope=args.parameter_scope,
                q_weight_scale=args.q_weight_scale,
                max_q_weight=args.max_q_weight,
                hierarchy_weight=args.hierarchy_weight,
                balance_sides=args.balance_sides,
                lambda_value=args.lambda_value,
                value_loss_calibrated=args.value_loss_calibrated,
                lambda_search_q=args.lambda_search_q,
                value_anchor_model=value_anchor_model,
                lambda_value_anchor=args.lambda_value_anchor,
                confidence_weight_floor=args.confidence_weight_floor,
                us_policy_weight=args.us_policy_weight,
                ussr_policy_weight=args.ussr_policy_weight,
                search_policy_mix=args.search_policy_mix,
                pairwise_min_q_gap=args.pairwise_min_q_gap,
                pairwise_q_scale=args.pairwise_q_scale,
                pairwise_max_pairs=args.pairwise_max_pairs,
            )
            for era in ("early", "mid", "late"):
                metrics[f"batch_era_{era}"] = float(
                    sum(sampler.era(example) == era for example in batch)
                )
                for side in ("us", "ussr"):
                    metrics[f"batch_side_era_{side}_{era}"] = float(
                        sum(
                            example.metadata.side == side
                            and sampler.era(example) == era
                            for example in batch
                        )
                    )
            metrics["update"] = float(update)
            history.append(metrics)
            if update == 1 or update == args.updates or update % 25 == 0:
                print({"event": "search_distill_progress", **metrics}, flush=True)
            if (
                args.checkpoint_every_updates > 0
                and update % args.checkpoint_every_updates == 0
            ):
                saved = save_checkpoint(algo, args.output_dir, f"updates-{update:06d}")
                save_search_optimizer_state(saved, {SHARED_POLICY_ID: optimizer})
                intermediate_checkpoints.append(
                    {
                        "update": update,
                        "checkpoint": str(Path(saved).resolve()),
                        "metrics": dict(metrics),
                    }
                )
                print(
                    {
                        "event": "search_distill_intermediate_checkpoint",
                        "update": update,
                        "checkpoint": saved,
                    },
                    flush=True,
                )
        after = predictions(policy, report_examples)
        frozen_parameter_hashes_after = parameter_hashes(
            policy.model, frozen_parameter_names
        )
        if frozen_parameter_hashes_after != frozen_parameter_hashes_before:
            changed = sorted(
                name
                for name in frozen_parameter_names
                if frozen_parameter_hashes_before.get(name)
                != frozen_parameter_hashes_after.get(name)
            )
            raise RuntimeError(f"frozen parameters changed during distillation: {changed}")
        checkpoint = save_checkpoint(algo, args.output_dir, "distilled")
        optimizer_path = save_search_optimizer_state(checkpoint, {SHARED_POLICY_ID: optimizer})
    finally:
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass
    report = {
        "source_checkpoint": str(args.restore_from.expanduser().resolve()),
        "search_replay": [str(path) for path in replay_paths],
        "anchor_replay": [str(path) for path in anchor_replay_paths],
        "discovered_examples": len(discovered_examples),
        "anchor_discovered_examples": len(anchor_discovered_examples),
        "quality_filtered_examples": len(quality_filtered_examples),
        "safe_anchor_examples": len(safe_anchor_examples),
        "examples": len(examples),
        "min_q_advantage": args.min_q_advantage,
        "min_teacher_visits": args.min_teacher_visits,
        "min_teacher_visit_share": args.min_teacher_visit_share,
        "max_atomic_horizon_cutoffs": args.max_atomic_horizon_cutoffs,
        "max_node_budget_cutoffs": args.max_node_budget_cutoffs,
        "max_examples_per_source_root": args.max_examples_per_source_root,
        "require_flip": args.require_flip,
        "lambda_search": args.lambda_search,
        "lambda_pairwise": args.lambda_pairwise,
        "pairwise_min_q_gap": args.pairwise_min_q_gap,
        "pairwise_q_scale": args.pairwise_q_scale,
        "pairwise_max_pairs": args.pairwise_max_pairs,
        "lambda_anchor": args.lambda_anchor,
        "search_policy_mix": args.search_policy_mix,
        "lambda_value": args.lambda_value,
        "lambda_search_q": args.lambda_search_q,
        "lambda_value_anchor": args.lambda_value_anchor,
        "confidence_weight_floor": args.confidence_weight_floor,
        "require_complete_game": args.require_complete_game,
        "root_only": args.root_only,
        "anchor_from_all_safe_replay": args.anchor_from_all_safe_replay,
        "anchor_replay_batch_size": args.anchor_replay_batch_size,
        "min_value_examples_per_batch": args.min_value_examples_per_batch,
        "separate_value_batch_size": args.separate_value_batch_size,
        "separate_value_population": len(value_population),
        "value_stratum_weights": (
            {
                f"{side}_{outcome}": weight
                for (side, outcome), weight in value_sampler.stratum_weights.items()
            }
            if value_sampler
            else {}
        ),
        "balance_sides": args.balance_sides,
        "us_policy_weight": args.us_policy_weight,
        "ussr_policy_weight": args.ussr_policy_weight,
        "parameter_scope": args.parameter_scope,
        "value_state_tower": bool(args.value_state_tower),
        "anchor_sample_fraction": anchor_fraction,
        "trainable_parameter_names": trainable_parameter_names,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_hashes_before": frozen_parameter_hashes_before,
        "frozen_parameter_hashes_after": frozen_parameter_hashes_after,
        "frozen_parameters_verified": True,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "optimizer_restored": optimizer_restored,
        "diversity_caps": diversity_caps,
        "min_search_examples_per_era": args.min_search_examples_per_era,
        "min_search_examples_per_side_era": args.min_search_examples_per_side_era,
        "q_weight_scale": args.q_weight_scale,
        "max_q_weight": args.max_q_weight,
        "hierarchy_weight": args.hierarchy_weight,
        "updates": args.updates,
        "intermediate_checkpoints": intermediate_checkpoints,
        "reported_examples": len(report_examples),
        "checkpoint": checkpoint,
        "optimizer_state": str(optimizer_path),
        "before": before,
        "after": after,
        "last_metrics": history[-1],
        "history": history,
    }
    (args.output_dir / "distillation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "search_distill_done",
                "checkpoint": checkpoint,
                "optimizer_state": str(optimizer_path),
                "discovered_examples": len(discovered_examples),
                "quality_filtered_examples": len(quality_filtered_examples),
                "examples": len(examples),
                "separate_value_population": len(value_population),
                "updates": args.updates,
                "last_metrics": history[-1],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
