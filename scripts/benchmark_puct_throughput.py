#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.mcts import SelectivePuctConfig, SelectivePuctPlanner, TorchPolicyValueEvaluator
from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import SHARED_POLICY_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark selective-PUCT state and inference throughput.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--game-log", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--seed-occurrence", type=int, default=1)
    parser.add_argument("--action-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--meaningful-depth", type=int, default=2)
    parser.add_argument("--max-atomic-steps", type=int, default=512)
    parser.add_argument("--root-repeats", type=int, default=3)
    parser.add_argument("--local-repeats", type=int, default=20)
    parser.add_argument("--nn-repeats", type=int, default=20)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--max-exact-chance-outcomes", type=int, default=36)
    parser.add_argument("--max-expanded-nodes", type=int, default=512)
    return parser.parse_args()


def algorithm_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        restore_from=args.checkpoint.expanduser().resolve(),
        num_gpus=args.num_gpus,
        hidden=256,
        history_layers=2,
        card_history_layers=2,
        history_attention_heads=4,
        history_dropout=0.05,
        graph_layers=2,
        graph_neighbor_hops=2,
        policy_temperature=1.0,
        ray_num_cpus=1,
        ray_object_store_memory=150_000_000,
    )


def env_config(*, search_mode: bool) -> dict[str, Any]:
    return {
        "force_setup_heuristic": False,
        "persistent_random_prob": 0.0,
        "heuristic_override_prob": 0.0,
        "scripted_side_prob": 0.0,
        "defcon_suicide_mode": "hard_filter",
        "terminal_reward_scale": 1.0,
        "reward_shaping_scale": 0.0,
        "nuke_death_penalty": 0.0,
        "scoring_card_held_penalty": 0.0,
        "max_episode_steps": 1200,
        "search_mode": search_mode,
        "log_action_details": False,
        "log_games_dir": None,
        "metrics_dir": None,
    }


def load_game(
    path: Path,
    seed: int,
    episode_index: int | None = None,
    occurrence: int = 1,
) -> dict[str, Any]:
    matched = 0
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record.get("kind") == "terminal"
                and int(record.get("seed", -1)) == seed
                and (episode_index is None or int(record.get("episode_index", -1)) == episode_index)
            ):
                matched += 1
                if matched == occurrence:
                    return record
    raise ValueError(
        f"no complete game occurrence={occurrence} with seed={seed}, "
        f"episode_index={episode_index} in {path}"
    )


def timed(callable_: Any, repeats: int) -> list[float]:
    values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        values.append(time.perf_counter() - started)
    return values


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "samples": len(values),
    }


def main() -> None:
    args = parse_args()
    game = load_game(args.game_log, args.seed, args.episode_index, args.seed_occurrence)
    if args.action_index > len(game.get("actions", [])):
        raise ValueError("action index exceeds recorded game length")

    replay_env = TwilightStruggleEnv(env_config(search_mode=False))
    search_env = TwilightStruggleEnv(env_config(search_mode=True))
    algo = build_algorithm(algorithm_args(args))
    evaluator = TorchPolicyValueEvaluator(algo.get_policy(SHARED_POLICY_ID))
    try:
        obs, _ = replay_env.reset(seed=args.seed)
        for record in game["actions"][: args.action_index]:
            obs, _reward, terminated, truncated, _info = replay_env.step(int(record["action_index"]))
            if terminated or truncated:
                raise RuntimeError("record terminated before requested benchmark position")
        portable = replay_env.snapshot(compact=True)

        root_times = timed(lambda: search_env.restore(portable), args.root_repeats)
        search_env.restore(portable)
        local_capture_times = timed(lambda: search_env.snapshot_local(compact=True), args.local_repeats)
        local = search_env.snapshot_local(compact=True)
        root_legal_count = len(search_env.legal_actions)
        local_restore_times = timed(lambda: search_env.restore_local(local, encode=False), args.local_repeats)

        evaluator.cache.clear()
        cold_nn_times = []
        for _ in range(args.nn_repeats):
            evaluator.cache.clear()
            cold_nn_times.extend(timed(lambda: evaluator.raw(obs, len(search_env.legal_actions)), 1))
        warm_nn_times = timed(lambda: evaluator.raw(obs, len(search_env.legal_actions)), args.nn_repeats)

        config = SelectivePuctConfig(
            initial_simulations=args.simulations,
            medium_simulations=args.simulations,
            max_simulations=args.simulations,
            belief_particles=args.belief_particles,
            meaningful_depth=args.meaningful_depth,
            max_atomic_steps=args.max_atomic_steps,
            max_exact_chance_outcomes=args.max_exact_chance_outcomes,
            max_expanded_nodes=args.max_expanded_nodes,
        )
        evaluator.cache.clear()
        planner = SelectivePuctPlanner(evaluator, search_env, config, seed=args.seed + 17)
        decision = planner.plan(portable, obs, force_search=True)
        bridge_metrics = search_env.bridge.checkpoint_metrics()
        payload = {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "game_log": str(args.game_log.expanduser().resolve()),
            "seed": args.seed,
            "episode_index": args.episode_index,
            "seed_occurrence": args.seed_occurrence,
            "action_index": args.action_index,
            "history_actions": len(portable["bridge"]["actions"]),
            "legal_actions": root_legal_count,
            "root_reconstruction": stats(root_times),
            "local_checkpoint_capture": stats(local_capture_times),
            "local_checkpoint_restore": stats(local_restore_times),
            "nn_cold_evaluation": stats(cold_nn_times),
            "nn_cached_evaluation": stats(warm_nn_times),
            "search": {
                "seconds": decision.latency_seconds,
                "simulations": decision.simulations,
                "simulations_per_second": decision.simulations / max(decision.latency_seconds, 1e-12),
                "expanded_nodes": decision.expanded_nodes,
                "exact_chance_nodes": decision.exact_chance_nodes,
                "sampled_large_chance_nodes": decision.sampled_large_chance_nodes,
                "raw_action": decision.raw_action,
                "search_action": decision.action,
                "action_flipped": decision.action_flipped,
                "root_q_gap": decision.root_q_gap,
                "q_advantage_over_raw": decision.q_advantage_over_raw,
                "triggers": list(decision.triggers),
                "hierarchical_path": list(decision.hierarchical_path),
                "q_gate_rejected": decision.q_gate_rejected,
                "atomic_horizon_cutoffs": decision.atomic_horizon_cutoffs,
                "node_budget_cutoffs": decision.node_budget_cutoffs,
                "max_meaningful_depth_reached": decision.max_meaningful_depth_reached,
                "max_turn_delta": decision.max_turn_delta,
                "cross_turn_leaf_evaluations": decision.cross_turn_leaf_evaluations,
            },
            "bridge_checkpoint_metrics": bridge_metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    finally:
        replay_env.close()
        search_env.close()
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
