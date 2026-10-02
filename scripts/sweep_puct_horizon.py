#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.frozen_diagnostic import find_action_index
from struggle_ai.mcts import SelectivePuctConfig, SelectivePuctPlanner, TorchPolicyValueEvaluator
from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import SHARED_POLICY_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep independent PUCT horizons and simulation budgets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--game-log", type=Path, action="append", default=[])
    parser.add_argument("--position-manifest", type=Path, default=None)
    parser.add_argument(
        "--position",
        action="append",
        default=[],
        help="seed:zero_based_action_index[:zero_based_game_occurrence]",
    )
    parser.add_argument("--config", action="append", required=True, help="meaningful_depth:simulations")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--max-atomic-steps", type=int, default=512)
    parser.add_argument(
        "--max-expanded-nodes",
        type=int,
        default=16384,
        help="Per-root safety cap; configurations reporting cutoffs must not be treated as converged.",
    )
    parser.add_argument("--max-exact-chance-outcomes", type=int, default=36)
    parser.add_argument("--child-initial", type=int, default=4)
    parser.add_argument("--child-max", type=int, default=24)
    parser.add_argument("--seed", type=int, default=8_120_001)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    return parser.parse_args()


def parse_pair(value: str, label: str) -> tuple[int, int]:
    try:
        left, right = value.split(":", 1)
        result = int(left), int(right)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid {label} {value!r}; expected integer:integer") from exc
    if min(result) <= 0:
        raise SystemExit(f"invalid {label} {value!r}; values must be positive")
    return result


def parse_position(value: str) -> tuple[int, int, int]:
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise SystemExit(
            f"invalid --position {value!r}; expected seed:action_index[:game_occurrence]"
        )
    try:
        seed, action_index = int(parts[0]), int(parts[1])
        occurrence = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise SystemExit(f"invalid --position {value!r}; fields must be integers") from exc
    if seed < 0 or action_index < 0 or occurrence < 0:
        raise SystemExit(f"invalid --position {value!r}; fields must be non-negative")
    return seed, action_index, occurrence


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


def env_config() -> dict[str, Any]:
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
        "search_mode": True,
        "log_action_details": False,
        "log_games_dir": None,
        "metrics_dir": None,
    }


def load_games(
    paths: list[Path], positions: list[tuple[int, int, int]]
) -> dict[tuple[int, int], dict[str, Any]]:
    requested = {(seed, occurrence) for seed, _action_index, occurrence in positions}
    occurrences: dict[int, int] = {}
    games: dict[tuple[int, int], dict[str, Any]] = {}
    for path in paths:
        with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("kind") != "terminal":
                    continue
                seed = int(record.get("seed", -1))
                occurrence = occurrences.get(seed, 0)
                occurrences[seed] = occurrence + 1
                key = (seed, occurrence)
                if key in requested:
                    games[key] = record
    missing = requested - set(games)
    if missing:
        raise ValueError(f"complete games not found for seed/occurrence pairs: {sorted(missing)}")
    return games


def replay_position(game: dict[str, Any], action_index: int) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    env = TwilightStruggleEnv(env_config())
    try:
        obs, _ = env.reset(seed=int(game["seed"]))
        for step_index, record in enumerate(game["actions"][:action_index]):
            logged_index = find_action_index(env.legal_actions, record["action"])
            obs, _reward, terminated, truncated, _info = env.step(logged_index)
            if terminated or truncated:
                raise RuntimeError(f"game terminated before requested action {action_index} at {step_index}")
        root = dict(env.last_obs or {})
        return env.snapshot(compact=True), obs, copy.deepcopy(env.legal_actions), root
    finally:
        env.close()


def q_map(result: dict[str, Any]) -> dict[str, float]:
    return {str(row["action"]): float(row["Q"]) for row in result["root_table"]}


def comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for left_index, left in enumerate(results):
        for right in results[left_index + 1 :]:
            left_q = q_map(left)
            right_q = q_map(right)
            shared = sorted(set(left_q) & set(right_q))
            left_values = np.asarray([left_q[key] for key in shared], dtype=np.float64)
            right_values = np.asarray([right_q[key] for key in shared], dtype=np.float64)
            correlation = 1.0
            if len(shared) > 1 and float(left_values.std()) > 1e-12 and float(right_values.std()) > 1e-12:
                correlation = float(np.corrcoef(left_values, right_values)[0, 1])
            output.append(
                {
                    "left": left["config"],
                    "right": right["config"],
                    "action_agreement": left["search_action_text"] == right["search_action_text"],
                    "shared_actions": len(shared),
                    "q_correlation": correlation,
                    "q_linf": float(np.max(np.abs(left_values - right_values))) if len(shared) else None,
                }
            )
    return output


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.position_manifest is not None:
        manifest = json.loads(args.position_manifest.expanduser().resolve().read_text(encoding="utf-8"))
        for entry in manifest.get("positions", []):
            args.game_log.append(Path(str(entry["game_log"])))
            args.position.append(
                f"{int(entry['seed'])}:{int(entry['action_index'])}:{int(entry.get('game_occurrence', 0))}"
            )
    args.game_log = list(dict.fromkeys(args.game_log))
    if not args.game_log or not args.position:
        raise SystemExit("provide --position-manifest or both --game-log and --position")
    positions = [parse_position(value) for value in args.position]
    configs = [parse_pair(value, "--config") for value in args.config]
    games = load_games(args.game_log, positions)
    roots = {
        (seed, action_index, occurrence): replay_position(games[(seed, occurrence)], action_index)
        for seed, action_index, occurrence in positions
    }
    algo = build_algorithm(algorithm_args(args))
    evaluator = TorchPolicyValueEvaluator(algo.get_policy(SHARED_POLICY_ID))
    search_env = TwilightStruggleEnv(env_config())
    payload: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "game_logs": [str(path.expanduser().resolve()) for path in args.game_log],
        "belief_particles": args.belief_particles,
        "max_atomic_steps": args.max_atomic_steps,
        "max_expanded_nodes": args.max_expanded_nodes,
        "positions": [],
    }
    try:
        for position_number, (seed, action_index, occurrence) in enumerate(positions):
            snapshot, obs, legal_actions, root = roots[(seed, action_index, occurrence)]
            position_payload = {
                "seed": seed,
                "action_index": action_index,
                "game_occurrence": occurrence,
                "root": {
                    "turn": int(root.get("turn") or 0),
                    "action_round": int(root.get("action_round") or 0),
                    "side": str(root.get("side") or ""),
                    "phase": str(root.get("phase") or ""),
                    "prompt": str(root.get("prompt") or ""),
                    "vp": int(root.get("vp") or 0),
                    "defcon": int(root.get("defcon") or 0),
                    "hand": list(root.get("hand") or []),
                    "legal_actions": legal_actions,
                },
                "results": [],
                "comparisons": [],
            }
            payload["positions"].append(position_payload)
            for config_number, (depth, simulations) in enumerate(configs):
                config = SelectivePuctConfig(
                    initial_simulations=simulations,
                    medium_simulations=simulations,
                    max_simulations=simulations,
                    belief_particles=args.belief_particles,
                    meaningful_depth=depth,
                    forced_response_depth=depth,
                    max_atomic_steps=args.max_atomic_steps,
                    max_expanded_nodes=args.max_expanded_nodes,
                    child_initial=args.child_initial,
                    child_max=args.child_max,
                    max_exact_chance_outcomes=args.max_exact_chance_outcomes,
                    min_q_advantage_to_flip=0.0,
                )
                evaluator.cache.clear()
                planner = SelectivePuctPlanner(
                    evaluator,
                    search_env,
                    config,
                    seed=args.seed + 1009 * position_number + 9176 * config_number,
                )
                started = time.perf_counter()
                decision = planner.plan(snapshot, obs, force_search=True)
                result = {
                    "config": {"depth": depth, "simulations": simulations},
                    "seconds": time.perf_counter() - started,
                    "simulations_per_second": simulations / max(decision.latency_seconds, 1e-12),
                    "raw_action": decision.raw_action,
                    "raw_action_text": str(legal_actions[decision.raw_action]),
                    "search_action": decision.action,
                    "search_action_text": str(legal_actions[decision.action]),
                    "action_flipped": decision.action_flipped,
                    "searched": decision.searched,
                    "trigger_reasons": list(decision.triggers),
                    "root_q_gap": decision.root_q_gap,
                    "q_advantage_over_raw": decision.q_advantage_over_raw,
                    "expanded_nodes": decision.expanded_nodes,
                    "node_budget_cutoffs": decision.node_budget_cutoffs,
                    "atomic_horizon_cutoffs": decision.atomic_horizon_cutoffs,
                    "exact_chance_nodes": decision.exact_chance_nodes,
                    "sampled_large_chance_nodes": decision.sampled_large_chance_nodes,
                    "max_meaningful_depth_reached": decision.max_meaningful_depth_reached,
                    "max_turn_delta": decision.max_turn_delta,
                    "cross_turn_leaf_evaluations": decision.cross_turn_leaf_evaluations,
                    "terminal_evaluations": decision.terminal_evaluations,
                    "evidence_gate_rejected": decision.evidence_gate_rejected,
                    "hierarchical_path": list(decision.hierarchical_path),
                    "root_table": decision.budgets.get(simulations, []),
                }
                position_payload["results"].append(result)
                position_payload["comparisons"] = comparisons(position_payload["results"])
                write_output(args.output, payload)
                print({"event": "horizon_sweep_result", "seed": seed, "action_index": action_index, **result}, flush=True)
    finally:
        search_env.close()
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
