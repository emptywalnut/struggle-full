#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.mcts import SelectivePuctConfig, SelectivePuctPlanner, TorchPolicyValueEvaluator
from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import SHARED_POLICY_ID, compute_algo_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired full-game evaluation of selective PUCT against the raw NN.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games-per-side", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=81_100_000)
    parser.add_argument("--initial-simulations", type=int, default=32)
    parser.add_argument("--medium-simulations", type=int, default=128)
    parser.add_argument("--max-simulations", type=int, default=512)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--meaningful-depth", type=int, default=2)
    parser.add_argument("--max-atomic-steps", type=int, default=512)
    parser.add_argument("--forced-response-depth", type=int, default=1)
    parser.add_argument("--child-initial", type=int, default=4)
    parser.add_argument("--child-max", type=int, default=24)
    parser.add_argument("--search-temperature", type=float, default=0.0)
    parser.add_argument("--medium-visit-share-threshold", type=float, default=0.62)
    parser.add_argument("--medium-q-gap-threshold", type=float, default=0.05)
    parser.add_argument("--max-visit-share-threshold", type=float, default=0.55)
    parser.add_argument("--max-q-gap-threshold", type=float, default=0.03)
    parser.add_argument(
        "--escalate-on-disagreement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--disable-trigger", action="append", default=[])
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--card-history-layers", type=int, default=2)
    parser.add_argument("--history-attention-heads", type=int, default=4)
    parser.add_argument("--history-dropout", type=float, default=0.05)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-neighbor-hops", type=int, default=2)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    return parser.parse_args()


def algorithm_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        restore_from=args.checkpoint.expanduser().resolve(),
        num_gpus=args.num_gpus,
        hidden=args.hidden,
        history_layers=args.history_layers,
        card_history_layers=args.card_history_layers,
        history_attention_heads=args.history_attention_heads,
        history_dropout=args.history_dropout,
        graph_layers=args.graph_layers,
        graph_neighbor_hops=args.graph_neighbor_hops,
        policy_temperature=args.policy_temperature,
        ray_num_cpus=1,
    )


def env_config(args: argparse.Namespace, log_dir: Path | None, *, search_mode: bool = False) -> dict[str, Any]:
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
        "max_episode_steps": args.max_episode_steps,
        "search_mode": search_mode,
        "log_games_dir": str(log_dir) if log_dir else None,
        "log_games_every": 1,
        "log_action_details": log_dir is not None,
        "metrics_dir": None,
    }


def play_game(
    algo: Any,
    planner: SelectivePuctPlanner,
    args: argparse.Namespace,
    *,
    seed: int,
    planned_side: str | None,
    label: str,
) -> dict[str, Any]:
    env = TwilightStruggleEnv(env_config(args, args.output_dir / label / "game_logs"))
    searched = flips = simulations = 0
    search_latency = 0.0
    search_events: list[dict[str, Any]] = []
    game_started = time.perf_counter()
    try:
        obs, _info = env.reset(seed=seed)
        terminated = truncated = False
        final_info: dict[str, Any] = {}
        while not (terminated or truncated):
            side = str((env.last_obs or {}).get("side") or "ussr")
            if planned_side == side:
                decision = planner.plan(env.snapshot(compact=True), obs)
                action = decision.action
                searched += int(decision.searched)
                flips += int(decision.action_flipped)
                simulations += decision.simulations
                search_latency += decision.latency_seconds
                if decision.searched or "search_continuation" in decision.triggers:
                    event = {
                        "episode_step": env.episode_step,
                        "turn": int((env.last_obs or {}).get("turn") or 0),
                        "action_round": int((env.last_obs or {}).get("action_round") or 0),
                        "side": side,
                        "triggers": list(decision.triggers),
                        "simulations": decision.simulations,
                        "raw_action": decision.raw_action,
                        "search_action": decision.action,
                        "raw_action_text": str(env.legal_actions[decision.raw_action]),
                        "search_action_text": str(env.legal_actions[decision.action]),
                        "action_flipped": decision.action_flipped,
                "root_q_gap": decision.root_q_gap,
                "q_advantage_over_raw": decision.q_advantage_over_raw,
                        "latency_seconds": decision.latency_seconds,
                        "hierarchical_path": list(decision.hierarchical_path),
                        "max_meaningful_depth_reached": decision.max_meaningful_depth_reached,
                        "max_turn_delta": decision.max_turn_delta,
                        "cross_turn_leaf_evaluations": decision.cross_turn_leaf_evaluations,
                    }
                    search_events.append(event)
                    print({"event": "selective_search_action", "label": label, "seed": seed, **event}, flush=True)
            else:
                action = compute_algo_action(algo, obs, SHARED_POLICY_ID)
            obs, _reward, terminated, truncated, final_info = env.step(action)
        return {
            "label": label,
            "seed": seed,
            "planned_side": planned_side,
            "winner": final_info.get("winner"),
            "terminal_reason": final_info.get("terminal_reason"),
            "steps": env.episode_step,
            "searched_decisions": searched,
            "action_flips": flips,
            "simulations": simulations,
            "search_latency_seconds": search_latency,
            "wall_clock_seconds": time.perf_counter() - game_started,
            "search_events": search_events,
        }
    finally:
        env.close()


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    p = wins / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denom
    half = z * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def elo_delta(score: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, score))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def summarize(records: list[dict[str, Any]], *, search_match: bool) -> dict[str, Any]:
    side_key = "planned_side" if search_match else "evaluated_side"
    wins = sum(record["winner"] == record.get(side_key) for record in records)
    games = len(records)
    rate = wins / games if games else 0.0
    lo, hi = wilson_interval(wins, games)
    return {
        "games": games,
        "wins": wins,
        "win_rate": rate,
        "win_rate_ci95_low": lo,
        "win_rate_ci95_high": hi,
        "elo_delta_vs_opponent": elo_delta(rate) if games else 0.0,
        "nuke_terminal_rate": sum(record["terminal_reason"] in {"nuclear_war", "thermonuclear war", "Cuban Missile Crisis"} for record in records) / max(1, games),
        "average_steps": float(np.mean([record["steps"] for record in records])) if records else 0.0,
        "average_simulations_per_game": float(np.mean([record.get("simulations", 0) for record in records])) if records else 0.0,
        "average_search_latency_seconds_per_game": float(
            np.mean([record.get("search_latency_seconds", 0.0) for record in records])
        ) if records else 0.0,
        "action_flip_rate": sum(record.get("action_flips", 0) for record in records)
        / max(1, sum(record.get("searched_decisions", 0) for record in records)),
        "average_wall_clock_seconds_per_game": float(
            np.mean([record.get("wall_clock_seconds", 0.0) for record in records])
        ) if records else 0.0,
        "terminal_reason_distribution": dict(Counter(str(record.get("terminal_reason") or "unknown") for record in records)),
    }


def summarize_triggers(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    for record in records:
        won = record.get("winner") == record.get("planned_side")
        for event in record.get("search_events", []):
            for trigger in event.get("triggers", []):
                grouped[str(trigger)].append((event, won))
    result: dict[str, dict[str, float]] = {}
    for trigger, values in sorted(grouped.items()):
        events = [value[0] for value in values]
        result[trigger] = {
            "frequency": float(len(events)),
            "flip_rate": sum(bool(event.get("action_flipped")) for event in events) / len(events),
            "average_root_q_gap": float(np.mean([event.get("root_q_gap", 0.0) for event in events])),
            "average_simulations": float(np.mean([event.get("simulations", 0) for event in events])),
            "total_simulations": float(sum(event.get("simulations", 0) for event in events)),
            "average_latency_seconds": float(np.mean([event.get("latency_seconds", 0.0) for event in events])),
            "downstream_controlled_side_win_rate": sum(value[1] for value in values) / len(values),
        }
    return result


def summarize_paired_strength(
    raw_records: list[dict[str, Any]],
    search_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(raw_records) != len(search_records):
        raise ValueError("paired evaluation record counts differ")
    improved = regressed = unchanged_wins = unchanged_losses = 0
    for raw, searched in zip(raw_records, search_records, strict=True):
        side = str(searched["planned_side"])
        raw_won = raw.get("winner") == side
        search_won = searched.get("winner") == side
        if search_won and not raw_won:
            improved += 1
        elif raw_won and not search_won:
            regressed += 1
        elif search_won:
            unchanged_wins += 1
        else:
            unchanged_losses += 1
    games = len(raw_records)
    return {
        "games": games,
        "improved": improved,
        "regressed": regressed,
        "unchanged_wins": unchanged_wins,
        "unchanged_losses": unchanged_losses,
        "net_improvement_rate": (improved - regressed) / max(1, games),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    enabled = tuple(
        trigger
        for trigger in SelectivePuctConfig().enabled_triggers
        if trigger not in set(args.disable_trigger)
    )
    config = SelectivePuctConfig(
        initial_simulations=args.initial_simulations,
        medium_simulations=args.medium_simulations,
        max_simulations=args.max_simulations,
        belief_particles=args.belief_particles,
        meaningful_depth=args.meaningful_depth,
        max_atomic_steps=args.max_atomic_steps,
        forced_response_depth=args.forced_response_depth,
        child_initial=args.child_initial,
        child_max=args.child_max,
        temperature=args.search_temperature,
        enabled_triggers=enabled,
        medium_visit_share_threshold=args.medium_visit_share_threshold,
        medium_q_gap_threshold=args.medium_q_gap_threshold,
        max_visit_share_threshold=args.max_visit_share_threshold,
        max_q_gap_threshold=args.max_q_gap_threshold,
        escalate_on_disagreement=args.escalate_on_disagreement,
    )
    algo = build_algorithm(algorithm_args(args))
    evaluator = TorchPolicyValueEvaluator(algo.get_policy(SHARED_POLICY_ID))
    search_env = TwilightStruggleEnv(env_config(args, None, search_mode=True))
    planner = SelectivePuctPlanner(evaluator, search_env, config, seed=args.seed_base + 17)
    raw_records: list[dict[str, Any]] = []
    search_records: list[dict[str, Any]] = []
    try:
        for side_index, side in enumerate(("us", "ussr")):
            for game_index in range(args.games_per_side):
                seed = args.seed_base + side_index * 1_000_003 + game_index
                raw_record = play_game(algo, planner, args, seed=seed, planned_side=None, label="raw_vs_raw")
                raw_record["evaluated_side"] = side
                raw_records.append(raw_record)
                search_records.append(play_game(algo, planner, args, seed=seed, planned_side=side, label="search_vs_raw"))
                print(
                    {
                        "event": "selective_eval_progress",
                        "side": side,
                        "game": game_index + 1,
                        "games_per_side": args.games_per_side,
                        "raw_winner": raw_records[-1]["winner"],
                        "search_winner": search_records[-1]["winner"],
                        **planner.telemetry.metrics(),
                    },
                    flush=True,
                )
    finally:
        search_env.close()
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass
    payload = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "config": vars(args) | {"checkpoint": str(args.checkpoint), "output_dir": str(args.output_dir)},
        "raw_vs_raw": summarize(raw_records, search_match=False),
        "selective_puct_vs_raw": summarize(search_records, search_match=True),
        "paired_strength": summarize_paired_strength(raw_records, search_records),
        "planner": planner.telemetry.metrics(),
        "triggers": summarize_triggers(search_records),
        "games": {"raw_vs_raw": raw_records, "selective_puct_vs_raw": search_records},
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "games"}, default=str), flush=True)


if __name__ == "__main__":
    main()
