#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.mcts import SelectivePuctConfig, SelectivePuctPlanner, TorchPolicyValueEvaluator
from struggle_ai.search_replay import (
    CompleteGameReplayTransaction,
    SearchReplayWriter,
    examples_from_decision,
)
from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import SHARED_POLICY_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect complete-game selective-PUCT self-play replay.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=84_000_000)
    parser.add_argument("--initial-simulations", type=int, default=32)
    parser.add_argument("--medium-simulations", type=int, default=32)
    parser.add_argument("--max-simulations", type=int, default=32)
    parser.add_argument("--meaningful-depth", type=int, default=2)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--max-atomic-steps", type=int, default=512)
    parser.add_argument("--max-expanded-nodes", type=int, default=512)
    parser.add_argument("--child-initial", type=int, default=4)
    parser.add_argument("--child-max", type=int, default=24)
    parser.add_argument("--search-temperature", type=float, default=1.0)
    parser.add_argument("--behavior-temperature", type=float, default=0.75)
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--low-confidence-sample-rate", type=float, default=0.05)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--deep-candidate-q-gap", type=float, default=0.03)
    parser.add_argument("--deep-candidate-max", type=int, default=256)
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


def search_config(args: argparse.Namespace) -> SelectivePuctConfig:
    return SelectivePuctConfig(
        initial_simulations=args.initial_simulations,
        medium_simulations=args.medium_simulations,
        max_simulations=args.max_simulations,
        belief_particles=args.belief_particles,
        meaningful_depth=args.meaningful_depth,
        max_atomic_steps=args.max_atomic_steps,
        max_expanded_nodes=args.max_expanded_nodes,
        child_initial=args.child_initial,
        child_max=args.child_max,
        temperature=args.search_temperature,
        behavior_temperature=args.behavior_temperature,
        low_confidence_sample_rate=args.low_confidence_sample_rate,
    )


def is_deep_candidate(event: dict[str, Any], q_gap_threshold: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    triggers = set(event["triggers"])
    if abs(float(event["root_q_gap"])) <= q_gap_threshold:
        reasons.append("small_q_gap")
    if event["action_flipped"]:
        reasons.append("raw_search_disagreement")
    for trigger in (
        "scoring_card",
        "scoring_pressure",
        "forced_response",
        "opponent_event",
        "chance_heavy",
    ):
        if trigger in triggers:
            reasons.append(trigger)
    return bool(reasons), reasons


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = args.output_dir / "search_replay"
    log_dir = args.output_dir / "game_logs"
    writer = SearchReplayWriter(replay_dir, shard_size=args.shard_size)
    algo = build_algorithm(algorithm_args(args))
    policy = algo.get_policy(SHARED_POLICY_ID)
    config = search_config(args)
    planners: dict[str, SelectivePuctPlanner] = {}
    search_envs: list[TwilightStruggleEnv] = []
    for offset, side in enumerate(("us", "ussr")):
        search_env = TwilightStruggleEnv(env_config(args, None, search_mode=True))
        search_envs.append(search_env)
        planners[side] = SelectivePuctPlanner(
            TorchPolicyValueEvaluator(policy),
            search_env,
            config,
            seed=args.seed_base + 10_003 * (offset + 1),
        )

    game_records: list[dict[str, Any]] = []
    terminal_reasons: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    side_searches: Counter[str] = Counter()
    deep_candidates: list[dict[str, Any]] = []
    committed_examples = discarded_games = total_decisions = total_searches = 0
    partial_search_labels_rejected = 0
    action_flips = behavior_teacher_disagreements = total_simulations = 0
    total_search_seconds = total_q_advantage = 0.0
    searched_state_kinds: set[tuple[int, int, str, tuple[str, ...]]] = set()
    started = time.perf_counter()
    try:
        for game_index in range(args.games):
            seed = args.seed_base + game_index
            source_game = f"puct-selfplay-{game_index:06d}-seed-{seed}"
            transaction = CompleteGameReplayTransaction(writer)
            env = TwilightStruggleEnv(env_config(args, log_dir))
            game_started = time.perf_counter()
            game_events: list[dict[str, Any]] = []
            final_info: dict[str, Any] = {}
            terminated = truncated = False
            try:
                obs, _ = env.reset(seed=seed)
                while not (terminated or truncated):
                    side = str((env.last_obs or {}).get("side") or "ussr")
                    legal_actions = [dict(action) for action in env.legal_actions]
                    decision = planners[side].plan(
                        env.snapshot(compact=True),
                        obs,
                        force_search=args.force_search,
                    )
                    total_decisions += 1
                    if decision.searched:
                        total_searches += 1
                        action_flips += int(decision.action_flipped)
                        behavior_teacher_disagreements += int(decision.action != decision.teacher_action)
                        total_simulations += int(decision.simulations)
                        total_search_seconds += float(decision.latency_seconds)
                        total_q_advantage += float(decision.q_advantage_over_raw)
                        searched_state_kinds.add(
                            (
                                int((env.last_obs or {}).get("turn") or 0),
                                int((env.last_obs or {}).get("action_round") or 0),
                                side,
                                tuple(decision.triggers),
                            )
                        )
                        side_searches[side] += 1
                        for trigger in decision.triggers:
                            trigger_counts[trigger] += 1
                        examples = examples_from_decision(
                            obs,
                            legal_actions,
                            side,
                            decision,
                            search_depth=decision.meaningful_depth,
                            source_game=source_game,
                            source_step=env.episode_step,
                            behavior_policy="selective_puct_selfplay",
                            behavior_temperature=args.behavior_temperature,
                        )
                        transaction.add_decision(examples, decision)
                        event = {
                            "source_game": source_game,
                            "seed": seed,
                            "action_index": env.episode_step,
                            "side": side,
                            "turn": int((env.last_obs or {}).get("turn") or 0),
                            "action_round": int((env.last_obs or {}).get("action_round") or 0),
                            "triggers": list(decision.triggers),
                            "simulations": decision.simulations,
                            "meaningful_depth": decision.meaningful_depth,
                            "raw_action": decision.raw_action,
                            "teacher_action": decision.teacher_action,
                            "behavior_action": decision.action,
                            "action_flipped": decision.action_flipped,
                            "behavior_differs_from_teacher": decision.action != decision.teacher_action,
                            "root_q_gap": decision.root_q_gap,
                            "q_advantage_over_raw": decision.q_advantage_over_raw,
                            "latency_seconds": decision.latency_seconds,
                            "max_turn_delta": decision.max_turn_delta,
                        }
                        selected, reasons = is_deep_candidate(event, args.deep_candidate_q_gap)
                        if selected and len(deep_candidates) < args.deep_candidate_max:
                            deep_candidates.append({**event, "adjudication_reasons": reasons})
                        game_events.append(event)
                    obs, _reward, terminated, truncated, final_info = env.step(decision.action)
                winner = str(final_info.get("winner") or "")
                reason = str(final_info.get("terminal_reason") or "unknown")
                if truncated:
                    transaction.invalidate("truncated_game")
                elif winner not in {"us", "ussr"}:
                    transaction.invalidate("missing_terminal_winner")
                committed = transaction.commit(winner=winner, terminal_reason=reason) if not truncated else 0
                if committed == 0 and transaction.invalid_reason is not None:
                    discarded_games += 1
                committed_examples += committed
                partial_search_labels_rejected += transaction.partial_decisions_rejected
                terminal_reasons[reason] += 1
                record = {
                    "game_index": game_index,
                    "source_game": source_game,
                    "seed": seed,
                    "winner": winner,
                    "terminal_reason": reason,
                    "steps": env.episode_step,
                    "searched_decisions": len(game_events),
                    "committed_examples": committed,
                    "discard_reason": transaction.invalid_reason,
                    "partial_search_labels_rejected": transaction.partial_decisions_rejected,
                    "wall_clock_seconds": time.perf_counter() - game_started,
                }
                game_records.append(record)
                with (args.output_dir / "games.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                writer.flush()
                print({"event": "puct_selfplay_game", **record}, flush=True)
            finally:
                env.close()
    finally:
        manifest = writer.close()
        for env in search_envs:
            env.close()
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass

    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "games_requested": args.games,
        "games_completed": len(game_records),
        "discarded_games": discarded_games,
        "partial_search_labels_rejected": partial_search_labels_rejected,
        "committed_examples": committed_examples,
        "decisions": total_decisions,
        "searched_decisions": total_searches,
        "search_fraction": total_searches / max(1, total_decisions),
        "search_flip_rate": action_flips / max(1, total_searches),
        "behavior_teacher_disagreement_rate": behavior_teacher_disagreements / max(1, total_searches),
        "average_simulations_per_search": total_simulations / max(1, total_searches),
        "average_search_latency_seconds": total_search_seconds / max(1, total_searches),
        "average_q_advantage_over_raw": total_q_advantage / max(1, total_searches),
        "distinct_trigger_state_buckets": len(searched_state_kinds),
        "searches_by_side": dict(side_searches),
        "trigger_counts": dict(trigger_counts),
        "terminal_reasons": dict(terminal_reasons),
        "elapsed_seconds": time.perf_counter() - started,
        "replay_manifest": str(manifest),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    game_log_paths = sorted(log_dir.glob("*.jsonl"))
    adjudication_positions = [
        {
            **candidate,
            "game_log": str(game_log_paths[0]) if game_log_paths else None,
            "game_occurrence": 0,
        }
        for candidate in deep_candidates
    ]
    (args.output_dir / "deep_adjudication_candidates.json").write_text(
        json.dumps({"positions": adjudication_positions}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
