#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.mcts import SelectivePuctConfig, SelectivePuctPlanner, TorchPolicyValueEvaluator
from struggle_ai.search_replay import SearchReplayWriter, examples_from_decision
from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import SHARED_POLICY_ID, compute_algo_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired, side-swapped evaluation of two raw/search agents.")
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--opponent-label", default="opponent")
    parser.add_argument("--candidate-search", action="store_true")
    parser.add_argument("--opponent-search", action="store_true")
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-pairs", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=82_000_000)
    parser.add_argument("--initial-simulations", type=int, default=32)
    parser.add_argument("--medium-simulations", type=int, default=32)
    parser.add_argument("--max-simulations", type=int, default=32)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--meaningful-depth", type=int, default=2)
    parser.add_argument("--max-atomic-steps", type=int, default=512)
    parser.add_argument("--forced-response-depth", type=int, default=1)
    parser.add_argument("--child-initial", type=int, default=4)
    parser.add_argument("--child-max", type=int, default=24)
    parser.add_argument("--disable-trigger", action="append", default=[])
    parser.add_argument("--low-confidence-sample-rate", type=float, default=0.05)
    parser.add_argument("--allocation-min-remaining", type=int, default=3)
    parser.add_argument("--search-setup-allocations", action="store_true")
    parser.add_argument("--max-scoring-searches-per-side-turn", type=int, default=1)
    parser.add_argument("--max-exact-chance-outcomes", type=int, default=36)
    parser.add_argument("--max-expanded-nodes", type=int, default=512)
    parser.add_argument("--max-chance-searches-per-side-round", type=int, default=1)
    parser.add_argument("--min-q-advantage-to-flip", type=float, default=0.0)
    parser.add_argument("--medium-visit-share-threshold", type=float, default=0.62)
    parser.add_argument("--medium-q-gap-threshold", type=float, default=0.05)
    parser.add_argument("--max-visit-share-threshold", type=float, default=0.55)
    parser.add_argument("--max-q-gap-threshold", type=float, default=0.03)
    parser.add_argument(
        "--escalate-on-disagreement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--num-gpus-per-checkpoint", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--search-replay-dir", type=Path, default=None)
    parser.add_argument("--search-replay-only-flips", action="store_true")
    parser.add_argument("--search-replay-min-q-advantage", type=float, default=0.0)
    return parser.parse_args()


def algorithm_args(checkpoint: Path, num_gpus: float) -> SimpleNamespace:
    return SimpleNamespace(
        restore_from=checkpoint.expanduser().resolve(),
        num_gpus=num_gpus,
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


@dataclass
class Agent:
    label: str
    algo: Any
    planner: SelectivePuctPlanner | None
    force_search: bool
    replay_writer: SearchReplayWriter | None = None
    replay_only_flips: bool = False
    replay_min_q_advantage: float = 0.0
    decision_count: int = 0

    def action(self, env: TwilightStruggleEnv, obs: dict[str, np.ndarray]) -> tuple[int, dict[str, Any] | None]:
        self.decision_count += 1
        if self.planner is None:
            return compute_algo_action(self.algo, obs, SHARED_POLICY_ID), None
        legal_actions = [dict(action) for action in env.legal_actions]
        decision = self.planner.plan(env.snapshot(compact=True), obs, force_search=self.force_search)
        replay_selected = decision.searched and (
            (not self.replay_only_flips or decision.action_flipped)
            and decision.q_advantage_over_raw >= self.replay_min_q_advantage
        )
        if replay_selected and self.replay_writer is not None:
            for example in examples_from_decision(
                    obs,
                    legal_actions,
                    str((env.last_obs or {}).get("side") or "ussr"),
                    decision,
                    search_depth=decision.meaningful_depth,
                    source_game=f"seed-{env.episode_seed}-{self.label}-{(env.last_obs or {}).get('side')}",
                    source_step=env.episode_step,
                ):
                self.replay_writer.add(example)
        event = None
        if decision.searched or "search_continuation" in decision.triggers:
            event = {
                "episode_step": env.episode_step,
                "turn": int((env.last_obs or {}).get("turn") or 0),
                "action_round": int((env.last_obs or {}).get("action_round") or 0),
                "side": str((env.last_obs or {}).get("side") or ""),
                "triggers": list(decision.triggers),
                "simulations": decision.simulations,
                "raw_action": decision.raw_action,
                "search_action": decision.action,
                "raw_action_text": str(env.legal_actions[decision.raw_action]),
                "search_action_text": str(env.legal_actions[decision.action]),
                "action_flipped": decision.action_flipped,
                "root_q_gap": decision.root_q_gap,
                "q_advantage_over_raw": decision.q_advantage_over_raw,
                "expanded_nodes": decision.expanded_nodes,
                "exact_chance_nodes": decision.exact_chance_nodes,
                "sampled_large_chance_nodes": decision.sampled_large_chance_nodes,
                "q_gate_rejected": decision.q_gate_rejected,
                "atomic_horizon_cutoffs": decision.atomic_horizon_cutoffs,
                "node_budget_cutoffs": decision.node_budget_cutoffs,
                "max_meaningful_depth_reached": decision.max_meaningful_depth_reached,
                "max_turn_delta": decision.max_turn_delta,
                "cross_turn_leaf_evaluations": decision.cross_turn_leaf_evaluations,
                "latency_seconds": decision.latency_seconds,
                "hierarchical_path": list(decision.hierarchical_path),
            }
            if decision.searched:
                print({"event": "puct_decision", "agent": self.label, **event}, flush=True)
        return decision.action, event


def make_search_config(args: argparse.Namespace) -> SelectivePuctConfig:
    enabled = tuple(
        trigger
        for trigger in SelectivePuctConfig().enabled_triggers
        if trigger not in set(args.disable_trigger)
    )
    return SelectivePuctConfig(
        initial_simulations=args.initial_simulations,
        medium_simulations=args.medium_simulations,
        max_simulations=args.max_simulations,
        belief_particles=args.belief_particles,
        meaningful_depth=args.meaningful_depth,
        max_atomic_steps=args.max_atomic_steps,
        forced_response_depth=args.forced_response_depth,
        child_initial=args.child_initial,
        child_max=args.child_max,
        enabled_triggers=enabled,
        low_confidence_sample_rate=args.low_confidence_sample_rate,
        allocation_min_remaining=args.allocation_min_remaining,
        search_setup_allocations=args.search_setup_allocations,
        max_scoring_searches_per_side_turn=args.max_scoring_searches_per_side_turn,
        max_exact_chance_outcomes=args.max_exact_chance_outcomes,
        max_expanded_nodes=args.max_expanded_nodes,
        max_chance_searches_per_side_round=args.max_chance_searches_per_side_round,
        min_q_advantage_to_flip=args.min_q_advantage_to_flip,
        medium_visit_share_threshold=args.medium_visit_share_threshold,
        medium_q_gap_threshold=args.medium_q_gap_threshold,
        max_visit_share_threshold=args.max_visit_share_threshold,
        max_q_gap_threshold=args.max_q_gap_threshold,
        escalate_on_disagreement=args.escalate_on_disagreement,
    )


def play_game(
    args: argparse.Namespace,
    *,
    seed: int,
    orientation: str,
    candidate: Agent,
    opponent: Agent,
) -> dict[str, Any]:
    candidate_side = "us" if orientation == "candidate_us" else "ussr"
    agents = {candidate_side: candidate, "ussr" if candidate_side == "us" else "us": opponent}
    env = TwilightStruggleEnv(env_config(args, args.output_dir / "game_logs"))
    started = time.perf_counter()
    candidate_start_decisions = candidate.decision_count
    opponent_start_decisions = opponent.decision_count
    events: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    try:
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            side = str((env.last_obs or {}).get("side") or "ussr")
            agent = agents[side]
            action, event = agent.action(env, obs)
            if event is not None:
                events.append({"agent": agent.label, **event})
            obs, _reward, terminated, truncated, final_info = env.step(action)
        winner = str(final_info.get("winner") or "")
        candidate_score = (
            1.0 if winner == candidate_side
            else 0.0 if winner in {"us", "ussr"}
            else 0.5
        )
        return {
            "seed": seed,
            "orientation": orientation,
            "candidate_side": candidate_side,
            "winner": winner,
            "terminal_reason": final_info.get("terminal_reason"),
            "steps": env.episode_step,
            "wall_clock_seconds": time.perf_counter() - started,
            "search_events": events,
            "candidate_won": candidate_score == 1.0,
            "draw": candidate_score == 0.5,
            "candidate_score": candidate_score,
            "candidate_decisions": candidate.decision_count - candidate_start_decisions,
            "opponent_decisions": opponent.decision_count - opponent_start_decisions,
        }
    finally:
        env.close()


def elo_delta(score: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, score))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def wilson_score_interval(score_sum: float, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    proportion = float(score_sum) / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / games + z * z / (4.0 * games * games)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_hoeffding_interval(score: float, pairs: int, alpha: float = 0.05) -> tuple[float, float]:
    if pairs <= 0:
        return 0.0, 1.0
    radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * pairs))
    return max(0.0, score - radius), min(1.0, score + radius)


def bootstrap_pair_ci(records: list[dict[str, Any]], samples: int, seed: int) -> tuple[float, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[int(record["seed"])].append(float(record["candidate_score"]))
    pair_scores = np.asarray([np.mean(values) for _, values in sorted(grouped.items())], dtype=np.float64)
    if not len(pair_scores):
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    draws = rng.choice(pair_scores, size=(samples, len(pair_scores)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def trigger_summary(records: list[dict[str, Any]], candidate_label: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for record in records:
        for event in record.get("search_events", []):
            candidate_score = float(record["candidate_score"])
            searching_agent_score = candidate_score if event.get("agent") == candidate_label else 1.0 - candidate_score
            for trigger in event.get("triggers", []):
                grouped[str(trigger)].append((event, searching_agent_score))
    return {
        trigger: {
            "frequency": float(len(values)),
            "flip_rate": sum(bool(event.get("action_flipped")) for event, _ in values) / len(values),
            "average_root_q_gap": float(np.mean([event.get("root_q_gap", 0.0) for event, _ in values])),
            "average_q_advantage_over_raw": float(
                np.mean([event.get("q_advantage_over_raw", 0.0) for event, _ in values])
            ),
            "total_simulations": float(sum(event.get("simulations", 0) for event, _ in values)),
            "average_latency_seconds": float(np.mean([event.get("latency_seconds", 0.0) for event, _ in values])),
            "downstream_searching_agent_score": sum(score for _, score in values) / len(values),
        }
        for trigger, values in sorted(grouped.items())
    }


def summarize(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    score = float(np.mean([record["candidate_score"] for record in records])) if records else 0.0
    ci_low, ci_high = bootstrap_pair_ci(records, args.bootstrap_samples, args.seed_base + 991)
    score_sum = sum(float(record["candidate_score"]) for record in records)
    wilson_low, wilson_high = wilson_score_interval(score_sum, len(records))
    pair_count = len({record["seed"] for record in records})
    hoeffding_low, hoeffding_high = paired_hoeffding_interval(score, pair_count)
    by_side = {}
    for side in ("us", "ussr"):
        subset = [record for record in records if record["candidate_side"] == side]
        by_side[side] = {
            "games": len(subset),
            "wins": sum(record["candidate_won"] for record in subset),
            "score": float(np.mean([record["candidate_score"] for record in subset])) if subset else 0.0,
        }
    events = [event for record in records for event in record.get("search_events", [])]
    searched = [event for event in events if event.get("simulations", 0) > 0]
    return {
        "games": len(records),
        "seed_pairs": len({record["seed"] for record in records}),
        "wins": sum(record["candidate_won"] for record in records),
        "draws": sum(bool(record.get("draw")) for record in records),
        "score": score,
        "paired_bootstrap_ci95_low": ci_low,
        "paired_bootstrap_ci95_high": ci_high,
        "game_wilson_ci95_low": wilson_low,
        "game_wilson_ci95_high": wilson_high,
        "paired_hoeffding_ci95_low": hoeffding_low,
        "paired_hoeffding_ci95_high": hoeffding_high,
        "elo_delta": elo_delta(score),
        "elo_ci95_low": elo_delta(ci_low),
        "elo_ci95_high": elo_delta(ci_high),
        "by_candidate_side": by_side,
        "terminal_reason_distribution": dict(Counter(str(record.get("terminal_reason") or "unknown") for record in records)),
        "average_wall_clock_seconds_per_game": float(np.mean([record["wall_clock_seconds"] for record in records])) if records else 0.0,
        "average_searches_per_game": len(searched) / max(1, len(records)),
        "average_simulations_per_game": sum(event.get("simulations", 0) for event in events) / max(1, len(records)),
        "average_search_seconds_per_game": sum(event.get("latency_seconds", 0.0) for event in events) / max(1, len(records)),
        "fraction_candidate_decisions_searched": len(
            [event for event in searched if event.get("agent") == args.candidate_label]
        ) / max(1, sum(record.get("candidate_decisions", 0) for record in records)) if records else 0.0,
        "search_flip_rate": sum(bool(event.get("action_flipped")) for event in searched) / max(1, len(searched)),
        "triggers": trigger_summary(records, args.candidate_label),
    }


def write_outputs(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    payload = {
        "candidate": {
            "label": args.candidate_label,
            "checkpoint": str(args.candidate_checkpoint.expanduser().resolve()),
            "search": args.candidate_search,
        },
        "opponent": {
            "label": args.opponent_label,
            "checkpoint": str(args.opponent_checkpoint.expanduser().resolve()),
            "search": args.opponent_search,
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": summarize(args, records),
        "records": records,
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "games.jsonl"
    records = [json.loads(line) for line in records_path.open("r", encoding="utf-8")] if records_path.exists() else []
    completed = {(int(record["seed"]), str(record["orientation"])) for record in records}

    same_checkpoint = args.candidate_checkpoint.expanduser().resolve() == args.opponent_checkpoint.expanduser().resolve()
    candidate_algo = build_algorithm(algorithm_args(args.candidate_checkpoint, args.num_gpus_per_checkpoint))
    opponent_algo = candidate_algo if same_checkpoint else build_algorithm(
        algorithm_args(args.opponent_checkpoint, args.num_gpus_per_checkpoint)
    )
    config = make_search_config(args)
    candidate_search_env = TwilightStruggleEnv(env_config(args, None, search_mode=True)) if args.candidate_search else None
    opponent_search_env = TwilightStruggleEnv(env_config(args, None, search_mode=True)) if args.opponent_search else None
    replay_writer = SearchReplayWriter(args.search_replay_dir) if args.search_replay_dir else None
    candidate = Agent(
        args.candidate_label,
        candidate_algo,
        SelectivePuctPlanner(
            TorchPolicyValueEvaluator(candidate_algo.get_policy(SHARED_POLICY_ID)), candidate_search_env, config, seed=args.seed_base + 17
        ) if candidate_search_env else None,
        args.force_search,
        replay_writer,
        args.search_replay_only_flips,
        args.search_replay_min_q_advantage,
    )
    opponent = Agent(
        args.opponent_label,
        opponent_algo,
        SelectivePuctPlanner(
            TorchPolicyValueEvaluator(opponent_algo.get_policy(SHARED_POLICY_ID)), opponent_search_env, config, seed=args.seed_base + 37
        ) if opponent_search_env else None,
        args.force_search,
        replay_writer,
        args.search_replay_only_flips,
        args.search_replay_min_q_advantage,
    )
    try:
        for pair_index in range(args.seed_pairs):
            seed = args.seed_base + pair_index
            for orientation in ("candidate_us", "candidate_ussr"):
                if (seed, orientation) in completed:
                    continue
                record = play_game(
                    args,
                    seed=seed,
                    orientation=orientation,
                    candidate=candidate,
                    opponent=opponent,
                )
                records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if replay_writer is not None:
                    replay_writer.flush()
                write_outputs(args, records)
                print({"event": "paired_eval_game", **record, "search_events": len(record["search_events"])}, flush=True)
    finally:
        if replay_writer is not None:
            replay_writer.close()
        if candidate_search_env:
            candidate_search_env.close()
        if opponent_search_env:
            opponent_search_env.close()
        candidate_algo.stop()
        if opponent_algo is not candidate_algo:
            opponent_algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass
    write_outputs(args, records)
    print(json.dumps(summarize(args, records), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
