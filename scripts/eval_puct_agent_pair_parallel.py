#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import random
import resource
import sys
import time
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from struggle_ai.batched_inference import (
    STAT_COUNT,
    BatchedInferenceClient,
    inference_server_main,
    inference_stats_payload,
)
from struggle_ai.baselines.steam_builtin import BridgeProxySearchRunner, SteamProxyConfig
from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.mcts import (
    ConvergedAdaptivePuctConfig,
    ConvergedAdaptivePuctPlanner,
    SelectivePuctConfig,
    SelectivePuctPlanner,
)
from struggle_ai.runtime_limits import configure_search_recursion_limit
from struggle_ai.policies import choose_heuristic_action


def elo_delta(score: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, score))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def wilson_score_interval(score_sum: float, games: int) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    proportion = score_sum / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / games + z * z / (4.0 * games * games)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_hoeffding_interval(score: float, pairs: int) -> tuple[float, float]:
    if pairs <= 0:
        return 0.0, 1.0
    radius = math.sqrt(math.log(40.0) / (2.0 * pairs))
    return max(0.0, score - radius), min(1.0, score + radius)


def bootstrap_pair_ci(
    records: list[dict[str, Any]], samples: int, seed: int
) -> tuple[float, float]:
    grouped: dict[int, list[float]] = {}
    for record in records:
        grouped.setdefault(int(record["seed"]), []).append(float(record["candidate_score"]))
    pair_scores = np.asarray(
        [np.mean(values) for _, values in sorted(grouped.items())], dtype=np.float64
    )
    if not len(pair_scores):
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    draws = rng.choice(pair_scores, size=(samples, len(pair_scores)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel matched, side-swapped D2 evaluation with shared dual-GPU inference."
    )
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-us-policy-state", type=Path)
    parser.add_argument("--candidate-ussr-policy-state", type=Path)
    parser.add_argument(
        "--candidate-value-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used only for candidate leaf values.",
    )
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument(
        "--opponent-kind",
        choices=("checkpoint", "steam_proxy", "heuristic", "random"),
        default="checkpoint",
    )
    parser.add_argument(
        "--opponent-value-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used only for opponent leaf values.",
    )
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--opponent-label", default="opponent")
    parser.add_argument("--steam-search-depth", type=int, default=2)
    parser.add_argument("--steam-max-candidates", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-pairs", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=92_000_000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--heap-trim-every-decisions", type=int, default=8)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--replicas-per-checkpoint", type=int, default=2)
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    parser.add_argument("--max-inference-delay-ms", type=float, default=3.0)
    parser.add_argument("--inference-cache-capacity", type=int, default=65_536)
    parser.add_argument("--inference-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--initial-simulations", type=int, default=32)
    parser.add_argument(
        "--search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable to run batched raw-policy evaluation without constructing PUCT trees.",
    )
    parser.add_argument(
        "--candidate-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override --search for the candidate only.",
    )
    parser.add_argument(
        "--opponent-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override --search for the opponent only.",
    )
    parser.add_argument("--medium-simulations", type=int, default=32)
    parser.add_argument("--max-simulations", type=int, default=32)
    parser.add_argument("--meaningful-depth", type=int, default=2)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument(
        "--adaptive-converged-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use D2 budget/deeper search only when Q rankings converge; fail back shallow.",
    )
    parser.add_argument(
        "--candidate-adaptive-converged-search",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--opponent-adaptive-converged-search",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--adaptive-budget-simulations", type=int, default=128)
    parser.add_argument("--adaptive-deep-depth", type=int, default=4)
    parser.add_argument("--adaptive-deep-simulations", type=int, default=128)
    parser.add_argument("--adaptive-deep-max-expanded-nodes", type=int, default=262_144)
    parser.add_argument("--adaptive-min-rank-correlation", type=float, default=0.70)
    parser.add_argument("--adaptive-meaningful-q-gap", type=float, default=0.01)
    parser.add_argument("--adaptive-escalation-q-gap", type=float, default=0.03)
    parser.add_argument("--max-atomic-steps", type=int, default=512)
    parser.add_argument("--max-expanded-nodes", type=int, default=512)
    for model in ("candidate", "opponent"):
        parser.add_argument(f"--{model}-simulations", type=int, default=None)
        parser.add_argument(f"--{model}-meaningful-depth", type=int, default=None)
        parser.add_argument(f"--{model}-max-atomic-steps", type=int, default=None)
        parser.add_argument(f"--{model}-max-expanded-nodes", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--max-episode-steps", type=int, default=1_200)
    parser.add_argument(
        "--log-games",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write detailed real-game transcripts under output-dir/game_logs.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse identity-verified complete games from output-dir/progress.json.",
    )
    return parser.parse_args()


def resolve_search_modes(args: Any) -> dict[str, bool]:
    shared = bool(args.search)
    return {
        "candidate": shared if args.candidate_search is None else bool(args.candidate_search),
        "opponent": shared if args.opponent_search is None else bool(args.opponent_search),
    }


def checkpoint_paths(args: Any) -> dict[str, str]:
    side_states = (args.candidate_us_policy_state, args.candidate_ussr_policy_state)
    if any(side_states) and not all(side_states):
        raise ValueError("both --candidate-us-policy-state and --candidate-ussr-policy-state are required")
    if all(side_states):
        paths = {
            f"candidate_{role}_{side}": str(Path(state).expanduser().resolve())
            for side, state in zip(("us", "ussr"), side_states, strict=True)
            for role in ("policy", "value")
        }
    else:
        candidate_value = args.candidate_value_checkpoint or args.candidate_checkpoint
        paths = {
            "candidate_policy": str(Path(args.candidate_checkpoint).expanduser().resolve()),
            "candidate_value": str(Path(candidate_value).expanduser().resolve()),
        }
    if args.opponent_kind == "checkpoint":
        if args.opponent_checkpoint is None:
            raise ValueError("--opponent-checkpoint is required for --opponent-kind checkpoint")
        opponent_value = args.opponent_value_checkpoint or args.opponent_checkpoint
        paths.update({
            "opponent_policy": str(Path(args.opponent_checkpoint).expanduser().resolve()),
            "opponent_value": str(Path(opponent_value).expanduser().resolve()),
        })
    return paths


def proxy_observation(observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep the proxy root catalog identical to the environment's safety-filtered catalog."""
    return {**observation, "legal_actions": legal_actions}


def resolve_adaptive_modes(args: Any) -> dict[str, bool]:
    shared = bool(args.adaptive_converged_search)
    return {
        "candidate": (
            shared
            if args.candidate_adaptive_converged_search is None
            else bool(args.candidate_adaptive_converged_search)
        ),
        "opponent": (
            shared
            if args.opponent_adaptive_converged_search is None
            else bool(args.opponent_adaptive_converged_search)
        ),
    }


def env_config(args: Any, *, search_mode: bool) -> dict[str, Any]:
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
        "max_episode_steps": int(args.max_episode_steps),
        "search_mode": bool(search_mode),
        "log_games_dir": None,
        "log_action_details": False,
        "metrics_dir": None,
    }


def resolved_search_config(args: Any, model: str) -> dict[str, int]:
    simulations = getattr(args, f"{model}_simulations", None)
    depth = getattr(args, f"{model}_meaningful_depth", None)
    atomic_steps = getattr(args, f"{model}_max_atomic_steps", None)
    expanded_nodes = getattr(args, f"{model}_max_expanded_nodes", None)
    return {
        "initial_simulations": int(args.initial_simulations if simulations is None else simulations),
        "medium_simulations": int(args.medium_simulations if simulations is None else simulations),
        "max_simulations": int(args.max_simulations if simulations is None else simulations),
        "meaningful_depth": int(args.meaningful_depth if depth is None else depth),
        "max_atomic_steps": int(args.max_atomic_steps if atomic_steps is None else atomic_steps),
        "max_expanded_nodes": int(
            args.max_expanded_nodes if expanded_nodes is None else expanded_nodes
        ),
    }


def search_config(args: Any, model: str = "candidate") -> SelectivePuctConfig:
    resolved = resolved_search_config(args, model)
    return SelectivePuctConfig(
        initial_simulations=resolved["initial_simulations"],
        medium_simulations=resolved["medium_simulations"],
        max_simulations=resolved["max_simulations"],
        belief_particles=int(args.belief_particles),
        meaningful_depth=resolved["meaningful_depth"],
        max_atomic_steps=resolved["max_atomic_steps"],
        max_expanded_nodes=resolved["max_expanded_nodes"],
        behavior_temperature=0.0,
    )


def adaptive_search_configs(
    args: Any, model: str
) -> tuple[SelectivePuctConfig, SelectivePuctConfig, SelectivePuctConfig]:
    base = search_config(args, model)
    budget_simulations = int(args.adaptive_budget_simulations)
    deep_simulations = int(args.adaptive_deep_simulations)
    budget = replace(
        base,
        initial_simulations=budget_simulations,
        medium_simulations=budget_simulations,
        max_simulations=budget_simulations,
    )
    deep = replace(
        base,
        initial_simulations=deep_simulations,
        medium_simulations=deep_simulations,
        max_simulations=deep_simulations,
        meaningful_depth=int(args.adaptive_deep_depth),
        max_expanded_nodes=int(args.adaptive_deep_max_expanded_nodes),
    )
    return base, budget, deep


def write_worker_heartbeat(
    output_dir: Path,
    worker_id: int,
    *,
    seed: int,
    orientation: str,
    candidate_side: str,
    side: str,
    model: str,
    stage: str,
    episode_step: int,
    observation: dict[str, Any] | None,
    search_seconds: float | None = None,
) -> None:
    heartbeat_dir = output_dir / "worker_heartbeats"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "worker_id": int(worker_id),
        "pid": os.getpid(),
        "seed": int(seed),
        "orientation": orientation,
        "candidate_side": candidate_side,
        "side": side,
        "model": model,
        "stage": stage,
        "episode_step": int(episode_step),
        "turn": int((observation or {}).get("turn") or 0),
        "action_round": int((observation or {}).get("action_round") or 0),
        "phase": str((observation or {}).get("phase") or ""),
        "prompt": str((observation or {}).get("prompt") or "")[:256],
        "rss_max_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    if search_seconds is not None:
        payload["search_seconds"] = float(search_seconds)
    path = heartbeat_dir / f"worker-{worker_id:03d}.json"
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _worker_main(
    worker_id: int,
    args_payload: dict[str, Any],
    task_queue: Any,
    request_queues: dict[str, Any],
    response_queues: dict[str, Any],
    result_queue: Any,
) -> None:
    args = SimpleNamespace(**args_payload)
    atomic_horizons = [int(args.max_atomic_steps)]
    for model in ("candidate", "opponent"):
        configured = getattr(args, f"{model}_max_atomic_steps", None)
        if configured is not None:
            atomic_horizons.append(int(configured))
    configure_search_recursion_limit(max(atomic_horizons))
    canonical_clients = {
        model: BatchedInferenceClient(
            worker_id=worker_id,
            request_queue=request_queue,
            response_queue=response_queues[model],
            timeout_seconds=float(args.inference_timeout_seconds),
        )
        for model, request_queue in request_queues.items()
    }
    clients = {
        model: canonical_clients[canonical]
        for model, canonical in args.canonical_model.items()
    }

    class SplitPolicyValueEvaluator:
        def __init__(self, policy_client: Any, value_client: Any) -> None:
            self.policy_client = policy_client
            self.value_client = value_client

        def raw(self, obs: dict[str, np.ndarray], legal_count: int) -> dict[str, Any]:
            policy = self.policy_client.raw(obs, legal_count)
            value = self.value_client.raw(obs, legal_count)
            return {
                "logits": policy["logits"],
                "probabilities": policy["probabilities"],
                "raw_value": float(value["raw_value"]),
                "calibrated_value": float(value["calibrated_value"]),
            }

        def __call__(self, obs: dict[str, np.ndarray], legal_count: int) -> tuple[np.ndarray, float]:
            result = self.raw(obs, legal_count)
            return result["probabilities"], float(result["raw_value"])

        def value_us(self, obs: dict[str, np.ndarray], side: str) -> tuple[float, float]:
            return self.value_client.value_us(obs, side)

    evaluators = {
        (model, side): SplitPolicyValueEvaluator(
            clients.get(f"{model}_policy_{side}", clients.get(f"{model}_policy")),
            clients.get(f"{model}_value_{side}", clients.get(f"{model}_value")),
        )
        for model in ("candidate", "opponent")
        for side in ("us", "ussr")
        if f"{model}_policy_{side}" in clients or f"{model}_policy" in clients
    }
    checkpoint_agents = tuple(sorted({model for model, _side in evaluators}))
    configs = {model: search_config(args, model) for model in checkpoint_agents}
    game_env_config = env_config(args, search_mode=False)
    if bool(args.log_games):
        game_env_config["log_games_dir"] = str(
            Path(args.output_dir) / "game_logs" / f"worker-{worker_id:03d}"
        )
        game_env_config["log_action_details"] = True
    game_env = TwilightStruggleEnv(game_env_config)
    search_envs = {
        (model, side): TwilightStruggleEnv(env_config(args, search_mode=True))
        for model in checkpoint_agents
        for side in ("us", "ussr")
    }
    planners: dict[tuple[str, str], Any] = {}
    for model in checkpoint_agents:
        for side in ("us", "ussr"):
            planner_seed = (
                int(args.seed_base)
                + worker_id * 100_003
                + (0 if model == "candidate" else 10_003)
                + (0 if side == "us" else 101)
            )
            if resolve_adaptive_modes(args)[model]:
                base_config, budget_config, deep_config = adaptive_search_configs(args, model)
                component_planners = [
                    SelectivePuctPlanner(
                        evaluators[(model, side)],
                        search_envs[(model, side)],
                        config,
                        seed=planner_seed,
                    )
                    for config in (base_config, budget_config, deep_config)
                ]
                planners[(model, side)] = ConvergedAdaptivePuctPlanner(
                    *component_planners,
                    ConvergedAdaptivePuctConfig(
                        min_rank_correlation=float(args.adaptive_min_rank_correlation),
                        meaningful_q_gap=float(args.adaptive_meaningful_q_gap),
                        escalation_q_gap=float(args.adaptive_escalation_q_gap),
                    ),
                    seed=planner_seed,
                )
            else:
                planners[(model, side)] = SelectivePuctPlanner(
                    evaluators[(model, side)],
                    search_envs[(model, side)],
                    configs[model],
                    seed=planner_seed,
                )
    search_by_model = resolve_search_modes(args)
    opponent_rng = random.Random(int(args.seed_base) + worker_id * 100_003 + 29)
    steam_runners = {
        side: BridgeProxySearchRunner(
            game_env.bridge,
            SteamProxyConfig(
                seed=int(args.seed_base) + worker_id * 100_003 + (0 if side == "us" else 101),
                search_state_depth=int(args.steam_search_depth),
                max_candidates=int(args.steam_max_candidates),
            ),
        )
        for side in ("us", "ussr")
    } if args.opponent_kind == "steam_proxy" else {}
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            seed = int(task["seed"])
            orientation = str(task["orientation"])
            candidate_side = "us" if orientation == "candidate_us" else "ussr"
            model_by_side = {
                candidate_side: "candidate",
                "ussr" if candidate_side == "us" else "us": "opponent",
            }
            for planner in planners.values():
                if isinstance(planner, ConvergedAdaptivePuctPlanner):
                    planner.reset()
                else:
                    planner.pending_continuation.clear()
            started = time.perf_counter()
            searched = simulations = flips = 0
            atomic_cutoffs = node_cutoffs = 0
            candidate_searched = candidate_simulations = candidate_flips = 0
            candidate_atomic_cutoffs = candidate_node_cutoffs = 0
            opponent_searched = opponent_simulations = opponent_flips = 0
            opponent_atomic_cutoffs = opponent_node_cutoffs = 0
            convergence_statuses: Counter[str] = Counter()
            adaptive_fallbacks = 0
            obs, _ = game_env.reset(seed=seed)
            terminated = truncated = False
            final_info: dict[str, Any] = {}
            write_worker_heartbeat(
                Path(args.output_dir),
                worker_id,
                seed=seed,
                orientation=orientation,
                candidate_side=candidate_side,
                side=str((game_env.last_obs or {}).get("side") or "ussr"),
                model="game",
                stage="game_start",
                episode_step=game_env.episode_step,
                observation=game_env.last_obs,
            )
            while not (terminated or truncated):
                side = str((game_env.last_obs or {}).get("side") or "ussr")
                model = model_by_side[side]
                planner = planners.get((model, side))
                if model == "opponent" and args.opponent_kind == "steam_proxy":
                    search_started = time.perf_counter()
                    proxy_decision = steam_runners[side].choose_action(proxy_observation(
                        game_env.last_obs or {}, game_env.legal_actions
                    ))
                    action = int(proxy_decision.action_index)
                    opponent_searched += 1
                    opponent_simulations += int(proxy_decision.expanded_nodes)
                    searched += 1
                    simulations += int(proxy_decision.expanded_nodes)
                    convergence_statuses["steam_reconstructed_proxy"] += 1
                    write_worker_heartbeat(
                        Path(args.output_dir),
                        worker_id,
                        seed=seed,
                        orientation=orientation,
                        candidate_side=candidate_side,
                        side=side,
                        model=model,
                        stage="steam_proxy_complete",
                        episode_step=game_env.episode_step,
                        observation=game_env.last_obs,
                        search_seconds=time.perf_counter() - search_started,
                    )
                elif model == "opponent" and args.opponent_kind == "heuristic":
                    action = int(choose_heuristic_action(
                        game_env.legal_actions, opponent_rng, game_env.last_obs or {}
                    ))
                elif model == "opponent" and args.opponent_kind == "random":
                    action = int(opponent_rng.randrange(len(game_env.legal_actions)))
                elif search_by_model[model]:
                    write_worker_heartbeat(
                        Path(args.output_dir),
                        worker_id,
                        seed=seed,
                        orientation=orientation,
                        candidate_side=candidate_side,
                        side=side,
                        model=model,
                        stage="search_start",
                        episode_step=game_env.episode_step,
                        observation=game_env.last_obs,
                    )
                    search_started = time.perf_counter()
                    assert planner is not None
                    decision = planner.plan(game_env.snapshot(compact=True), obs)
                    write_worker_heartbeat(
                        Path(args.output_dir),
                        worker_id,
                        seed=seed,
                        orientation=orientation,
                        candidate_side=candidate_side,
                        side=side,
                        model=model,
                        stage="search_complete",
                        episode_step=game_env.episode_step,
                        observation=game_env.last_obs,
                        search_seconds=time.perf_counter() - search_started,
                    )
                    action = int(decision.action)
                    searched += int(decision.searched)
                    simulations += int(decision.simulations)
                    flips += int(decision.action_flipped)
                    atomic_cutoffs += int(decision.atomic_horizon_cutoffs)
                    node_cutoffs += int(decision.node_budget_cutoffs)
                    if decision.convergence_status:
                        convergence_statuses[str(decision.convergence_status)] += 1
                    adaptive_fallbacks += int(decision.adaptive_fallback)
                    if model == "candidate":
                        candidate_searched += int(decision.searched)
                        candidate_simulations += int(decision.simulations)
                        candidate_flips += int(decision.action_flipped)
                        candidate_atomic_cutoffs += int(decision.atomic_horizon_cutoffs)
                        candidate_node_cutoffs += int(decision.node_budget_cutoffs)
                    else:
                        opponent_searched += int(decision.searched)
                        opponent_simulations += int(decision.simulations)
                        opponent_flips += int(decision.action_flipped)
                        opponent_atomic_cutoffs += int(decision.atomic_horizon_cutoffs)
                        opponent_node_cutoffs += int(decision.node_budget_cutoffs)
                else:
                    legal_count = len(game_env.legal_actions)
                    action = int(np.argmax(evaluators[(model, side)].raw(obs, legal_count)["probabilities"]))
                obs, _, terminated, truncated, final_info = game_env.step(action)
                if terminated or truncated or game_env.episode_step % 8 == 0:
                    write_worker_heartbeat(
                        Path(args.output_dir),
                        worker_id,
                        seed=seed,
                        orientation=orientation,
                        candidate_side=candidate_side,
                        side=side,
                        model=model,
                        stage="game_complete" if terminated or truncated else "step_complete",
                        episode_step=game_env.episode_step,
                        observation=game_env.last_obs,
                    )
                trim_every = max(0, int(args.heap_trim_every_decisions))
                if trim_every and game_env.episode_step % trim_every == 0:
                    gc.collect()
                    try:
                        ctypes.CDLL(None).malloc_trim(0)
                    except (AttributeError, OSError):
                        pass
            winner = str(final_info.get("winner") or "")
            score = 1.0 if winner == candidate_side else 0.0 if winner in {"us", "ussr"} else 0.5
            result_queue.put(
                {
                    "event": "game_complete",
                    "record": {
                        "seed": seed,
                        "orientation": orientation,
                        "candidate_side": candidate_side,
                        "winner": winner,
                        "terminal_reason": str(final_info.get("terminal_reason") or "unknown"),
                        "steps": int(game_env.episode_step),
                        "candidate_score": score,
                        "candidate_won": score == 1.0,
                        "draw": score == 0.5,
                        "searched_decisions": searched,
                        "simulations": simulations,
                        "action_flips": flips,
                        "atomic_horizon_cutoffs": atomic_cutoffs,
                        "node_budget_cutoffs": node_cutoffs,
                        "candidate_searched_decisions": candidate_searched,
                        "candidate_simulations": candidate_simulations,
                        "candidate_action_flips": candidate_flips,
                        "candidate_atomic_horizon_cutoffs": candidate_atomic_cutoffs,
                        "candidate_node_budget_cutoffs": candidate_node_cutoffs,
                        "opponent_searched_decisions": opponent_searched,
                        "opponent_simulations": opponent_simulations,
                        "opponent_action_flips": opponent_flips,
                        "opponent_atomic_horizon_cutoffs": opponent_atomic_cutoffs,
                        "opponent_node_budget_cutoffs": opponent_node_cutoffs,
                        "convergence_statuses": dict(convergence_statuses),
                        "adaptive_fallbacks": adaptive_fallbacks,
                        "wall_clock_seconds": time.perf_counter() - started,
                    },
                }
            )
    except BaseException as exc:
        result_queue.put(
            {
                "event": "worker_error",
                "worker_id": worker_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "recursion_limit": sys.getrecursionlimit(),
            }
        )
        raise
    finally:
        game_env.close()
        for env in search_envs.values():
            env.close()
        result_queue.put({"event": "worker_done", "worker_id": worker_id})


def summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    score = float(np.mean([record["candidate_score"] for record in records]))
    ci_low, ci_high = bootstrap_pair_ci(records, args.bootstrap_samples, args.seed_base + 991)
    wilson_low, wilson_high = wilson_score_interval(
        sum(float(record["candidate_score"]) for record in records), len(records)
    )
    hoeffding_low, hoeffding_high = paired_hoeffding_interval(score, args.seed_pairs)
    by_side = {}
    for side in ("us", "ussr"):
        selected = [record for record in records if record["candidate_side"] == side]
        by_side[side] = {
            "games": len(selected),
            "wins": sum(bool(record["candidate_won"]) for record in selected),
            "score": float(np.mean([record["candidate_score"] for record in selected])),
        }
    simulations = sum(int(record["simulations"]) for record in records)
    searched = sum(int(record["searched_decisions"]) for record in records)
    candidate_simulations = sum(int(record.get("candidate_simulations", 0)) for record in records)
    candidate_searched = sum(
        int(record.get("candidate_searched_decisions", 0)) for record in records
    )
    candidate_flips = sum(int(record.get("candidate_action_flips", 0)) for record in records)
    opponent_simulations = sum(int(record.get("opponent_simulations", 0)) for record in records)
    opponent_searched = sum(
        int(record.get("opponent_searched_decisions", 0)) for record in records
    )
    opponent_flips = sum(int(record.get("opponent_action_flips", 0)) for record in records)
    atomic_cutoffs = sum(int(record.get("atomic_horizon_cutoffs", 0)) for record in records)
    node_cutoffs = sum(int(record.get("node_budget_cutoffs", 0)) for record in records)
    candidate_atomic_cutoffs = sum(
        int(record.get("candidate_atomic_horizon_cutoffs", 0)) for record in records
    )
    candidate_node_cutoffs = sum(
        int(record.get("candidate_node_budget_cutoffs", 0)) for record in records
    )
    opponent_atomic_cutoffs = sum(
        int(record.get("opponent_atomic_horizon_cutoffs", 0)) for record in records
    )
    opponent_node_cutoffs = sum(
        int(record.get("opponent_node_budget_cutoffs", 0)) for record in records
    )
    convergence_statuses: Counter[str] = Counter()
    for record in records:
        convergence_statuses.update(record.get("convergence_statuses") or {})
    adaptive_fallbacks = sum(int(record.get("adaptive_fallbacks", 0)) for record in records)
    return {
        "games": len(records),
        "seed_pairs": args.seed_pairs,
        "wins": sum(bool(record["candidate_won"]) for record in records),
        "draws": sum(bool(record["draw"]) for record in records),
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
        "terminal_reason_distribution": dict(Counter(record["terminal_reason"] for record in records)),
        "average_wall_clock_seconds_per_game": float(np.mean([record["wall_clock_seconds"] for record in records])),
        "average_searches_per_game": searched / len(records),
        "average_simulations_per_game": simulations / len(records),
        "search_flip_rate": sum(int(record["action_flips"]) for record in records) / max(1, searched),
        "atomic_horizon_cutoffs": atomic_cutoffs,
        "node_budget_cutoffs": node_cutoffs,
        "cutoff_rate_per_search": (atomic_cutoffs + node_cutoffs) / max(1, searched),
        "candidate_average_searches_per_game": candidate_searched / len(records),
        "candidate_average_simulations_per_game": candidate_simulations / len(records),
        "candidate_search_flip_rate": candidate_flips / max(1, candidate_searched),
        "candidate_atomic_horizon_cutoffs": candidate_atomic_cutoffs,
        "candidate_node_budget_cutoffs": candidate_node_cutoffs,
        "opponent_average_searches_per_game": opponent_searched / len(records),
        "opponent_average_simulations_per_game": opponent_simulations / len(records),
        "opponent_search_flip_rate": opponent_flips / max(1, opponent_searched),
        "opponent_atomic_horizon_cutoffs": opponent_atomic_cutoffs,
        "opponent_node_budget_cutoffs": opponent_node_cutoffs,
        "convergence_statuses": dict(convergence_statuses),
        "adaptive_fallbacks": adaptive_fallbacks,
        "adaptive_fallback_rate": adaptive_fallbacks / max(1, searched),
    }


def write_progress_snapshot(
    output_dir: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    identity: dict[str, Any] | None = None,
) -> None:
    """Persist completed games without presenting an interrupted gate as final."""
    ordered = sorted(records, key=lambda record: (int(record["seed"]), str(record["orientation"])))
    scores = [float(record["candidate_score"]) for record in ordered]
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in ordered:
        records_by_seed.setdefault(int(record["seed"]), []).append(record)
    complete_pair_scores = [
        float(np.mean([float(record["candidate_score"]) for record in pair]))
        for pair in records_by_seed.values()
        if {str(record["orientation"]) for record in pair}
        == {"candidate_us", "candidate_ussr"}
    ]
    by_side: dict[str, dict[str, float | int]] = {}
    for side in ("us", "ussr"):
        side_scores = [
            float(record["candidate_score"])
            for record in ordered
            if record["candidate_side"] == side
        ]
        by_side[side] = {
            "games": len(side_scores),
            "score": float(np.mean(side_scores)) if side_scores else 0.0,
        }
    payload = {
        "status": "partial",
        "expected_games": int(args.seed_pairs) * 2,
        "completed_games": len(ordered),
        "score": float(np.mean(scores)) if scores else 0.0,
        "complete_seed_pairs": len(complete_pair_scores),
        "complete_pair_score": float(np.mean(complete_pair_scores))
        if complete_pair_scores
        else None,
        "by_candidate_side": by_side,
        "terminal_reason_distribution": dict(
            Counter(str(record["terminal_reason"]) for record in ordered)
        ),
        "average_wall_clock_seconds_per_completed_game": float(
            np.mean([float(record.get("wall_clock_seconds", 0.0)) for record in ordered])
        ) if ordered else 0.0,
        "average_searches_per_completed_game": float(
            np.mean([float(record.get("searched_decisions", 0.0)) for record in ordered])
        ) if ordered else 0.0,
        "average_simulations_per_completed_game": float(
            np.mean([float(record.get("simulations", 0.0)) for record in ordered])
        ) if ordered else 0.0,
        "search_flip_rate": sum(int(record.get("action_flips", 0)) for record in ordered)
        / max(1, sum(int(record.get("searched_decisions", 0)) for record in ordered)),
        "atomic_horizon_cutoffs": sum(
            int(record.get("atomic_horizon_cutoffs", 0)) for record in ordered
        ),
        "node_budget_cutoffs": sum(
            int(record.get("node_budget_cutoffs", 0)) for record in ordered
        ),
        "records": ordered,
        "identity": identity,
    }
    target = output_dir / "progress.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def worker_error_detail(event: dict[str, Any]) -> str:
    detail = f"worker {event['worker_id']}: {event['error']}"
    if event.get("traceback"):
        detail += f"\n{event['traceback']}"
    if event.get("recursion_limit"):
        detail += f"\nworker_recursion_limit={event['recursion_limit']}"
    return detail


def write_failure(output_dir: Path, error: str) -> None:
    (output_dir / "failure.json").write_text(
        json.dumps({"status": "failed", "error": error, "timestamp": time.time()}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_resumable_records(
    output_dir: Path,
    args: argparse.Namespace,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    progress_path = output_dir / "progress.json"
    if not args.resume or not progress_path.is_file():
        return []
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    if payload.get("identity") != identity:
        raise ValueError(
            f"cannot resume {progress_path}: evaluation identity is missing or different"
        )
    expected = {
        (int(args.seed_base) + offset, orientation)
        for offset in range(int(args.seed_pairs))
        for orientation in ("candidate_us", "candidate_ussr")
    }
    records = list(payload.get("records") or [])
    keys = [(int(record["seed"]), str(record["orientation"])) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"cannot resume {progress_path}: duplicate game records")
    unexpected = sorted(set(keys) - expected)
    if unexpected:
        raise ValueError(f"cannot resume {progress_path}: unexpected tasks {unexpected}")
    return records


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpus = [part.strip() for part in args.gpus.split(",") if part.strip()]
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    if args.opponent_kind != "checkpoint":
        args.opponent_search = False
        args.opponent_adaptive_converged_search = False
    checkpoint_by_model = checkpoint_paths(args)
    candidate_models = {
        key.removeprefix("candidate_"): value
        for key, value in checkpoint_by_model.items()
        if key.startswith("candidate_")
    }
    evaluation_identity = {
        "candidate_models": candidate_models,
        "opponent_kind": args.opponent_kind,
        "opponent_policy": checkpoint_by_model.get("opponent_policy"),
        "opponent_value": checkpoint_by_model.get("opponent_value"),
        "steam_search_depth": int(args.steam_search_depth),
        "steam_max_candidates": int(args.steam_max_candidates),
        "seed_base": int(args.seed_base),
        "seed_pairs": int(args.seed_pairs),
        "search": bool(args.search),
        **resolve_search_modes(args),
        "initial_simulations": int(args.initial_simulations),
        "medium_simulations": int(args.medium_simulations),
        "max_simulations": int(args.max_simulations),
        "meaningful_depth": int(args.meaningful_depth),
        "belief_particles": int(args.belief_particles),
        "max_atomic_steps": int(args.max_atomic_steps),
        "max_expanded_nodes": int(args.max_expanded_nodes),
        "candidate_search_config": resolved_search_config(args, "candidate"),
        "opponent_search_config": resolved_search_config(args, "opponent"),
        "adaptive_converged_search": bool(args.adaptive_converged_search),
        **{f"adaptive_{key}": value for key, value in resolve_adaptive_modes(args).items()},
        "adaptive_budget_simulations": int(args.adaptive_budget_simulations),
        "adaptive_deep_depth": int(args.adaptive_deep_depth),
        "adaptive_deep_simulations": int(args.adaptive_deep_simulations),
        "adaptive_deep_max_expanded_nodes": int(args.adaptive_deep_max_expanded_nodes),
        "adaptive_min_rank_correlation": float(args.adaptive_min_rank_correlation),
        "adaptive_meaningful_q_gap": float(args.adaptive_meaningful_q_gap),
        "adaptive_escalation_q_gap": float(args.adaptive_escalation_q_gap),
        "max_episode_steps": int(args.max_episode_steps),
        "log_games": bool(args.log_games),
    }
    records = load_resumable_records(args.output_dir, args, evaluation_identity)
    completed_keys = {
        (int(record["seed"]), str(record["orientation"])) for record in records
    }
    remaining_tasks = int(args.seed_pairs) * 2 - len(completed_keys)
    worker_count = min(max(1, int(args.workers)), max(1, remaining_tasks))
    canonical_model: dict[str, str] = {}
    for model, checkpoint in checkpoint_by_model.items():
        canonical_model[model] = next(
            (
                existing
                for existing, existing_checkpoint in checkpoint_by_model.items()
                if existing < model and existing_checkpoint == checkpoint
            ),
            model,
        )
    unique_models = sorted(set(canonical_model.values()))
    response_queues = {
        model: [context.Queue(maxsize=8) for _ in range(worker_count)]
        for model in unique_models
    }
    request_queues = {
        model: context.Queue(maxsize=max(256, worker_count * 32))
        for model in unique_models
    }
    control_queue = context.Queue()
    servers: list[Any] = []
    server_stats: list[tuple[str, str, Any]] = []
    for model in unique_models:
        for replica in range(max(1, int(args.replicas_per_checkpoint))):
            gpu = gpus[len(servers) % len(gpus)]
            stats = context.Array("d", STAT_COUNT, lock=False)
            server_id = len(servers)
            server = context.Process(
                target=inference_server_main,
                kwargs={
                    "server_id": server_id,
                    "gpu": gpu,
                    "checkpoint": checkpoint_by_model[model],
                    "request_queue": request_queues[model],
                    "response_queues": response_queues[model],
                    "control_queue": control_queue,
                    "stats": stats,
                    "max_batch_size": args.max_inference_batch_size,
                    "max_batch_delay_ms": args.max_inference_delay_ms,
                    "cache_capacity": args.inference_cache_capacity,
                },
                name=f"{model}-eval-inference-gpu-{gpu}",
            )
            servers.append(server)
            server_stats.append((model, gpu, stats))
    for server in servers:
        server.start()
    ready = 0
    deadline = time.monotonic() + 300.0
    while ready < len(servers):
        event = control_queue.get(timeout=max(0.1, deadline - time.monotonic()))
        print(event, flush=True)
        if event["event"] == "inference_server_error":
            write_failure(args.output_dir, str(event["error"]))
            raise RuntimeError(event["error"])
        ready += int(event["event"] == "inference_server_ready")

    for seed_offset in range(args.seed_pairs):
        seed = args.seed_base + seed_offset
        for orientation in ("candidate_us", "candidate_ussr"):
            if (seed, orientation) not in completed_keys:
                task_queue.put({"seed": seed, "orientation": orientation})
    for _ in range(worker_count):
        task_queue.put(None)
    args_payload = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    args_payload["canonical_model"] = canonical_model
    workers = [
        context.Process(
            target=_worker_main,
            args=(
                index,
                args_payload,
                task_queue,
                {
                    model: request_queues[model]
                    for model in unique_models
                },
                {
                    model: response_queues[model][index]
                    for model in unique_models
                },
                result_queue,
            ),
            name=f"paired-puct-eval-{index:03d}",
        )
        for index in range(worker_count)
    ]
    for worker in workers:
        worker.start()
    errors: list[str] = []
    try:
        while any(worker.is_alive() for worker in workers):
            try:
                event = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if event["event"] == "game_complete":
                records.append(event["record"])
                write_progress_snapshot(args.output_dir, args, records, evaluation_identity)
                print({"event": "parallel_paired_eval_game", **event["record"]}, flush=True)
            elif event["event"] == "worker_error":
                errors.append(worker_error_detail(event))
        for worker in workers:
            worker.join(timeout=5)
        while True:
            try:
                event = result_queue.get_nowait()
            except queue.Empty:
                break
            if event["event"] == "game_complete":
                records.append(event["record"])
                write_progress_snapshot(args.output_dir, args, records, evaluation_identity)
            elif event["event"] == "worker_error":
                errors.append(worker_error_detail(event))
        failed = [f"{worker.name} exit={worker.exitcode}" for worker in workers if worker.exitcode]
        if errors or failed:
            detail = "; ".join(errors + failed)
            write_failure(args.output_dir, detail)
            raise RuntimeError(detail)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for model in unique_models:
            for _ in range(max(1, int(args.replicas_per_checkpoint))):
                request_queues[model].put(None)
        for server in servers:
            server.join(timeout=30)
            if server.is_alive():
                server.terminate()
    records.sort(key=lambda record: (int(record["seed"]), str(record["orientation"])))
    result = {
        "candidate": {
            "label": args.candidate_label,
            "models": candidate_models,
            "search": resolve_search_modes(args)["candidate"],
        },
        "opponent": {
            "label": args.opponent_label,
            "kind": args.opponent_kind,
            "policy_checkpoint": checkpoint_by_model.get("opponent_policy"),
            "value_checkpoint": checkpoint_by_model.get("opponent_value"),
            "search": resolve_search_modes(args)["opponent"],
            "steam_search_depth": args.steam_search_depth if args.opponent_kind == "steam_proxy" else None,
            "steam_max_candidates": args.steam_max_candidates if args.opponent_kind == "steam_proxy" else None,
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": summary(args, records),
        "records": records,
        "inference_servers": [
            {"model": model, **inference_stats_payload(index, gpu, stats)}
            for index, (model, gpu, stats) in enumerate(server_stats)
        ],
    }
    (args.output_dir / "failure.json").unlink(missing_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "games.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
