from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.env import TwilightStruggleEnv, TwilightStruggleMultiAgentEnv
from struggle_ai.mcts import ChanceAwareMCTS, TorchPolicyValueEvaluator
from struggle_ai.rllib_masked_model import register_masked_model
from struggle_ai.train_rllib import (
    SHARED_POLICY_ID,
    checkpoint_elo_ratings,
    configure_focused_head_only,
    copy_compatible_or_widened_weights,
    evaluate_policy,
    initialize_elo_state,
    load_policy_weights,
    pull_focused_actor_toward_anchor,
    save_elo_state,
    save_checkpoint,
)


@dataclass
class SearchSample:
    obs: dict[str, np.ndarray]
    policy: np.ndarray
    side: str
    root_value_us: float
    value_target: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chance-aware belief-MCTS expert iteration for Twilight Struggle.")
    parser.add_argument("--restore-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-seconds", type=float, default=172800.0)
    parser.add_argument("--max-games", type=int, default=1_000_000)
    parser.add_argument("--max-decisions", type=int, default=None, help="Debug-only global decision bound.")
    parser.add_argument("--seed", type=int, default=2_608_020)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--search-depth", type=int, default=12)
    parser.add_argument("--belief-particles", type=int, default=4)
    parser.add_argument(
        "--max-children",
        type=int,
        default=0,
        help="Restrict each search node to the policy's top-k actions; zero keeps every legal action.",
    )
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-fraction", type=float, default=0.20)
    parser.add_argument("--search-discount", type=float, default=1.0)
    parser.add_argument("--early-temperature", type=float, default=1.0)
    parser.add_argument("--late-temperature", type=float, default=0.25)
    parser.add_argument("--temperature-through-turn", type=int, default=3)
    parser.add_argument("--replay-capacity", type=int, default=2048)
    parser.add_argument("--min-replay-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--updates-per-game", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--entropy-coeff", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--checkpoint-every-seconds", type=float, default=3600.0)
    parser.add_argument("--eval-every-seconds", type=float, default=14400.0)
    parser.add_argument("--eval-games-per-side", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--log-games-every", type=int, default=1)
    parser.add_argument("--progress-every-decisions", type=int, default=10)
    parser.add_argument("--wandb-project", default="twilight-struggle-mcts")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--elo-path", type=Path, default=None)
    parser.add_argument("--elo-seed-from", type=Path, default=None)
    parser.add_argument("--elo-k-factor", type=float, default=32.0)
    parser.add_argument(
        "--retained-elo-min-delta",
        type=float,
        default=0.0,
        help="Required improvement in the lower of US/USSR Elo before retaining an MCTS candidate.",
    )
    parser.add_argument(
        "--rollback-on-eval-reject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore retained weights and clear optimizer/replay after a candidate fails the Elo-floor gate.",
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--card-history-layers", type=int, default=2)
    parser.add_argument("--history-attention-heads", type=int, default=4)
    parser.add_argument("--history-dropout", type=float, default=0.05)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-neighbor-hops", type=int, default=2)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--train-focus-side", choices=("both", "us", "ussr"), default="both")
    parser.add_argument(
        "--focused-head-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze shared encoders and the opposite actor while training one side.",
    )
    parser.add_argument(
        "--focused-actor-anchor-pullback",
        type=float,
        default=0.0,
        help="After each Expert Iteration update, interpolate the focused actor toward its initial weights.",
    )
    args = parser.parse_args()
    if args.focused_head_only and args.train_focus_side == "both":
        parser.error("--focused-head-only requires --train-focus-side us or ussr")
    if not 0.0 <= args.focused_actor_anchor_pullback <= 1.0:
        parser.error("--focused-actor-anchor-pullback must be in [0, 1]")
    if args.focused_actor_anchor_pullback > 0.0 and not args.focused_head_only:
        parser.error("--focused-actor-anchor-pullback requires --focused-head-only")
    return args


def build_algorithm(args: argparse.Namespace):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    register_masked_model()
    ray_init_kwargs = {"ignore_reinit_error": True, "include_dashboard": False}
    diagnostic_cpus = getattr(args, "ray_num_cpus", None)
    if diagnostic_cpus is not None:
        ray_init_kwargs["num_cpus"] = int(diagnostic_cpus)
    object_store_memory = getattr(args, "ray_object_store_memory", None)
    if object_store_memory is not None:
        ray_init_kwargs["object_store_memory"] = int(object_store_memory)
    ray.init(**ray_init_kwargs)
    probe = TwilightStruggleEnv({"force_setup_heuristic": False, "log_games_dir": None, "metrics_dir": None})
    policy_spec = (None, probe.observation_space, probe.action_space, {})
    probe.close()
    env_name = f"twilight_struggle_mcts_{id(args)}"
    register_env(env_name, lambda cfg: TwilightStruggleMultiAgentEnv(cfg))
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env_name, env_config={"force_setup_heuristic": False, "log_games_dir": None, "metrics_dir": None})
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .env_runners(num_env_runners=0)
        .training(
            model={
                "custom_model": "twilight_masked_model",
                "custom_model_config": {
                    "hidden": args.hidden,
                    "model_arch": "transformer_history",
                    "history_layers": args.history_layers,
                    "card_history_layers": args.card_history_layers,
                    "history_attention_heads": args.history_attention_heads,
                    "history_dropout": args.history_dropout,
                    "graph_layers": args.graph_layers,
                    "graph_neighbor_hops": args.graph_neighbor_hops,
                    "heuristic_prior_scale": 2.0,
                    "setup_heuristic_prior_scale": 0.0,
                    "policy_temperature": args.policy_temperature,
                    "side_specific_heads": True,
                    "value_adapter_hidden": int(getattr(args, "value_adapter_hidden", 0)),
                    "value_state_tower": bool(getattr(args, "value_state_tower", False)),
                },
            }
        )
        .multi_agent(
            policies={SHARED_POLICY_ID: policy_spec},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: SHARED_POLICY_ID,
            policies_to_train=[SHARED_POLICY_ID],
        )
    )
    algo = config.build()
    policy = algo.get_policy(SHARED_POLICY_ID)
    source_weights = load_policy_weights(args.restore_from.resolve(), SHARED_POLICY_ID)
    migrated, matched, widened, skipped, missing, unexpected = copy_compatible_or_widened_weights(
        policy.get_weights(),
        source_weights,
    )
    if skipped:
        raise RuntimeError(f"MCTS checkpoint has incompatible tensors: {skipped[:5]}")
    policy.set_weights(migrated)
    print(
        {
            "event": "mcts_checkpoint_migrated",
            "matched_count": len(matched),
            "widened_count": len(widened),
            "missing_source_count": len(missing),
            "unexpected_source_count": len(unexpected),
        },
        flush=True,
    )
    return algo


def pad_search_policy(policy: np.ndarray, action_count: int, output_size: int) -> np.ndarray:
    target = np.zeros(output_size, dtype=np.float32)
    target[:action_count] = policy[:action_count]
    total = float(target.sum())
    if total <= 0.0:
        target[:action_count] = 1.0 / action_count
    else:
        target /= total
    return target


def torch_batch(policy: Any, samples: list[SearchSample]):
    import torch

    device = next(policy.model.parameters()).device
    obs = {
        key: torch.as_tensor(np.stack([sample.obs[key] for sample in samples]), dtype=torch.float32, device=device)
        for key in samples[0].obs
    }
    policy_targets = torch.as_tensor(np.stack([sample.policy for sample in samples]), dtype=torch.float32, device=device)
    value_targets = torch.as_tensor([sample.value_target for sample in samples], dtype=torch.float32, device=device)
    return obs, policy_targets, value_targets


def sample_side_balanced_replay(
    population: list[SearchSample],
    batch_size: int,
    rng: random.Random,
    focus_side: str = "both",
) -> list[SearchSample]:
    """Sample one acting side, or balance both sides when both are trained."""
    if batch_size <= 0 or not population:
        return []
    if focus_side not in {"both", "us", "ussr"}:
        raise ValueError(f"unsupported replay focus side: {focus_side}")

    if focus_side != "both":
        focused = [sample for sample in population if sample.side == focus_side]
        if not focused:
            return []
        return rng.sample(focused, min(batch_size, len(focused)))

    target_size = min(batch_size, len(population))
    by_side = {
        "us": [index for index, sample in enumerate(population) if sample.side == "us"],
        "ussr": [index for index, sample in enumerate(population) if sample.side == "ussr"],
    }
    selected: list[int] = []
    us_target = target_size // 2
    ussr_target = target_size - us_target
    selected.extend(rng.sample(by_side["us"], min(us_target, len(by_side["us"]))))
    selected.extend(rng.sample(by_side["ussr"], min(ussr_target, len(by_side["ussr"]))))

    if len(selected) < target_size:
        selected_set = set(selected)
        remaining = [index for index in range(len(population)) if index not in selected_set]
        selected.extend(rng.sample(remaining, target_size - len(selected)))

    rng.shuffle(selected)
    return [population[index] for index in selected]


def train_updates(
    policy: Any,
    optimizer: Any,
    replay: deque[SearchSample],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    if len(replay) < args.min_replay_size:
        return {}
    model = policy.model
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "grad_norm": 0.0,
        "value_abs_mean": 0.0,
        "value_saturation_ratio": 0.0,
        "us_value_mean": 0.0,
        "ussr_value_mean": 0.0,
        "us_target_mean": 0.0,
        "ussr_target_mean": 0.0,
    }
    model.train()
    population = list(replay)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for _ in range(args.updates_per_game):
        batch = sample_side_balanced_replay(
            population,
            args.batch_size,
            rng,
            focus_side=args.train_focus_side,
        )
        if not batch:
            return {}
        obs, policy_targets, value_targets = torch_batch(policy, batch)
        logits, _state = model({"obs": obs}, [], None)
        values = model.value_function()
        log_probs = functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()
        value_loss = functional.mse_loss(values, value_targets)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = policy_loss + args.value_loss_weight * value_loss - args.entropy_coeff * entropy
        us_mask = torch.as_tensor([sample.side == "us" for sample in batch], dtype=torch.bool, device=values.device)
        ussr_mask = ~us_mask
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
        optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["policy_loss"] += float(policy_loss.detach().cpu())
        totals["value_loss"] += float(value_loss.detach().cpu())
        totals["entropy"] += float(entropy.detach().cpu())
        totals["grad_norm"] += float(grad_norm.detach().cpu())
        totals["value_abs_mean"] += float(values.detach().abs().mean().cpu())
        totals["value_saturation_ratio"] += float((values.detach().abs() >= 0.95).float().mean().cpu())
        if bool(us_mask.any().item()):
            totals["us_value_mean"] += float(values.detach()[us_mask].mean().cpu())
            totals["us_target_mean"] += float(value_targets[us_mask].mean().cpu())
        if bool(ussr_mask.any().item()):
            totals["ussr_value_mean"] += float(values.detach()[ussr_mask].mean().cpu())
            totals["ussr_target_mean"] += float(value_targets[ussr_mask].mean().cpu())
    return {f"train/{key}": value / args.updates_per_game for key, value in totals.items()}


def save_training_checkpoint(
    algo: Any,
    optimizer: Any,
    output_dir: Path,
    games: int,
    decisions: int,
    elapsed: float,
    label: str | None = None,
) -> str:
    import torch

    checkpoint = save_checkpoint(
        algo,
        output_dir / "mcts_checkpoints",
        label or f"decisions-{decisions:010d}-games-{games:07d}",
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "games": games,
            "decisions": decisions,
            "elapsed_seconds": elapsed,
        },
        Path(checkpoint) / "mcts_trainer_state.pt",
    )
    return checkpoint


def evaluation_metrics(
    algo: Any,
    initial_weights: dict[str, Any],
    args: argparse.Namespace,
    seed_base: int,
    elo_state: dict[str, Any],
    current_elo_ratings: dict[str, float],
    initial_elo_ratings: dict[str, float],
):
    env_config = {
        "force_setup_heuristic": False,
        "persistent_random_prob": 0.0,
        "heuristic_override_prob": 0.0,
        "scripted_side_prob": 0.0,
        "defcon_suicide_mode": "hard_filter",
        "max_episode_steps": args.max_episode_steps,
        "log_games_dir": None,
        "metrics_dir": None,
    }
    return evaluate_policy(
        algo=algo,
        benchmark_weights={"initial": {SHARED_POLICY_ID: initial_weights}},
        env_config=env_config,
        eval_games=args.eval_games_per_side,
        min_games_per_side=args.eval_games_per_side,
        max_episode_steps=args.max_episode_steps,
        seed_base=seed_base,
        side_policy_ids={"us": SHARED_POLICY_ID, "ussr": SHARED_POLICY_ID},
        eval_label=f"mcts_games_{seed_base}",
        progress_every_games=10,
        elo_state=elo_state,
        elo_k_factor=args.elo_k_factor,
        include_random=True,
        elo_current_ratings=current_elo_ratings,
        elo_opponent_ratings={"initial": initial_elo_ratings},
    )


def selection_score(eval_metrics: dict[str, Any]) -> float:
    us_rate = float(eval_metrics.get("eval_vs_initial/us_side_win_rate", 0.0))
    ussr_rate = float(eval_metrics.get("eval_vs_initial/ussr_side_win_rate", 0.0))
    overall = float(eval_metrics.get("eval_vs_initial/win_rate", 0.0))
    nuke = float(eval_metrics.get("eval_vs_initial/nuke_terminal_rate", 0.0))
    return 1000.0 * min(us_rate, ussr_rate) + 100.0 * overall - 50.0 * nuke


def mcts_eval_elo_ratings(eval_metrics: dict[str, Any]) -> dict[str, float]:
    return {
        side: float(eval_metrics[f"elo/{side}/current_rating"])
        for side in ("us", "ussr", "bot")
        if f"elo/{side}/current_rating" in eval_metrics
    }


def mcts_candidate_improves_elo_floor(
    candidate_ratings: dict[str, float],
    retained_ratings: dict[str, float],
    min_delta: float = 0.0,
) -> bool:
    if not all(side in candidate_ratings and side in retained_ratings for side in ("us", "ussr")):
        return False
    candidate_floor = min(candidate_ratings["us"], candidate_ratings["ussr"])
    retained_floor = min(retained_ratings["us"], retained_ratings["ussr"])
    return candidate_floor > retained_floor + float(min_delta)


def rollback_mcts_candidate(
    policy: Any,
    optimizer: Any,
    replay: deque[SearchSample],
    retained_weights: dict[str, Any],
) -> dict[str, int]:
    policy.set_weights(copy.deepcopy(retained_weights))
    optimizer_states_cleared = 0
    state = getattr(optimizer, "state", None)
    if state is not None:
        state.clear()
        optimizer_states_cleared = 1
    replay_items_cleared = len(replay)
    replay.clear()
    return {
        "optimizer_states_cleared": optimizer_states_cleared,
        "replay_items_cleared": replay_items_cleared,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    import torch

    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    elo_path = (args.elo_path or (args.output_dir / "elo_unified.json")).expanduser().resolve()
    elo_seed_path = args.elo_seed_from.expanduser().resolve() if args.elo_seed_from else None
    elo_state, elo_state_source = initialize_elo_state(elo_path, elo_seed_path, args.elo_k_factor)
    print(
        {
            "event": "elo_state_initialized",
            "source": elo_state_source,
            "elo_path": str(elo_path),
            "seed_path": str(elo_seed_path) if elo_seed_path else None,
            "prior_matches": len(elo_state.get("matches", [])),
        },
        flush=True,
    )
    (args.output_dir / "game_logs").mkdir(exist_ok=True)
    (args.output_dir / "metrics").mkdir(exist_ok=True)
    write_json(args.output_dir / "mcts_config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    algo = build_algorithm(args)
    policy = algo.get_policy(SHARED_POLICY_ID)
    # build_algorithm may widen input tensors for a newer feature schema.
    initial_weights = copy.deepcopy(policy.get_weights())
    initial_elo_ratings = checkpoint_elo_ratings(elo_state, args.restore_from)
    current_elo_ratings = dict(initial_elo_ratings)
    print(
        {
            "event": "mcts_elo_parent_initialized",
            "checkpoint": str(args.restore_from),
            "ratings": initial_elo_ratings,
        },
        flush=True,
    )
    if args.focused_head_only:
        freeze_report = configure_focused_head_only(policy.model, args.train_focus_side)
        print(
            {
                "event": "mcts_focused_head_only_configured",
                "focus_side": args.train_focus_side,
                **freeze_report,
            },
            flush=True,
        )
    trainable_parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("MCTS optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    evaluator = TorchPolicyValueEvaluator(policy)
    search = ChanceAwareMCTS(
        evaluator,
        simulations=args.simulations,
        max_depth=args.search_depth,
        cpuct=args.cpuct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_fraction=args.dirichlet_fraction,
        discount=args.search_discount,
        belief_particles=args.belief_particles,
        max_children=args.max_children,
        seed=args.seed + 1,
    )
    env_config = {
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
        "log_games_dir": str(args.output_dir / "game_logs"),
        "log_games_every": args.log_games_every,
        "metrics_dir": str(args.output_dir / "metrics"),
    }
    search_config = {**env_config, "log_games_dir": None, "metrics_dir": None, "max_episode_step_penalty": 0.0}
    actual_env = TwilightStruggleEnv(env_config)
    search_env = TwilightStruggleEnv(search_config)
    replay: deque[SearchSample] = deque(maxlen=args.replay_capacity)
    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            dir=str(args.output_dir),
            config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        )
    start = time.monotonic()
    last_checkpoint = start
    last_eval = start
    games = 0
    decisions = 0
    retained_weights = copy.deepcopy(initial_weights)
    retained_checkpoint_path = str(args.restore_from)
    retained_elo_ratings = dict(initial_elo_ratings)
    retained_side_floor = min(
        float(retained_elo_ratings.get("us", 1500.0)),
        float(retained_elo_ratings.get("ussr", 1500.0)),
    )
    best_score = retained_side_floor
    write_json(
        args.output_dir / "best_checkpoint.json",
        {
            "checkpoint": retained_checkpoint_path,
            "score": best_score,
            "games": 0,
            "decisions": 0,
            "selection_metric": "elo_side_floor",
            "metrics": {
                "eval/elo_us": retained_elo_ratings.get("us"),
                "eval/elo_ussr": retained_elo_ratings.get("ussr"),
                "eval/elo_side_floor": retained_side_floor,
                "eval/source": "inherited_parent",
            },
        },
    )
    last_checkpoint_path: str | None = None
    try:
        while games < args.max_games and time.monotonic() - start < args.stop_seconds:
            game_seed = args.seed + games * 1009
            obs, _info = actual_env.reset(seed=game_seed)
            episode_samples: list[SearchSample] = []
            final_info: dict[str, Any] = {}
            terminated = truncated = False
            while not (terminated or truncated):
                if args.max_decisions is not None and decisions >= args.max_decisions:
                    break
                root_snapshot = actual_env.snapshot(compact=True)
                turn = int((actual_env.last_obs or {}).get("turn") or 1)
                temperature = args.early_temperature if turn <= args.temperature_through_turn else args.late_temperature
                result = search.search(search_env, root_snapshot, obs, temperature=temperature)
                legal_count = int(obs["action_mask"].sum())
                episode_samples.append(
                    SearchSample(
                        obs={key: value.copy() for key, value in obs.items()},
                        policy=pad_search_policy(result.policy, legal_count, actual_env.action_space.n),
                        side=str((actual_env.last_obs or {}).get("side") or "ussr"),
                        root_value_us=result.root_value_us,
                    )
                )
                obs, _reward, terminated, truncated, final_info = actual_env.step(result.action)
                decisions += 1
                if args.progress_every_decisions > 0 and decisions % args.progress_every_decisions == 0:
                    elapsed = time.monotonic() - start
                    print(
                        {
                            "event": "mcts_progress",
                            "games": games,
                            "decisions": decisions,
                            "decisions_per_second": decisions / max(elapsed, 1e-9),
                            "turn": turn,
                            "root_side": episode_samples[-1].side,
                            "root_value_us": result.root_value_us,
                            "expanded_nodes": result.expanded_nodes,
                            "max_depth": result.max_depth_reached,
                            "belief_particles": result.belief_particles,
                            "chance_samples": sum(result.chance_roll_histogram.values()),
                        },
                        flush=True,
                    )
                if wandb_run:
                    wandb_run.log(
                        {
                            "selfplay/decisions": decisions,
                            "selfplay/game": games + 1,
                            "search/root_value_us": result.root_value_us,
                            "search/expanded_nodes": result.expanded_nodes,
                            "search/max_depth": result.max_depth_reached,
                            "search/belief_particles": result.belief_particles,
                            "search/chance_samples": sum(result.chance_roll_histogram.values()),
                            "search/root_visit_entropy": float(-np.sum(result.policy * np.log(result.policy + 1e-12))),
                            "replay/size": len(replay),
                        },
                        step=decisions,
                    )
                if time.monotonic() - start >= args.stop_seconds:
                    # Finish the current game so every value target is auditable.
                    pass
            if args.max_decisions is not None and decisions >= args.max_decisions and not (terminated or truncated):
                print({"event": "debug_decision_limit", "decisions": decisions, "partial_game_discarded": 1}, flush=True)
                break
            winner = str(final_info.get("winner") or "")
            for sample in episode_samples:
                if winner == "us":
                    sample.value_target = 1.0 if sample.side == "us" else -1.0
                elif winner == "ussr":
                    sample.value_target = 1.0 if sample.side == "ussr" else -1.0
                else:
                    sample.value_target = 0.0
                replay.append(sample)
            games += 1
            train_metrics = train_updates(policy, optimizer, replay, args, rng)
            if train_metrics and args.focused_actor_anchor_pullback > 0.0:
                pullback_report = pull_focused_actor_toward_anchor(
                    policy,
                    initial_weights,
                    args.train_focus_side,
                    args.focused_actor_anchor_pullback,
                )
                train_metrics.update(
                    {f"train/anchor_pullback_{key}": value for key, value in pullback_report.items()}
                )
            replay_us = sum(sample.side == "us" for sample in replay)
            replay_ussr = sum(sample.side == "ussr" for sample in replay)
            game_metrics = {
                "selfplay/games": games,
                "selfplay/completed_decisions": decisions,
                "selfplay/game_steps": len(episode_samples),
                "selfplay/us_win": 1.0 if winner == "us" else 0.0,
                "selfplay/ussr_win": 1.0 if winner == "ussr" else 0.0,
                "selfplay/nuke_terminal": 1.0 if final_info.get("terminal_reason") in {"nuclear_war", "thermonuclear war", "Cuban Missile Crisis"} else 0.0,
                "selfplay/scoring_card_held_terminal": 1.0 if final_info.get("terminal_reason") == "scoring card held" else 0.0,
                "replay/size": len(replay),
                "replay/us_positions": replay_us,
                "replay/ussr_positions": replay_ussr,
                **train_metrics,
            }
            print({"event": "mcts_game_done", "winner": winner, "reason": final_info.get("terminal_reason"), **game_metrics}, flush=True)
            if wandb_run:
                wandb_run.log(game_metrics, step=decisions)

            now = time.monotonic()
            if now - last_checkpoint >= args.checkpoint_every_seconds:
                last_checkpoint_path = save_training_checkpoint(algo, optimizer, args.output_dir, games, decisions, now - start)
                last_checkpoint = now
                print({"event": "mcts_checkpoint", "checkpoint": last_checkpoint_path, "games": games, "decisions": decisions}, flush=True)
                if wandb_run:
                    wandb_run.summary["latest_checkpoint"] = last_checkpoint_path
            if now - last_eval >= args.eval_every_seconds:
                # Bind evaluation metrics to the exact weights being measured.
                # Reusing the last periodic checkpoint could associate a best
                # score with weights from up to checkpoint_every_seconds ago.
                last_checkpoint_path = save_training_checkpoint(
                    algo,
                    optimizer,
                    args.output_dir,
                    games,
                    decisions,
                    now - start,
                )
                last_checkpoint = now
                print(
                    {
                        "event": "mcts_checkpoint",
                        "checkpoint": last_checkpoint_path,
                        "games": games,
                        "decisions": decisions,
                        "reason": "pre_eval_exact_weights",
                    },
                    flush=True,
                )
                eval_metrics = evaluation_metrics(
                    algo,
                    initial_weights,
                    args,
                    args.seed + games * 100_000,
                    elo_state,
                    current_elo_ratings,
                    initial_elo_ratings,
                )
                current_elo_ratings = mcts_eval_elo_ratings(eval_metrics)
                save_elo_state(elo_path, elo_state)
                score = selection_score(eval_metrics)
                eval_metrics["eval/selection_score"] = score
                candidate_side_floor = min(
                    float(current_elo_ratings.get("us", 1500.0)),
                    float(current_elo_ratings.get("ussr", 1500.0)),
                )
                improved = mcts_candidate_improves_elo_floor(
                    current_elo_ratings,
                    retained_elo_ratings,
                    args.retained_elo_min_delta,
                )
                eval_metrics["eval/elo_side_floor"] = candidate_side_floor
                eval_metrics["eval/retained_elo_side_floor"] = retained_side_floor
                eval_metrics["eval/retained"] = int(improved)
                print({"event": "mcts_eval_done", "games": games, "decisions": decisions, **eval_metrics}, flush=True)
                if last_checkpoint_path and improved:
                    retained_weights = copy.deepcopy(policy.get_weights())
                    retained_checkpoint_path = last_checkpoint_path
                    retained_elo_ratings = dict(current_elo_ratings)
                    retained_side_floor = candidate_side_floor
                    best_score = retained_side_floor
                    write_json(
                        args.output_dir / "best_checkpoint.json",
                        {
                            "checkpoint": retained_checkpoint_path,
                            "score": best_score,
                            "games": games,
                            "decisions": decisions,
                            "selection_metric": "elo_side_floor",
                            "metrics": eval_metrics,
                        },
                    )
                    print(
                        {
                            "event": "mcts_candidate_retained",
                            "checkpoint": retained_checkpoint_path,
                            "elo_side_floor": retained_side_floor,
                            "ratings": retained_elo_ratings,
                        },
                        flush=True,
                    )
                elif args.rollback_on_eval_reject:
                    rollback_report = rollback_mcts_candidate(
                        policy,
                        optimizer,
                        replay,
                        retained_weights,
                    )
                    current_elo_ratings = dict(retained_elo_ratings)
                    print(
                        {
                            "event": "mcts_candidate_rejected_rollback",
                            "candidate_checkpoint": last_checkpoint_path,
                            "retained_checkpoint": retained_checkpoint_path,
                            "candidate_elo_side_floor": candidate_side_floor,
                            "retained_elo_side_floor": retained_side_floor,
                            **rollback_report,
                        },
                        flush=True,
                    )
                if wandb_run:
                    wandb_run.log(eval_metrics, step=decisions)
                    wandb_run.summary["best_checkpoint_score"] = best_score
                last_eval = now

        elapsed = time.monotonic() - start
        last_checkpoint_path = save_training_checkpoint(algo, optimizer, args.output_dir, games, decisions, elapsed)
        final_metrics = evaluation_metrics(
            algo,
            initial_weights,
            args,
            args.seed + 900_000_000,
            elo_state,
            current_elo_ratings,
            initial_elo_ratings,
        )
        save_elo_state(elo_path, elo_state)
        final_score = selection_score(final_metrics)
        final_metrics["eval/selection_score"] = final_score
        final_elo_ratings = mcts_eval_elo_ratings(final_metrics)
        final_side_floor = min(
            float(final_elo_ratings.get("us", 1500.0)),
            float(final_elo_ratings.get("ussr", 1500.0)),
        )
        final_improved = mcts_candidate_improves_elo_floor(
            final_elo_ratings,
            retained_elo_ratings,
            args.retained_elo_min_delta,
        )
        final_metrics["eval/elo_side_floor"] = final_side_floor
        final_metrics["eval/retained_elo_side_floor"] = retained_side_floor
        final_metrics["eval/retained"] = int(final_improved)
        if final_improved:
            retained_weights = copy.deepcopy(policy.get_weights())
            retained_checkpoint_path = last_checkpoint_path
            retained_elo_ratings = dict(final_elo_ratings)
            retained_side_floor = final_side_floor
            best_score = retained_side_floor
            write_json(
                args.output_dir / "best_checkpoint.json",
                {
                    "checkpoint": retained_checkpoint_path,
                    "score": best_score,
                    "games": games,
                    "decisions": decisions,
                    "selection_metric": "elo_side_floor",
                    "metrics": final_metrics,
                },
            )
        elif args.rollback_on_eval_reject:
            rollback_report = rollback_mcts_candidate(policy, optimizer, replay, retained_weights)
            retained_final_path = save_training_checkpoint(
                algo,
                optimizer,
                args.output_dir,
                games,
                decisions,
                elapsed,
                label=f"retained-final-decisions-{decisions:010d}-games-{games:07d}",
            )
            retained_checkpoint_path = retained_final_path
            final_metrics.update(
                {f"eval/rollback_{key}": value for key, value in rollback_report.items()}
            )
        write_json(
            args.output_dir / "final_summary.json",
            {
                "games": games,
                "decisions": decisions,
                "elapsed_seconds": elapsed,
                "checkpoint": retained_checkpoint_path,
                "latest_candidate_checkpoint": last_checkpoint_path,
                "retained_elo_ratings": retained_elo_ratings,
                "eval": final_metrics,
            },
        )
        print({"event": "mcts_training_done", "games": games, "decisions": decisions, "checkpoint": retained_checkpoint_path, **final_metrics}, flush=True)
        if wandb_run:
            wandb_run.summary.update(
                {
                    "games": games,
                    "decisions": decisions,
                    "checkpoint": retained_checkpoint_path,
                    "best_checkpoint_score": best_score,
                    **final_metrics,
                }
            )
    finally:
        actual_env.close()
        search_env.close()
        algo.stop()
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
