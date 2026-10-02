#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.frozen_diagnostic import find_action_index
from struggle_ai.mcts import SelectivePuctConfig, SelectivePuctPlanner, TorchPolicyValueEvaluator
from struggle_ai.search_replay import SearchReplayWriter, examples_from_decision, with_terminal_outcome
from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import SHARED_POLICY_ID


DEFAULT_TRIGGERS = (
    "forced_response",
    "opponent_event",
    "influence_allocation",
    "scoring_card",
    "chance_heavy",
    "low_confidence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate targeted exact-PUCT supervision from complete game logs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--game-log", type=Path, action="append", default=[])
    parser.add_argument("--position-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-searches", type=int, default=None)
    parser.add_argument(
        "--position",
        action="append",
        default=[],
        help="Restrict search to seed:zero_based_action_index[:game_occurrence]; may be repeated.",
    )
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--initial-simulations", type=int, default=32)
    parser.add_argument("--medium-simulations", type=int, default=128)
    parser.add_argument("--max-simulations", type=int, default=512)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--meaningful-depth", type=int, default=2)
    parser.add_argument("--max-atomic-steps", type=int, default=512)
    parser.add_argument("--forced-response-depth", type=int, default=1)
    parser.add_argument("--child-initial", type=int, default=4)
    parser.add_argument("--child-max", type=int, default=24)
    parser.add_argument("--max-exact-chance-outcomes", type=int, default=36)
    parser.add_argument("--max-expanded-nodes", type=int, default=512)
    parser.add_argument("--search-temperature", type=float, default=1.0)
    parser.add_argument("--trigger", action="append", choices=DEFAULT_TRIGGERS, default=[])
    parser.add_argument("--only-action-flips", action="store_true")
    parser.add_argument("--seed", type=int, default=8_110_001)
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
    return parser.parse_args()


def iter_games(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        resolved = path.expanduser().resolve()
        with resolved.open("r", encoding="utf-8") as handle:
            if resolved.suffix == ".json":
                payload = json.load(handle)
                records = payload if isinstance(payload, list) else [payload]
            else:
                records = [json.loads(line) for line in handle if line.strip()]
        for record in records:
            if record.get("kind") == "terminal" and record.get("actions"):
                yield record


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


def parse_positions(values: list[str]) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()
    for value in values:
        parts = value.split(":")
        if len(parts) not in {2, 3}:
            raise SystemExit(
                f"invalid --position {value!r}; expected seed:action_index[:game_occurrence]"
            )
        try:
            seed, step = int(parts[0]), int(parts[1])
            occurrence = int(parts[2]) if len(parts) == 3 else 0
            result.add((seed, step, occurrence))
        except (ValueError, TypeError) as exc:
            raise SystemExit(
                f"invalid --position {value!r}; expected seed:action_index[:game_occurrence]"
            ) from exc
    return result


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
    if not args.game_log:
        raise SystemExit("provide --game-log or --position-manifest")
    selected_positions = parse_positions(args.position)
    triggers = tuple(args.trigger or DEFAULT_TRIGGERS)
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
        enabled_triggers=triggers,
        max_exact_chance_outcomes=args.max_exact_chance_outcomes,
        max_expanded_nodes=args.max_expanded_nodes,
    )
    algo = build_algorithm(algorithm_args(args))
    policy = algo.get_policy(SHARED_POLICY_ID)
    evaluator = TorchPolicyValueEvaluator(policy)
    replay_env = TwilightStruggleEnv(env_config())
    search_env = TwilightStruggleEnv(env_config())
    planner = SelectivePuctPlanner(evaluator, search_env, config, seed=args.seed)
    games = examples = searched = discarded = 0
    writer = SearchReplayWriter(args.output_dir, shard_size=args.shard_size)
    try:
        seed_occurrences: dict[int, int] = {}
        for game in iter_games(args.game_log):
            if args.max_games is not None and games >= args.max_games:
                break
            game_seed = int(game["seed"])
            game_occurrence = seed_occurrences.get(game_seed, 0)
            seed_occurrences[game_seed] = game_occurrence + 1
            obs, _info = replay_env.reset(seed=game_seed)
            for step_index, record in enumerate(game["actions"]):
                if args.max_examples is not None and examples >= args.max_examples:
                    break
                if args.max_searches is not None and searched >= args.max_searches:
                    break
                root_snapshot = replay_env.snapshot(compact=True)
                legal_actions = copy.deepcopy(replay_env.legal_actions)
                selected = not selected_positions or (
                    game_seed,
                    step_index,
                    game_occurrence,
                ) in selected_positions
                if selected:
                    print(
                        {
                            "event": "search_replay_search_started",
                            "seed": game_seed,
                            "game_occurrence": game_occurrence,
                            "step": step_index,
                            "turn": int((replay_env.last_obs or {}).get("turn") or 0),
                            "action_round": int((replay_env.last_obs or {}).get("action_round") or 0),
                            "side": str((replay_env.last_obs or {}).get("side") or ""),
                        },
                        flush=True,
                    )
                decision = (
                    planner.plan(
                        root_snapshot,
                        obs,
                        force_search=bool(selected_positions),
                    )
                    if selected
                    else None
                )
                if decision is not None and decision.searched:
                    searched += 1
                    print(
                        {
                            "event": "search_replay_search_finished",
                            "seed": game_seed,
                            "game_occurrence": game_occurrence,
                            "step": step_index,
                            "turn": int((replay_env.last_obs or {}).get("turn") or 0),
                            "action_round": int(
                                (replay_env.last_obs or {}).get("action_round") or 0
                            ),
                            "side": str((replay_env.last_obs or {}).get("side") or ""),
                            "simulations": decision.simulations,
                            "latency_seconds": decision.latency_seconds,
                            "action_flipped": decision.action_flipped,
                            "root_q_gap": decision.root_q_gap,
                            "q_advantage_over_raw": decision.q_advantage_over_raw,
                            "max_meaningful_depth_reached": (
                                decision.max_meaningful_depth_reached
                            ),
                            "max_turn_delta": decision.max_turn_delta,
                            "cross_turn_leaf_evaluations": (
                                decision.cross_turn_leaf_evaluations
                            ),
                            "atomic_horizon_cutoffs": decision.atomic_horizon_cutoffs,
                            "node_budget_cutoffs": decision.node_budget_cutoffs,
                        },
                        flush=True,
                    )
                    search_complete = not (
                        decision.atomic_horizon_cutoffs or decision.node_budget_cutoffs
                    )
                    if search_complete and (not args.only_action_flips or decision.action_flipped):
                        generated = examples_from_decision(
                                obs,
                                legal_actions,
                                str((replay_env.last_obs or {}).get("side") or "ussr"),
                                decision,
                                search_depth=decision.meaningful_depth,
                                source_game=f"seed-{game_seed}-occurrence-{game_occurrence}",
                                source_step=step_index,
                            )
                        game_info = dict(game.get("info") or {})
                        winner = str(game.get("winner") or game_info.get("winner") or "")
                        if winner in {"us", "ussr"}:
                            generated = with_terminal_outcome(
                                generated,
                                winner=winner,
                                terminal_reason=str(
                                    game.get("terminal_reason")
                                    or game_info.get("terminal_reason")
                                    or "unknown"
                                ),
                            )
                        for example in generated:
                            if args.max_examples is not None and examples >= args.max_examples:
                                break
                            writer.add(example)
                            examples += 1
                        # Deep roots can take minutes. Persist each completed
                        # root so an interruption never discards its labels.
                        writer.flush()
                    else:
                        discarded += 1
                logged_index = find_action_index(replay_env.legal_actions, record["action"])
                obs, _reward, terminated, truncated, _final_info = replay_env.step(logged_index)
                if terminated or truncated:
                    break
            games += 1
            print(
                {
                    "event": "search_replay_progress",
                    "games": games,
                    "searched_states": searched,
                    "examples": examples,
                    "discarded": discarded,
                    **planner.telemetry.metrics(),
                },
                flush=True,
            )
            if args.max_examples is not None and examples >= args.max_examples:
                break
            if args.max_searches is not None and searched >= args.max_searches:
                break
    finally:
        manifest = writer.close()
        replay_env.close()
        search_env.close()
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass
    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "manifest": str(manifest),
        "games": games,
        "searched_states": searched,
        "examples": examples,
        "discarded": discarded,
        "planner": planner.telemetry.metrics(),
        "config": vars(args) | {"checkpoint": str(args.checkpoint), "game_log": [str(path) for path in args.game_log], "output_dir": str(args.output_dir)},
    }
    (args.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "search_replay_done", **summary}, default=str), flush=True)


if __name__ == "__main__":
    main()
