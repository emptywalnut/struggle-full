#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frozen_diagnostics import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_LOG,
    POSITIONS,
    algorithm_args,
    counterfactuals,
    env_config,
    hand_mask_report,
    logged_macro,
    raw_report,
    replay_root,
    targeted_action_generation_probe,
)
from struggle_ai.env import TwilightStruggleEnv  # noqa: E402
from struggle_ai.frozen_diagnostic import (  # noqa: E402
    ExactBeliefPuct,
    FrozenNetwork,
    action_text,
    observation_digest,
    root_side,
)
from struggle_ai.train_mcts import build_algorithm  # noqa: E402
from struggle_ai.train_rllib import SHARED_POLICY_ID  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one independently saveable Iran diagnostic component.")
    parser.add_argument("--mode", choices=("counterfactual", "puct"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--game-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--budgets", default="32,128,512")
    parser.add_argument("--counterfactual-top-k", type=int, default=5)
    parser.add_argument("--child-top-k", type=int, default=16)
    parser.add_argument("--depth1-budget", type=int, default=128)
    parser.add_argument("--seed", type=int, default=8_808_800)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    return parser.parse_args()


def puct_dict(run: object, trace_limit: int) -> dict[str, object]:
    return {
        "budgets": run.budgets,
        "traces": run.traces[:trace_limit],
        "expanded_nodes": run.expanded_nodes,
        "exact_chance_nodes": run.exact_chance_nodes,
        "max_exact_outcomes": run.max_exact_outcomes,
        "boundary_evaluations": run.boundary_evaluations,
        "terminal_evaluations": run.terminal_evaluations,
        "chance_cache_hits": run.chance_cache_hits,
    }


def main() -> None:
    args = parse_args()
    build_args = SimpleNamespace(checkpoint=args.checkpoint, num_gpus=args.num_gpus)
    algo = build_algorithm(algorithm_args(build_args))
    env = TwilightStruggleEnv(env_config())
    try:
        network = FrozenNetwork(algo.get_policy(SHARED_POLICY_ID))
        games = [
            json.loads(line)
            for line in args.game_log.resolve().read_text(encoding="utf-8").splitlines()
            if line
        ]
        spec = POSITIONS["iran_coup_t7"]
        game = games[spec.game - 1]
        obs, snapshot, target = replay_root(env, game, spec.action_index)
        player = root_side(env.last_obs)
        raw, raw_order = raw_report(env, network, obs, int(target["action_index"]))
        result: dict[str, object] = {
            "position": spec.name,
            "seed": game["seed"],
            "game": spec.game,
            "logged_game_action_index": spec.action_index,
            "checkpoint": str(args.checkpoint.resolve()),
            "frozen": True,
            "training_updates": 0,
            "root": {
                "turn": target["before"].get("turn"),
                "action_round": target["before"].get("action_round"),
                "side": player,
                "phase": target["before"].get("phase"),
                "prompt": target["before"].get("prompt"),
                "vp": target["before"].get("vp"),
                "defcon": target["before"].get("defcon"),
                "observation_digest": observation_digest(obs),
                "information_mask": hand_mask_report(obs, player),
                "logged_action": action_text(target["action"]),
                "logged_macro": [action_text(action) for action in logged_macro(game, spec.action_index)],
            },
            "raw": raw,
            "action_generation": targeted_action_generation_probe(env, snapshot, game, spec),
        }
        if args.mode == "counterfactual":
            result["counterfactuals"] = counterfactuals(
                env,
                network,
                snapshot,
                obs,
                game,
                spec,
                raw_order,
                args.counterfactual_top_k,
                args.belief_particles,
                args.seed + spec.game * 1_000_003 + spec.action_index,
            )
        else:
            budgets = tuple(sorted({int(value) for value in args.budgets.split(",") if value.strip()}))
            depth2 = ExactBeliefPuct(
                network,
                belief_particles=args.belief_particles,
                max_meaningful_decisions=2,
                child_top_k=args.child_top_k,
                seed=args.seed + spec.action_index * 1009,
            ).run(env, snapshot, obs, budgets=budgets)
            print(json.dumps({"event": "iran_puct_depth2_done"}), flush=True)
            depth1 = ExactBeliefPuct(
                network,
                belief_particles=args.belief_particles,
                max_meaningful_decisions=1,
                child_top_k=args.child_top_k,
                seed=args.seed + spec.action_index * 2017,
            ).run(env, snapshot, obs, budgets=(args.depth1_budget,))
            result["puct_depth1"] = puct_dict(depth1, 30)
            result["puct_depth2"] = puct_dict(depth2, 50)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "iran_component_done", "mode": args.mode, "output": str(args.output)}), flush=True)
    finally:
        env.close()
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
