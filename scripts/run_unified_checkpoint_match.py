#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import (
    SHARED_POLICY_ID,
    EvalPolicySwitcher,
    load_policy_weights,
    restore_policy_model_modes,
    set_policy_models_evaluation_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fully logged games between two unified side-head checkpoints."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--opponent-label", default="opponent")
    parser.add_argument("--games-per-side", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2_608_080)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    return parser.parse_args()


def algorithm_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        restore_from=args.candidate.resolve(),
        num_gpus=args.num_gpus,
        hidden=256,
        history_layers=2,
        card_history_layers=2,
        history_attention_heads=4,
        history_dropout=0.05,
        graph_layers=2,
        graph_neighbor_hops=2,
        policy_temperature=1.0,
    )


def main() -> None:
    args = parse_args()
    if args.games_per_side <= 0:
        raise ValueError("--games-per-side must be positive")
    args.output_dir = args.output_dir.resolve()
    game_log_dir = args.output_dir / "game_logs"
    metrics_dir = args.output_dir / "metrics"
    game_log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    algo = build_algorithm(algorithm_args(args))
    current_weights = {
        SHARED_POLICY_ID: algo.get_policy(SHARED_POLICY_ID).get_weights()
    }
    opponent_weights = {
        SHARED_POLICY_ID: load_policy_weights(args.opponent.resolve(), SHARED_POLICY_ID)
    }
    switcher = EvalPolicySwitcher(
        algo,
        current_weights,
        {"opponent": opponent_weights},
        partial_compatible_load=True,
    )
    prior_modes = set_policy_models_evaluation_mode(algo, [SHARED_POLICY_ID])
    env = TwilightStruggleEnv(
        {
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
            "log_games_dir": str(game_log_dir),
            "log_games_every": 1,
            "metrics_dir": str(metrics_dir),
        }
    )
    records: list[dict[str, Any]] = []
    try:
        run_number = 0
        for candidate_side in ("us", "ussr"):
            for side_game in range(args.games_per_side):
                run_number += 1
                seed = args.seed + run_number - 1
                obs, _info = env.reset(seed=seed)
                terminated = truncated = False
                final_info: dict[str, Any] = {}
                while not (terminated or truncated):
                    side = str((env.last_obs or {}).get("side") or "ussr")
                    which = "current" if side == candidate_side else "opponent"
                    action = switcher.action(SHARED_POLICY_ID, which, obs)
                    obs, _reward, terminated, truncated, final_info = env.step(action)
                record = {
                    "run": run_number,
                    "seed": seed,
                    "candidate_side": candidate_side,
                    "us_model": args.candidate_label if candidate_side == "us" else args.opponent_label,
                    "ussr_model": args.candidate_label if candidate_side == "ussr" else args.opponent_label,
                    "winner": final_info.get("winner"),
                    "terminal_reason": final_info.get("terminal_reason"),
                    "vp": final_info.get("vp"),
                    "defcon": final_info.get("defcon"),
                    "steps": env.episode_step,
                }
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        env.close()
        switcher.restore_current()
        restore_policy_model_modes(prior_modes)
        algo.stop()
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass

    payload = {
        "candidate": {
            "label": args.candidate_label,
            "checkpoint": str(args.candidate.resolve()),
        },
        "opponent": {
            "label": args.opponent_label,
            "checkpoint": str(args.opponent.resolve()),
        },
        "games_per_side": args.games_per_side,
        "records": records,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index_lines = [
        f"candidate: {args.candidate_label}",
        f"opponent: {args.opponent_label}",
        "",
    ]
    for record in records:
        index_lines.append(
            "Run {run}: US={us_model}; USSR={ussr_model}; winner={winner}; "
            "reason={terminal_reason}; VP={vp}; DEFCON={defcon}; steps={steps}".format(**record)
        )
    (args.output_dir / "match_index.txt").write_text(
        "\n".join(index_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
