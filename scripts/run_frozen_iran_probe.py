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

from scripts.run_frozen_diagnostics import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LOG,
    algorithm_args,
    env_config,
    iran_exact_probe,
    replay_root,
    summarize_values,
)
from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.frozen_diagnostic import (
    ExactChanceStepper,
    FrozenNetwork,
    action_text,
    find_action_index,
    root_side,
    terminal_value_us,
)
from struggle_ai.train_mcts import build_algorithm
from struggle_ai.train_rllib import SHARED_POLICY_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the frozen K-particle Iran exact-chance probe.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--game-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=8_808_800)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    return parser.parse_args()


def action_with_value(env: TwilightStruggleEnv, value: str) -> int:
    matches = [index for index, action in enumerate(env.legal_actions) if str(action.get("value")) == value]
    if len(matches) != 1:
        raise RuntimeError(f"expected one action {value!r}, got {matches}")
    return matches[0]


def iran_mode_probe(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    game: dict[str, object],
    particles: int,
    seed: int,
) -> dict[str, object]:
    spec_index = 494
    _obs, root_snapshot, target = replay_root(env, game, spec_index)
    env.step(find_action_index(env.legal_actions, target["action"]))
    modes = [action_text(action) for action in env.legal_actions]
    space_legal = any(str(action.get("value")) == "space" for action in env.legal_actions)
    env.step(action_with_value(env, "ops"))
    ops_modes = [action_text(action) for action in env.legal_actions]
    placement_obs, _reward, _terminated, _truncated, _info = env.step(action_with_value(env, "place"))
    placement_snapshot = env.snapshot(compact=True)
    raw = network.raw(placement_obs, len(env.legal_actions))
    order = list(raw["probabilities"].argsort()[::-1])
    placement_labels = [action_text(action) for action in env.legal_actions]
    stepper = ExactChanceStepper()
    rows = []
    for action_index in order[:5]:
        values = []
        sides_after = []
        for particle in range(particles):
            env.restore(placement_snapshot)
            particle_obs = env.redeterminize("us", seed + particle * 104729)
            particle_snapshot = env.snapshot(compact=True)
            outcomes = stepper.outcomes(env, particle_snapshot, int(action_index))
            value = 0.0
            for outcome in outcomes:
                env.restore(outcome.snapshot)
                side_after = root_side(env.last_obs)
                sides_after.append(side_after)
                branch = (
                    terminal_value_us(outcome.info)
                    if outcome.terminated or outcome.truncated
                    else network.value_us(outcome.obs, side_after)[1]
                )
                value += outcome.probability * branch
            values.append(value)
        rows.append(
            {
                "action": placement_labels[int(action_index)],
                "actor_probability": float(raw["probabilities"][action_index]),
                "actor_logit": float(raw["logits"][action_index]),
                "Q_after_us": summarize_values(values, "us"),
                "sides_after": sorted(set(sides_after)),
            }
        )
    env.restore(root_snapshot)
    return {
        "kitchen_modes": modes,
        "space_legal": space_legal,
        "ops_modes": ops_modes,
        "placement_legal_count": len(raw["probabilities"]),
        "top5_influence": rows,
    }


def main() -> None:
    args = parse_args()
    build_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        num_gpus=args.num_gpus,
    )
    algo = build_algorithm(algorithm_args(build_args))
    env = TwilightStruggleEnv(env_config())
    try:
        network = FrozenNetwork(algo.get_policy(SHARED_POLICY_ID))
        games = [
            json.loads(line)
            for line in args.game_log.resolve().read_text(encoding="utf-8").splitlines()
            if line
        ]
        probe = iran_exact_probe(
            env,
            network,
            games[3],
            494,
            args.belief_particles,
            args.seed + 494,
        )
        mode_probe = iran_mode_probe(
            env,
            network,
            games[3],
            args.belief_particles,
            args.seed + 494 * 17,
        )
        result = {
            "checkpoint": str(args.checkpoint.resolve()),
            "game_log": str(args.game_log.resolve()),
            "position": "iran_coup_t7",
            "belief_particles": args.belief_particles,
            "frozen": True,
            "training_updates": 0,
            "iran_exact_chance": probe,
            "iran_mode_probe": mode_probe,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "iran_exact_done", "output": str(args.output)}, sort_keys=True))
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
