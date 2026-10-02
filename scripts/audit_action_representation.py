#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.features import ACTION_SEMANTIC_FEATURE_OFFSET
from struggle_ai.frozen_diagnostic import find_action_index
from struggle_ai.mcts import action_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 61-D action aliases on replayed legal candidate sets.")
    parser.add_argument("--game-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=4)
    parser.add_argument("--max-collision-examples", type=int, default=20)
    return parser.parse_args()


def iter_games(path: Path):
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") == "terminal" and record.get("actions"):
                yield record


def env_config() -> dict[str, Any]:
    return {
        "force_setup_heuristic": False,
        "persistent_random_prob": 0.0,
        "heuristic_override_prob": 0.0,
        "scripted_side_prob": 0.0,
        "defcon_suicide_mode": "hard_filter",
        "max_episode_steps": 1200,
        "log_games_dir": None,
        "metrics_dir": None,
    }


def signature(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def main() -> None:
    args = parse_args()
    env = TwilightStruggleEnv(env_config())
    states = legal_actions = pairs = legacy_alias_pairs = semantic_resolved_pairs = full_alias_pairs = 0
    categories: Counter[str] = Counter()
    collision_examples: list[dict[str, Any]] = []
    semantic_examples: list[dict[str, Any]] = []
    olympics: list[dict[str, Any]] = []
    games = 0
    try:
        for game in iter_games(args.game_log):
            if games >= args.max_games:
                break
            obs, _ = env.reset(seed=int(game["seed"]))
            for step, record in enumerate(game["actions"]):
                count = len(env.legal_actions)
                features = np.asarray(obs["action_features"][:count])
                labels = [action_text(action) for action in env.legal_actions]
                states += 1
                legal_actions += count
                pairs += count * (count - 1) // 2
                for left in range(count):
                    for right in range(left + 1, count):
                        legacy_equal = signature(features[left, :ACTION_SEMANTIC_FEATURE_OFFSET]) == signature(
                            features[right, :ACTION_SEMANTIC_FEATURE_OFFSET]
                        )
                        if not legacy_equal:
                            continue
                        legacy_alias_pairs += 1
                        semantic_equal = signature(features[left, ACTION_SEMANTIC_FEATURE_OFFSET:]) == signature(
                            features[right, ACTION_SEMANTIC_FEATURE_OFFSET:]
                        )
                        example = {
                            "seed": int(game["seed"]),
                            "step": step,
                            "left": labels[left],
                            "right": labels[right],
                            "prompt": str((env.last_obs or {}).get("prompt") or ""),
                        }
                        if semantic_equal:
                            full_alias_pairs += 1
                            categories["full_alias"] += 1
                            if len(collision_examples) < args.max_collision_examples:
                                collision_examples.append(example)
                        else:
                            semantic_resolved_pairs += 1
                            categories["semantic_resolved"] += 1
                            if len(semantic_examples) < args.max_collision_examples:
                                semantic_examples.append(example)
                if {"boycott", "participate"}.issubset(set(labels)):
                    boycott = labels.index("boycott")
                    participate = labels.index("participate")
                    olympics.append(
                        {
                            "seed": int(game["seed"]),
                            "step": step,
                            "legacy_l2": float(
                                np.linalg.norm(
                                    features[boycott, :ACTION_SEMANTIC_FEATURE_OFFSET]
                                    - features[participate, :ACTION_SEMANTIC_FEATURE_OFFSET]
                                )
                            ),
                            "semantic_l2": float(
                                np.linalg.norm(
                                    features[boycott, ACTION_SEMANTIC_FEATURE_OFFSET:]
                                    - features[participate, ACTION_SEMANTIC_FEATURE_OFFSET:]
                                )
                            ),
                            "full_l2": float(np.linalg.norm(features[boycott] - features[participate])),
                        }
                    )
                logged_index = find_action_index(env.legal_actions, record["action"])
                obs, _reward, terminated, truncated, _info = env.step(logged_index)
                if terminated or truncated:
                    break
            games += 1
    finally:
        env.close()
    payload = {
        "games": games,
        "states": states,
        "legal_actions": legal_actions,
        "candidate_pairs": pairs,
        "legacy_alias_pairs": legacy_alias_pairs,
        "semantic_resolved_pairs": semantic_resolved_pairs,
        "full_alias_pairs": full_alias_pairs,
        "semantic_resolution_rate": semantic_resolved_pairs / max(1, legacy_alias_pairs),
        "full_alias_rate_among_all_pairs": full_alias_pairs / max(1, pairs),
        "olympics": olympics,
        "semantic_resolved_examples": semantic_examples,
        "full_alias_examples": collision_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if not key.endswith("examples")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
