#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.batched_inference import load_checkpoint_model
from struggle_ai.frozen_diagnostic import (
    ExactBeliefPuct,
    ExactChanceStepper,
    FrozenNetwork,
    action_text,
    find_action_index,
    is_meaningful_root,
    observation_digest,
    root_side,
    terminal_value_us,
)


DEFAULT_LOG = Path(
    "runs/eval/mcts-best23522-vs-balanced-parent-full-logs-20260808-171844/"
    "game_logs/games-126410.jsonl"
)
DEFAULT_CHECKPOINT = Path(
    "runs/mcts/belief-chance-sideheads-usfocus-topk4-gpu0-48h-20260803-063914/"
    "mcts_checkpoints/checkpoints/decisions-0000023522-games-0000054"
)


@dataclass(frozen=True)
class PositionSpec:
    name: str
    game: int
    action_index: int
    depth: int = 2


POSITIONS = {
    spec.name: spec
    for spec in (
        PositionSpec("olympics_t1", 1, 19),
        PositionSpec("olympics_t3", 2, 151),
        PositionSpec("olympics_t5", 3, 286),
        PositionSpec("destalinization_t1", 2, 41),
        PositionSpec("europe_scoring_t2", 1, 91, 3),
        PositionSpec("mexico_t5_04", 4, 330),
        PositionSpec("mexico_t5_05", 4, 348),
        PositionSpec("iran_coup_t7", 4, 494),
        PositionSpec("final_scoring_t10", 4, 738, 1),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen actor/critic/belief/exact-chance diagnostics.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--value-checkpoint",
        type=Path,
        default=None,
        help="Optional independent checkpoint used only for critic leaf values.",
    )
    parser.add_argument("--game-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positions", default=",".join(POSITIONS))
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--budgets", default="32,128,512")
    parser.add_argument("--counterfactual-top-k", type=int, default=5)
    parser.add_argument("--child-top-k", type=int, default=16)
    parser.add_argument("--depth1-budget", type=int, default=128)
    parser.add_argument("--depth3-budget", type=int, default=32)
    parser.add_argument("--terminal-max-nodes-per-root", type=int, default=512)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument(
        "--root-search-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run only raw-policy and depth-2 PUCT root diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=8_808_800)
    return parser.parse_args()


def algorithm_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        restore_from=args.checkpoint.resolve(),
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
    )


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
        "log_games_dir": None,
        "metrics_dir": None,
    }


def replay_root(
    env: TwilightStruggleEnv,
    game: dict[str, Any],
    target_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    obs, _info = env.reset(seed=int(game["seed"]))
    for index, record in enumerate(game["actions"][:target_index]):
        obs, _reward, terminated, truncated, info = env.step(int(record["action_index"]))
        if terminated or truncated:
            raise RuntimeError(f"replay terminated at action {index}: {info}")
    target = game["actions"][target_index]
    before = target["before"]
    actual = env.last_obs or {}
    fields = ("turn", "action_round", "side", "phase", "prompt", "vp", "defcon")
    mismatch = {key: (before.get(key), actual.get(key)) for key in fields if before.get(key) != actual.get(key)}
    if mismatch:
        raise RuntimeError(f"replay mismatch before action {target_index}: {mismatch}")
    matched = find_action_index(env.legal_actions, target["action"])
    if matched != int(target["action_index"]):
        raise RuntimeError(
            f"action ordering mismatch at {target_index}: logged={target['action_index']} replay={matched}"
        )
    return obs, env.snapshot(compact=True), target


def logged_macro(game: dict[str, Any], target_index: int) -> list[dict[str, Any]]:
    records = game["actions"]
    result = [copy.deepcopy(records[target_index]["action"])]
    for record in records[target_index + 1 :]:
        if is_meaningful_root(record.get("before")):
            break
        result.append(copy.deepcopy(record["action"]))
    return result


class GreedyCounterfactual:
    def __init__(self, network: FrozenNetwork, *, max_atomic_steps: int = 96) -> None:
        self.network = network
        self.max_atomic_steps = max_atomic_steps
        self.stepper = ExactChanceStepper()

    def _walk(
        self,
        env: TwilightStruggleEnv,
        snapshot: dict[str, Any],
        obs: dict[str, np.ndarray],
        *,
        decisions_used: int,
        max_decisions: int,
        atomic_steps: int,
        forced_actions: tuple[dict[str, Any], ...],
        sequence: tuple[str, ...],
        state_loaded: bool = False,
    ) -> list[tuple[float, float, tuple[str, ...]]]:
        if not state_loaded:
            env.restore(snapshot)
        if not forced_actions and atomic_steps > 0 and is_meaningful_root(env.last_obs):
            if decisions_used >= max_decisions:
                value = self.network.value_us(obs, root_side(env.last_obs))[1]
                return [(1.0, value, sequence)]
        if atomic_steps >= self.max_atomic_steps:
            raise RuntimeError(f"greedy counterfactual exceeded atomic step bound at {env.last_obs}")

        if forced_actions:
            action_index = find_action_index(env.legal_actions, forced_actions[0])
            remaining = forced_actions[1:]
        else:
            raw = self.network.raw(obs, len(env.legal_actions))
            action_index = int(np.argmax(raw["probabilities"]))
            remaining = ()
        action_label = action_text(env.legal_actions[action_index])
        next_decisions = decisions_used + (1 if is_meaningful_root(env.last_obs) else 0)
        outcomes = self.stepper.outcomes(env, snapshot, action_index)
        rows: list[tuple[float, float, tuple[str, ...]]] = []
        for outcome in outcomes:
            next_sequence = sequence + (action_label,)
            if outcome.terminated or outcome.truncated:
                rows.append((outcome.probability, terminal_value_us(outcome.info), next_sequence))
                continue
            children = self._walk(
                env,
                outcome.snapshot,
                outcome.obs,
                decisions_used=next_decisions,
                max_decisions=max_decisions,
                atomic_steps=atomic_steps + 1,
                forced_actions=remaining,
                sequence=next_sequence,
                state_loaded=outcome.current_in_env,
            )
            rows.extend((outcome.probability * probability, value, child_sequence) for probability, value, child_sequence in children)
        return rows

    def evaluate(
        self,
        env: TwilightStruggleEnv,
        particle_snapshot: dict[str, Any],
        particle_obs: dict[str, np.ndarray],
        root_action: dict[str, Any],
        *,
        max_decisions: int,
        forced_macro: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        forced = tuple(forced_macro or [root_action])
        rows = self._walk(
            env,
            particle_snapshot,
            particle_obs,
            decisions_used=0,
            max_decisions=max_decisions,
            atomic_steps=0,
            forced_actions=forced,
            sequence=(),
            state_loaded=False,
        )
        total = sum(probability for probability, _value, _sequence in rows)
        if not math_isclose(total, 1.0):
            raise RuntimeError(f"counterfactual probability mass is {total}")
        q_us = sum(probability * value for probability, value, _sequence in rows)
        traces = sorted(rows, key=lambda row: -row[0])[:8]
        return {
            "Q_us": q_us,
            "traces": [
                {"probability": probability, "Q_us": value, "sequence": list(sequence)}
                for probability, value, sequence in traces
            ],
        }


def math_isclose(value: float, target: float) -> bool:
    return abs(value - target) <= 1e-7 * max(1.0, abs(value), abs(target))


def summarize_values(values: list[float], root_player: str) -> dict[str, float]:
    perspective = values if root_player == "us" else [-value for value in values]
    return {
        "mean": float(statistics.fmean(perspective)),
        "std": float(statistics.pstdev(perspective)) if len(perspective) > 1 else 0.0,
        "min": float(min(perspective)),
        "max": float(max(perspective)),
    }


def counterfactuals(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    root_snapshot: dict[str, Any],
    root_obs: dict[str, np.ndarray],
    game: dict[str, Any],
    spec: PositionSpec,
    raw_order: list[int],
    top_k: int,
    particles: int,
    seed: int,
) -> list[dict[str, Any]]:
    root_player = root_side(env.last_obs)
    logged = game["actions"][spec.action_index]
    selected = [int(index) for index in raw_order[:top_k]]
    if int(logged["action_index"]) not in selected:
        selected.append(int(logged["action_index"]))
    logged_sequence = logged_macro(game, spec.action_index)
    evaluator = GreedyCounterfactual(network)
    output = []
    for action_index in selected:
        env.restore(root_snapshot)
        action = copy.deepcopy(env.legal_actions[action_index])
        use_logged = action_index == int(logged["action_index"])
        depth_values: dict[int, list[float]] = {1: [], 2: []}
        traces: dict[int, list[dict[str, Any]]] = {}
        for particle in range(particles):
            env.restore(root_snapshot)
            particle_obs = env.redeterminize(root_player, seed + particle * 104729)
            particle_snapshot = env.snapshot(compact=True)
            for depth in (1, 2):
                result = evaluator.evaluate(
                    env,
                    particle_snapshot,
                    particle_obs,
                    action,
                    max_decisions=depth,
                    forced_macro=logged_sequence if use_logged else None,
                )
                depth_values[depth].append(float(result["Q_us"]))
                if particle == 0:
                    traces[depth] = result["traces"]
        output.append(
            {
                "action_index": action_index,
                "action": action_text(action),
                "logged": use_logged,
                "depth1": summarize_values(depth_values[1], root_player),
                "depth2": summarize_values(depth_values[2], root_player),
                "particle_rank_values_depth1": [
                    value if root_player == "us" else -value for value in depth_values[1]
                ],
                "traces_depth1_particle0": traces.get(1, []),
                "traces_depth2_particle0": traces.get(2, []),
            }
        )
    return output


def hand_mask_report(obs: dict[str, np.ndarray], player: str) -> dict[str, Any]:
    us_count = float(np.asarray(obs.get("us_hand", [])).sum())
    ussr_count = float(np.asarray(obs.get("ussr_hand", [])).sum())
    opponent_count = ussr_count if player == "us" else us_count
    return {
        "acting_side": player,
        "us_hand_visible_count": us_count,
        "ussr_hand_visible_count": ussr_count,
        "opponent_hand_masked": opponent_count == 0.0,
    }


def raw_report(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    obs: dict[str, np.ndarray],
    logged_action_index: int,
) -> tuple[dict[str, Any], list[int]]:
    raw = network.raw(obs, len(env.legal_actions))
    order = list(np.argsort(-raw["probabilities"]))
    rows = []
    for rank, index in enumerate(order[:10], 1):
        rows.append(
            {
                "rank": rank,
                "action_index": int(index),
                "action": action_text(env.legal_actions[int(index)]),
                "probability": float(raw["probabilities"][index]),
                "logit": float(raw["logits"][index]),
                "logged": int(index) == logged_action_index,
            }
        )
    logged_rank = order.index(logged_action_index) + 1
    return (
        {
            "raw_critic": raw["raw_value"],
            "calibrated_critic": raw["calibrated_value"],
            "logged_action_rank": logged_rank,
            "top10": rows,
        },
        order,
    )


def targeted_action_generation_probe(
    env: TwilightStruggleEnv,
    root_snapshot: dict[str, Any],
    game: dict[str, Any],
    spec: PositionSpec,
) -> dict[str, Any]:
    env.restore(root_snapshot)
    labels = [action_text(action) for action in env.legal_actions]
    probe: dict[str, Any] = {"root_legal_count": len(labels), "root_actions": labels}
    records = game["actions"]
    if spec.name.startswith("olympics"):
        probe["participate_present"] = any(str(action.get("value")) == "participate" for action in env.legal_actions)
        probe["boycott_present"] = any(str(action.get("value")) == "boycott" for action in env.legal_actions)
    if spec.name == "destalinization_t1":
        action_index = find_action_index(env.legal_actions, records[spec.action_index]["action"])
        env.step(action_index)
        probe["destal_modes"] = [action_text(action) for action in env.legal_actions]
        probe["space_present"] = any(str(action.get("value")) == "space" for action in env.legal_actions)
    if spec.name in {"mexico_t5_04", "mexico_t5_05"}:
        offset = 3 if spec.name == "mexico_t5_04" else 4
        env.restore(root_snapshot)
        for record in records[spec.action_index : spec.action_index + offset]:
            env.step(find_action_index(env.legal_actions, record["action"]))
        countries = [str(action.get("label") or action.get("value")) for action in env.legal_actions]
        probe["first_placement_legal_count"] = len(countries)
        probe["first_placement_countries"] = countries
        probe["non_mexico_placement_present"] = any(country.lower() != "mexico" for country in countries)
    if spec.name == "iran_coup_t7":
        env.restore(root_snapshot)
        for record in records[spec.action_index : spec.action_index + 3]:
            env.step(find_action_index(env.legal_actions, record["action"]))
        probe["coup_targets"] = [str(action.get("label") or action.get("value")) for action in env.legal_actions]
        probe["iran_present"] = any(str(action.get("value")) == "iran" for action in env.legal_actions)
    if spec.name == "final_scoring_t10":
        env.restore(root_snapshot)
        first = records[spec.action_index]["action"]
        env.step(find_action_index(env.legal_actions, first))
        probe["card_modes"] = [action_text(action) for action in env.legal_actions]
    env.restore(root_snapshot)
    return probe


def mexico_marginal_probe(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    game: dict[str, Any],
    indices: list[int],
    particles: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for target_index in indices:
        obs, snapshot, logged = replay_root(env, game, target_index)
        player = root_side(env.last_obs)
        side_before = player
        before_us = network.value_us(obs, player)[1]
        countries = {str(country.get("id")): country for country in (env.last_obs or {}).get("countries", [])}
        mexico = countries.get("mexico", {})
        raw = network.raw(obs, len(env.legal_actions))
        order = list(np.argsort(-raw["probabilities"]))
        wanted = [int(index) for index in order[:8]]
        logged_index = int(logged["action_index"])
        if logged_index not in wanted:
            wanted.append(logged_index)
        action_rows = []
        stepper = ExactChanceStepper()
        for action_index in wanted:
            env.restore(snapshot)
            action_label = action_text(env.legal_actions[action_index])
            values = []
            sides_after: list[str] = []
            for particle in range(particles):
                env.restore(snapshot)
                particle_obs = env.redeterminize(player, seed + target_index * 1009 + particle * 104729)
                particle_snapshot = env.snapshot(compact=True)
                matched_index = action_index
                outcomes = stepper.outcomes(env, particle_snapshot, matched_index)
                value = 0.0
                for outcome in outcomes:
                    env.restore(outcome.snapshot)
                    side_after = root_side(env.last_obs)
                    if outcome.terminated or outcome.truncated:
                        branch = terminal_value_us(outcome.info)
                    else:
                        branch = network.value_us(outcome.obs, side_after)[1]
                    value += outcome.probability * branch
                    sides_after.append(side_after)
                values.append(value)
            summary = summarize_values(values, "us")
            action_rows.append(
                {
                    "action_index": action_index,
                    "action": action_label,
                    "actor_probability": float(raw["probabilities"][action_index]),
                    "Q_after_us": summary,
                    "delta_V_mean": summary["mean"] - before_us,
                    "side_before": side_before,
                    "sides_after": sorted(set(sides_after)),
                    "crosses_turn_boundary": any(side != side_before for side in sides_after),
                    "logged": action_index == logged_index,
                }
            )
        rows.append(
            {
                "game_action_index": target_index,
                "prompt": (env.last_obs or {}).get("prompt"),
                "mexico_us": mexico.get("us"),
                "mexico_ussr": mexico.get("ussr"),
                "V_before_us": before_us,
                "placements": sorted(action_rows, key=lambda row: -row["Q_after_us"]["mean"]),
            }
        )
    return rows


def iran_exact_probe(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    game: dict[str, Any],
    target_index: int,
    particles: int,
    seed: int,
) -> dict[str, Any]:
    _obs, root_snapshot, _logged = replay_root(env, game, target_index)
    records = game["actions"]
    root_player = root_side(env.last_obs)
    face_values: dict[int, list[float]] = {face: [] for face in range(1, 7)}
    face_states: dict[int, dict[str, Any]] = {}
    for particle in range(particles):
        env.restore(root_snapshot)
        env.redeterminize(root_player, seed + particle * 104729)
        for record in records[target_index : target_index + 3]:
            env.step(find_action_index(env.legal_actions, record["action"]))
        coup_snapshot = env.snapshot(compact=True)
        iran_index = find_action_index(env.legal_actions, records[target_index + 3]["action"])
        for face in range(1, 7):
            env.restore(coup_snapshot)
            next_obs, _reward, terminated, truncated, info = env.step_with_forced_rolls(iran_index, [face])
            if terminated or truncated:
                value_us = terminal_value_us(info)
            else:
                value_us = network.value_us(next_obs, root_side(env.last_obs))[1]
            face_values[face].append(value_us)
            if particle == 0:
                countries = {
                    str(country.get("id")): country
                    for country in (env.last_obs or {}).get("countries", [])
                }
                iran = countries.get("iran", {})
                face_states[face] = {
                    "us_influence": iran.get("us"),
                    "ussr_influence": iran.get("ussr"),
                    "defcon": (env.last_obs or {}).get("defcon"),
                    "chance_rolls": info.get("chance_rolls"),
                }
    rows = []
    exact_values = []
    for face in range(1, 7):
        summary = summarize_values(face_values[face], "us")
        exact_values.append(summary["mean"])
        rows.append({"roll": face, "Q_us": summary, "state_particle0": face_states[face]})
    exact_q = float(statistics.fmean(exact_values))
    rng = np.random.default_rng(seed + 64)
    sampled_faces = rng.integers(1, 7, size=64)
    mc_values = [exact_values[int(face) - 1] for face in sampled_faces]
    return {
        "rolls": rows,
        "Q_exact": exact_q,
        "mc64": {
            "mean": float(statistics.fmean(mc_values)),
            "std": float(statistics.pstdev(mc_values)),
            "face_histogram": {str(face): int(np.sum(sampled_faces == face)) for face in range(1, 7)},
        },
    }


def terminal_progressive_search(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    root_snapshot: dict[str, Any],
    *,
    max_wide_actions: int = 16,
    max_nodes_per_root: int = 5000,
) -> dict[str, Any]:
    stepper = ExactChanceStepper()
    env.restore(root_snapshot)
    root_obs = env.encode_current_observation(env.last_obs or {})
    root_actions = copy.deepcopy(env.legal_actions)
    root_rows: list[dict[str, Any]] = []
    total_nodes = 0
    total_edges = 0
    total_limit_leaves = 0
    total_unexpected_boundaries = 0
    total_truncated_actions = 0

    for root_index, root_action in enumerate(root_actions):
        nodes = 0
        edges = 0
        limit_leaves = 0
        unexpected_boundaries = 0
        truncated_actions = 0

        def solve(
            snapshot: dict[str, Any],
            obs: dict[str, np.ndarray],
            *,
            state_loaded: bool = False,
        ) -> tuple[float, list[str], dict[str, Any]]:
            nonlocal nodes, edges, limit_leaves, unexpected_boundaries, truncated_actions
            if not state_loaded:
                env.restore(snapshot)
            if edges >= max_nodes_per_root:
                limit_leaves += 1
                return network.value_us(obs, root_side(env.last_obs))[1], [], {"source": "critic_node_limit"}
            nodes += 1
            side = root_side(env.last_obs)
            legal_count = len(env.legal_actions)
            actions = copy.deepcopy(env.legal_actions)
            if legal_count <= max_wide_actions:
                selected = list(range(legal_count))
            else:
                raw = network.raw(obs, legal_count)
                selected = [int(index) for index in np.argsort(-raw["probabilities"])[:max_wide_actions]]
            choices: list[tuple[float, list[str], dict[str, Any]]] = []
            for selected_offset, action_index in enumerate(selected):
                if edges >= max_nodes_per_root:
                    truncated_actions += len(selected) - selected_offset
                    break
                edges += 1
                env.restore(snapshot)
                label = action_text(actions[action_index])
                outcomes = stepper.outcomes(env, snapshot, action_index)
                expected = 0.0
                representative: tuple[float, list[str], dict[str, Any]] | None = None
                sources: set[str] = set()
                for outcome in outcomes:
                    if outcome.terminated or outcome.truncated:
                        branch_value = terminal_value_us(outcome.info)
                        branch_sequence: list[str] = []
                        branch_meta = {
                            "source": "terminal",
                            "winner": outcome.info.get("winner"),
                            "terminal_reason": outcome.info.get("terminal_reason"),
                            "vp": outcome.info.get("vp"),
                            "defcon": outcome.info.get("defcon"),
                        }
                    else:
                        env.restore(outcome.snapshot)
                        if is_meaningful_root(env.last_obs):
                            unexpected_boundaries += 1
                            branch_value = network.value_us(outcome.obs, root_side(env.last_obs))[1]
                            branch_sequence = []
                            branch_meta = {"source": "critic_unexpected_boundary"}
                        else:
                            branch_value, branch_sequence, branch_meta = solve(
                                outcome.snapshot,
                                outcome.obs,
                                state_loaded=outcome.current_in_env,
                            )
                    expected += outcome.probability * branch_value
                    sources.add(str(branch_meta.get("source")))
                    candidate = (outcome.probability, branch_sequence, branch_meta)
                    if representative is None or candidate[0] > representative[0]:
                        representative = candidate
                assert representative is not None
                choices.append(
                    (
                        expected,
                        [label, *representative[1]],
                        {**representative[2], "branch_sources": sorted(sources)},
                    )
                )
            if not choices:
                raise RuntimeError(f"terminal search reached state without legal actions: {env.last_obs}")
            return (max(choices, key=lambda row: row[0]) if side == "us" else min(choices, key=lambda row: row[0]))

        env.restore(root_snapshot)
        root_outcomes = stepper.outcomes(env, root_snapshot, root_index)
        root_q = 0.0
        representative_root: tuple[float, list[str], dict[str, Any]] | None = None
        for outcome in root_outcomes:
            if outcome.terminated or outcome.truncated:
                value = terminal_value_us(outcome.info)
                sequence: list[str] = []
                meta = {
                    "source": "terminal",
                    "winner": outcome.info.get("winner"),
                    "terminal_reason": outcome.info.get("terminal_reason"),
                    "vp": outcome.info.get("vp"),
                    "defcon": outcome.info.get("defcon"),
                }
            else:
                value, sequence, meta = solve(
                    outcome.snapshot,
                    outcome.obs,
                    state_loaded=outcome.current_in_env,
                )
            root_q += outcome.probability * value
            candidate = (outcome.probability, sequence, meta)
            if representative_root is None or candidate[0] > representative_root[0]:
                representative_root = candidate
        assert representative_root is not None
        root_rows.append(
            {
                "action_index": root_index,
                "action": action_text(root_action),
                "Q_us": float(root_q),
                "representative_best_response_sequence": [action_text(root_action), *representative_root[1]],
                "representative_terminal": representative_root[2],
                "expanded_nodes": nodes,
                "expanded_action_edges": edges,
                "node_limit_leaves": limit_leaves,
                "truncated_progressive_actions": truncated_actions,
                "unexpected_meaningful_boundaries": unexpected_boundaries,
            }
        )
        total_nodes += nodes
        total_edges += edges
        total_limit_leaves += limit_leaves
        total_unexpected_boundaries += unexpected_boundaries
        total_truncated_actions += truncated_actions
    root_rows.sort(key=lambda row: -row["Q_us"])
    return {
        "expanded_nodes": total_nodes,
        "expanded_action_edges": total_edges,
        "node_limit_per_root": max_nodes_per_root,
        "node_limit_leaves": total_limit_leaves,
        "truncated_progressive_actions": total_truncated_actions,
        "unexpected_meaningful_boundaries": total_unexpected_boundaries,
        "root_action_summary": root_rows,
        "exact_chance_nodes": stepper.enumerated_nodes,
        "max_exact_outcomes": stepper.max_outcomes,
    }


def run_position(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    game: dict[str, Any],
    spec: PositionSpec,
    args: argparse.Namespace,
    budgets: tuple[int, ...],
) -> dict[str, Any]:
    obs, snapshot, target = replay_root(env, game, spec.action_index)
    print(json.dumps({"event": "diagnostic_stage", "position": spec.name, "stage": "root_replayed"}), flush=True)
    player = root_side(env.last_obs)
    raw, raw_order = raw_report(env, network, obs, int(target["action_index"]))
    generation = targeted_action_generation_probe(env, snapshot, game, spec)
    print(json.dumps({"event": "diagnostic_stage", "position": spec.name, "stage": "raw_and_action_gen_done"}), flush=True)
    cfs = []
    if not args.root_search_only:
        cfs = counterfactuals(
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
        print(json.dumps({"event": "diagnostic_stage", "position": spec.name, "stage": "counterfactuals_done"}), flush=True)
    search = ExactBeliefPuct(
        network,
        belief_particles=args.belief_particles,
        max_meaningful_decisions=2 if spec.depth > 1 else 1,
        child_top_k=args.child_top_k,
        seed=args.seed + spec.action_index * 1009,
    ).run(env, snapshot, obs, budgets=budgets)
    print(json.dumps({"event": "diagnostic_stage", "position": spec.name, "stage": "depth2_puct_done"}), flush=True)
    depth1 = None
    if not args.root_search_only:
        depth1 = ExactBeliefPuct(
            network,
            belief_particles=args.belief_particles,
            max_meaningful_decisions=1,
            child_top_k=args.child_top_k,
            seed=args.seed + spec.action_index * 2017,
        ).run(env, snapshot, obs, budgets=(args.depth1_budget,))
        print(json.dumps({"event": "diagnostic_stage", "position": spec.name, "stage": "depth1_puct_done"}), flush=True)
    depth3 = None
    if spec.depth >= 3 and not args.root_search_only:
        depth3 = ExactBeliefPuct(
            network,
            belief_particles=args.belief_particles,
            max_meaningful_decisions=3,
            child_top_k=args.child_top_k,
            seed=args.seed + spec.action_index * 3011,
        ).run(env, snapshot, obs, budgets=(args.depth3_budget,))
        print(json.dumps({"event": "diagnostic_stage", "position": spec.name, "stage": "depth3_puct_done"}), flush=True)
    result: dict[str, Any] = {
        "position": spec.name,
        "seed": game["seed"],
        "game": spec.game,
        "logged_game_action_index": spec.action_index,
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
        "action_generation": generation,
        "counterfactuals": cfs,
        "puct_depth2": {
            "budgets": search.budgets,
            "traces": search.traces[:50],
            "expanded_nodes": search.expanded_nodes,
            "exact_chance_nodes": search.exact_chance_nodes,
            "max_exact_outcomes": search.max_exact_outcomes,
            "boundary_evaluations": search.boundary_evaluations,
            "terminal_evaluations": search.terminal_evaluations,
            "chance_cache_hits": search.chance_cache_hits,
        },
        "network_cache": {
            "hits": network.cache_hits,
            "misses": network.cache_misses,
            "entries": len(network.cache),
        },
    }
    if depth1 is not None:
        result["puct_depth1"] = {
            "budgets": depth1.budgets,
            "traces": depth1.traces[:30],
            "expanded_nodes": depth1.expanded_nodes,
            "exact_chance_nodes": depth1.exact_chance_nodes,
            "chance_cache_hits": depth1.chance_cache_hits,
        }
    if spec.name == "mexico_t5_04" and not args.root_search_only:
        print(json.dumps({"event": "diagnostic_stage", "position": spec.name, "stage": "marginals_start"}), flush=True)
        result["mexico_marginals"] = mexico_marginal_probe(
            env, network, game, [333, 334, 335, 336], args.belief_particles, args.seed
        )
    if spec.name == "mexico_t5_05" and not args.root_search_only:
        print(json.dumps({"event": "diagnostic_stage", "position": spec.name, "stage": "marginals_start"}), flush=True)
        result["mexico_marginals"] = mexico_marginal_probe(
            env, network, game, [352, 353, 354, 355], args.belief_particles, args.seed
        )
    if spec.name == "iran_coup_t7" and not args.root_search_only:
        print(json.dumps({"event": "diagnostic_stage", "position": spec.name, "stage": "iran_exact_start"}), flush=True)
        result["iran_exact_chance"] = iran_exact_probe(
            env, network, game, spec.action_index, args.belief_particles, args.seed + spec.action_index
        )
    if spec.name == "final_scoring_t10" and not args.root_search_only:
        print(json.dumps({"event": "diagnostic_stage", "position": spec.name, "stage": "terminal_search_start"}), flush=True)
        result["terminal_progressive_search"] = terminal_progressive_search(
            env,
            network,
            snapshot,
            max_nodes_per_root=args.terminal_max_nodes_per_root,
        )
    if depth3 is not None:
        result["puct_depth3"] = {
            "budgets": depth3.budgets,
            "traces": depth3.traces[:50],
            "expanded_nodes": depth3.expanded_nodes,
            "exact_chance_nodes": depth3.exact_chance_nodes,
            "max_exact_outcomes": depth3.max_exact_outcomes,
            "boundary_evaluations": depth3.boundary_evaluations,
            "terminal_evaluations": depth3.terminal_evaluations,
            "chance_cache_hits": depth3.chance_cache_hits,
        }
    return result


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = [name.strip() for name in args.positions.split(",") if name.strip()]
    unknown = sorted(set(names) - set(POSITIONS))
    if unknown:
        raise ValueError(f"unknown positions: {unknown}")
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",") if value.strip()}))
    games = [json.loads(line) for line in args.game_log.resolve().read_text(encoding="utf-8").splitlines() if line]
    device = "cuda:0" if args.num_gpus > 0 else "cpu"
    policy_model = load_checkpoint_model(args.checkpoint.resolve(), device)
    value_checkpoint = (args.value_checkpoint or args.checkpoint).resolve()
    value_model = (
        policy_model
        if value_checkpoint == args.checkpoint.resolve()
        else load_checkpoint_model(value_checkpoint, device)
    )
    network = FrozenNetwork(policy_model, value_model)
    env = TwilightStruggleEnv(env_config())
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "value_checkpoint": str(value_checkpoint),
        "game_log": str(args.game_log.resolve()),
        "belief_particles": args.belief_particles,
        "budgets": budgets,
        "positions": names,
        "frozen": True,
        "training_updates": 0,
        "results": {},
    }
    (args.output_dir / "config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        for name in names:
            spec = POSITIONS[name]
            print(json.dumps({"event": "position_start", "position": name}), flush=True)
            result = run_position(env, network, games[spec.game - 1], spec, args, budgets)
            path = args.output_dir / f"{name}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest["results"][name] = str(path)
            (args.output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            final_budget = max(budgets)
            top = result["puct_depth2"]["budgets"][final_budget][0]
            print(
                json.dumps(
                    {
                        "event": "position_done",
                        "position": name,
                        "raw_choice": result["raw"]["top10"][0]["action"],
                        "search_choice": top["action"],
                        "search_N": top["N"],
                        "search_Q": top["Q"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
