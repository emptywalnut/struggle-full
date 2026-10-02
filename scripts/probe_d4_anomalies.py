#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.batched_inference import load_checkpoint_model
from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.frozen_diagnostic import FrozenNetwork, action_text, find_action_index, observation_digest
from struggle_ai.mcts import (
    ExactBeliefPuctSession,
    ExactChanceEnumerator,
    SelectivePuctConfig,
    local_search_snapshot,
    restore_search_state,
    root_side,
    validate_information_safe_observation,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = ROOT / "runs/eval/best-d4-vs-d4-4games-gpu45-20260818/game_logs"
DEFAULT_CHECKPOINT = (
    ROOT
    / "runs/puct_selfplay/cycle0-student-balanced2games-100updates-gpu4-20260812/checkpoints/distilled"
)


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    source_log: str
    root_action_index: int
    detail_action_index: int
    target_action_index: int | None
    kind: str
    suspicious_country: str
    logged_macro_length: int
    detail_macro_length: int
    source_policy: str
    note: str


SPECS = {
    spec.name: spec
    for spec in (
        ProbeSpec(
            name="ussr_uk_overcontrol",
            source_log="worker-001/games-3961640.jsonl",
            root_action_index=341,
            detail_action_index=345,
            target_action_index=None,
            kind="influence",
            suspicious_country="uk",
            logged_macro_length=8,
            detail_macro_length=4,
            source_policy="pre-campaign-policyvalueheads-d4",
            note="USSR starts a four-influence NATO allocation with UK at US 5 / USSR 12.",
        ),
        ProbeSpec(
            name="us_italy_overcontrol",
            source_log="worker-001/games-3961640.jsonl",
            root_action_index=577,
            detail_action_index=580,
            target_action_index=None,
            kind="influence",
            suspicious_country="italy",
            logged_macro_length=7,
            detail_macro_length=4,
            source_policy="accepted-parent-d4",
            note="US starts a four-influence China allocation with Italy at US 15 / USSR 2.",
        ),
        ProbeSpec(
            name="ussr_mexico_realignment",
            source_log="worker-003/games-3961642.jsonl",
            root_action_index=496,
            detail_action_index=498,
            target_action_index=500,
            kind="realignment",
            suspicious_country="mexico",
            logged_macro_length=7,
            detail_macro_length=4,
            source_policy="accepted-parent-d4",
            note="USSR chooses Junta realignment, then targets Mexico three times against a +2 US adjacency bonus.",
        ),
        ProbeSpec(
            name="us_venezuela_realignment",
            source_log="worker-001/games-3961640.jsonl",
            root_action_index=324,
            detail_action_index=326,
            target_action_index=327,
            kind="realignment",
            suspicious_country="venezuela",
            logged_macro_length=6,
            detail_macro_length=4,
            source_policy="accepted-parent-d4",
            note="US chooses a three-Ops realignment, then targets Venezuela three times against a +2 USSR adjacency bonus.",
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen accepted-parent probes for D4 gameplay anomalies.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positions", default=",".join(SPECS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--belief-particles", type=int, default=32)
    parser.add_argument("--matrix", default="2:32:2,4:128:2,6:512:2,8:2048:1")
    parser.add_argument("--search-seed", type=int, default=98_180_000)
    parser.add_argument(
        "--paired-matrix-seeds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the same root seed for every depth/budget in a repeat. This is "
            "required when the matrix is used to reproduce converged adaptive search."
        ),
    )
    parser.add_argument("--search-root", choices=("primary", "detail", "target"), default="primary")
    parser.add_argument("--max-atomic-steps", type=int, default=1024)
    parser.add_argument("--max-expanded-nodes", type=int, default=8192)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--root-cpuct", type=float, default=0.9)
    parser.add_argument("--root-prior-temperature", type=float, default=2.0)
    parser.add_argument("--root-initial-children", type=int, default=6)
    parser.add_argument("--root-max-children", type=int, default=32)
    parser.add_argument("--root-widening-c", type=float, default=1.5)
    parser.add_argument("--root-widening-alpha", type=float, default=0.5)
    parser.add_argument("--root-q-probe-weight", type=float, default=0.5)
    parser.add_argument("--root-q-prior-mix", type=float, default=0.25)
    parser.add_argument("--root-q-prior-temperature", type=float, default=0.02)
    parser.add_argument("--root-q-probe-particles", type=int, default=1)
    parser.add_argument("--root-prior-protected-children", type=int, default=2)
    parser.add_argument("--root-selection-min-visits", type=int, default=3)
    parser.add_argument("--root-selection-visit-fraction", type=float, default=0.0)
    parser.add_argument("--root-selection-min-q-advantage", type=float, default=0.002)
    parser.add_argument("--refresh-counterfactuals", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


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
        "log_games_dir": None,
        "metrics_dir": None,
    }


def parse_matrix(value: str) -> tuple[tuple[int, int, int], ...]:
    rows = []
    for item in value.split(","):
        depth, budget, repeats = (int(part) for part in item.split(":"))
        if depth < 1 or budget < 1 or repeats < 1:
            raise ValueError(f"invalid search matrix item: {item}")
        rows.append((depth, budget, repeats))
    return tuple(rows)


def load_game(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 1:
        raise RuntimeError(f"expected one complete game in {path}, found {len(records)}")
    game = records[0]
    if game.get("kind") != "terminal":
        raise RuntimeError(f"source game is not terminal: {path}")
    return game


def country_state(obs: dict[str, Any], country_id: str) -> dict[str, Any]:
    for country in obs.get("countries", []):
        if str(country.get("id")) == country_id:
            return {
                key: country.get(key)
                for key in ("id", "name", "us", "ussr", "control", "stability", "bg", "region")
            }
    raise KeyError(country_id)


def replay_root(
    env: TwilightStruggleEnv,
    game: dict[str, Any],
    target_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    obs, _info = env.reset(seed=int(game["seed"]))
    for index, record in enumerate(game["actions"][:target_index]):
        obs, _reward, terminated, truncated, info = env.step(find_action_index(env.legal_actions, record["action"]))
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
            f"legal ordering mismatch at action {target_index}: logged={target['action_index']} replay={matched}"
        )
    validate_information_safe_observation(obs, root_side(env.last_obs))
    return obs, env.snapshot(compact=True), target


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(payload.encode("ascii"), digest_size=16).hexdigest()


def write_snapshot_fixture(
    output_dir: Path,
    spec: ProbeSpec,
    filename: str,
    snapshot: dict[str, Any],
) -> tuple[Path, str]:
    fixture_dir = output_dir / "fixtures" / spec.name
    fixture_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = fixture_dir / filename
    with gzip.open(snapshot_path, "wt", encoding="utf-8") as handle:
        json.dump(snapshot, handle, sort_keys=True)
    return snapshot_path, snapshot_digest(snapshot)


def information_mask_report(obs: dict[str, np.ndarray], side: str) -> dict[str, Any]:
    return {
        "acting_side": side,
        "us_hand_visible": float(np.asarray(obs.get("us_hand", [])).sum()),
        "ussr_hand_visible": float(np.asarray(obs.get("ussr_hand", [])).sum()),
    }


def save_fixture(
    output_dir: Path,
    spec: ProbeSpec,
    source_path: Path,
    game: dict[str, Any],
    env: TwilightStruggleEnv,
    obs: dict[str, np.ndarray],
    snapshot: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    fixture_dir = output_dir / "fixtures" / spec.name
    snapshot_path, digest = write_snapshot_fixture(output_dir, spec, "snapshot.json.gz", snapshot)
    fixture = {
        "name": spec.name,
        "source_log": str(source_path.resolve()),
        "source_seed": int(game["seed"]),
        "source_policy": spec.source_policy,
        "root_action_index": spec.root_action_index,
        "detail_action_index": spec.detail_action_index,
        "kind": spec.kind,
        "note": spec.note,
        "before": copy.deepcopy(target["before"]),
        "logged_action": copy.deepcopy(target["action"]),
        "logged_action_index": int(target["action_index"]),
        "legal_actions": copy.deepcopy(env.legal_actions),
        "legal_action_count": len(env.legal_actions),
        "observation_digest": observation_digest(obs),
        "snapshot_digest": digest,
        "snapshot": str(snapshot_path.resolve()),
        "information_mask": information_mask_report(obs, root_side(env.last_obs)),
        "suspicious_country": country_state(env.last_obs or {}, spec.suspicious_country),
    }
    (fixture_dir / "fixture.json").write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fixture


def root_value(network: FrozenNetwork, obs: dict[str, np.ndarray], side: str) -> dict[str, float]:
    raw_us, calibrated_us = network.value_us(obs, side)
    sign = 1.0 if side == "us" else -1.0
    return {
        "raw_us": float(raw_us),
        "calibrated_us": float(calibrated_us),
        "raw_root": float(sign * raw_us),
        "calibrated_root": float(sign * calibrated_us),
    }


def raw_policy_report(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    obs: dict[str, np.ndarray],
    logged_action: dict[str, Any],
) -> dict[str, Any]:
    logged_index = find_action_index(env.legal_actions, logged_action)
    raw = network.raw(obs, len(env.legal_actions))
    probabilities = np.asarray(raw["probabilities"], dtype=np.float64)
    order = np.argsort(-probabilities)
    entropy = -float(np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0)))
    rows = [
        {
            "rank": rank,
            "action_index": int(index),
            "action": action_text(env.legal_actions[int(index)]),
            "probability": float(probabilities[index]),
            "logit": float(raw["logits"][index]),
            "logged": int(index) == logged_index,
        }
        for rank, index in enumerate(order[: min(15, len(order))], 1)
    ]
    return {
        "legal_count": len(env.legal_actions),
        "entropy": entropy,
        "normalized_entropy": entropy / max(math.log(max(2, len(probabilities))), 1e-12),
        "top1_probability": float(probabilities[order[0]]),
        "top1_top2_gap": float(probabilities[order[0]] - probabilities[order[1]]) if len(order) > 1 else 1.0,
        "logged_action_rank": int(np.where(order == logged_index)[0][0]) + 1,
        "logged_action_probability": float(probabilities[logged_index]),
        "critic": root_value(network, obs, root_side(env.last_obs)),
        "top_actions": rows,
    }


def logged_path_report(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    snapshot: dict[str, Any],
    game: dict[str, Any],
    start: int,
    length: int,
) -> list[dict[str, Any]]:
    env.restore(snapshot)
    rows = []
    for offset, record in enumerate(game["actions"][start : start + length]):
        wanted = record["action"]
        index = find_action_index(env.legal_actions, wanted)
        raw = network.raw(env.encode_current_observation(env.last_obs or {}), len(env.legal_actions))
        order = np.argsort(-raw["probabilities"])
        rows.append(
            {
                "offset": offset,
                "prompt": str((env.last_obs or {}).get("prompt") or ""),
                "side": root_side(env.last_obs),
                "action": action_text(env.legal_actions[index]),
                "rank": int(np.where(order == index)[0][0]) + 1,
                "probability": float(raw["probabilities"][index]),
                "logit": float(raw["logits"][index]),
                "legal_count": len(env.legal_actions),
            }
        )
        _obs, _reward, terminated, truncated, info = env.step(index)
        if terminated or truncated:
            rows[-1]["terminal"] = info
            break
    return rows


def evaluate_loaded_state(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    obs: dict[str, np.ndarray],
    root_player: str,
    terminated: bool = False,
    truncated: bool = False,
    info: dict[str, Any] | None = None,
) -> float:
    if terminated or truncated:
        winner = str((info or {}).get("winner") or "")
        value_us = 1.0 if winner == "us" else -1.0 if winner == "ussr" else 0.0
    else:
        _raw_us, value_us = network.value_us(obs, root_side(env.last_obs))
    return float(value_us if root_player == "us" else -value_us)


def influence_counterfactuals(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    snapshot: dict[str, Any],
    spec: ProbeSpec,
    beam_width: int,
) -> dict[str, Any]:
    root_obs = env.restore(snapshot)
    player = root_side(env.last_obs)
    before = root_value(network, root_obs, player)["calibrated_root"]
    first_actions = copy.deepcopy(env.legal_actions)
    raw = network.raw(root_obs, len(first_actions))
    first_rows = []
    for index, action in enumerate(first_actions):
        env.restore(snapshot)
        next_obs, _reward, terminated, truncated, info = env.step(index)
        q = evaluate_loaded_state(env, network, next_obs, player, terminated, truncated, info)
        first_rows.append(
            {
                "action": action_text(action),
                "country": str(action.get("value") or ""),
                "policy_probability": float(raw["probabilities"][index]),
                "V_after_root": q,
                "delta_V": q - before,
                "suspicious": str(action.get("value") or "") == spec.suspicious_country,
            }
        )

    env.restore(snapshot)
    current_snapshot = snapshot
    curve = []
    for amount in range(1, spec.detail_macro_length + 1):
        env.restore(current_snapshot)
        candidates = [
            index for index, action in enumerate(env.legal_actions)
            if str(action.get("value") or "") == spec.suspicious_country
        ]
        if len(candidates) != 1:
            break
        obs, _reward, terminated, truncated, info = env.step(candidates[0])
        q = evaluate_loaded_state(env, network, obs, player, terminated, truncated, info)
        curve.append(
            {
                "influence_added": amount,
                "country": country_state(env.last_obs or {}, spec.suspicious_country),
                "side_after": root_side(env.last_obs),
                "prompt_after": str((env.last_obs or {}).get("prompt") or ""),
                "acting_side_changed": root_side(env.last_obs) != player,
                "V_after_root": q,
                "delta_from_root": q - before,
                "incremental_delta": q - (curve[-1]["V_after_root"] if curve else before),
            }
        )
        current_snapshot = env.snapshot(compact=True)

    ranked = sorted(first_rows, key=lambda row: -row["V_after_root"])
    probe_countries = [spec.suspicious_country]
    probe_countries.extend(
        str(row["country"])
        for row in ranked
        if str(row["country"]) != spec.suspicious_country
    )
    allocation_rows = []
    for country_id in probe_countries[: max(2, beam_width + 1)]:
        env.restore(snapshot)
        path: list[str] = []
        value = before
        complete = False
        for _depth in range(spec.detail_macro_length):
            prompt = str((env.last_obs or {}).get("prompt") or "").lower()
            if not prompt.startswith("place "):
                complete = True
                break
            candidates = [
                index
                for index, action in enumerate(env.legal_actions)
                if str(action.get("value") or "") == country_id
            ]
            if len(candidates) != 1:
                break
            action = copy.deepcopy(env.legal_actions[candidates[0]])
            obs, _reward, terminated, truncated, info = env.step_with_forced_rolls(candidates[0], [])
            path.append(action_text(action))
            value = evaluate_loaded_state(env, network, obs, player, terminated, truncated, info)
            if terminated or truncated:
                complete = True
                break
        if not complete:
            complete = not str((env.last_obs or {}).get("prompt") or "").lower().startswith("place ")
        allocation_rows.append(
            {
                "allocation": path,
                "country": country_id,
                "complete": complete,
                "V_after_root": value,
                "delta_V": value - before,
            }
        )
    allocation_rows.sort(key=lambda row: -row["V_after_root"])
    env.clear_runtime_checkpoints()
    return {
        "V_before_root": before,
        "first_placement_values": sorted(first_rows, key=lambda row: -row["V_after_root"]),
        "suspicious_country_marginal_curve": curve,
        "complete_allocation_method": "repeat one strategically selected country until allocation completion",
        "critic_complete_allocation_probes": allocation_rows,
    }


def exact_realignment_targets(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    detail_snapshot: dict[str, Any],
    suspicious_country: str,
) -> dict[str, Any]:
    obs = env.restore(detail_snapshot)
    player = root_side(env.last_obs)
    before = root_value(network, obs, player)["calibrated_root"]
    actions = copy.deepcopy(env.legal_actions)
    raw = network.raw(obs, len(actions))
    stepper = ExactChanceEnumerator(max_outcomes=36)
    local = local_search_snapshot(env)
    rows = []
    for index, action in enumerate(actions):
        restore_search_state(env, local)
        outcomes = stepper.outcomes(env, local, index)
        q = 0.0
        outcome_rows = []
        for outcome in outcomes:
            restore_search_state(env, outcome.snapshot)
            value = evaluate_loaded_state(
                env,
                network,
                outcome.obs,
                player,
                outcome.terminated,
                outcome.truncated,
                outcome.info,
            )
            q += outcome.probability * value
            outcome_rows.append(
                {
                    "probability": outcome.probability,
                    "chance_rolls": list(outcome.chance_rolls),
                    "V_root": value,
                }
            )
        rows.append(
            {
                "action": action_text(action),
                "country": str(action.get("value") or ""),
                "policy_probability": float(raw["probabilities"][index]),
                "Q_exact_immediate": q,
                "delta_V": q - before,
                "outcome_count": len(outcomes),
                "suspicious": str(action.get("value") or "") == suspicious_country,
                "outcomes": outcome_rows if str(action.get("value") or "") == suspicious_country else None,
            }
        )
    env.clear_runtime_checkpoints()
    return {
        "V_before_root": before,
        "exact_chance_nodes": stepper.enumerated_nodes,
        "sampled_large_chance_nodes": stepper.sampled_large_nodes,
        "targets": sorted(rows, key=lambda row: -row["Q_exact_immediate"]),
    }


def search_once(
    env: TwilightStruggleEnv,
    network: FrozenNetwork,
    snapshot: dict[str, Any],
    depth: int,
    budget: int,
    particles: int,
    seed: int,
    max_atomic_steps: int,
    max_expanded_nodes: int,
    root_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obs = env.restore(snapshot)
    config = SelectivePuctConfig(
        initial_simulations=budget,
        medium_simulations=budget,
        max_simulations=budget,
        belief_particles=particles,
        meaningful_depth=depth,
        max_atomic_steps=max_atomic_steps,
        max_expanded_nodes=max_expanded_nodes,
        temperature=0.0,
        behavior_temperature=0.0,
        **(root_options or {}),
    )
    session = ExactBeliefPuctSession(network, config, seed=seed)
    started = time.perf_counter()
    session.begin(env, snapshot, obs)
    assert session.root_node is not None
    prior_order = [int(index) for index in session.root_node.ordered_actions]
    first_visit: dict[int, int] = {}
    evolution: list[dict[str, Any]] = []
    checkpoints = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, budget}
    previous_choice: int | None = None
    for simulation in range(1, budget + 1):
        session.advance_to(env, simulation)
        visits, q_values = session.root_arrays()
        for index in np.flatnonzero(visits > 0):
            first_visit.setdefault(int(index), simulation)
        current_choice, selection = session.select_root_action()
        if simulation in checkpoints or current_choice != previous_choice:
            top = sorted(
                range(len(visits)),
                key=lambda index: (-float(visits[index]), -float(q_values[index]), -float(session.root_node.priors[index])),
            )[:10]
            evolution.append(
                {
                    "simulation": simulation,
                    "selected_action": session.root_node.labels[current_choice],
                    "selection": selection,
                    "active_count": int(session.root_node.active.sum()),
                    "visited_count": int(np.sum(visits > 0)),
                    "top": [
                        {
                            "action": session.root_node.labels[index],
                            "N": float(visits[index]),
                            "Q": float(q_values[index]),
                            "P": float(session.root_node.priors[index]),
                        }
                        for index in top
                    ],
                }
            )
        previous_choice = current_choice
    latency = time.perf_counter() - started
    visits, q_root = session.root_arrays()
    choice, root_selection = session.select_root_action()
    table = session.root_table()
    visited = np.flatnonzero(visits > 0)
    child_nodes = []
    for key, node in session.nodes.items():
        if node is session.root_node:
            continue
        child_nodes.append(
            {
                "information_state": key,
                "legal_count": len(node.priors),
                "active_count": int(node.active.sum()),
                "visited_count": int(np.sum(node.visits > 0)),
                "total_visits": float(node.visits.sum()),
            }
        )
    child_nodes.sort(key=lambda row: (-row["legal_count"], -row["total_visits"]))
    q_visited = sorted((float(q_root[index]) for index in visited), reverse=True)
    result = {
        "depth": depth,
        "budget": budget,
        "belief_particles": particles,
        "seed": seed,
        "choice": session.root_node.labels[choice],
        "choice_action_index": choice,
        "choice_Q": float(q_root[choice]),
        "root_selection": root_selection,
        "root_Q_gap": q_visited[0] - q_visited[1] if len(q_visited) > 1 else 0.0,
        "root_table": table,
        "root_legal_count": len(visits),
        "root_expanded_actions": int(len(visited)),
        "root_unvisited_actions": int(len(visits) - len(visited)),
        "prior_order": [
            {
                "rank": rank,
                "action_index": index,
                "action": session.root_node.labels[index],
                "P": float(session.root_node.priors[index]),
                "first_visit_simulation": first_visit.get(index),
            }
            for rank, index in enumerate(prior_order, 1)
        ],
        "root_candidate_admission": sorted(
            session.root_admission_events,
            key=lambda row: int(row["admission_rank"]),
        ),
        "visit_q_evolution": evolution,
        "root_active_actions": int(session.root_node.active.sum()),
        "best_trace": list(session.best_trace(choice)),
        "expanded_nodes": len(session.nodes),
        "child_progressive_widening": child_nodes[:20],
        "exact_chance_nodes": session.stepper.enumerated_nodes,
        "sampled_large_chance_nodes": session.stepper.sampled_large_nodes,
        "root_probe_exact_chance_nodes": session.root_probe_stepper.enumerated_nodes,
        "root_probe_sampled_large_chance_nodes": session.root_probe_stepper.sampled_large_nodes,
        "max_exact_outcomes": session.stepper.max_outcomes,
        "chance_cache_hits": session.stepper.chance_cache_hits,
        "boundary_evaluations": session.boundary_evaluations,
        "terminal_evaluations": session.terminal_evaluations,
        "atomic_horizon_cutoffs": session.atomic_horizon_cutoffs,
        "node_budget_cutoffs": session.node_budget_cutoffs,
        "max_meaningful_depth_reached": session.max_meaningful_depth_reached,
        "max_turn_delta": session.max_turn_delta,
        "cross_turn_leaf_evaluations": session.cross_turn_leaf_evaluations,
        "latency_seconds": latency,
        "simulations_per_second": budget / max(latency, 1e-9),
    }
    env.clear_runtime_checkpoints()
    return result


def run_position(
    args: argparse.Namespace,
    spec: ProbeSpec,
    matrix: tuple[tuple[int, int, int], ...],
    network: FrozenNetwork,
) -> dict[str, Any]:
    output_path = args.output_dir / f"{spec.name}.json"
    expected_searches = sum(repeats for _depth, _budget, repeats in matrix)
    existing: dict[str, Any] | None = None
    searches_complete = False
    if args.resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        searches_complete = len(existing.get("search", [])) >= expected_searches
    source_path = (args.log_root / spec.source_log).resolve()
    game = load_game(source_path)
    env = TwilightStruggleEnv(env_config())
    try:
        obs, snapshot, target = replay_root(env, game, spec.root_action_index)
        if existing is None:
            env.restore(snapshot)
            fixture = save_fixture(args.output_dir, spec, source_path, game, env, obs, snapshot, target)
            raw = raw_policy_report(env, network, obs, target["action"])
            path = logged_path_report(
                env,
                network,
                snapshot,
                game,
                spec.root_action_index,
                spec.logged_macro_length,
            )
            detail_obs, detail_snapshot, detail_target = replay_root(env, game, spec.detail_action_index)
            detail_actor = raw_policy_report(env, network, detail_obs, detail_target["action"])
            detail_fixture = {
                "action_index": spec.detail_action_index,
                "before": detail_target["before"],
                "logged_action": detail_target["action"],
                "legal_actions": copy.deepcopy(env.legal_actions),
                "observation_digest": observation_digest(detail_obs),
                "suspicious_country": country_state(env.last_obs or {}, spec.suspicious_country),
            }
            counterfactuals: dict[str, Any]
            if spec.kind == "influence":
                counterfactuals = influence_counterfactuals(
                    env,
                    network,
                    detail_snapshot,
                    spec,
                    args.beam_width,
                )
            else:
                if spec.target_action_index is None:
                    raise RuntimeError(f"realignment probe {spec.name} has no target action index")
                target_obs, target_snapshot, target_action = replay_root(env, game, spec.target_action_index)
                counterfactuals = {
                    "target_fixture": {
                        "action_index": spec.target_action_index,
                        "before": target_action["before"],
                        "logged_action": target_action["action"],
                        "legal_actions": copy.deepcopy(env.legal_actions),
                        "observation_digest": observation_digest(target_obs),
                    },
                    "realignment_targets": exact_realignment_targets(
                        env,
                        network,
                        target_snapshot,
                        spec.suspicious_country,
                    ),
                }
            result = {
                "spec": asdict(spec),
                "checkpoint": str(args.checkpoint.resolve()),
                "fixture": fixture,
                "raw_actor": raw,
                "detail_fixture": detail_fixture,
                "detail_actor": detail_actor,
                "logged_hierarchical_path": path,
                "counterfactuals": counterfactuals,
                "search_root": args.search_root,
                "search": [],
            }
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            result = existing
            detail_obs, detail_snapshot, detail_target = replay_root(env, game, spec.detail_action_index)
        detail_snapshot_path, detail_snapshot_digest = write_snapshot_fixture(
            args.output_dir,
            spec,
            "detail_snapshot.json.gz",
            detail_snapshot,
        )
        result["detail_fixture"]["snapshot"] = str(detail_snapshot_path.resolve())
        result["detail_fixture"]["snapshot_digest"] = detail_snapshot_digest
        result["detail_fixture"]["information_mask"] = information_mask_report(
            detail_obs,
            str(result["detail_fixture"]["before"]["side"]),
        )
        target_snapshot: dict[str, Any] | None = None
        if spec.target_action_index is not None:
            target_obs, target_snapshot, target_action = replay_root(env, game, spec.target_action_index)
            target_snapshot_path, target_snapshot_digest = write_snapshot_fixture(
                args.output_dir,
                spec,
                "target_snapshot.json.gz",
                target_snapshot,
            )
            result["counterfactuals"]["target_fixture"]["snapshot"] = str(target_snapshot_path.resolve())
            result["counterfactuals"]["target_fixture"]["snapshot_digest"] = target_snapshot_digest
            result["counterfactuals"]["target_fixture"]["information_mask"] = information_mask_report(
                target_obs,
                str(result["counterfactuals"]["target_fixture"]["before"]["side"]),
            )
            result["target_actor"] = raw_policy_report(
                env,
                network,
                target_obs,
                target_action["action"],
            )
        if args.refresh_counterfactuals:
            if spec.kind == "influence":
                result["counterfactuals"] = influence_counterfactuals(
                    env,
                    network,
                    detail_snapshot,
                    spec,
                    args.beam_width,
                )
            else:
                if target_snapshot is None:
                    raise RuntimeError(f"realignment probe {spec.name} has no target search root")
                result["counterfactuals"]["realignment_targets"] = exact_realignment_targets(
                    env,
                    network,
                    target_snapshot,
                    spec.suspicious_country,
                )
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if searches_complete:
            return result
        if args.search_root == "primary":
            search_snapshot = snapshot
        elif args.search_root == "detail":
            search_snapshot = detail_snapshot
        else:
            if target_snapshot is None:
                raise RuntimeError(f"probe {spec.name} has no target search root")
            search_snapshot = target_snapshot
        completed_keys = {
            (int(row["depth"]), int(row["budget"]), int(row["repeat"]))
            for row in result.get("search", [])
        }
        for depth, budget, repeats in matrix:
            for repeat in range(repeats):
                if (depth, budget, repeat) in completed_keys:
                    continue
                seed = args.search_seed + 10_000_019 * list(SPECS).index(spec.name) + repeat
                if not args.paired_matrix_seeds:
                    seed += 100_003 * depth
                print(
                    json.dumps(
                        {
                            "event": "search_start",
                            "position": spec.name,
                            "depth": depth,
                            "budget": budget,
                            "repeat": repeat,
                            "seed": seed,
                        }
                    ),
                    flush=True,
                )
                search = search_once(
                    env,
                    network,
                    search_snapshot,
                    depth,
                    budget,
                    args.belief_particles,
                    seed,
                    args.max_atomic_steps,
                    args.max_expanded_nodes,
                    {
                        "root_cpuct": args.root_cpuct,
                        "root_prior_temperature": args.root_prior_temperature,
                        "root_initial_children": args.root_initial_children,
                        "root_max_children": args.root_max_children,
                        "root_progressive_widening_c": args.root_widening_c,
                        "root_progressive_widening_alpha": args.root_widening_alpha,
                        "root_q_probe_weight": args.root_q_probe_weight,
                        "root_q_prior_mix": args.root_q_prior_mix,
                        "root_q_prior_temperature": args.root_q_prior_temperature,
                        "root_q_probe_particles": args.root_q_probe_particles,
                        "root_prior_protected_children": args.root_prior_protected_children,
                        "root_selection_min_visits": args.root_selection_min_visits,
                        "root_selection_visit_fraction": args.root_selection_visit_fraction,
                        "root_selection_min_q_advantage": args.root_selection_min_q_advantage,
                    },
                )
                search["repeat"] = repeat
                result["search"].append(search)
                output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "event": "search_done",
                            "position": spec.name,
                            "depth": depth,
                            "budget": budget,
                            "repeat": repeat,
                            "choice": search["choice"],
                            "Q": search["choice_Q"],
                            "seconds": search["latency_seconds"],
                            "cutoffs": search["atomic_horizon_cutoffs"] + search["node_budget_cutoffs"],
                        }
                    ),
                    flush=True,
                )
        return result
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.log_root = args.log_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = [name.strip() for name in args.positions.split(",") if name.strip()]
    unknown = sorted(set(names) - set(SPECS))
    if unknown:
        raise ValueError(f"unknown positions: {unknown}")
    matrix = parse_matrix(args.matrix)
    model = load_checkpoint_model(args.checkpoint, args.device)
    network = FrozenNetwork(model)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "matrix_parsed": matrix,
        "frozen": True,
        "training_updates": 0,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"config": str((args.output_dir / "config.json").resolve()), "results": {}}
    for name in names:
        print(json.dumps({"event": "position_start", "position": name}), flush=True)
        result = run_position(args, SPECS[name], matrix, network)
        path = args.output_dir / f"{name}.json"
        manifest["results"][name] = str(path.resolve())
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "position_done",
                    "position": name,
                    "raw_choice": result["raw_actor"]["top_actions"][0]["action"],
                    "search_runs": len(result["search"]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
