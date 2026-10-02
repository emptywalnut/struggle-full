#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.batched_inference import load_checkpoint_model
from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.frozen_diagnostic import FrozenNetwork, action_text
from struggle_ai.mcts import q_aware_root_action, root_side, validate_information_safe_observation
from struggle_ai.search_replay import (
    SearchReplayExample,
    SearchReplayMetadata,
    SearchReplayWriter,
    json_safe_action,
    padded,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Q-aware actor replay from frozen anomaly probes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--q-temperature", type=float, default=0.02)
    parser.add_argument("--visit-weight", type=float, default=0.20)
    parser.add_argument("--q-weight", type=float, default=0.60)
    parser.add_argument("--selected-weight", type=float, default=0.20)
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


def load_snapshot(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def q_softmax(q_values: np.ndarray, eligible: np.ndarray, temperature: float) -> np.ndarray:
    result = np.zeros_like(q_values, dtype=np.float64)
    if not bool(eligible.any()):
        return result
    shifted = (q_values[eligible] - float(np.max(q_values[eligible]))) / max(float(temperature), 1e-6)
    weights = np.exp(np.clip(shifted, -60.0, 0.0))
    result[eligible] = weights / max(float(weights.sum()), 1e-30)
    return result


def counterfactual_q(
    result: dict[str, Any],
    labels: tuple[str, ...],
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = fallback.copy()
    known = np.zeros(len(labels), dtype=np.bool_)
    counterfactuals = result.get("counterfactuals", {})
    rows = counterfactuals.get("first_placement_values")
    value_key = "V_after_root"
    if rows is None:
        rows = counterfactuals.get("realignment_targets", {}).get("targets", [])
        value_key = "Q_exact_immediate"
    by_label = {str(row["action"]): float(row[value_key]) for row in rows or ()}
    for index, label in enumerate(labels):
        if label in by_label:
            values[index] = by_label[label]
            known[index] = True
    return values, known


def build_example(
    name: str,
    results: list[dict[str, Any]],
    network: FrozenNetwork,
    args: argparse.Namespace,
) -> tuple[SearchReplayExample, dict[str, Any]]:
    representative = results[0]
    kind = str(representative["spec"]["kind"])
    fixture = (
        representative["detail_fixture"]
        if kind == "influence"
        else representative["counterfactuals"]["target_fixture"]
    )
    snapshot = load_snapshot(Path(fixture["snapshot"]))
    env = TwilightStruggleEnv(env_config())
    try:
        observation = env.restore(snapshot)
        side = root_side(env.last_obs)
        validate_information_safe_observation(observation, side)
        legal_actions = list(env.legal_actions)
        labels = tuple(action_text(action) for action in legal_actions)
        raw = network.raw(observation, len(legal_actions))
        raw_policy = np.asarray(raw["probabilities"], dtype=np.float64)
        visits = np.zeros(len(labels), dtype=np.float64)
        q_numerator = np.zeros(len(labels), dtype=np.float64)
        q_denominator = np.zeros(len(labels), dtype=np.float64)
        used_runs = []
        for result in results:
            for search in result.get("search", []):
                cutoffs = int(search["atomic_horizon_cutoffs"]) + int(search["node_budget_cutoffs"])
                if cutoffs:
                    raise RuntimeError(f"{name} contains cutoff-limited teacher search")
                table = {int(row["action_index"]): row for row in search["root_table"]}
                if len(table) != len(labels):
                    raise RuntimeError(f"{name} legal action count changed across probe results")
                for index in range(len(labels)):
                    count = float(table[index]["N"])
                    visits[index] += count
                    q_numerator[index] += count * float(table[index]["Q"])
                    q_denominator[index] += count
                used_runs.append(
                    {
                        "depth": int(search["depth"]),
                        "budget": int(search["budget"]),
                        "repeat": int(search["repeat"]),
                        "choice": str(search["choice"]),
                    }
                )
        if not used_runs:
            raise RuntimeError(f"{name} has no search runs")
        search_q = np.divide(
            q_numerator,
            q_denominator,
            out=np.zeros_like(q_numerator),
            where=q_denominator > 0.0,
        )
        search_selected, selection = q_aware_root_action(
            visits,
            search_q,
            raw_policy,
            labels,
            min_visits=3,
            visit_fraction=0.25,
            min_q_advantage=0.002,
        )
        visit_policy = visits / max(float(visits.sum()), 1e-30)
        eligible = visits >= max(3.0, float(visits.max()) * 0.25)
        search_q_policy = q_softmax(search_q, eligible, args.q_temperature)
        ranking_q, ranking_known = counterfactual_q(representative, labels, search_q)
        counterfactual_q_policy = q_softmax(ranking_q, ranking_known, args.q_temperature)
        q_policy = 0.5 * search_q_policy + 0.5 * counterfactual_q_policy
        raw_action = int(np.argmax(raw_policy))
        ranking_best = int(np.argmax(np.where(ranking_known, ranking_q, -np.inf)))
        corrected_selected = (
            search_selected
            if ranking_q[search_selected] + 1e-12 >= ranking_q[raw_action]
            else ranking_best
        )
        selected_policy = np.zeros_like(visits)
        selected_policy[corrected_selected] = 1.0
        weights = np.asarray(
            [args.visit_weight, args.q_weight, args.selected_weight],
            dtype=np.float64,
        )
        weights /= max(float(weights.sum()), 1e-30)
        search_policy = weights[0] * visit_policy + weights[1] * q_policy + weights[2] * selected_policy
        search_policy /= max(float(search_policy.sum()), 1e-30)
        selected = int(np.argmax(search_policy))
        hierarchy_level = "influence_allocation" if kind == "influence" else "realignment_target"
        output_size = int(np.asarray(observation["action_mask"]).shape[0])
        metadata = SearchReplayMetadata(
            side=side,
            trigger_categories=(hierarchy_level, "anomaly_actor_correction"),
            legal_actions=tuple(json_safe_action(action) for action in legal_actions),
            hierarchical_path=({"side": side, "action": labels[selected], "level": hierarchy_level},),
            raw_action=raw_action,
            search_action=selected,
            search_depth=max(int(run["depth"]) for run in used_runs),
            simulation_budget=sum(int(run["budget"]) for run in used_runs),
            root_q_gap=float(selection["q_advantage"]),
            action_flipped=selected != raw_action,
            source_game=f"frozen-anomaly:{name}",
            source_step=int(fixture["action_index"]),
            q_advantage_over_raw=float(ranking_q[ranking_best] - ranking_q[raw_action]),
            hierarchy_index=1,
            hierarchy_level=hierarchy_level,
            teacher_action_visits=float(visits[selected]),
            total_visits=float(visits.sum()),
            teacher_visit_share=float(visits[selected] / max(float(visits.sum()), 1e-30)),
            behavior_policy="corrected_q_aware_frozen_probe",
        )
        example = SearchReplayExample(
            observation={key: np.asarray(value, dtype=np.float32).copy() for key, value in observation.items()},
            raw_policy=padded(raw_policy, output_size),
            visit_counts=padded(visits, output_size),
            search_policy=padded(search_policy, output_size),
            q_values=padded(ranking_q, output_size),
            metadata=metadata,
        )
        report = {
            "name": name,
            "side": side,
            "hierarchy_level": hierarchy_level,
            "raw_action": labels[raw_action],
            "teacher_action": labels[selected],
            "teacher_probability": float(search_policy[selected]),
            "raw_teacher_probability": float(raw_policy[selected]),
            "raw_suspicious_probability": float(raw_policy[raw_action]),
            "q_advantage_over_raw": float(ranking_q[ranking_best] - ranking_q[raw_action]),
            "selection": selection,
            "search_selected_action": labels[search_selected],
            "counterfactual_best_action": labels[ranking_best],
            "corrected_selected_action": labels[corrected_selected],
            "runs": used_runs,
        }
        return example, report
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in args.result:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        grouped[str(payload["spec"]["name"])].append(payload)
    network = FrozenNetwork(load_checkpoint_model(args.checkpoint.expanduser().resolve(), args.device))
    reports = []
    with SearchReplayWriter(args.output_dir, shard_size=16) as writer:
        for name in sorted(grouped):
            example, report = build_example(name, grouped[name], network, args)
            writer.add(example)
            reports.append(report)
    report_path = args.output_dir / "actor_correction_replay_report.json"
    report_path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "actor_correction_replay_done", "examples": len(reports), "report": str(report_path)}))


if __name__ == "__main__":
    main()
