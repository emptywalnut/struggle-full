#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def wilson_score_interval(score_sum: float, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    proportion = float(score_sum) / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / games + z * z / (4.0 * games * games)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def score_to_elo(score: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, float(score)))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paired PUCT evaluations into a four-agent Elo table.")
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument("--result-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="parent_raw")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=8_211_001)
    return parser.parse_args()


def resolve_result_paths(results: list[Path], directories: list[Path]) -> list[Path]:
    paths = [path.expanduser().resolve() for path in results]
    for directory in directories:
        paths.extend(directory.expanduser().resolve().glob("shard-*/results.json"))
    unique = sorted(set(paths))
    if not unique:
        raise ValueError("no results.json inputs were found")
    return unique


def load_matches(paths: list[Path]) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    agents: set[str] = set()
    matches: list[dict[str, Any]] = []
    incomplete_blocks: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        candidate = str(payload["candidate"]["label"])
        opponent = str(payload["opponent"]["label"])
        agents.update((candidate, opponent))
        records_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload.get("records", []):
            records_by_seed[int(record["seed"])].append(record)
        for seed, seed_records in records_by_seed.items():
            orientations = {str(record["orientation"]) for record in seed_records}
            expected = {"candidate_us", "candidate_ussr"}
            if len(seed_records) != 2 or orientations != expected:
                incomplete_blocks.append({
                    "source": str(path.expanduser().resolve()),
                    "seed": seed,
                    "orientations": sorted(orientations),
                    "record_count": len(seed_records),
                })
                continue
            for record in seed_records:
                matches.append(
                    {
                        "candidate": candidate,
                        "opponent": opponent,
                        "score": float(record["candidate_score"]),
                        "seed": seed,
                        "orientation": str(record["orientation"]),
                        "terminal_reason": str(record.get("terminal_reason") or "unknown"),
                        "wall_clock_seconds": float(record.get("wall_clock_seconds") or 0.0),
                        "candidate_decisions": int(record.get("candidate_decisions") or 0),
                        "opponent_decisions": int(record.get("opponent_decisions") or 0),
                        "search_events": list(record.get("search_events") or []),
                        "source": str(path.expanduser().resolve()),
                    }
                )
    return sorted(agents), matches, incomplete_blocks


def fit_elo(agents: list[str], matches: list[dict[str, Any]], baseline: str) -> dict[str, float]:
    if baseline not in agents:
        raise ValueError(f"baseline {baseline!r} is not present in {agents}")
    free = [agent for agent in agents if agent != baseline]
    free_index = {agent: index for index, agent in enumerate(free)}
    ratings = np.zeros(len(free), dtype=np.float64)
    scale = math.log(10.0) / 400.0
    for _ in range(80):
        gradient = np.zeros_like(ratings)
        hessian = np.eye(len(free), dtype=np.float64) * -1e-6
        for match in matches:
            candidate = match["candidate"]
            opponent = match["opponent"]
            candidate_rating = ratings[free_index[candidate]] if candidate != baseline else 0.0
            opponent_rating = ratings[free_index[opponent]] if opponent != baseline else 0.0
            probability = 1.0 / (1.0 + math.exp(-scale * (candidate_rating - opponent_rating)))
            derivative = scale * (float(match["score"]) - probability)
            curvature = -(scale**2) * probability * (1.0 - probability)
            vector = np.zeros_like(ratings)
            if candidate != baseline:
                vector[free_index[candidate]] += 1.0
            if opponent != baseline:
                vector[free_index[opponent]] -= 1.0
            gradient += derivative * vector
            hessian += curvature * np.outer(vector, vector)
        if not len(free):
            break
        step = np.linalg.solve(hessian, gradient)
        ratings -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return {baseline: 0.0, **{agent: float(ratings[index]) for agent, index in free_index.items()}}


def bootstrap_elo(
    agents: list[str],
    matches: list[dict[str, Any]],
    baseline: str,
    samples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    blocks: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        blocks[(match["source"], match["seed"])].append(match)
    values = list(blocks.values())
    if not values:
        return {agent: (0.0, 0.0) for agent in agents}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled: list[dict[str, Any]] = []
        for index in rng.integers(0, len(values), size=len(values)):
            sampled.extend(values[int(index)])
        try:
            ratings = fit_elo(agents, sampled, baseline)
        except np.linalg.LinAlgError:
            continue
        for agent, rating in ratings.items():
            draws[agent].append(rating)
    return {
        agent: (
            float(np.quantile(draws[agent], 0.025)) if draws[agent] else 0.0,
            float(np.quantile(draws[agent], 0.975)) if draws[agent] else 0.0,
        )
        for agent in agents
    }


def main() -> None:
    args = parse_args()
    result_paths = resolve_result_paths(args.result, args.result_dir)
    agents, matches, incomplete_blocks = load_matches(result_paths)
    ratings = fit_elo(agents, matches, args.baseline)
    intervals = bootstrap_elo(agents, matches, args.baseline, args.bootstrap_samples, args.seed)
    records_by_agent: dict[str, list[float]] = defaultdict(list)
    records_by_agent_side: dict[tuple[str, str], list[float]] = defaultdict(list)
    for match in matches:
        records_by_agent[match["candidate"]].append(match["score"])
        records_by_agent[match["opponent"]].append(1.0 - match["score"])
        candidate_side = "us" if match["orientation"] == "candidate_us" else "ussr"
        opponent_side = "ussr" if candidate_side == "us" else "us"
        records_by_agent_side[(match["candidate"], candidate_side)].append(match["score"])
        records_by_agent_side[(match["opponent"], opponent_side)].append(1.0 - match["score"])
    table = []
    for agent in sorted(agents, key=lambda item: ratings[item], reverse=True):
        agent_scores = records_by_agent[agent]
        score_low, score_high = wilson_score_interval(sum(agent_scores), len(agent_scores))
        us_scores = records_by_agent_side[(agent, "us")]
        us_low, us_high = wilson_score_interval(sum(us_scores), len(us_scores))
        ussr_scores = records_by_agent_side[(agent, "ussr")]
        ussr_low, ussr_high = wilson_score_interval(sum(ussr_scores), len(ussr_scores))
        table.append({
            "agent": agent,
            "elo": ratings[agent],
            "elo_ci95_low": intervals[agent][0],
            "elo_ci95_high": intervals[agent][1],
            "games": len(agent_scores),
            "score": float(np.mean(agent_scores)) if agent_scores else 0.0,
            "score_wilson_ci95_low": score_low,
            "score_wilson_ci95_high": score_high,
            "us_games": len(us_scores),
            "us_score": float(np.mean(us_scores)) if us_scores else 0.0,
            "us_score_wilson_ci95_low": us_low,
            "us_score_wilson_ci95_high": us_high,
            "us_matchup_elo_delta": score_to_elo(float(np.mean(us_scores))) if us_scores else 0.0,
            "us_matchup_elo_ci95_low": score_to_elo(us_low) if us_scores else 0.0,
            "us_matchup_elo_ci95_high": score_to_elo(us_high) if us_scores else 0.0,
            "ussr_games": len(ussr_scores),
            "ussr_score": float(np.mean(ussr_scores)) if ussr_scores else 0.0,
            "ussr_score_wilson_ci95_low": ussr_low,
            "ussr_score_wilson_ci95_high": ussr_high,
            "ussr_matchup_elo_delta": score_to_elo(float(np.mean(ussr_scores))) if ussr_scores else 0.0,
            "ussr_matchup_elo_ci95_low": score_to_elo(ussr_low) if ussr_scores else 0.0,
            "ussr_matchup_elo_ci95_high": score_to_elo(ussr_high) if ussr_scores else 0.0,
        })
    searched_events: list[tuple[dict[str, Any], float]] = []
    for match in matches:
        for event in match["search_events"]:
            if int(event.get("simulations") or 0) <= 0:
                continue
            agent_score = match["score"] if event.get("agent") == match["candidate"] else 1.0 - match["score"]
            searched_events.append((event, agent_score))
    trigger_values: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for event, agent_score in searched_events:
        for trigger in event.get("triggers", []):
            trigger_values[str(trigger)].append((event, agent_score))
    trigger_metrics = {
        trigger: {
            "frequency": len(values),
            "flip_rate": float(np.mean([bool(event.get("action_flipped")) for event, _ in values])),
            "average_q_advantage_over_raw": float(
                np.mean([float(event.get("q_advantage_over_raw") or 0.0) for event, _ in values])
            ),
            "simulations": int(sum(int(event.get("simulations") or 0) for event, _ in values)),
            "average_expanded_nodes": float(
                np.mean([int(event.get("expanded_nodes") or 0) for event, _ in values])
            ),
            "exact_chance_nodes": int(
                sum(int(event.get("exact_chance_nodes") or 0) for event, _ in values)
            ),
            "sampled_large_chance_nodes": int(
                sum(int(event.get("sampled_large_chance_nodes") or 0) for event, _ in values)
            ),
            "atomic_horizon_cutoffs": int(
                sum(int(event.get("atomic_horizon_cutoffs") or 0) for event, _ in values)
            ),
            "node_budget_cutoffs": int(
                sum(int(event.get("node_budget_cutoffs") or 0) for event, _ in values)
            ),
            "average_max_meaningful_depth_reached": float(
                np.mean([int(event.get("max_meaningful_depth_reached") or 0) for event, _ in values])
            ),
            "cross_turn_search_fraction": float(
                np.mean([int(event.get("max_turn_delta") or 0) > 0 for event, _ in values])
            ),
            "cross_turn_leaf_evaluations": int(
                sum(int(event.get("cross_turn_leaf_evaluations") or 0) for event, _ in values)
            ),
            "average_latency_seconds": float(
                np.mean([float(event.get("latency_seconds") or 0.0) for event, _ in values])
            ),
            "downstream_searching_agent_score": float(np.mean([score for _, score in values])),
        }
        for trigger, values in sorted(trigger_values.items())
    }
    payload = {
        "baseline": args.baseline,
        "matches": len(matches),
        "paired_seed_blocks": len({(match["source"], match["seed"]) for match in matches}),
        "incomplete_seed_blocks": incomplete_blocks,
        "terminal_reason_distribution": dict(Counter(match["terminal_reason"] for match in matches)),
        "compute": {
            "average_wall_clock_seconds_per_game": float(
                np.mean([match["wall_clock_seconds"] for match in matches])
            ) if matches else 0.0,
            "median_wall_clock_seconds_per_game": float(
                np.median([match["wall_clock_seconds"] for match in matches])
            ) if matches else 0.0,
            "p95_wall_clock_seconds_per_game": float(
                np.quantile([match["wall_clock_seconds"] for match in matches], 0.95)
            ) if matches else 0.0,
            "average_searches_per_game": len(searched_events) / max(1, len(matches)),
            "average_simulations_per_game": sum(
                int(event.get("simulations") or 0) for event, _ in searched_events
            ) / max(1, len(matches)),
            "average_search_seconds_per_game": sum(
                float(event.get("latency_seconds") or 0.0) for event, _ in searched_events
            ) / max(1, len(matches)),
            "median_search_latency_seconds": float(
                np.median([float(event.get("latency_seconds") or 0.0) for event, _ in searched_events])
            ) if searched_events else 0.0,
            "p95_search_latency_seconds": float(
                np.quantile(
                    [float(event.get("latency_seconds") or 0.0) for event, _ in searched_events],
                    0.95,
                )
            ) if searched_events else 0.0,
            "max_search_latency_seconds": max(
                (float(event.get("latency_seconds") or 0.0) for event, _ in searched_events),
                default=0.0,
            ),
            "p95_expanded_nodes_per_search": float(
                np.quantile(
                    [int(event.get("expanded_nodes") or 0) for event, _ in searched_events],
                    0.95,
                )
            ) if searched_events else 0.0,
            "max_expanded_nodes_per_search": max(
                (int(event.get("expanded_nodes") or 0) for event, _ in searched_events),
                default=0,
            ),
            "atomic_horizon_cutoffs": int(
                sum(int(event.get("atomic_horizon_cutoffs") or 0) for event, _ in searched_events)
            ),
            "node_budget_cutoffs": int(
                sum(int(event.get("node_budget_cutoffs") or 0) for event, _ in searched_events)
            ),
            "search_flip_rate": float(
                np.mean([bool(event.get("action_flipped")) for event, _ in searched_events])
            ) if searched_events else 0.0,
            "average_max_meaningful_depth_reached": float(
                np.mean([int(event.get("max_meaningful_depth_reached") or 0) for event, _ in searched_events])
            ) if searched_events else 0.0,
            "cross_turn_search_fraction": float(
                np.mean([int(event.get("max_turn_delta") or 0) > 0 for event, _ in searched_events])
            ) if searched_events else 0.0,
            "average_cross_turn_leaf_evaluations_per_search": float(
                np.mean([int(event.get("cross_turn_leaf_evaluations") or 0) for event, _ in searched_events])
            ) if searched_events else 0.0,
        },
        "triggers": trigger_metrics,
        "ratings": table,
        "sources": [str(path) for path in result_paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
