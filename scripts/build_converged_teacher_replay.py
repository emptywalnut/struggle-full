#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from struggle_ai.search_convergence import visited_q_top
from struggle_ai.search_replay import (
    SearchReplayExample,
    SearchReplayWriter,
    load_search_replay,
)


ALLOWED_STATUSES = {
    "BUDGET_CONVERGENCE",
    "HORIZON_CONVERGENCE",
    "ENSEMBLE_CONVERGENCE",
}

LOCAL_DECISION_LEVELS = frozenset(
    {
        "influence_allocation",
        "coup_target",
        "realignment_target",
        "forced_response",
    }
)


def era_for_example(example: SearchReplayExample) -> str:
    global_features = np.asarray(example.observation.get("global", ()), dtype=np.float32)
    if global_features.size <= 2:
        return "unknown"
    turn = int(round(float(global_features[2]) * 10.0))
    if turn <= 3:
        return "early"
    if turn <= 7:
        return "mid"
    return "late"


def crosses_independent_macro_boundary(example: SearchReplayExample) -> bool:
    """Reject downstream targets that belong to a later strategic macro."""

    metadata = example.metadata
    if int(metadata.hierarchy_index) <= 0:
        return False
    if not metadata.hierarchical_path:
        return True
    root_level = str(metadata.hierarchical_path[0].get("level") or "")
    current_level = str(metadata.hierarchy_level or "")
    if current_level == "card":
        return True
    return bool(
        root_level in LOCAL_DECISION_LEVELS and current_level != root_level
    )


def lacks_required_horizon_validation(
    example: SearchReplayExample,
    require_horizon_triggers: frozenset[str],
) -> bool:
    return bool(
        require_horizon_triggers.intersection(example.metadata.trigger_categories)
        and example.metadata.convergence_status != "HORIZON_CONVERGENCE"
    )


def load_excluded_roots(paths: Iterable[Path]) -> set[tuple[str, int]]:
    excluded: set[tuple[str, int]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("positions", []):
            source_game = row.get("source_game")
            source_step = row.get("action_index")
            if source_game is None or source_step is None:
                raise ValueError(f"exclusion position lacks source identity in {path}: {row}")
            excluded.add((str(source_game), int(source_step)))
    return excluded


def q_softened_example(
    example: SearchReplayExample,
    *,
    temperature: float,
    min_q_gap: float,
    min_total_visits: float = 0.0,
    min_best_action_visits: float = 0.0,
    require_horizon_triggers: frozenset[str] = frozenset(),
    require_terminal_triggers: frozenset[str] = frozenset(),
    require_min_depth: dict[str, int] | None = None,
) -> SearchReplayExample | None:
    metadata = example.metadata
    if crosses_independent_macro_boundary(example):
        return None
    if metadata.convergence_status not in ALLOWED_STATUSES:
        return None
    if metadata.atomic_horizon_cutoffs or metadata.node_budget_cutoffs:
        return None
    if not metadata.complete_game:
        return None
    if metadata.evidence_gate_rejected:
        return None
    if lacks_required_horizon_validation(example, require_horizon_triggers):
        return None
    if (
        require_terminal_triggers.intersection(metadata.trigger_categories)
        and metadata.terminal_evaluations <= 0
    ):
        return None
    for trigger, depth in (require_min_depth or {}).items():
        if trigger in metadata.trigger_categories and metadata.search_depth < depth:
            return None
    legal = example.legal_count
    visits = np.asarray(example.visit_counts[:legal], dtype=np.float64)
    q_values = np.asarray(example.q_values[:legal], dtype=np.float64)
    valid = visits > 0.0
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return None
    best, gap = visited_q_top(visits, q_values)
    if float(visits.sum()) < float(min_total_visits):
        return None
    if float(visits[best]) < float(min_best_action_visits):
        return None
    if gap < float(min_q_gap):
        return None
    logits = q_values[indices] / max(float(temperature), 1e-6)
    logits -= np.max(logits)
    weights = np.exp(logits)
    policy = np.zeros_like(example.search_policy, dtype=np.float32)
    policy[indices] = (weights / weights.sum()).astype(np.float32)
    pseudo_visits = np.zeros_like(example.visit_counts, dtype=np.float32)
    pseudo_visits[:legal] = policy[:legal] * max(float(example.visit_counts.sum()), 1.0)
    raw_action = int(metadata.raw_action)
    return SearchReplayExample(
        observation=example.observation,
        raw_policy=example.raw_policy,
        visit_counts=pseudo_visits,
        search_policy=policy,
        q_values=example.q_values,
        metadata=replace(
            metadata,
            search_action=best,
            root_q_gap=gap,
            q_advantage_over_raw=float(q_values[best] - q_values[raw_action]),
            action_flipped=best != raw_action,
            teacher_action_visits=float(pseudo_visits[best]),
            total_visits=float(pseudo_visits.sum()),
            teacher_visit_share=float(policy[best]),
            behavior_policy="adaptive_converged_q_teacher",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Q-soft actor targets from converged complete-game search replay.")
    parser.add_argument("--search-replay", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q-temperature", type=float, default=0.05)
    parser.add_argument("--min-q-gap", type=float, default=0.005)
    parser.add_argument(
        "--min-total-visits",
        type=float,
        default=0.0,
        help="Reject low-evidence hierarchy nodes even when a one-visit Q gap looks large.",
    )
    parser.add_argument(
        "--require-min-depth-trigger",
        action="append",
        default=["scoring_pressure:8"],
        help="Require TRIGGER:DEPTH before admitting a teacher label.",
    )
    parser.add_argument(
        "--require-terminal-trigger",
        action="append",
        default=["scoring_pressure"],
        help="Reject this trigger unless search reached at least one terminal branch.",
    )
    parser.add_argument(
        "--min-best-action-visits",
        type=float,
        default=0.0,
        help="Require the Q-best teacher action itself to have this much search weight.",
    )
    parser.add_argument("--exclude-position-manifest", action="append", type=Path, default=[])
    parser.add_argument("--load-workers", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument(
        "--require-horizon-trigger",
        action="append",
        default=["scoring_card"],
        help=(
            "Reject labels carrying this trigger unless a deeper search agrees. "
            "Defaults to scoring_card because same-depth budget agreement does not "
            "validate end-of-turn card-disposal consequences."
        ),
    )
    args = parser.parse_args()

    examples = load_search_replay(args.search_replay, workers=max(1, int(args.load_workers)))
    excluded_roots = load_excluded_roots(args.exclude_position_manifest)
    require_horizon_triggers = frozenset(str(item) for item in args.require_horizon_trigger)
    require_terminal_triggers = frozenset(str(item) for item in args.require_terminal_trigger)
    require_min_depth: dict[str, int] = {}
    for item in args.require_min_depth_trigger:
        trigger, separator, raw_depth = str(item).partition(":")
        if not separator or not trigger or int(raw_depth) <= 0:
            raise SystemExit(f"invalid --require-min-depth-trigger {item!r}; expected TRIGGER:DEPTH")
        require_min_depth[trigger] = int(raw_depth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    source_games: set[str] = set()
    with SearchReplayWriter(args.output_dir / "search_replay", shard_size=int(args.shard_size)) as writer:
        for example in examples:
            counts["input"] += 1
            source_identity = (
                str(example.metadata.source_game),
                int(example.metadata.source_step),
            ) if example.metadata.source_game is not None and example.metadata.source_step is not None else None
            if source_identity in excluded_roots:
                counts["rejected"] += 1
                counts["rejected:particle_variance"] += 1
                continue
            if crosses_independent_macro_boundary(example):
                counts["rejected"] += 1
                counts["rejected:cross_macro_hierarchy"] += 1
                continue
            if lacks_required_horizon_validation(example, require_horizon_triggers):
                counts["rejected"] += 1
                counts["rejected:requires_horizon_validation"] += 1
                continue
            if (
                require_terminal_triggers.intersection(example.metadata.trigger_categories)
                and example.metadata.terminal_evaluations <= 0
            ):
                counts["rejected"] += 1
                counts["rejected:requires_terminal_evaluation"] += 1
                continue
            if any(
                trigger in example.metadata.trigger_categories
                and example.metadata.search_depth < depth
                for trigger, depth in require_min_depth.items()
            ):
                counts["rejected"] += 1
                counts["rejected:requires_minimum_depth"] += 1
                continue
            converted = q_softened_example(
                example,
                temperature=float(args.q_temperature),
                min_q_gap=float(args.min_q_gap),
                min_total_visits=float(args.min_total_visits),
                min_best_action_visits=float(args.min_best_action_visits),
                require_horizon_triggers=require_horizon_triggers,
                require_terminal_triggers=require_terminal_triggers,
                require_min_depth=require_min_depth,
            )
            if converted is None:
                counts["rejected"] += 1
                continue
            writer.add(converted)
            counts["accepted"] += 1
            counts[f"side:{converted.metadata.side}"] += 1
            era = era_for_example(converted)
            counts[f"era:{era}"] += 1
            counts[f"side_era:{converted.metadata.side}:{era}"] += 1
            counts[f"status:{converted.metadata.convergence_status}"] += 1
            counts[f"hierarchy:{converted.metadata.hierarchy_level}"] += 1
            counts[f"flipped:{converted.metadata.action_flipped}"] += 1
            source_games.add(str(converted.metadata.source_game))
    summary = {
        **dict(counts),
        "source_games": len(source_games),
        "q_temperature": float(args.q_temperature),
        "min_q_gap": float(args.min_q_gap),
        "min_total_visits": float(args.min_total_visits),
        "min_best_action_visits": float(args.min_best_action_visits),
        "excluded_particle_variance_roots": len(excluded_roots),
        "require_horizon_triggers": sorted(require_horizon_triggers),
        "require_terminal_triggers": sorted(require_terminal_triggers),
        "require_min_depth": dict(sorted(require_min_depth.items())),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
