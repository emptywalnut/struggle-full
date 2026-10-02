#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.search_replay import SearchReplayExample, load_search_replay


SEED_RE = re.compile(r"seed-(\d+)")


def source_seed(source_game: str | None) -> int:
    match = SEED_RE.search(str(source_game or ""))
    if match is None:
        raise ValueError(f"source_game has no seed: {source_game!r}")
    return int(match.group(1))


def root_category(example: SearchReplayExample) -> str:
    actions = example.metadata.legal_actions
    decisions = {str(action.get("decision") or "") for action in actions}
    prompts = " ".join(str(action.get("prompt") or "").lower() for action in actions)
    if decisions & {"country_click", "country_mouseup"}:
        if "realign" in prompts:
            return "realignment_target"
        if "coup" in prompts:
            return "coup_target"
        return "influence_allocation"
    triggers = set(example.metadata.trigger_categories)
    for name in (
        "scoring_card",
        "scoring_pressure",
        "opponent_event",
        "forced_response",
        "chance_heavy",
    ):
        if name in triggers:
            return name
    return "other"


def root_era(example: SearchReplayExample) -> int:
    global_features = np.asarray(example.observation["global"])
    turn = max(1, int(round(float(global_features[2]) * 10.0)))
    return 0 if turn <= 3 else (1 if turn <= 7 else 2)


def select_roots(
    examples: list[SearchReplayExample],
    *,
    max_roots: int,
    max_per_game: int,
    held_scoring_sources: set[str] | None = None,
) -> list[SearchReplayExample]:
    roots = [
        example
        for example in examples
        if example.metadata.hierarchy_index == 0
        and example.metadata.complete_game
        and example.metadata.atomic_horizon_cutoffs == 0
        and example.metadata.node_budget_cutoffs == 0
        and example.legal_count >= 2
    ]
    roots.sort(
        key=lambda example: (
            -int(root_category(example) in {"influence_allocation", "realignment_target", "coup_target"}),
            -example.legal_count,
            -abs(float(example.metadata.root_q_gap)),
            str(example.metadata.source_game),
            int(example.metadata.source_step or -1),
        )
    )
    selected: list[SearchReplayExample] = []
    selected_ids: set[int] = set()
    game_counts: Counter[str] = Counter()
    bucket_counts: Counter[tuple[str, int, str]] = Counter()
    deferred: list[SearchReplayExample] = []
    per_bucket = max(1, max_roots // 24)

    # Keep one scoring-pressure diagnostic from a game that actually ended by
    # holding a scoring card. It remains ordinary search evidence and is not
    # automatically eligible for teacher supervision.
    held_sources = held_scoring_sources or set()
    held_candidates = [
        example
        for example in roots
        if str(example.metadata.source_game) in held_sources
        and root_category(example) in {"scoring_card", "scoring_pressure"}
    ]
    held_candidates.sort(
        key=lambda example: (
            -int(example.metadata.source_step or -1),
            -abs(float(example.metadata.root_q_gap)),
            str(example.metadata.source_game),
        )
    )
    def reserve(example: SearchReplayExample) -> None:
        source = str(example.metadata.source_game)
        bucket = (example.metadata.side, root_era(example), root_category(example))
        selected.append(example)
        selected_ids.add(id(example))
        game_counts[source] += 1
        bucket_counts[bucket] += 1

    reserved = held_candidates[0] if held_candidates and max_roots > 0 else None
    if reserved is not None:
        reserve(reserved)
        if len(selected) >= max_roots:
            return selected

    # Preserve enough roots for the fixed panel to exercise every war era.
    # This runs before high-branching roots consume a long game's quota.
    era_order = sorted(
        range(3),
        key=lambda target: len(
            {
                str(example.metadata.source_game)
                for example in roots
                if root_era(example) == target
            }
        ),
    )
    for target_era in era_order:
        while (
            sum(root_era(example) == target_era for example in selected) < 2
            and len(selected) < max_roots
        ):
            candidates = [
                example
                for example in roots
                if id(example) not in selected_ids
                and root_era(example) == target_era
                and game_counts[str(example.metadata.source_game)] < max_per_game
            ]
            if not candidates:
                break
            era_side_counts = Counter(
                example.metadata.side
                for example in selected
                if root_era(example) == target_era
            )
            candidates.sort(
                key=lambda example: (
                    era_side_counts[example.metadata.side],
                    game_counts[str(example.metadata.source_game)],
                    -abs(float(example.metadata.root_q_gap)),
                    str(example.metadata.source_game),
                    int(example.metadata.source_step or -1),
                )
            )
            reserve(candidates[0])

    for example in roots:
        if id(example) in selected_ids:
            continue
        source = str(example.metadata.source_game)
        era = root_era(example)
        bucket = (example.metadata.side, era, root_category(example))
        if game_counts[source] >= max_per_game or bucket_counts[bucket] >= per_bucket:
            deferred.append(example)
            continue
        selected.append(example)
        game_counts[source] += 1
        bucket_counts[bucket] += 1
        if len(selected) >= max_roots:
            return selected
    for example in deferred:
        source = str(example.metadata.source_game)
        if game_counts[source] >= max_per_game:
            continue
        selected.append(example)
        game_counts[source] += 1
        if len(selected) >= max_roots:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select balanced roots for multi-depth rescoring.")
    parser.add_argument(
        "--d4-replay",
        action="append",
        type=Path,
        required=True,
        help="Search-replay manifest. Repeat to merge independent collections.",
    )
    parser.add_argument(
        "--completed-games",
        action="append",
        type=Path,
        required=True,
        help="Completed-game directory. Repeat to merge independent collections.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-roots", type=int, default=256)
    parser.add_argument("--max-per-game", type=int, default=20)
    parser.add_argument("--load-workers", type=int, default=16)
    args = parser.parse_args()

    games: dict[str, dict[str, Any]] = {}
    for directory in args.completed_games:
        for path in sorted(directory.expanduser().resolve().glob("game-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            source = str(payload["source_game"])
            if source in games:
                raise ValueError(f"duplicate complete-game source across collections: {source}")
            games[source] = payload

    examples = load_search_replay(
        tuple(path.expanduser().resolve() for path in args.d4_replay),
        workers=max(1, int(args.load_workers)),
        metadata_predicate=lambda metadata: metadata.hierarchy_index == 0,
    )
    held_scoring_sources = {
        source
        for source, payload in games.items()
        if str(payload.get("terminal_reason") or "") == "scoring card held"
    }
    selected = select_roots(
        examples,
        max_roots=args.max_roots,
        max_per_game=args.max_per_game,
        held_scoring_sources=held_scoring_sources,
    )

    positions = []
    for example in selected:
        source = str(example.metadata.source_game)
        game = games.get(source)
        if game is None or not game.get("game_log"):
            raise ValueError(f"no complete game log for {source}")
        positions.append(
            {
                "source_game": source,
                "seed": source_seed(source),
                "game_occurrence": 0,
                "game_log": game["game_log"],
                "action_index": int(example.metadata.source_step),
                "side": example.metadata.side,
                "category": root_category(example),
                "legal_count": example.legal_count,
                "d4_q_gap": float(example.metadata.root_q_gap),
            }
        )
    counts = Counter((row["side"], row["category"]) for row in positions)
    payload = {
        "source_replays": [str(path.expanduser().resolve()) for path in args.d4_replay],
        "completed_game_dirs": [
            str(path.expanduser().resolve()) for path in args.completed_games
        ],
        "positions": positions,
        "summary": {
            "selected_roots": len(positions),
            "by_side_category": {f"{side}:{category}": count for (side, category), count in sorted(counts.items())},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
