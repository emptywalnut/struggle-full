from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_logged_episode(path: Path, *, seed: int) -> dict[str, Any]:
    """Load the matching episode from a worker JSONL reused across games."""

    matches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            if int(episode.get("seed", -1)) == int(seed):
                matches.append(episode)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one game-log episode for seed={seed} in {path}, found {len(matches)}"
        )
    return matches[0]


def searched_root_steps(manifest: Path) -> dict[str, set[int]]:
    """Index cutoff-free root searches without loading observation tensors."""

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    roots: dict[str, set[int]] = defaultdict(set)
    for shard in payload.get("shards", []):
        shard_path = manifest.parent / str(shard["path"])
        with np.load(shard_path, allow_pickle=False) as arrays:
            metadata_rows = arrays["metadata_json"]
            if len(metadata_rows) != int(shard["examples"]):
                raise RuntimeError(f"metadata count mismatch in {shard_path}")
            for raw in metadata_rows:
                metadata = json.loads(str(raw))
                if int(metadata.get("hierarchy_index", -1)) != 0:
                    continue
                source = str(metadata.get("source_game") or "")
                source_step = metadata.get("source_step")
                if not source or source_step is None:
                    raise RuntimeError(f"root replay row lacks source identity in {shard_path}")
                roots[source].add(int(source_step))
    return dict(roots)


def _card_choice_actions(episode: dict[str, Any]) -> list[dict[str, Any]]:
    choices: list[dict[str, Any]] = []
    for row in episode.get("actions", []):
        action = dict(row.get("action") or {})
        prompt = str(action.get("prompt") or "").lower()
        if str(action.get("decision")) != "list":
            continue
        if "pick a card" not in prompt and "pick headline card" not in prompt:
            continue
        choices.append(row)
    return choices


def replay_legal_action_counts(episode: dict[str, Any]) -> dict[int, int]:
    """Recover actor-visible legal counts for logs written before they were persisted."""

    from struggle_ai.env import TwilightStruggleEnv

    config = {
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
        "search_mode": False,
        "log_games_dir": None,
        "metrics_dir": None,
    }
    env = TwilightStruggleEnv(config)
    counts: dict[int, int] = {}
    actions = list(episode.get("actions", []))
    try:
        env.reset(seed=int(episode["seed"]))
        for action_index, row in enumerate(actions):
            step = int(row.get("step", action_index))
            counts[step] = len(env.legal_actions)
            logged = dict(row.get("action") or {})
            matches = [
                index
                for index, action in enumerate(env.legal_actions)
                if str(action.get("decision") or "")
                == str(logged.get("decision") or "")
                and str(action.get("value") or "") == str(logged.get("value") or "")
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"cannot recover legal count at step={step}: "
                    f"logged action has {len(matches)} legal matches"
                )
            _, _, terminated, truncated, _ = env.step(matches[0])
            if (terminated or truncated) and action_index != len(actions) - 1:
                raise RuntimeError(
                    f"legacy replay ended at step={step} before the persisted action log"
                )
    finally:
        env.close()
    return counts


def required_card_reroots(
    episode: dict[str, Any],
    *,
    winner: str,
    terminal_reason: str,
    legal_action_counts: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Return card choices that must be independent search roots.

    With force-search enabled, headlines and every non-final card choice have
    multiple strategic candidates. A final non-scoring choice that ends in a
    held-scoring-card loss also necessarily had the scoring card as an
    alternative. The last ordinary choice in other games can legitimately be
    the sole legal card and is therefore reported separately, not required.
    """

    choices = _card_choice_actions(episode)
    legal_action_counts = legal_action_counts or {}

    def is_strategic(row: dict[str, Any]) -> bool:
        step = int(row["step"])
        persisted = row.get("legal_action_count_before")
        count = legal_action_counts.get(step) if persisted is None else int(persisted)
        # Preserve fail-closed behavior only when no count can be reconstructed.
        return count is None or int(count) > 1

    ordinary: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    required_reasons: dict[int, str] = {}
    for row in choices:
        prompt = str((row.get("action") or {}).get("prompt") or "").lower()
        step = int(row["step"])
        if "headline" in prompt and is_strategic(row):
            required_reasons[step] = "headline"
        else:
            ordinary[(str(row.get("side")), int(row.get("turn") or 0))].append(row)

    for rows in ordinary.values():
        rows.sort(key=lambda item: int(item["step"]))
        for row in rows[:-1]:
            if is_strategic(row):
                required_reasons[int(row["step"])] = "non_final_card_choice"

    normalized_reason = terminal_reason.strip().lower().replace("_", " ")
    if "scoring card held" in normalized_reason and winner in {"us", "ussr"}:
        loser = "ussr" if winner == "us" else "us"
        loser_choices = sorted(
            (
                row
                for (side, _turn), rows in ordinary.items()
                if side == loser
                for row in rows
            ),
            key=lambda item: int(item["step"]),
        )
        scoring_ids = {
            str(card.get("id"))
            for card in episode.get("cards", [])
            if bool(card.get("scoring"))
        }
        if loser_choices:
            final_choice = loser_choices[-1]
            selected = str((final_choice.get("action") or {}).get("value") or "")
            if selected not in scoring_ids and is_strategic(final_choice):
                required_reasons[int(final_choice["step"])] = "held_scoring_card_alternative"

    required: list[dict[str, Any]] = []
    for row in choices:
        step = int(row["step"])
        reason = required_reasons.get(step)
        if reason is None:
            continue
        action = dict(row.get("action") or {})
        required.append(
            {
                "step": step,
                "side": str(row.get("side") or ""),
                "turn": int(row.get("turn") or 0),
                "action_round": int(row.get("action_round") or 0),
                "label": str(action.get("label") or ""),
                "value": str(action.get("value") or ""),
                "prompt": str(action.get("prompt") or ""),
                "requirement": reason,
            }
        )
    return required


def audit_search_reroots(
    collections: Iterable[Path],
    *,
    legacy_source_games: Iterable[str] = (),
    expected_corrected_games: int | None = None,
) -> dict[str, Any]:
    """Cross-check complete game logs against committed anchor search roots."""

    legacy = {str(source) for source in legacy_source_games}
    game_reports: list[dict[str, Any]] = []
    for collection in collections:
        collection = Path(collection)
        merged_manifest = collection / "anchor_replay/manifest.json"
        manifests = (
            [merged_manifest]
            if merged_manifest.is_file()
            else sorted((collection / "workers").glob("worker-*/anchor_replay/manifest.json"))
        )
        if not manifests:
            raise RuntimeError(f"collection has no anchor replay manifests: {collection}")
        roots_by_source: dict[str, set[int]] = defaultdict(set)
        for manifest in manifests:
            for source, steps in searched_root_steps(manifest).items():
                roots_by_source[source].update(steps)
        for record_path in sorted((collection / "completed_games").glob("game-*.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            source = str(record["source_game"])
            log_path = Path(str(record.get("game_log") or ""))
            if not log_path.is_file():
                raise RuntimeError(f"missing game log for {source}: {log_path}")
            episode = load_logged_episode(log_path, seed=int(record["seed"]))
            card_choices = _card_choice_actions(episode)
            missing_counts = any(
                row.get("legal_action_count_before") is None for row in card_choices
            )
            legal_action_counts = (
                replay_legal_action_counts(episode) if missing_counts else {}
            )
            required = required_card_reroots(
                episode,
                winner=str(record.get("winner") or ""),
                terminal_reason=str(record.get("terminal_reason") or ""),
                legal_action_counts=legal_action_counts,
            )
            root_steps = roots_by_source.get(source, set())
            boundary_events = {
                int(item["source_step"]): dict(item)
                for item in record.get("independent_boundary_events", [])
            }
            attempted_steps = {
                step
                for step, item in boundary_events.items()
                if bool(item.get("searched")) and not bool(item.get("continued"))
            }
            verified_steps = set(root_steps) | attempted_steps
            missing = [item for item in required if int(item["step"]) not in verified_steps]
            invalid_boundary_events = [
                {
                    "step": int(item["step"]),
                    "requirement": item["requirement"],
                    "boundary_event": boundary_events[int(item["step"])],
                }
                for item in required
                if int(item["step"]) in boundary_events
                and (
                    not bool(boundary_events[int(item["step"])].get("searched"))
                    or bool(boundary_events[int(item["step"])].get("continued"))
                )
            ]
            game_reports.append(
                {
                    "collection": str(collection.resolve()),
                    "source_game": source,
                    "seed": int(record["seed"]),
                    "legacy_planner_semantics": source in legacy,
                    "search_root_count": len(root_steps),
                    "required_card_reroot_count": len(required),
                    "missing_required_card_reroots": missing,
                    "invalid_independent_boundary_events": invalid_boundary_events,
                }
            )

    corrected = [row for row in game_reports if not row["legacy_planner_semantics"]]
    if expected_corrected_games is not None and len(corrected) != int(expected_corrected_games):
        raise RuntimeError(
            f"corrected-game reroot audit inventory mismatch: "
            f"{len(corrected)}/{int(expected_corrected_games)}"
        )
    corrected_missing = [
        {"source_game": row["source_game"], **missing}
        for row in corrected
        for missing in row["missing_required_card_reroots"]
    ]
    corrected_invalid = [
        {"source_game": row["source_game"], **invalid}
        for row in corrected
        for invalid in row.get("invalid_independent_boundary_events", [])
    ]
    legacy_missing = [
        {"source_game": row["source_game"], **missing}
        for row in game_reports
        if row["legacy_planner_semantics"]
        for missing in row["missing_required_card_reroots"]
    ]
    summary = {
        "games": len(game_reports),
        "legacy_games": sum(bool(row["legacy_planner_semantics"]) for row in game_reports),
        "corrected_games": len(corrected),
        "required_card_reroots": sum(
            int(row["required_card_reroot_count"]) for row in game_reports
        ),
        "corrected_missing_required_card_reroots": corrected_missing,
        "corrected_invalid_independent_boundary_events": corrected_invalid,
        "legacy_missing_required_card_reroots": legacy_missing,
        "passed": not corrected_missing and not corrected_invalid,
        "game_reports": game_reports,
    }
    if corrected_missing or corrected_invalid:
        raise RuntimeError(
            "corrected games skipped required independent card search roots: "
            f"missing={corrected_missing[:8]}, invalid={corrected_invalid[:8]}"
        )
    return summary
