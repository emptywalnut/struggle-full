from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from struggle_ai.bridge_client import TwilightBridgeClient
from struggle_ai.search_semantics_audit import load_logged_episode


SEMANTIC_OBSERVATION_FIELDS = (
    "turn",
    "action_round",
    "current_player",
    "side",
    "phase",
    "prompt",
    "queue_top",
    "vp",
    "defcon",
    "event_name",
)


def _action_signature(action: dict[str, Any], *, include_prompt: bool) -> tuple[str, ...]:
    fields = ("type", "decision", "value", "prompt") if include_prompt else (
        "type",
        "decision",
        "value",
    )
    return tuple(str(action.get(field) or "") for field in fields)


def _match_logged_action(
    legal_actions: list[dict[str, Any]], logged_action: dict[str, Any]
) -> dict[str, Any] | None:
    for include_prompt in (True, False):
        signature = _action_signature(logged_action, include_prompt=include_prompt)
        matches = [
            action
            for action in legal_actions
            if _action_signature(action, include_prompt=include_prompt) == signature
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for field in ("vp", "defcon"):
        if before.get(field) != after.get(field):
            delta[field] = {"before": before.get(field), "after": after.get(field)}
    before_countries = {
        str(country["id"]): country
        for country in before.get("countries", [])
        if country.get("id")
    }
    changes = []
    for country in after.get("countries", []):
        country_id = str(country.get("id") or "")
        previous = before_countries.get(country_id)
        if not previous or (
            previous.get("us") == country.get("us")
            and previous.get("ussr") == country.get("ussr")
        ):
            continue
        changes.append(
            {
                "id": country_id,
                "name": country.get("name") or country_id,
                "us_before": previous.get("us"),
                "us_after": country.get("us"),
                "ussr_before": previous.get("ussr"),
                "ussr_after": country.get("ussr"),
                "control_before": previous.get("control"),
                "control_after": country.get("control"),
            }
        )
    if changes:
        delta["countries"] = changes
    return delta


def replay_episode_semantics(
    episode: dict[str, Any],
    *,
    client_factory: Callable[[], TwilightBridgeClient] = TwilightBridgeClient,
) -> dict[str, Any]:
    """Replay an episode and compare every persisted semantic observation."""

    seed = int(episode["seed"])
    actions = list(episode.get("actions") or [])
    last_result: dict[str, Any] = {}
    with client_factory() as bridge:
        observation = bridge.reset(seed)
        initial_log = bridge.log() if hasattr(bridge, "log") else {"log": []}
        log_length = len(initial_log.get("log") or [])
        for action_index, row in enumerate(actions):
            before_observation = observation
            logged_action = dict(row.get("action") or {})
            legal_actions = list(observation.get("legal_actions") or [])
            action = _match_logged_action(legal_actions, logged_action)
            if action is None:
                return {
                    "passed": False,
                    "seed": seed,
                    "action_index": action_index,
                    "source_step": int(row.get("step", action_index)),
                    "reason": "logged_action_not_legal",
                    "logged_action": logged_action,
                    "legal_action_count": len(legal_actions),
                }

            last_result = bridge.step(action)
            observation = dict(last_result.get("observation") or {})
            recorded_after = dict(row.get("after") or {})
            differences = {
                field: {
                    "recorded": recorded_after.get(field),
                    "replayed": observation.get(field),
                }
                for field in SEMANTIC_OBSERVATION_FIELDS
                if field in recorded_after
                and recorded_after.get(field) != observation.get(field)
            }
            if differences:
                return {
                    "passed": False,
                    "seed": seed,
                    "action_index": action_index,
                    "source_step": int(row.get("step", action_index)),
                    "turn": int(row.get("turn") or recorded_after.get("turn") or 0),
                    "reason": "semantic_observation_mismatch",
                    "logged_action": logged_action,
                    "differences": differences,
                }

            if "state_delta" in row:
                recorded_delta = dict(row.get("state_delta") or {})
                replayed_delta = _state_delta(before_observation, observation)
                if recorded_delta != replayed_delta:
                    return {
                        "passed": False,
                        "seed": seed,
                        "action_index": action_index,
                        "source_step": int(row.get("step", action_index)),
                        "turn": int(row.get("turn") or recorded_after.get("turn") or 0),
                        "reason": "state_delta_mismatch",
                        "logged_action": logged_action,
                        "recorded_state_delta": recorded_delta,
                        "replayed_state_delta": replayed_delta,
                    }

            if "saito_log_delta" in row and hasattr(bridge, "log"):
                current_log = list((bridge.log().get("log") or []))
                replayed_log_delta = [str(item) for item in current_log[log_length:]]
                log_length = len(current_log)
                recorded_log_delta = [str(item) for item in row.get("saito_log_delta") or []]
                if recorded_log_delta != replayed_log_delta:
                    return {
                        "passed": False,
                        "seed": seed,
                        "action_index": action_index,
                        "source_step": int(row.get("step", action_index)),
                        "turn": int(row.get("turn") or recorded_after.get("turn") or 0),
                        "reason": "saito_log_delta_mismatch",
                        "logged_action": logged_action,
                        "recorded_saito_log_delta": recorded_log_delta,
                        "replayed_saito_log_delta": replayed_log_delta,
                    }

        terminal = dict(episode.get("info") or {})
        winner = str(episode.get("winner") or terminal.get("winner") or "")
        replay_winner = str(last_result.get("winner") or "")
        if winner in {"us", "ussr"} and replay_winner and winner != replay_winner:
            return {
                "passed": False,
                "seed": seed,
                "action_index": len(actions),
                "source_step": len(actions),
                "reason": "terminal_winner_mismatch",
                "recorded_winner": winner,
                "replayed_winner": replay_winner,
            }

    return {"passed": True, "seed": seed, "actions_replayed": len(actions)}


def audit_record_semantics(record_path: Path) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    source_game = str(record["source_game"])
    seed = int(record["seed"])
    log_path = Path(str(record.get("game_log") or ""))
    if not log_path.is_file():
        return {
            "passed": False,
            "source_game": source_game,
            "seed": seed,
            "reason": "missing_game_log",
            "game_log": str(log_path),
        }
    try:
        episode = load_logged_episode(log_path, seed=seed)
        result = replay_episode_semantics(episode)
    except Exception as exc:  # Fail closed while preserving a durable diagnostic.
        result = {
            "passed": False,
            "seed": seed,
            "reason": "replay_exception",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"source_game": source_game, "game_log": str(log_path.resolve()), **result}


def audit_bridge_replay_semantics(
    collections: Iterable[Path],
    *,
    expected_games: int | None = None,
    workers: int = 16,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Replay committed games in parallel against the current bridge."""

    records = [
        path
        for collection in map(Path, collections)
        for path in sorted((collection / "completed_games").glob("game-*.json"))
    ]
    if expected_games is not None and len(records) != int(expected_games):
        raise RuntimeError(
            f"bridge replay audit inventory mismatch: {len(records)}/{int(expected_games)}"
        )
    max_workers = max(1, min(int(workers), len(records) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        reports = list(executor.map(audit_record_semantics, records))
    failures = [report for report in reports if not bool(report.get("passed"))]
    summary = {
        "games": len(reports),
        "passed_games": len(reports) - len(failures),
        "failed_games": len(failures),
        "workers": max_workers,
        "passed": not failures,
        "failures": failures,
        "game_reports": reports,
    }
    if failures and raise_on_failure:
        raise RuntimeError(
            "committed games do not replay under current bridge semantics: "
            f"{failures[:4]}"
        )
    return summary
