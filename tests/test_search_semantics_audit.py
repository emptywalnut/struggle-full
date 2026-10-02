from __future__ import annotations

import json

import numpy as np
import pytest

from struggle_ai.search_semantics_audit import audit_search_reroots, required_card_reroots


def action(
    step: int,
    side: str,
    turn: int,
    ar: int,
    label: str,
    prompt: str,
    value: str,
    legal_action_count: int = 2,
):
    return {
        "step": step,
        "legal_action_count_before": legal_action_count,
        "side": side,
        "turn": turn,
        "action_round": ar,
        "action": {
            "decision": "list",
            "label": label,
            "value": value,
            "prompt": prompt,
        },
    }


def test_required_reroots_include_headline_and_non_final_card_choices() -> None:
    episode = {
        "actions": [
            action(10, "ussr", 1, 0, "Suez", "USSR pick headline card", "suez"),
            action(11, "us", 1, 0, "Olympic", "US pick headline card", "olympic"),
            action(20, "ussr", 1, 1, "Comecon", "USSR pick a card: ", "comecon"),
            action(30, "ussr", 1, 2, "China", "USSR pick a card: ", "china"),
        ],
        "cards": [],
    }

    required = required_card_reroots(episode, winner="us", terminal_reason="vp")
    assert [(row["step"], row["requirement"]) for row in required] == [
        (10, "headline"),
        (11, "headline"),
        (20, "non_final_card_choice"),
    ]


def test_held_scoring_loss_requires_final_non_scoring_card_reroot() -> None:
    episode = {
        "actions": [
            action(20, "us", 2, 4, "Asia Scoring", "US pick a card: ", "asia"),
            action(30, "us", 2, 5, "Blockade", "US pick a card: ", "blockade"),
        ],
        "cards": [{"id": "asia", "scoring": True}],
    }

    required = required_card_reroots(
        episode, winner="ussr", terminal_reason="scoring card held"
    )
    assert [(row["step"], row["requirement"]) for row in required] == [
        (20, "non_final_card_choice"),
        (30, "held_scoring_card_alternative"),
    ]


def test_final_single_card_choice_is_not_assumed_to_need_search() -> None:
    episode = {
        "actions": [action(30, "us", 2, 5, "Blockade", "US pick a card: ", "blockade")],
        "cards": [],
    }
    assert required_card_reroots(episode, winner="us", terminal_reason="vp") == []


def test_non_final_singleton_card_choice_does_not_require_search_root() -> None:
    episode = {
        "actions": [
            action(
                20,
                "ussr",
                2,
                4,
                "Europe Scoring",
                "USSR pick a card: ",
                "europe",
                legal_action_count=1,
            ),
            action(
                30,
                "ussr",
                2,
                5,
                "Five Year Plan",
                "USSR pick a card: ",
                "fiveyear",
            ),
        ],
        "cards": [],
    }

    assert required_card_reroots(episode, winner="us", terminal_reason="vp") == []


def test_collection_audit_fails_when_corrected_game_skips_required_root(tmp_path) -> None:
    collection = tmp_path / "collection"
    completed = collection / "completed_games"
    replay = collection / "anchor_replay"
    completed.mkdir(parents=True)
    replay.mkdir()
    episode = {
        "seed": 7,
        "actions": [
            action(10, "us", 1, 0, "Olympic", "US pick headline card", "olympic"),
            action(20, "us", 1, 1, "Marshall", "US pick a card: ", "marshall"),
            action(30, "us", 1, 2, "Blockade", "US pick a card: ", "blockade"),
        ],
        "cards": [],
    }
    game_log = tmp_path / "game.jsonl"
    game_log.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    (completed / "game-000001.json").write_text(
        json.dumps(
            {
                "source_game": "game-1",
                "seed": 7,
                "winner": "us",
                "terminal_reason": "vp",
                "game_log": str(game_log),
            }
        ),
        encoding="utf-8",
    )
    rows = [
        json.dumps(
            {
                "source_game": "game-1",
                "source_step": step,
                "hierarchy_index": 0,
            },
            separators=(",", ":"),
        )
        for step in (10,)
    ]
    np.savez_compressed(replay / "search-replay-000000.npz", metadata_json=np.asarray(rows))
    (replay / "manifest.json").write_text(
        json.dumps(
            {
                "shards": [
                    {"path": "search-replay-000000.npz", "examples": len(rows)}
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="skipped required independent card"):
        audit_search_reroots([collection], expected_corrected_games=1)


def _write_boundary_audit_fixture(tmp_path, boundary_event: dict) -> object:
    collection = tmp_path / "collection"
    completed = collection / "completed_games"
    replay = collection / "anchor_replay"
    completed.mkdir(parents=True)
    replay.mkdir()
    episode = {
        "seed": 8,
        "actions": [
            action(10, "us", 1, 0, "Olympic", "US pick headline card", "olympic"),
        ],
        "cards": [],
    }
    game_log = tmp_path / "game.jsonl"
    game_log.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    (completed / "game-000001.json").write_text(
        json.dumps(
            {
                "source_game": "game-1",
                "seed": 8,
                "winner": "us",
                "terminal_reason": "vp",
                "game_log": str(game_log),
                "independent_boundary_events": [boundary_event],
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        replay / "search-replay-000000.npz",
        metadata_json=np.asarray([], dtype=str),
    )
    (replay / "manifest.json").write_text(
        json.dumps(
            {
                "shards": [
                    {"path": "search-replay-000000.npz", "examples": 0}
                ]
            }
        ),
        encoding="utf-8",
    )
    return collection


def test_collection_audit_accepts_cutoff_search_attempt_at_independent_root(tmp_path) -> None:
    collection = _write_boundary_audit_fixture(
        tmp_path,
        {
            "source_step": 10,
            "side": "us",
            "boundary_kind": "headline",
            "legal_action_count": 2,
            "searched": True,
            "continued": False,
            "atomic_horizon_cutoffs": 1,
            "node_budget_cutoffs": 0,
        },
    )

    result = audit_search_reroots([collection], expected_corrected_games=1)
    assert result["passed"] is True


def test_collection_audit_rejects_continuation_at_independent_root(tmp_path) -> None:
    collection = _write_boundary_audit_fixture(
        tmp_path,
        {
            "source_step": 10,
            "side": "us",
            "boundary_kind": "headline",
            "legal_action_count": 2,
            "searched": False,
            "continued": True,
            "atomic_horizon_cutoffs": 0,
            "node_budget_cutoffs": 0,
        },
    )

    with pytest.raises(RuntimeError, match="invalid="):
        audit_search_reroots([collection], expected_corrected_games=1)
