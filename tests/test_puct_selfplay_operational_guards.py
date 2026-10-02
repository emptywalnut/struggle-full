from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.collect_puct_selfplay_parallel import (
    _acquire_collection_lock,
    _clear_independent_root_continuations,
    _failed_processes,
)


def test_collection_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    first = _acquire_collection_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            _acquire_collection_lock(tmp_path)
    finally:
        first.close()

    replacement = _acquire_collection_lock(tmp_path)
    replacement.close()


def test_failed_processes_reports_only_exited_children() -> None:
    pending = SimpleNamespace(name="pending", pid=None, exitcode=None)
    exited = SimpleNamespace(name="inference-gpu-4", pid=123, exitcode=-9)
    assert _failed_processes([pending]) == []
    assert _failed_processes([exited]) == ["inference-gpu-4 pid=123 exit=-9"]


def test_independent_card_root_clears_every_cached_planner_continuation() -> None:
    planners = {
        depth: {
            side: SimpleNamespace(
                pending_continuation=[{"side": side, "action": "stale"}],
                pending_continuation_seed=17,
            )
            for side in ("us", "ussr")
        }
        for depth in ("d4", "d6")
    }
    observation = {
        "side": "us",
        "current_player": "us",
        "phase": "action",
        "turn": 4,
        "action_round": 1,
        "prompt": "US pick a card",
        "hand": [],
    }
    legal_actions = [
        {"decision": "list", "value": "abmtreaty"},
        {"decision": "list", "value": "panamacanal"},
    ]

    boundary = _clear_independent_root_continuations(
        planners,
        side="us",
        last_obs=observation,
        legal_actions=legal_actions,
    )

    assert boundary == "card_choice"
    assert all(not group["us"].pending_continuation for group in planners.values())
    assert all(
        group["us"].pending_continuation_seed is None for group in planners.values()
    )
    assert all(group["ussr"].pending_continuation for group in planners.values())


def test_downstream_mode_boundary_preserves_current_macro_continuation() -> None:
    planner = SimpleNamespace(
        pending_continuation=[{"side": "us", "action": "ops"}],
        pending_continuation_seed=23,
    )
    planners = {"d4": {"us": planner}}
    observation = {
        "side": "us",
        "current_player": "us",
        "phase": "action",
        "turn": 4,
        "action_round": 1,
        "prompt": "US playing ABM Treaty",
        "hand": [],
    }
    legal_actions = [
        {"decision": "option", "value": "event"},
        {"decision": "option", "value": "ops"},
    ]

    boundary = _clear_independent_root_continuations(
        planners,
        side="us",
        last_obs=observation,
        legal_actions=legal_actions,
    )

    assert boundary == "mode"
    assert planner.pending_continuation == [{"side": "us", "action": "ops"}]
    assert planner.pending_continuation_seed == 23
