from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.generate_search_replay_parallel import (
    build_root_tasks,
    filter_completed_tasks,
    root_key,
    reset_planner_for_root,
    root_search_seed,
)


def test_root_search_seed_is_config_independent_and_position_specific() -> None:
    assert root_search_seed(100, 7, 11) == root_search_seed(100, 7, 11)
    assert root_search_seed(100, 7, 11) != root_search_seed(100, 7, 12)


def test_reset_planner_reproduces_rng_and_clears_runtime_state() -> None:
    planner = SimpleNamespace(
        seed=0,
        rng=np.random.default_rng(1),
        search_index=9,
        pending_continuation=[{"action": "x"}],
        pending_continuation_seed=3,
        scoring_search_counts={"x": 1},
        chance_search_counts={"x": 1},
        allocation_search_counts={"x": 1},
    )
    reset_planner_for_root(planner, 42)
    first = planner.rng.integers(0, 1_000_000, size=8)
    reset_planner_for_root(planner, 42)
    second = planner.rng.integers(0, 1_000_000, size=8)
    np.testing.assert_array_equal(first, second)
    assert planner.search_index == 0
    assert planner.pending_continuation == []
    assert planner.pending_continuation_seed is None
    assert planner.scoring_search_counts == {}
    assert planner.chance_search_counts == {}
    assert planner.allocation_search_counts == {}


def test_build_root_tasks_can_split_roots_from_the_same_game() -> None:
    positions = [
        {
            "game_log": "/tmp/game.jsonl",
            "seed": 7,
            "game_occurrence": 0,
            "action_index": action_index,
        }
        for action_index in (20, 10, 30)
    ]
    grouped = build_root_tasks(positions, roots_per_task=0)
    split = build_root_tasks(positions, roots_per_task=1)
    assert len(grouped) == 1
    assert [row["action_index"] for row in grouped[0]["positions"]] == [10, 20, 30]
    assert len(split) == 3
    assert [task["positions"][0]["action_index"] for task in split] == [10, 20, 30]


def test_filter_completed_tasks_preserves_only_unfinished_roots() -> None:
    positions = [
        {
            "game_log": "/tmp/game.jsonl",
            "seed": 7,
            "game_occurrence": 0,
            "action_index": action_index,
        }
        for action_index in (10, 20, 30)
    ]
    tasks = build_root_tasks(positions, roots_per_task=1)
    pending = filter_completed_tasks(tasks, {root_key(7, 0, 20)})
    assert [task["positions"][0]["action_index"] for task in pending] == [10, 30]
