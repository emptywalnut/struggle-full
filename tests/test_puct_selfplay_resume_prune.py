from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.collect_puct_selfplay_parallel import (
    _drain_results,
    _prune_interrupted_worker_replay,
    _required_worker_recursion_limit,
    _stop_inference_servers,
)
from struggle_ai.mcts import SelectiveSearchDecision
from struggle_ai.search_replay import (
    SearchReplayWriter,
    example_from_decision,
    load_search_replay,
    with_terminal_outcome,
)


def completed_example(source_game: str, winner: str = "us"):
    observation = {
        "action_mask": np.ones(2, dtype=np.float32),
        "state": np.ones(1, dtype=np.float32),
    }
    decision = SelectiveSearchDecision(
        action=1,
        raw_action=0,
        raw_policy=np.array([0.6, 0.4]),
        search_policy=np.array([0.25, 0.75]),
        visits=np.array([8.0, 24.0]),
        q_root=np.array([-0.1, 0.2]),
        searched=True,
        triggers=("allocation",),
        simulations=32,
        budgets={},
        root_q_gap=0.3,
        action_flipped=True,
        latency_seconds=0.1,
        expanded_nodes=3,
        exact_chance_nodes=0,
        belief_particles=32,
        meaningful_depth=2,
        hierarchical_path=(),
    )
    example = example_from_decision(
        observation,
        [{"value": "a"}, {"value": "b"}],
        "us",
        decision,
        search_depth=2,
        source_game=source_game,
        source_step=1,
    )
    return with_terminal_outcome(
        [example], winner=winner, terminal_reason="final_scoring"
    )[0]


def write_replay(path: Path, *examples) -> None:
    with SearchReplayWriter(path, shard_size=1) as writer:
        for example in examples:
            writer.add(example)


def test_worker_recursion_limit_scales_above_atomic_horizon() -> None:
    assert _required_worker_recursion_limit(512) == 4096
    assert _required_worker_recursion_limit(1536) == 8192


def test_worker_error_preserves_traceback() -> None:
    class OneEventQueue:
        def __init__(self):
            self.events = [
                {
                    "event": "worker_error",
                    "worker_id": 3,
                    "error": "RecursionError('boom')",
                    "traceback": "Traceback: worker stack",
                    "recursion_limit": 8192,
                }
            ]

        def get_nowait(self):
            if self.events:
                return self.events.pop()
            import queue

            raise queue.Empty

    errors = _drain_results(OneEventQueue(), [], [])
    assert errors == [
        "worker 3: RecursionError('boom')\n"
        "Traceback: worker stack\nworker_recursion_limit=8192"
    ]


def test_inference_server_cleanup_skips_unstarted_and_terminates_stuck() -> None:
    class RequestQueue:
        def __init__(self):
            self.items = []

        def put(self, item, timeout):
            self.items.append((item, timeout))

    class Server:
        def __init__(self, states):
            self.states = iter(states)
            self.current = None
            self.joins = []
            self.terminated = False

        def is_alive(self):
            self.current = next(self.states, self.current)
            if self.current == "unstarted":
                raise AssertionError
            return bool(self.current)

        def join(self, timeout):
            self.joins.append(timeout)

        def terminate(self):
            self.terminated = True

    queue_ = RequestQueue()
    clean = Server((True, False))
    stuck = Server((True, True, False))
    unstarted = Server(("unstarted",))

    _stop_inference_servers(queue_, [clean, stuck, unstarted])

    assert queue_.items == [(None, 1.0), (None, 1.0)]
    assert clean.joins == [30]
    assert not clean.terminated
    assert stuck.joins == [30, 5]
    assert stuck.terminated


def test_resume_prune_removes_orphan_and_wrong_worker_labels(tmp_path) -> None:
    worker0 = tmp_path / "workers/worker-000"
    worker1 = tmp_path / "workers/worker-001"
    valid = completed_example("game-valid")
    orphan = completed_example("game-orphan")
    write_replay(worker0 / "search_replay", valid, orphan)
    write_replay(worker0 / "anchor_replay", valid, orphan)
    write_replay(worker1 / "search_replay", valid)
    write_replay(worker1 / "anchor_replay", valid)
    stale = worker0 / "staging/teacher/game-orphan"
    stale.mkdir(parents=True)
    (stale / "partial").write_text("partial", encoding="utf-8")

    stats = _prune_interrupted_worker_replay(
        tmp_path,
        [
            {
                "worker_id": 0,
                "source_game": "game-valid",
                "winner": "us",
                "terminal_reason": "final_scoring",
                "committed_examples": 1,
                "anchor_committed_examples": 1,
            }
        ],
        shard_size=2,
    )

    assert [item.metadata.source_game for item in load_search_replay([worker0 / "search_replay"])] == [
        "game-valid"
    ]
    assert load_search_replay([worker1 / "search_replay"]) == []
    assert not (worker0 / "staging").exists()
    assert stats["search_replay_retained"] == 1
    assert stats["search_replay_dropped"] == 2
    assert stats["anchor_replay_retained"] == 1
    assert stats["anchor_replay_dropped"] == 2


def test_resume_prune_fails_closed_when_completed_labels_are_missing(tmp_path) -> None:
    (tmp_path / "workers/worker-000").mkdir(parents=True)
    try:
        _prune_interrupted_worker_replay(
            tmp_path,
            [
                {
                    "worker_id": 0,
                    "source_game": "game-valid",
                    "winner": "us",
                    "terminal_reason": "final_scoring",
                    "committed_examples": 1,
                    "anchor_committed_examples": 0,
                }
            ],
            shard_size=2,
        )
    except RuntimeError as error:
        assert "teacher=0/1" in str(error)
    else:
        raise AssertionError("missing committed labels were accepted")


def test_resume_prune_discards_corrupt_uncommitted_manifest(tmp_path) -> None:
    replay = tmp_path / "workers/worker-000/search_replay"
    replay.mkdir(parents=True)
    (replay / "manifest.json").write_text("", encoding="utf-8")
    (replay / "orphan.npz").write_bytes(b"partial")

    stats = _prune_interrupted_worker_replay(tmp_path, [], shard_size=2)

    assert not replay.exists()
    assert stats["search_replay_corrupt_uncommitted_pruned"] == 1


def test_resume_prune_rejects_corrupt_manifest_with_committed_labels(tmp_path) -> None:
    replay = tmp_path / "workers/worker-000/search_replay"
    replay.mkdir(parents=True)
    (replay / "manifest.json").write_text("", encoding="utf-8")

    try:
        _prune_interrupted_worker_replay(
            tmp_path,
            [
                {
                    "worker_id": 0,
                    "source_game": "game-valid",
                    "winner": "us",
                    "terminal_reason": "final_scoring",
                    "committed_examples": 1,
                    "anchor_committed_examples": 0,
                }
            ],
            shard_size=2,
        )
    except RuntimeError as error:
        assert "corrupt replay manifest contains committed-game labels" in str(error)
    else:
        raise AssertionError("corrupt committed replay was silently discarded")
