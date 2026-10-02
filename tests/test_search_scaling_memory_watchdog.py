from scripts.watch_search_scaling_collection_memory import (
    available_memory_gib,
    collector_is_absent,
    restart_command,
    should_take_over,
    tail_is_stalled,
)


def test_takeover_requires_consecutive_low_memory_samples() -> None:
    assert not should_take_over([250.0, 270.0, 250.0], 260.0, 3)
    assert should_take_over([300.0, 250.0, 240.0, 230.0], 260.0, 3)


def test_collector_absence_requires_consecutive_polls() -> None:
    assert not collector_is_absent([False, True, True], 3)
    assert not collector_is_absent([True, True], 3)
    assert collector_is_absent([False, True, True, True], 3)


def test_restart_preserves_collection_identity_settings() -> None:
    command = restart_command(8)
    assert command[command.index("--workers") + 1] == "8"
    assert command[command.index("--seed-base") + 1] == "97300000"
    assert "--no-adaptive-on-flip" in command
    assert command[command.index("--inference-replicas-per-gpu") + 1] == "1"


def test_available_memory_reads_linux_meminfo(tmp_path) -> None:
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 2048 kB\nMemAvailable: 1048576 kB\n")
    assert available_memory_gib(path) == 1.0


def test_tail_stall_requires_only_a_small_stale_remainder(tmp_path) -> None:
    completed = tmp_path / "completed_games"
    workers = tmp_path / "workers" / "worker-000"
    completed.mkdir(parents=True)
    workers.mkdir(parents=True)
    for game_index in range(3):
        (completed / f"game-{game_index:06d}.json").write_text(
            '{"game_index": %d}\n' % game_index
        )
    (workers / "heartbeat.json").write_text(
        '{"game_index": 3, "updated_at": 100.0}\n'
    )

    stalled, missing, ages = tail_is_stalled(
        output_dir=tmp_path,
        expected_games=4,
        max_remaining=1,
        stale_seconds=50.0,
        now=200.0,
    )
    assert stalled
    assert missing == [3]
    assert ages == {3: 100.0}

    stalled, _, _ = tail_is_stalled(
        output_dir=tmp_path,
        expected_games=4,
        max_remaining=1,
        stale_seconds=150.0,
        now=200.0,
    )
    assert not stalled


def test_tail_stall_never_interrupts_a_broad_incomplete_shard(tmp_path) -> None:
    completed = tmp_path / "completed_games"
    completed.mkdir(parents=True)
    (completed / "game-000000.json").write_text('{"game_index": 0}\n')
    stalled, missing, _ = tail_is_stalled(
        output_dir=tmp_path,
        expected_games=4,
        max_remaining=2,
        stale_seconds=1.0,
        now=200.0,
    )
    assert not stalled
    assert missing == [1, 2, 3]
