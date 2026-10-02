from pathlib import Path

from scripts.watch_search_reroots import (
    collections_with_completed_games,
    transient_inventory_mismatch,
)


def test_reroot_watcher_skips_only_shards_without_committed_games(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    committed = tmp_path / "committed"
    (empty / "completed_games").mkdir(parents=True)
    completed_dir = committed / "completed_games"
    completed_dir.mkdir(parents=True)
    (completed_dir / "game-000003.json").write_text("{}\n", encoding="utf-8")

    assert collections_with_completed_games([empty, committed]) == [committed]


def test_reroot_watcher_does_not_treat_uncommitted_staging_as_complete(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"
    staging = collection / "workers/worker-000/staging/anchor/game-000000"
    staging.mkdir(parents=True)
    (staging / "manifest.json").write_text("{}\n", encoding="utf-8")

    assert collections_with_completed_games([collection]) == []


def test_reroot_watcher_retries_only_atomic_inventory_window() -> None:
    assert transient_inventory_mismatch(
        RuntimeError("corrected-game reroot audit inventory mismatch: 63/62")
    )
    assert not transient_inventory_mismatch(
        RuntimeError("collection has no anchor replay manifests: /tmp/collection")
    )
    assert not transient_inventory_mismatch(ValueError("inventory mismatch"))
