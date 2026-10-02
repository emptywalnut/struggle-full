from __future__ import annotations

import json
from pathlib import Path

from scripts.collect_puct_selfplay_parallel import (
    _merge_replay_manifests,
    _resume_throughput_totals,
)


def _write_worker_manifest(root: Path, worker_id: int, examples: int) -> Path:
    replay = root / "workers" / f"worker-{worker_id:03d}" / "search_replay"
    replay.mkdir(parents=True)
    shard = replay / "shard-000000.npz"
    shard.write_bytes(f"worker-{worker_id}".encode())
    manifest = replay / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "format": "compressed_npz_shards",
                "examples": examples,
                "shard_size": examples,
                "shards": [{"path": shard.name, "examples": examples}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_merge_includes_durable_manifests_from_inactive_worker_ids(tmp_path: Path) -> None:
    current = _write_worker_manifest(tmp_path, worker_id=0, examples=3)
    _write_worker_manifest(tmp_path, worker_id=9, examples=5)

    merged = _merge_replay_manifests(
        tmp_path,
        [{"worker_id": 0, "replay_manifest": str(current)}],
    )

    payload = json.loads(merged.read_text(encoding="utf-8"))
    assert payload["examples"] == 8
    assert [item["path"] for item in payload["shards"]] == [
        "worker-000-000000.npz",
        "worker-009-000000.npz",
    ]
    assert (merged.parent / "worker-009-000000.npz").read_bytes() == b"worker-9"


def test_merge_atomically_replaces_stale_target_files(tmp_path: Path) -> None:
    current = _write_worker_manifest(tmp_path, worker_id=0, examples=2)
    target = tmp_path / "search_replay"
    target.mkdir()
    (target / "stale.npz").write_bytes(b"stale")
    (target / "manifest.json").write_text("{}", encoding="utf-8")

    merged = _merge_replay_manifests(
        tmp_path,
        [{"worker_id": 0, "replay_manifest": str(current)}],
    )

    assert not (target / "stale.npz").exists()
    assert json.loads(merged.read_text(encoding="utf-8"))["examples"] == 2


def test_resume_throughput_excludes_preexisting_completed_games() -> None:
    resumed = [
        {"game_index": 1, "simulations": 100, "searched_decisions": 10},
        {"game_index": 4, "simulations": 200, "searched_decisions": 20},
    ]
    records = [
        *resumed,
        {"game_index": 2, "simulations": 300, "searched_decisions": 30},
        {"game_index": 7, "simulations": 400, "searched_decisions": 40},
    ]

    assert _resume_throughput_totals(records, resumed) == {
        "new_games_completed": 2,
        "new_simulations": 700,
        "new_searched_decisions": 70,
    }
