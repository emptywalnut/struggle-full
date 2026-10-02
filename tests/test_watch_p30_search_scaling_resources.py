from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/watch_p30_search_scaling_resources.py"
SPEC = importlib.util.spec_from_file_location("watch_p30_resources", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_available_memory_gib(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 2048 kB\nMemAvailable: 1048576 kB\n", encoding="ascii")
    assert module.available_memory_gib(meminfo) == 1.0


def test_available_disk_gib(tmp_path: Path, monkeypatch) -> None:
    usage = type("DiskUsage", (), {"free": 7 * 2**30})()
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: usage)

    assert module.available_disk_gib(tmp_path) == 7.0


def test_process_age_seconds_handles_spaces_in_process_name(
    tmp_path: Path, monkeypatch
) -> None:
    proc = tmp_path / "123"
    proc.mkdir()
    # fields_after_comm[0] is field 3 (state); index 19 is field 22 (starttime).
    fields_after_comm = ["S", *(["0"] * 18), "500"]
    (proc / "stat").write_text(
        f"123 (search worker) {' '.join(fields_after_comm)}\n", encoding="ascii"
    )
    (tmp_path / "uptime").write_text("100.0 0.0\n", encoding="ascii")
    monkeypatch.setattr(module.os, "sysconf", lambda _name: 100)

    assert module.process_age_seconds(123, proc_root=tmp_path) == 95.0


def test_throttle_order_prefers_fourth_shard(monkeypatch) -> None:
    monkeypatch.setattr(module, "collection_worker_count", lambda collection: 16)
    assert module.choose_throttle_collection(8) == module.COLLECTIONS[3]


def test_throttle_order_skips_elastic_shard_already_at_floor(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "collection_worker_count",
        lambda collection: 8 if collection == module.COLLECTIONS[3] else 16,
    )
    assert module.choose_throttle_collection(8) == module.COLLECTIONS[2]


def test_elastic_shard_inherits_completed_shard_capacity(tmp_path, monkeypatch) -> None:
    collections = tuple(tmp_path / f"shard-{index}" for index in range(4))
    for collection in collections:
        collection.mkdir()
    workers = {collections[0]: 16, collections[1]: 16, collections[2]: 16, collections[3]: 8}
    monkeypatch.setattr(module, "COLLECTIONS", collections)
    monkeypatch.setattr(module, "ELASTIC_COLLECTION", collections[3])
    monkeypatch.setattr(module, "collection_worker_count", workers.__getitem__)

    assert module.desired_elastic_worker_count(
        target_total_workers=56, elastic_max_workers=32
    ) == 8

    (collections[0] / "collection_summary.json").write_text("{}", encoding="utf-8")
    assert module.desired_elastic_worker_count(
        target_total_workers=56, elastic_max_workers=32
    ) == 24

    (collections[1] / "collection_summary.json").write_text("{}", encoding="utf-8")
    assert module.desired_elastic_worker_count(
        target_total_workers=56, elastic_max_workers=32
    ) == 32


def test_completed_game_count_ignores_partial_staging(tmp_path: Path) -> None:
    completed = tmp_path / "completed_games"
    completed.mkdir()
    for index in range(3):
        (completed / f"game-{index:06d}.json").write_text("{}", encoding="utf-8")
    staging = tmp_path / "workers/worker-000/staging/teacher/game-000003"
    staging.mkdir(parents=True)
    (staging / "partial.npz").write_bytes(b"partial")

    assert module.completed_game_count(tmp_path) == 3


def test_newest_expected_worker_heartbeat_age_ignores_inactive_worker_ids(
    tmp_path: Path,
) -> None:
    for worker_id, updated_at in ((0, 100.0), (1, 150.0), (9, 199.0)):
        heartbeat = tmp_path / f"workers/worker-{worker_id:03d}/heartbeat.json"
        heartbeat.parent.mkdir(parents=True)
        heartbeat.write_text(
            __import__("json").dumps({"updated_at": updated_at}), encoding="utf-8"
        )

    assert module.newest_expected_worker_heartbeat_age(
        tmp_path, 2, now=200.0
    ) == 50.0


def test_newest_expected_worker_heartbeat_age_requires_a_valid_heartbeat(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "workers/worker-000/heartbeat.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{}", encoding="utf-8")

    assert module.newest_expected_worker_heartbeat_age(
        tmp_path, 1, now=200.0
    ) is None


def test_expected_worker_heartbeat_ages_reports_every_valid_worker(
    tmp_path: Path,
) -> None:
    for worker_id, updated_at in ((0, 100.0), (1, 150.0), (2, 175.0)):
        heartbeat = tmp_path / f"workers/worker-{worker_id:03d}/heartbeat.json"
        heartbeat.parent.mkdir(parents=True)
        heartbeat.write_text(
            __import__("json").dumps({"updated_at": updated_at}), encoding="utf-8"
        )
    malformed = tmp_path / "workers/worker-003/heartbeat.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{}", encoding="utf-8")

    assert module.expected_worker_heartbeat_ages(tmp_path, 4, now=200.0) == [
        100.0,
        50.0,
        25.0,
    ]


def test_worker_override_records_rebalance_reason(tmp_path: Path) -> None:
    module.write_worker_override(
        tmp_path,
        24,
        512.0,
        reason="completed_shard_capacity_rebalance",
    )
    payload = __import__("json").loads((tmp_path / ".collector_workers.json").read_text())
    assert payload["workers"] == 24
    assert payload["reason"] == "completed_shard_capacity_rebalance"
