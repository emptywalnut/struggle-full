#!/usr/bin/env python3
"""Keep the search-scaling collectors inside a safe, useful worker envelope."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from scripts.continue_p30_search_scaling_amortization import (
        COLLECTIONS,
        ROOT,
        collection_worker_count,
    )
except ModuleNotFoundError:
    from continue_p30_search_scaling_amortization import (  # type: ignore[no-redef]
        COLLECTIONS,
        ROOT,
        collection_worker_count,
    )


ELASTIC_COLLECTION = COLLECTIONS[3]
THROTTLE_ORDER = (ELASTIC_COLLECTION, COLLECTIONS[2], COLLECTIONS[0], COLLECTIONS[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-threshold-gib", type=float, default=240.0)
    parser.add_argument("--consecutive-polls", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--floor-workers", type=int, default=8)
    parser.add_argument("--termination-grace-seconds", type=float, default=30.0)
    parser.add_argument("--scale-up-available-gib", type=float, default=400.0)
    parser.add_argument("--scale-up-consecutive-polls", type=int, default=3)
    parser.add_argument("--target-total-workers", type=int, default=56)
    parser.add_argument("--elastic-max-workers", type=int, default=32)
    parser.add_argument("--disk-path", type=Path, default=ROOT)
    parser.add_argument("--min-free-disk-gib", type=float, default=4.0)
    parser.add_argument(
        "--worker-stale-telemetry-seconds",
        type=float,
        default=900.0,
        help="Report workers whose decision heartbeat exceeds a full adaptive-search envelope.",
    )
    parser.add_argument(
        "--max-all-workers-stale-seconds",
        type=float,
        default=1800.0,
        help=(
            "Restart a live collector only when every configured worker heartbeat "
            "is older than this threshold. Zero disables hang recovery."
        ),
    )
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True), flush=True)


def available_memory_gib(meminfo: Path = Path("/proc/meminfo")) -> float:
    for line in meminfo.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 2**30
    raise RuntimeError(f"MemAvailable is missing from {meminfo}")


def available_disk_gib(path: Path = ROOT) -> float:
    return float(shutil.disk_usage(path).free) / 2**30


def collector_main_pids(collection: Path) -> list[int]:
    relative = str(collection.relative_to(ROOT))
    canonical = str(collection)
    absolute = str(collection.resolve())
    matches: list[int] = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            argv = [
                item.decode("utf-8", errors="replace")
                for item in (proc_dir / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (OSError, ValueError):
            continue
        if "scripts/collect_puct_selfplay_parallel.py" not in argv:
            continue
        if relative in argv or canonical in argv or absolute in argv:
            matches.append(int(proc_dir.name))
    return sorted(matches)


def process_age_seconds(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> float:
    """Read a process age from procfs without depending on psutil."""

    raw_stat = (proc_root / str(int(pid)) / "stat").read_text(encoding="ascii")
    closing_parenthesis = raw_stat.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError(f"malformed proc stat for pid {pid}")
    fields_after_comm = raw_stat[closing_parenthesis + 2 :].split()
    start_ticks = int(fields_after_comm[19])
    uptime_seconds = float(
        (proc_root / "uptime").read_text(encoding="ascii").split()[0]
    )
    ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    return max(0.0, uptime_seconds - start_ticks / ticks_per_second)


def youngest_collector_age_seconds(collection: Path) -> float | None:
    ages: list[float] = []
    for pid in collector_main_pids(collection):
        try:
            ages.append(process_age_seconds(pid))
        except (OSError, ValueError, IndexError):
            continue
    return min(ages) if ages else None


def choose_throttle_collection(floor_workers: int) -> Path | None:
    for collection in THROTTLE_ORDER:
        if (collection / "collection_summary.json").is_file():
            continue
        if collection_worker_count(collection) > int(floor_workers):
            return collection
    return None


def write_worker_override(
    collection: Path,
    workers: int,
    available_gib: float,
    *,
    reason: str = "host_available_memory_below_threshold",
) -> None:
    path = collection / ".collector_workers.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "workers": int(workers),
                "reason": str(reason),
                "available_memory_gib": float(available_gib),
                "updated_at": time.time(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def completed_game_count(collection: Path) -> int:
    return sum(1 for _path in (collection / "completed_games").glob("game-*.json"))


def newest_expected_worker_heartbeat_age(
    collection: Path,
    workers: int,
    *,
    now: float | None = None,
) -> float | None:
    """Return the age of the newest heartbeat among currently configured workers."""

    timestamps: list[float] = []
    for worker_id in range(int(workers)):
        path = collection / "workers" / f"worker-{worker_id:03d}" / "heartbeat.json"
        if not path.is_file():
            continue
        try:
            timestamps.append(float(json.loads(path.read_text(encoding="utf-8"))["updated_at"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue
    if not timestamps:
        return None
    return max(0.0, float(time.time() if now is None else now) - max(timestamps))


def expected_worker_heartbeat_ages(
    collection: Path,
    workers: int,
    *,
    now: float | None = None,
) -> list[float]:
    """Return heartbeat ages for every configured worker with valid telemetry."""

    observed_at = float(time.time() if now is None else now)
    ages: list[float] = []
    for worker_id in range(int(workers)):
        path = collection / "workers" / f"worker-{worker_id:03d}" / "heartbeat.json"
        if not path.is_file():
            continue
        try:
            updated_at = float(json.loads(path.read_text(encoding="utf-8"))["updated_at"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue
        ages.append(max(0.0, observed_at - updated_at))
    return ages


def desired_elastic_worker_count(
    *,
    target_total_workers: int,
    elastic_max_workers: int,
) -> int:
    current = collection_worker_count(ELASTIC_COLLECTION)
    other_incomplete_workers = sum(
        collection_worker_count(collection)
        for collection in COLLECTIONS
        if collection != ELASTIC_COLLECTION
        and not (collection / "collection_summary.json").is_file()
    )
    return min(
        int(elastic_max_workers),
        max(current, int(target_total_workers) - other_incomplete_workers),
    )


def terminate_collector_groups(collection: Path, grace_seconds: float) -> list[int]:
    pids = collector_main_pids(collection)
    groups = sorted({os.getpgid(pid) for pid in pids})
    for group in groups:
        os.killpg(group, signal.SIGTERM)
    deadline = time.time() + max(1.0, float(grace_seconds))
    while collector_main_pids(collection) and time.time() < deadline:
        time.sleep(0.5)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return groups


def main() -> None:
    args = parse_args()
    low_samples = 0
    high_samples = 0
    hang_restart_times: dict[Path, float] = {}
    while True:
        if all((collection / "collection_summary.json").is_file() for collection in COLLECTIONS):
            emit("all_collections_complete")
            return
        available_gib = available_memory_gib()
        free_disk_gib = available_disk_gib(Path(args.disk_path))
        disk_low = free_disk_gib < float(args.min_free_disk_gib)
        low_samples = low_samples + 1 if available_gib < float(args.available_threshold_gib) else 0
        high_samples = (
            high_samples + 1
            if available_gib >= float(args.scale_up_available_gib) and not disk_low
            else 0
        )
        workers = {
            collection: collection_worker_count(collection) for collection in COLLECTIONS
        }
        heartbeat_ages = {
            collection: newest_expected_worker_heartbeat_age(
                collection, workers[collection]
            )
            for collection in COLLECTIONS
            if not (collection / "collection_summary.json").is_file()
        }
        all_heartbeat_ages = {
            collection: expected_worker_heartbeat_ages(
                collection, workers[collection]
            )
            for collection in heartbeat_ages
        }
        collector_ages = {
            collection: youngest_collector_age_seconds(collection)
            for collection in heartbeat_ages
        }
        emit(
            "resource_sample",
            available_memory_gib=available_gib,
            available_disk_gib=free_disk_gib,
            disk_path=str(Path(args.disk_path)),
            disk_low=disk_low,
            min_free_disk_gib=float(args.min_free_disk_gib),
            threshold_gib=float(args.available_threshold_gib),
            consecutive_low_samples=low_samples,
            consecutive_scale_up_samples=high_samples,
            workers={
                str(collection.relative_to(ROOT)): workers[collection]
                for collection in COLLECTIONS
            },
            newest_worker_heartbeat_age_seconds={
                str(collection.relative_to(ROOT)): age
                for collection, age in heartbeat_ages.items()
            },
            oldest_worker_heartbeat_age_seconds={
                str(collection.relative_to(ROOT)): (max(ages) if ages else None)
                for collection, ages in all_heartbeat_ages.items()
            },
            workers_without_valid_heartbeat={
                str(collection.relative_to(ROOT)): max(
                    0, workers[collection] - len(all_heartbeat_ages[collection])
                )
                for collection in all_heartbeat_ages
            },
            worker_stale_telemetry_seconds=float(args.worker_stale_telemetry_seconds),
            workers_stale_over_telemetry_threshold={
                str(collection.relative_to(ROOT)): sum(
                    age > float(args.worker_stale_telemetry_seconds)
                    for age in all_heartbeat_ages[collection]
                )
                for collection in all_heartbeat_ages
            },
            youngest_collector_age_seconds={
                str(collection.relative_to(ROOT)): age
                for collection, age in collector_ages.items()
            },
        )
        if disk_low:
            emit(
                "disk_headroom_low",
                available_disk_gib=free_disk_gib,
                min_free_disk_gib=float(args.min_free_disk_gib),
                action="scale_up_suppressed_active_games_preserved",
            )
        stale_threshold = float(args.max_all_workers_stale_seconds)
        stale_collection = next(
            (
                collection
                for collection in COLLECTIONS
                if stale_threshold > 0.0
                and heartbeat_ages.get(collection) is not None
                and float(heartbeat_ages[collection]) >= stale_threshold
                and collector_ages.get(collection) is not None
                and float(collector_ages[collection]) >= stale_threshold
                and time.time() - hang_restart_times.get(collection, 0.0)
                >= stale_threshold
            ),
            None,
        )
        if stale_collection is not None:
            groups = terminate_collector_groups(
                stale_collection, float(args.termination_grace_seconds)
            )
            hang_restart_times[stale_collection] = time.time()
            emit(
                "collector_restarted_after_all_workers_stale",
                collection=str(stale_collection.relative_to(ROOT)),
                workers=workers[stale_collection],
                newest_heartbeat_age_seconds=heartbeat_ages[stale_collection],
                youngest_collector_age_seconds=collector_ages[stale_collection],
                threshold_seconds=stale_threshold,
                process_groups=groups,
            )
            low_samples = 0
            high_samples = 0
        elif low_samples >= int(args.consecutive_polls):
            collection = choose_throttle_collection(int(args.floor_workers))
            if collection is None:
                emit("memory_low_no_further_safe_backoff", available_memory_gib=available_gib)
                low_samples = 0
            else:
                write_worker_override(collection, int(args.floor_workers), available_gib)
                groups = terminate_collector_groups(
                    collection, float(args.termination_grace_seconds)
                )
                emit(
                    "collector_throttled",
                    collection=str(collection.relative_to(ROOT)),
                    workers=int(args.floor_workers),
                    process_groups=groups,
                    available_memory_gib=available_gib,
                )
                low_samples = 0
                high_samples = 0
        elif (
            high_samples >= int(args.scale_up_consecutive_polls)
            and not (ELASTIC_COLLECTION / "collection_summary.json").is_file()
        ):
            current = collection_worker_count(ELASTIC_COLLECTION)
            desired = desired_elastic_worker_count(
                target_total_workers=int(args.target_total_workers),
                elastic_max_workers=int(args.elastic_max_workers),
            )
            remaining = max(0, 32 - completed_game_count(ELASTIC_COLLECTION))
            # Do not discard an already sufficient final wave merely to expose
            # more idle process slots. Rebalance only when extra concurrent
            # games can reduce the remaining number of waves.
            if desired > current and remaining > current:
                write_worker_override(
                    ELASTIC_COLLECTION,
                    desired,
                    available_gib,
                    reason="completed_shard_capacity_rebalance",
                )
                groups = terminate_collector_groups(
                    ELASTIC_COLLECTION, float(args.termination_grace_seconds)
                )
                emit(
                    "collector_scaled_up",
                    collection=str(ELASTIC_COLLECTION.relative_to(ROOT)),
                    previous_workers=current,
                    workers=desired,
                    remaining_games=remaining,
                    target_total_workers=int(args.target_total_workers),
                    process_groups=groups,
                    available_memory_gib=available_gib,
                )
                high_samples = 0
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
