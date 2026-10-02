#!/usr/bin/env python3
"""Fail-safe memory takeover for the active search-scaling collection shard."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075"
OUTPUT = ROOT / "runs/puct_selfplay/search-scaling-phase3u75-selective-adaptive-d4d6-32games-w12-gpu45-20260821"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-threshold-gib", type=float, default=260.0)
    parser.add_argument("--consecutive-polls", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--session", default="ts-scale-selfplay")
    parser.add_argument("--restart-workers", type=int, default=8)
    parser.add_argument("--tail-max-remaining", type=int, default=2)
    parser.add_argument("--tail-stale-seconds", type=float, default=43200.0)
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True), flush=True)


def should_take_over(samples_gib: list[float], threshold_gib: float, consecutive: int) -> bool:
    return len(samples_gib) >= consecutive and all(
        value < threshold_gib for value in samples_gib[-consecutive:]
    )


def collector_is_absent(samples: list[bool], consecutive: int) -> bool:
    """Require repeated absence observations before restarting a dead collector."""
    return len(samples) >= consecutive and all(samples[-consecutive:])


def completed_game_indices(output_dir: Path = OUTPUT) -> set[int]:
    result: set[int] = set()
    for path in (output_dir / "completed_games").glob("game-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result.add(int(payload["game_index"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return result


def tail_is_stalled(
    *,
    output_dir: Path,
    expected_games: int,
    max_remaining: int,
    stale_seconds: float,
    now: float | None = None,
) -> tuple[bool, list[int], dict[int, float | None]]:
    """Detect only a stale collection tail; never interrupt a broad active shard."""
    completed = completed_game_indices(output_dir)
    missing = sorted(set(range(expected_games)) - completed)
    if not missing or len(missing) > max_remaining:
        return False, missing, {}

    observed_at = time.time() if now is None else float(now)
    heartbeat_ages: dict[int, float | None] = {game_index: None for game_index in missing}
    for path in (output_dir / "workers").glob("worker-*/heartbeat.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            game_index = int(payload["game_index"])
            if game_index in heartbeat_ages:
                age = observed_at - float(payload["updated_at"])
                prior = heartbeat_ages[game_index]
                heartbeat_ages[game_index] = age if prior is None else min(prior, age)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    stalled = all(age is None or age >= stale_seconds for age in heartbeat_ages.values())
    return stalled, missing, heartbeat_ages


def available_memory_gib(meminfo: Path = Path("/proc/meminfo")) -> float:
    fields = {}
    for line in meminfo.read_text(encoding="ascii").splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.strip().split()[0])
    if "MemAvailable" not in fields:
        raise RuntimeError(f"MemAvailable is missing from {meminfo}")
    return fields["MemAvailable"] * 1024 / 2**30


def collector_pids() -> list[int]:
    marker = str(OUTPUT.relative_to(ROOT))
    output = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
    ).stdout
    pids = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        pid, command = fields
        if "collect_puct_selfplay_parallel.py" in command and marker in command:
            pids.append(int(pid))
    return sorted(pids)


def restart_command(workers: int) -> list[str]:
    return [
        sys.executable,
        "scripts/collect_puct_selfplay_parallel.py",
        "--checkpoint", str(PARENT),
        "--output-dir", str(OUTPUT),
        "--games", "32",
        "--seed-base", "97300000",
        "--gpus", "4,5",
        "--workers", str(workers),
        "--inference-replicas-per-gpu", "1",
        "--max-inference-batch-size", "32",
        "--max-inference-delay-ms", "3",
        "--inference-timeout-seconds", "300",
        "--initial-simulations", "128",
        "--medium-simulations", "128",
        "--max-simulations", "128",
        "--meaningful-depth", "4",
        "--belief-particles", "32",
        "--max-atomic-steps", "1536",
        "--max-expanded-nodes", "32768",
        "--child-initial", "4",
        "--child-max", "24",
        "--search-temperature", "1.0",
        "--behavior-temperature", "0.5",
        "--force-search",
        "--low-confidence-sample-rate", "0.0",
        "--max-episode-steps", "1200",
        "--shard-size", "128",
        "--adaptive-search",
        "--no-adaptive-on-flip",
        "--adaptive-budget-simulations", "512",
        "--adaptive-deep-depth", "6",
        "--adaptive-deep-simulations", "512",
        "--adaptive-min-rank-correlation", "0.70",
        "--adaptive-meaningful-q-gap", "0.01",
        "--adaptive-escalation-q-gap", "0.03",
    ]


def main() -> None:
    args = parse_args()
    samples: list[float] = []
    collector_absent_samples: list[bool] = []
    summary = OUTPUT / "collection_summary.json"
    takeover_reason: str | None = None
    takeover_detail: dict[str, Any] = {}
    while not summary.is_file():
        available_gib = available_memory_gib()
        samples.append(available_gib)
        active_collector_pids = collector_pids()
        collector_absent_samples.append(not active_collector_pids)
        emit(
            "memory_sample",
            available_gib=available_gib,
            threshold_gib=float(args.available_threshold_gib),
            collector_pids=active_collector_pids,
        )
        if should_take_over(
            samples, float(args.available_threshold_gib), int(args.consecutive_polls)
        ):
            takeover_reason = "host_available_memory_below_threshold"
            break
        if collector_is_absent(collector_absent_samples, int(args.consecutive_polls)):
            takeover_reason = "collector_process_absent"
            emit(
                "collector_absence_detected",
                consecutive_polls=int(args.consecutive_polls),
            )
            break
        tail_stalled, missing, heartbeat_ages = tail_is_stalled(
            output_dir=OUTPUT,
            expected_games=32,
            max_remaining=max(1, int(args.tail_max_remaining)),
            stale_seconds=max(60.0, float(args.tail_stale_seconds)),
        )
        if tail_stalled:
            takeover_reason = "stale_collection_tail"
            takeover_detail = {
                "missing_game_indices": missing,
                "heartbeat_ages_seconds": heartbeat_ages,
            }
            emit("tail_stall_detected", **takeover_detail)
            break
        time.sleep(max(1.0, float(args.poll_seconds)))
    if summary.is_file():
        emit("collection_completed_without_takeover")
        return

    old_pids = collector_pids()
    emit(
        "takeover_triggered",
        reason=takeover_reason,
        old_pids=old_pids,
        samples_gib=samples[-int(args.consecutive_polls):],
        **takeover_detail,
    )
    subprocess.run(["tmux", "kill-session", "-t", str(args.session)], check=False)
    deadline = time.time() + 120.0
    while collector_pids() and time.time() < deadline:
        time.sleep(1.0)
    if collector_pids():
        raise RuntimeError(f"collector did not stop cleanly: {collector_pids()}")
    if summary.is_file():
        emit("collection_completed_during_takeover")
        return

    command = restart_command(int(args.restart_workers))
    run_log = OUTPUT / "run-memory-takeover.log"
    shell = (
        f"cd {shlex.quote(str(ROOT))} && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"{shlex.join(command)} 2>&1 | tee -a {shlex.quote(str(run_log))}"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", str(args.session), shell], check=True
    )
    marker = {
        "status": "restarted",
        "reason": takeover_reason,
        "old_pids": old_pids,
        "available_samples_gib": samples[-int(args.consecutive_polls):],
        "restart_workers": int(args.restart_workers),
        "inference_replicas_per_gpu": 1,
        "command": command,
        **takeover_detail,
    }
    (OUTPUT / "MEMORY_TAKEOVER.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit("collector_restarted", workers=int(args.restart_workers), session=str(args.session))


if __name__ == "__main__":
    main()
