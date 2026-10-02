#!/usr/bin/env python3
"""Abort the optional supplemental shard before it pressures the primary shard."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from scripts.watch_search_scaling_collection_memory import (
        available_memory_gib,
        should_take_over,
    )
except ModuleNotFoundError:
    from watch_search_scaling_collection_memory import available_memory_gib, should_take_over


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/puct_selfplay/search-scaling-phase3u75-selective-adaptive-d4d6-32games-w4-gpu45-20260821"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-threshold-gib", type=float, default=320.0)
    parser.add_argument("--consecutive-polls", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--session", default="ts-scale-supp")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True), flush=True)


def matching_pids(output_dir: Path = OUTPUT) -> list[int]:
    marker = str(output_dir.relative_to(ROOT))
    output = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
    ).stdout
    return sorted(
        int(line.strip().split(maxsplit=1)[0])
        for line in output.splitlines()
        if len(line.strip().split(maxsplit=1)) == 2
        and "collect_puct_selfplay_parallel.py" in line
        and marker in line
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    samples: list[float] = []
    summary = output_dir / "collection_summary.json"
    while not summary.is_file():
        available = available_memory_gib()
        samples.append(available)
        emit("memory_sample", available_gib=available, pids=matching_pids(output_dir))
        if should_take_over(samples, args.available_threshold_gib, args.consecutive_polls):
            break
        time.sleep(max(1.0, args.poll_seconds))
    if summary.is_file():
        emit("supplemental_completed")
        return
    pids = matching_pids(output_dir)
    subprocess.run(["tmux", "kill-session", "-t", args.session], check=False)
    deadline = time.time() + 120.0
    while matching_pids(output_dir) and time.time() < deadline:
        time.sleep(1.0)
    if matching_pids(output_dir):
        raise RuntimeError(f"supplemental did not stop: {matching_pids(output_dir)}")
    destination = output_dir.with_name(
        output_dir.name + "-aborted-memory-pressure-" + time.strftime("%Y%m%d-%H%M%S")
    )
    output_dir.rename(destination)
    (destination / "ABORTED.json").write_text(
        json.dumps(
            {
                "status": "aborted",
                "reason": "protect_primary_shard_memory_margin",
                "available_samples_gib": samples[-args.consecutive_polls :],
                "old_pids": pids,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    emit("supplemental_aborted", destination=str(destination.relative_to(ROOT)))


if __name__ == "__main__":
    main()
