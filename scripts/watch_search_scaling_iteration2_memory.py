#!/usr/bin/env python3
"""Memory fail-safe for the iteration-2 adaptive complete-game collection."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.watch_search_scaling_collection_memory as watcher


PARENT = ROOT / "runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075"
OUTPUT = ROOT / "runs/puct_selfplay/search-scaling-iteration2-phase3u75-adaptive-d4d6-32games-gpu45-20260824"
CAMPAIGN = ROOT / "runs/search_distill/search-scaling-iteration2-phase3u75-u25-gpu4-20260824"

watcher.PARENT = PARENT
watcher.OUTPUT = OUTPUT


def restart_command(workers: int) -> list[str]:
    return [
        sys.executable,
        "scripts/collect_puct_selfplay_parallel.py",
        "--checkpoint", str(PARENT),
        "--output-dir", str(OUTPUT),
        "--games", "32",
        "--seed-base", "97600000",
        "--gpus", "4,5",
        "--workers", str(workers),
        "--inference-replicas-per-gpu", "2",
        "--max-inference-batch-size", "64",
        "--max-inference-delay-ms", "4",
        "--inference-timeout-seconds", "300",
        "--initial-simulations", "128",
        "--medium-simulations", "128",
        "--max-simulations", "128",
        "--meaningful-depth", "4",
        "--belief-particles", "32",
        "--max-atomic-steps", "1536",
        "--max-expanded-nodes", "131072",
        "--max-search-worker-rss-mib", "20480",
        "--max-search-seconds", "300",
        "--child-initial", "4",
        "--child-max", "24",
        "--search-temperature", "1.0",
        "--behavior-temperature", "0.35",
        "--force-search",
        "--low-confidence-sample-rate", "0.0",
        "--max-episode-steps", "1200",
        "--shard-size", "128",
        "--staging-shard-size", "32",
        "--heap-trim-every-decisions", "8",
        "--heartbeat-every-decisions", "1",
        "--adaptive-search",
        "--no-adaptive-on-flip",
        "--adaptive-budget-simulations", "512",
        "--adaptive-deep-depth", "6",
        "--adaptive-deep-simulations", "512",
        "--adaptive-min-rank-correlation", "0.70",
        "--adaptive-meaningful-q-gap", "0.01",
        "--adaptive-escalation-q-gap", "0.01",
    ]


watcher.restart_command = restart_command


def main() -> None:
    summary = OUTPUT / "collection_summary.json"
    # A takeover replaces the tmux collector process. Keep monitoring the
    # replacement instead of leaving the resumed shard unprotected.
    while not summary.is_file():
        watcher.main()
        if not summary.is_file():
            time.sleep(5)

    takeover = OUTPUT / "MEMORY_TAKEOVER.json"
    if not takeover.is_file():
        return
    ready = CAMPAIGN / "checkpoints/updates-000025/algorithm_state.pkl"
    if ready.is_file():
        return
    completed = subprocess.run(
        [sys.executable, "scripts/run_search_scaling_iteration2.py"],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"iteration-2 post-takeover pipeline failed: {completed.returncode}")


if __name__ == "__main__":
    main()
