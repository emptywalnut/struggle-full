#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable paired PUCT evaluation shards across selected GPUs.",
        epilog="Pass eval_puct_agent_pair.py arguments after a literal --.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--pairs-per-shard", type=int, default=1)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--capture-search-replay", action="store_true")
    parser.add_argument("pair_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.pair_args and args.pair_args[0] == "--":
        args.pair_args = args.pair_args[1:]
    forbidden = {"--output-dir", "--seed-pairs", "--seed-base", "--search-replay-dir"}
    if forbidden & set(args.pair_args):
        raise SystemExit(f"launcher owns these arguments: {sorted(forbidden)}")
    return args


def main() -> None:
    args = parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one device")
    args.output_root.mkdir(parents=True, exist_ok=True)
    queue = list(range(args.shards))
    running: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []

    def terminate_running() -> None:
        for entry in running:
            process = entry["process"]
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for entry in running:
            process = entry["process"]
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            if not entry["log_handle"].closed:
                entry["log_handle"].close()

    atexit.register(terminate_running)
    def handle_signal(signum: int, _frame: Any) -> None:
        terminate_running()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    capacity = len(gpus) * max(1, args.workers_per_gpu)
    while queue or running:
        while queue and len(running) < capacity:
            shard = queue.pop(0)
            gpu = gpus[shard % len(gpus)]
            output_dir = args.output_root / f"shard-{shard:03d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "scripts/eval_puct_agent_pair.py",
                *args.pair_args,
                "--output-dir",
                str(output_dir),
                "--seed-pairs",
                str(args.pairs_per_shard),
                "--seed-base",
                str(args.seed_base + shard * args.pairs_per_shard),
            ]
            if args.capture_search_replay:
                command.extend(
                    [
                        "--search-replay-dir",
                        str(output_dir / "search_replay"),
                        "--search-replay-only-flips",
                    ]
                )
            log_path = output_dir / "stdout.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            running.append(
                {
                    "shard": shard,
                    "gpu": gpu,
                    "process": process,
                    "log_handle": log_handle,
                    "log": str(log_path.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "started": time.time(),
                }
            )
            print({"event": "puct_shard_started", "shard": shard, "gpu": gpu, "pid": process.pid}, flush=True)
        time.sleep(1.0)
        survivors = []
        for entry in running:
            return_code = entry["process"].poll()
            if return_code is None:
                survivors.append(entry)
                continue
            entry["log_handle"].close()
            result = {
                key: value
                for key, value in entry.items()
                if key not in {"process", "log_handle"}
            }
            result.update(return_code=return_code, elapsed_seconds=time.time() - entry["started"])
            completed.append(result)
            print({"event": "puct_shard_finished", **result}, flush=True)
        running = survivors
        manifest = {
            "shards": args.shards,
            "pairs_per_shard": args.pairs_per_shard,
            "seed_base": args.seed_base,
            "gpus": gpus,
            "workers_per_gpu": args.workers_per_gpu,
            "capture_search_replay": args.capture_search_replay,
            "pair_args": args.pair_args,
            "completed": completed,
            "running": [
                {key: value for key, value in entry.items() if key not in {"process", "log_handle"}}
                for entry in running
            ],
        }
        (args.output_root / "shards.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    failures = [entry for entry in completed if entry["return_code"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} PUCT evaluation shards failed; inspect shards.json/stdout.log")


if __name__ == "__main__":
    main()
