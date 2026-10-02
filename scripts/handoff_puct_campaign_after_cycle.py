#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boundary-safe campaign upgrade after a completed cycle.")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--after-cycle", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def campaign_processes(campaign_dir: Path) -> list[int]:
    needle = str(campaign_dir)
    result = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    pids = []
    for line in result.stdout.splitlines():
        raw_pid, _, command = line.strip().partition(" ")
        if (
            "scripts/run_puct_multiday_campaign.py" in command
            and needle in command
            and int(raw_pid) != os.getpid()
        ):
            pids.append(int(raw_pid))
    return pids


def descendants(pid: int) -> list[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        child, parent = (int(value) for value in line.split())
        children.setdefault(parent, []).append(child)
    output: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        output.append(child)
        stack.extend(children.get(child, []))
    return output


def terminate_tree(pid: int) -> None:
    targets = list(reversed(descendants(pid))) + [pid]
    for target in targets:
        try:
            os.kill(target, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        alive = []
        for target in targets:
            try:
                os.kill(target, 0)
                alive.append(target)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(0.5)
    for target in alive:
        try:
            os.kill(target, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> None:
    args = parse_args()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    state_path = campaign_dir / "campaign_state.json"
    report_path = campaign_dir / "cycles" / f"cycle-{args.after_cycle:04d}" / "cycle_report.json"
    print(json.dumps({
        "event": "campaign_handoff_waiting",
        "cycle": args.after_cycle,
        "campaign_dir": str(campaign_dir),
    }), flush=True)
    while True:
        if state_path.is_file() and report_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if int(state.get("cycle", 0)) >= int(args.after_cycle):
                break
        time.sleep(max(1.0, float(args.poll_seconds)))
    print(json.dumps({"event": "cycle_boundary_reached", "cycle": args.after_cycle}), flush=True)
    pids = campaign_processes(campaign_dir)
    if len(pids) != 1:
        raise RuntimeError(f"expected exactly one campaign process, found {pids}")
    terminate_tree(pids[0])
    print(json.dumps({"event": "old_campaign_stopped", "pid": pids[0]}), flush=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/postprocess_puct_cycle.py",
            "--campaign-dir", str(campaign_dir),
            "--cycle", str(args.after_cycle),
            "--gpus", "4,5",
            "--compatibility-gpu", "5",
            "--sweep-seed-pairs", "4",
            "--eval-seed-pairs", "10",
            "--workers", "32",
        ],
        check=True,
    )
    print(json.dumps({"event": "cycle_postprocess_finished"}), flush=True)

    env = dict(os.environ)
    env.update(
        {
            "STAMP": "20260813-145350",
            "RUN_DIR": str(campaign_dir),
            "DURATION_HOURS": "72",
        }
    )
    os.execvpe("bash", ["bash", "scripts/launch_puct_multiday_gpu45.sh"], env)


if __name__ == "__main__":
    main()
