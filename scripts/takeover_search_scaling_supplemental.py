#!/usr/bin/env python3
"""Scale the supplemental collector to one 32-worker wave after the primary finishes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

try:
    from scripts.continue_search_scaling_amortization import (
        PRIMARY,
        ROOT,
        SUPPLEMENTAL,
        collector_command,
    )
except ModuleNotFoundError:
    from continue_search_scaling_amortization import (  # type: ignore[no-redef]
        PRIMARY,
        ROOT,
        SUPPLEMENTAL,
        collector_command,
    )


def emit(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True), flush=True)


def tmux_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def stop_tmux(session: str) -> None:
    if tmux_exists(session):
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)


def start_tmux(session: str, shell_command: str) -> None:
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell_command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--old-session", default="ts-scale-supp8")
    parser.add_argument("--old-memory-session", default="ts-scale-supp8-memory")
    parser.add_argument("--new-session", default="ts-scale-supp")
    parser.add_argument("--new-memory-session", default="ts-scale-supp-memory")
    args = parser.parse_args()

    primary_summary = PRIMARY / "collection_summary.json"
    supplemental_summary = SUPPLEMENTAL / "collection_summary.json"
    while not primary_summary.is_file():
        emit("waiting_for_primary", path=str(primary_summary.relative_to(ROOT)))
        time.sleep(max(1.0, float(args.poll_seconds)))

    if supplemental_summary.is_file():
        emit("supplemental_already_complete", path=str(supplemental_summary.relative_to(ROOT)))
        return

    # If the memory guard quarantined the optional shard, the campaign driver owns
    # the clean serial restart. Do not recreate the target behind its back.
    if not (SUPPLEMENTAL / "collection_identity.json").is_file():
        emit("supplemental_identity_absent_campaign_will_restart")
        return

    stop_tmux(str(args.old_memory_session))
    stop_tmux(str(args.old_session))
    time.sleep(5.0)
    stop_tmux(str(args.new_memory_session))
    stop_tmux(str(args.new_session))

    command = collector_command(
        SUPPLEMENTAL,
        games=32,
        workers=int(args.workers),
    )
    collector_shell = (
        f"cd {shlex.quote(str(ROOT))} && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"{shlex.join(command)} 2>&1 | tee -a "
        f"{shlex.quote(str(SUPPLEMENTAL / 'run-takeover-w32.log'))}"
    )
    start_tmux(str(args.new_session), collector_shell)
    time.sleep(3.0)

    guard_command = [
        sys.executable,
        str(ROOT / "scripts/watch_search_scaling_supplemental_memory.py"),
        "--output-dir", str(SUPPLEMENTAL),
        "--available-threshold-gib", "300",
        "--consecutive-polls", "3",
        "--poll-seconds", "60",
        "--session", str(args.new_session),
    ]
    guard_shell = (
        f"cd {shlex.quote(str(ROOT))} && "
        f"{shlex.join(guard_command)} 2>&1 | tee -a "
        f"{shlex.quote(str(SUPPLEMENTAL / 'memory-watch.log'))}"
    )
    start_tmux(str(args.new_memory_session), guard_shell)
    emit(
        "supplemental_takeover_started",
        workers=int(args.workers),
        completed_games=len(list((SUPPLEMENTAL / "completed_games").glob("game-*.json"))),
        output=str(SUPPLEMENTAL.relative_to(ROOT)),
    )


if __name__ == "__main__":
    main()
