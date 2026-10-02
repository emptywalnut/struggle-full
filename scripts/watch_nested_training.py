from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


LANES = {
    "ppo": ("PPO_SESSION", "PPO_RUN", "struggle_ai.train_rllib"),
    "mcts": ("MCTS_SESSION", "MCTS_RUN", "struggle_ai.train_mcts"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record health of the active PPO and MCTS lanes.")
    parser.add_argument(
        "--pointer",
        type=Path,
        default=Path("runs/current_nested_training.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/nested_training_health.jsonl"),
    )
    parser.add_argument("--interval-s", type=float, default=600.0)
    parser.add_argument("--stale-s", type=float, default=7200.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def read_pointer(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def tmux_alive(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def process_status(module: str, run_dir: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["pgrep", "-af", module],
        check=False,
        capture_output=True,
        text=True,
    )
    candidates: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        pid_text, _, command = line.partition(" ")
        if str(run_dir) not in command:
            continue
        try:
            candidates.append((int(pid_text), command))
        except ValueError:
            continue
    if not candidates:
        return {"alive": False, "pid": None, "uptime_seconds": None}

    # Prefer the Python trainer over its timeout wrapper.
    pid, command = next(
        ((pid, command) for pid, command in candidates if "python" in command and "timeout " not in command),
        candidates[-1],
    )
    uptime = subprocess.run(
        ["ps", "-o", "etimes=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "alive": True,
        "pid": pid,
        "uptime_seconds": int(uptime) if uptime.isdigit() else None,
    }


def newest_artifact_age(run_dir: Path, now: float) -> float | None:
    paths = [run_dir / "ppo_train.log", run_dir / "mcts_train.log"]
    paths.extend((run_dir / "metrics").glob("*.jsonl"))
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return now - max(mtimes) if mtimes else None


def error_flags(run_dir: Path) -> dict[str, bool]:
    log_path = run_dir / "ppo_train.log"
    if not log_path.exists():
        log_path = run_dir / "mcts_train.log"
    if not log_path.exists():
        return {"traceback": False, "cuda_oom": False, "nonfinite": False}
    with log_path.open("rb") as handle:
        handle.seek(max(0, os.fstat(handle.fileno()).st_size - 250_000))
        tail = handle.read().decode("utf-8", errors="replace").lower()
    return {
        "traceback": "traceback (most recent call last)" in tail,
        "cuda_oom": "cuda out of memory" in tail or "outofmemoryerror" in tail,
        "nonfinite": "'train/loss': nan" in tail or "'train/loss': inf" in tail,
    }


def retained_elo(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "best_checkpoint.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    details = manifest.get("score_details") or manifest.get("metrics") or {}
    us = details.get("us_elo", details.get("eval/elo_us"))
    ussr = details.get("ussr_elo", details.get("eval/elo_ussr"))
    side_floor = details.get("side_floor_elo", details.get("eval/elo_side_floor"))
    return {
        "checkpoint": manifest.get("checkpoint"),
        "us": us,
        "ussr": ussr,
        "side_floor": side_floor,
    }


def check(args: argparse.Namespace) -> dict[str, Any]:
    now = time.time()
    pointer = read_pointer(args.pointer)
    payload: dict[str, Any] = {
        "time": now,
        "pointer": str(args.pointer.resolve()),
        "lanes": {},
        "alerts": [],
        "both_lanes_48h": True,
        "checkpoint_over_1750_both_sides": False,
    }
    for lane, (session_key, run_key, module) in LANES.items():
        session = pointer[session_key]
        run_dir = Path(pointer[run_key])
        process = process_status(module, run_dir)
        artifact_age = newest_artifact_age(run_dir, now)
        flags = error_flags(run_dir)
        elo = retained_elo(run_dir)
        lane_payload = {
            "session": session,
            "run_dir": str(run_dir),
            "tmux_alive": tmux_alive(session),
            "process": process,
            "artifact_age_seconds": artifact_age,
            "errors": flags,
            "retained_elo": elo,
        }
        payload["lanes"][lane] = lane_payload
        uptime = process.get("uptime_seconds")
        lane_48h = bool(process.get("alive") and uptime is not None and uptime >= 172_800)
        payload["both_lanes_48h"] = payload["both_lanes_48h"] and lane_48h
        if not lane_payload["tmux_alive"]:
            payload["alerts"].append(f"{lane}:tmux_dead")
        if not process.get("alive"):
            payload["alerts"].append(f"{lane}:process_dead")
        if artifact_age is None or artifact_age > args.stale_s:
            payload["alerts"].append(f"{lane}:artifacts_stale")
        for flag, active in flags.items():
            if active:
                payload["alerts"].append(f"{lane}:{flag}")
        if elo and elo.get("us") is not None and elo.get("ussr") is not None:
            if float(elo["us"]) > 1750.0 and float(elo["ussr"]) > 1750.0:
                payload["checkpoint_over_1750_both_sides"] = True
    return payload


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        payload = check(args)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True), flush=True)
        if args.once:
            return
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
