#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

PAIR_FILE="${PAIR_FILE:-/tmp/struggle_sideheads_48h_pair.json}"
PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
STALE_SECONDS="${STALE_SECONDS:-5400}"
MIN_FREE_GB="${MIN_FREE_GB:-15}"
MIN_EVAL_SIDE_WIN_RATE="${MIN_EVAL_SIDE_WIN_RATE:-0.20}"
MIN_RANDOM_SIDE_WIN_RATE="${MIN_RANDOM_SIDE_WIN_RATE:-0.80}"
MAX_EVAL_SIDE_GAP="${MAX_EVAL_SIDE_GAP:-0.60}"
MAX_EVAL_NUKE_RATE="${MAX_EVAL_NUKE_RATE:-0.45}"

if [[ ! -f "$PAIR_FILE" ]]; then
  echo "missing pair metadata: $PAIR_FILE" >&2
  exit 2
fi

MONITOR_LOG=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["monitor_log"])' "$PAIR_FILE")
mkdir -p "$(dirname "$MONITOR_LOG")"

while true; do
  "$PYTHON" - \
    "$PAIR_FILE" \
    "$STALE_SECONDS" \
    "$MIN_FREE_GB" \
    "$MIN_EVAL_SIDE_WIN_RATE" \
    "$MIN_RANDOM_SIDE_WIN_RATE" \
    "$MAX_EVAL_SIDE_GAP" \
    "$MAX_EVAL_NUKE_RATE" <<'PY' | tee -a "$MONITOR_LOG"
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

pair_file = Path(sys.argv[1])
stale_seconds = float(sys.argv[2])
min_free_gb = float(sys.argv[3])
min_eval_side_win_rate = float(sys.argv[4])
min_random_side_win_rate = float(sys.argv[5])
max_eval_side_gap = float(sys.argv[6])
max_eval_nuke_rate = float(sys.argv[7])
pair = json.loads(pair_file.read_text())
payload = {"time": time.time(), "runs": {}, "alerts": []}

for item in pair["runs"]:
    run_dir = Path(item["run_dir"])
    train_log = run_dir / "ppo_train.log"
    rows = []
    for path in sorted((run_dir / "metrics").glob("terminal-*.jsonl")):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            pass
        else:
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A writer may be appending the final JSONL record.
                    continue
    window = rows[-100:]
    reasons = Counter(str(row.get("terminal_reason") or row.get("reason") or "unknown") for row in window)
    nuke = sum(reasons[key] for key in ("nuclear_war", "thermonuclear war", "Cuban Missile Crisis"))
    mechanical = sum(
        1
        for row in window
        if row.get("bridge_error")
        or row.get("no_legal_actions")
        or str(row.get("terminal_reason")) in {"bridge_error", "no_legal_actions"}
    )
    latest_iter = None
    latest_eval = None
    log_tail = ""
    if train_log.exists():
        log_text = train_log.read_text(errors="replace")
        log_tail = log_text[-50000:]
        for line in reversed(log_text.splitlines()):
            if line.startswith("{'iter'"):
                try:
                    cleaned = re.sub(r"np\.float(?:32|64)?\(([^()]*)\)", r"\1", line)
                    parsed = ast.literal_eval(cleaned)
                    latest_iter = {
                        key: parsed.get(key)
                        for key in (
                            "iter",
                            "episodes_seen",
                            "train_steps_seen",
                            "num_env_steps_sampled_lifetime",
                            "rollout/us_win_rate",
                            "rollout/ussr_win_rate",
                            "rollout/nuke_terminal_rate",
                            "rollout/bridge_error_rate",
                            "rollout/no_legal_actions_rate",
                            "rollout/timeout_rate",
                        )
                    }
                except (SyntaxError, ValueError):
                    latest_iter = line
                break
        for line in reversed(log_text.splitlines()):
            if "'event': 'checkpoint_eval_done'" not in line:
                continue
            try:
                cleaned = re.sub(r"np\.float(?:32|64)?\(([^()]*)\)", r"\1", line)
                parsed = ast.literal_eval(cleaned)
                opponents = {}
                for key, value in parsed.items():
                    match = re.fullmatch(r"eval_vs_(.+)/win_rate", key)
                    if not match:
                        continue
                    opponent = match.group(1)
                    prefix = f"eval_vs_{opponent}/"
                    opponents[opponent] = {
                        metric: parsed.get(prefix + metric)
                        for metric in (
                            "games",
                            "games_per_side",
                            "win_rate",
                            "us_side_win_rate",
                            "ussr_side_win_rate",
                            "nuke_terminal_rate",
                            "scoring_card_held_terminal_rate",
                            "timeout_rate",
                        )
                    }
                latest_eval = {
                    "checkpoint": parsed.get("checkpoint"),
                    "train_steps_seen": parsed.get("train_steps_seen"),
                    "episodes_seen": parsed.get("episodes_seen"),
                    "elo_us": parsed.get("eval/elo_us"),
                    "elo_ussr": parsed.get("eval/elo_ussr"),
                    "composite_score": parsed.get("eval/composite_score"),
                    "incumbent_score": parsed.get("best_checkpoint/score"),
                    "best_anchor_passed": parsed.get("best_anchor/passed"),
                    "best_anchor_updated": parsed.get("best_anchor/updated"),
                    "best_anchor_rollback_applied": parsed.get("best_anchor/rollback_applied"),
                    "opponents": opponents,
                }
            except (SyntaxError, ValueError):
                latest_eval = {"parse_error": True}
            break
    process = subprocess.run(
        ["pgrep", "-af", "struggle_ai.train_rllib"],
        check=False,
        capture_output=True,
        text=True,
    )
    alive = str(run_dir) in process.stdout
    artifact_mtimes = [path.stat().st_mtime for path in (train_log,) if path.exists()]
    artifact_mtimes.extend(path.stat().st_mtime for path in (run_dir / "metrics").glob("terminal-*.jsonl"))
    artifact_age = time.time() - max(artifact_mtimes) if artifact_mtimes else None
    has_oom = "CUDA out of memory" in log_tail or "OutOfMemoryError" in log_tail
    has_traceback = "Traceback (most recent call last)" in log_tail
    summary = {
        "mode": item["mode"],
        "session": item["session"],
        "run_dir": str(run_dir),
        "alive": alive,
        "games": len(rows),
        "last100_nuke_rate": nuke / len(window) if window else 0.0,
        "last100_mechanical_rate": mechanical / len(window) if window else 0.0,
        "last100_reasons": dict(reasons),
        "latest_iter": latest_iter,
        "latest_eval": latest_eval,
        "artifact_age_seconds": artifact_age,
        "has_oom": has_oom,
        "has_traceback": has_traceback,
    }
    payload["runs"][item["mode"]] = summary
    if not alive and not (run_dir / "exit_status.txt").exists():
        payload["alerts"].append(f"{item['mode']}:process_dead")
    if has_oom:
        payload["alerts"].append(f"{item['mode']}:cuda_oom")
    if has_traceback:
        payload["alerts"].append(f"{item['mode']}:traceback")
    if artifact_age is not None and artifact_age > stale_seconds and alive:
        payload["alerts"].append(f"{item['mode']}:stalled")
    if len(window) >= 50 and mechanical > 0:
        payload["alerts"].append(f"{item['mode']}:mechanical_terminal")
    if latest_eval and "opponents" in latest_eval:
        anchor_passed = latest_eval.get("best_anchor_passed")
        if anchor_passed is False and latest_eval.get("best_anchor_rollback_applied") != 1:
            payload["alerts"].append(f"{item['mode']}:anchor_reject_without_rollback")
        if anchor_passed is True and latest_eval.get("best_anchor_updated") != 1:
            payload["alerts"].append(f"{item['mode']}:anchor_pass_without_update")
        for opponent, result in latest_eval["opponents"].items():
            games_per_side = result.get("games_per_side") or 0
            us_rate = result.get("us_side_win_rate")
            ussr_rate = result.get("ussr_side_win_rate")
            nuke_rate = result.get("nuke_terminal_rate")
            if games_per_side < 10 or us_rate is None or ussr_rate is None:
                continue
            rejected_anchor_was_rolled_back = (
                opponent == "best_anchor"
                and anchor_passed is False
                and latest_eval.get("best_anchor_rollback_applied") == 1
            )
            if rejected_anchor_was_rolled_back:
                continue
            side_floor = min_random_side_win_rate if opponent == "random" else min_eval_side_win_rate
            if min(us_rate, ussr_rate) < side_floor:
                payload["alerts"].append(f"{item['mode']}:eval_{opponent}_side_collapse")
            if abs(us_rate - ussr_rate) > max_eval_side_gap:
                payload["alerts"].append(f"{item['mode']}:eval_{opponent}_side_imbalance")
            if nuke_rate is not None and nuke_rate > max_eval_nuke_rate:
                payload["alerts"].append(f"{item['mode']}:eval_{opponent}_nuke_high")

free_gb = shutil.disk_usage("/mnt2").free / (1024 ** 3)
payload["disk_free_gb"] = free_gb
if free_gb < min_free_gb:
    payload["alerts"].append("disk_low")

print(json.dumps(payload, sort_keys=True))
PY

  active=0
  while IFS= read -r session; do
    if tmux has-session -t "$session" 2>/dev/null; then
      active=1
    fi
  done < <("$PYTHON" -c 'import json,sys; [print(x["session"]) for x in json.load(open(sys.argv[1]))["runs"]]' "$PAIR_FILE")
  if [[ "$active" == "0" ]]; then
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done
