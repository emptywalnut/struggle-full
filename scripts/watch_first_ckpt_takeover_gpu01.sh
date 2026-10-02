#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

CURRENT_RUN_DIR="${CURRENT_RUN_DIR:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-noforcesetup-coolshape-mix50202010-gpu1-20260718-111629}"
CURRENT_SESSION="${CURRENT_SESSION:-struggle_noforcesetup_gpu1}"
CURRENT_WATCH_SESSION="${CURRENT_WATCH_SESSION:-struggle_noforcesetup_gpu1_watch}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-scripts/launch_unified_noforcesetup_coolshape_lowmem_gpu1.sh}"
TARGET_EPISODES="${TARGET_EPISODES:-5000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

INITIAL_CKPT="${INITIAL_CKPT:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-antidrift-from3002-gpu2-20260714-221128/ppo_checkpoints/checkpoints/steps-0000217905-episodes-00001010}"
TAKEOVER_SESSION="${TAKEOVER_SESSION:-struggle_noforcesetup_gpu01_takeover}"
TAKEOVER_RUN_DIR="${TAKEOVER_RUN_DIR:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-noforcesetup-coolshape-mix50202010-gpu01-takeover-$(date +%Y%m%d-%H%M%S)}"
DECISION_JSON="${TAKEOVER_RUN_DIR}.handoff_decision.json"
LOG_PATH="${TAKEOVER_RUN_DIR}.watch.log"

mkdir -p "$(dirname "$LOG_PATH")"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_PATH"
}

latest_eval_decision() {
  python - "$CURRENT_RUN_DIR" "$TARGET_EPISODES" "$INITIAL_CKPT" "$DECISION_JSON" <<'PY'
import ast
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
target_episodes = int(sys.argv[2])
initial_ckpt = str(Path(sys.argv[3]))
decision_path = Path(sys.argv[4])
log_path = run_dir / "ppo_train.log"
if not log_path.exists():
    sys.exit(1)

decision = None
for line in log_path.read_text(errors="ignore").splitlines():
    if "'event': 'checkpoint_eval_done'" not in line and '"event": "checkpoint_eval_done"' not in line:
        continue
    try:
        payload = ast.literal_eval(line)
    except Exception:
        continue
    episodes = int(payload.get("episodes_seen") or 0)
    if episodes < target_episodes:
        continue
    checkpoint = str(payload.get("checkpoint") or payload.get("checkpoint/path") or "")
    vs_initial = float(payload.get("eval_vs_initial/win_rate") or 0.0)
    us_vs_initial = float(payload.get("eval_vs_initial/us_side_win_rate") or 0.0)
    ussr_vs_initial = float(payload.get("eval_vs_initial/ussr_side_win_rate") or 0.0)
    side_floor = min(us_vs_initial, ussr_vs_initial)
    selected = checkpoint if checkpoint and vs_initial >= 0.5 and side_floor >= 0.5 else initial_ckpt
    decision = {
        "episodes_seen": episodes,
        "train_steps_seen": int(payload.get("train_steps_seen") or 0),
        "checkpoint": checkpoint,
        "initial_checkpoint": initial_ckpt,
        "eval_vs_initial_win_rate": vs_initial,
        "eval_vs_initial_us_side_win_rate": us_vs_initial,
        "eval_vs_initial_ussr_side_win_rate": ussr_vs_initial,
        "eval_vs_initial_side_floor_win_rate": side_floor,
        "selected_checkpoint": selected,
        "selected_reason": "first_checkpoint_not_worse_than_initial_on_both_sides" if selected == checkpoint else "fallback_initial_checkpoint",
    }

if not decision:
    sys.exit(1)
decision_path.parent.mkdir(parents=True, exist_ok=True)
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print(json.dumps(decision, sort_keys=True))
PY
}

stop_current_training() {
  log "stopping current training session=${CURRENT_SESSION}"
  if tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; then
    tmux send-keys -t "$CURRENT_SESSION" C-c || true
    sleep 20
  fi
  pids=$(pgrep -f "train_rllib.*${CURRENT_RUN_DIR}" || true)
  if [[ -n "$pids" ]]; then
    log "killing remaining train_rllib pids: ${pids//$'\n'/ }"
    kill $pids || true
    sleep 5
  fi
  pids=$(pgrep -f "train_rllib.*${CURRENT_RUN_DIR}" || true)
  if [[ -n "$pids" ]]; then
    log "force killing remaining train_rllib pids: ${pids//$'\n'/ }"
    kill -9 $pids || true
  fi
  if tmux has-session -t "$CURRENT_WATCH_SESSION" 2>/dev/null; then
    tmux kill-session -t "$CURRENT_WATCH_SESSION" || true
  fi
}

start_gpu01_takeover() {
  local selected_checkpoint="$1"
  log "starting GPU 0,1 takeover from ${selected_checkpoint}"
  if tmux has-session -t "$TAKEOVER_SESSION" 2>/dev/null; then
    log "takeover session already exists: ${TAKEOVER_SESSION}"
    return
  fi
  tmux new-session -d -s "$TAKEOVER_SESSION" \
    "cd /mnt2/users/kaile/hantao/game/struggle && GPU=0,1 NUM_GPUS=2 NUM_ENV_RUNNERS=16 TRAIN_BATCH_SIZE=4080 MINIBATCH_SIZE=240 ROLLOUT_FRAGMENT_LENGTH=255 RESTORE_FROM='${selected_checkpoint}' US_TEACHER='${selected_checkpoint}' USSR_TEACHER='${selected_checkpoint}' ANCHORS='[\"${selected_checkpoint}\"]' RUN_DIR='${TAKEOVER_RUN_DIR}' ${LAUNCH_SCRIPT}"
  log "takeover tmux session=${TAKEOVER_SESSION}"
  log "takeover run_dir=${TAKEOVER_RUN_DIR}"
}

log "watching first checkpoint eval in ${CURRENT_RUN_DIR}"
while true; do
  if decision=$(latest_eval_decision 2>/dev/null); then
    log "handoff decision: ${decision}"
    selected=$(python - "$DECISION_JSON" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["selected_checkpoint"])
PY
)
    stop_current_training
    start_gpu01_takeover "$selected"
    log "handoff complete"
    exit 0
  fi
  if ! pgrep -f "train_rllib.*${CURRENT_RUN_DIR}" >/dev/null; then
    log "current training is not running and no usable checkpoint eval decision was found"
    exit 3
  fi
  tail_line=$(grep -F "episodes_seen" "$CURRENT_RUN_DIR/ppo_train.log" 2>/dev/null | tail -1 || true)
  if [[ -n "$tail_line" ]]; then
    log "still waiting; latest=${tail_line:0:500}"
  else
    log "still waiting; no iteration line yet"
  fi
  sleep "$POLL_SECONDS"
done
