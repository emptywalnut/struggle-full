#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
BALANCED_SESSION="${BALANCED_SESSION:-struggle_sideheads_balanced_gpu0}"
USFOCUS_SESSION="${USFOCUS_SESSION:-struggle_sideheads_usfocus_gpu1}"
MONITOR_SESSION="${MONITOR_SESSION:-struggle_sideheads_48h_monitor}"
PAIR_FILE="${PAIR_FILE:-/tmp/struggle_sideheads_48h_pair.json}"

BALANCED_RUN="${BALANCED_RUN:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/sideheads-balanced-gpu0-48h-${STAMP}}"
USFOCUS_RUN="${USFOCUS_RUN:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/sideheads-usfocus-gpu1-48h-${STAMP}}"
MONITOR_LOG="${MONITOR_LOG:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/sideheads-pair-48h-${STAMP}.watchdog.jsonl}"

for session in "$BALANCED_SESSION" "$USFOCUS_SESSION" "$MONITOR_SESSION"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 2
  fi
done

python - "$PAIR_FILE" "$STAMP" "$BALANCED_SESSION" "$BALANCED_RUN" "$USFOCUS_SESSION" "$USFOCUS_RUN" "$MONITOR_LOG" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "stamp": sys.argv[2],
    "runs": [
        {"mode": "balanced", "session": sys.argv[3], "run_dir": sys.argv[4], "gpu": 0},
        {"mode": "usfocus", "session": sys.argv[5], "run_dir": sys.argv[6], "gpu": 1},
    ],
    "monitor_log": sys.argv[7],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

tmux new-session -d -s "$BALANCED_SESSION" \
  "cd /mnt2/users/kaile/hantao/game/struggle && MODE=balanced GPU=0 STAMP='$STAMP' RUN_DIR='$BALANCED_RUN' scripts/launch_side_specific_heads_48h.sh"
tmux new-session -d -s "$USFOCUS_SESSION" \
  "cd /mnt2/users/kaile/hantao/game/struggle && MODE=usfocus GPU=1 STAMP='$STAMP' RUN_DIR='$USFOCUS_RUN' scripts/launch_side_specific_heads_48h.sh"
tmux new-session -d -s "$MONITOR_SESSION" \
  "cd /mnt2/users/kaile/hantao/game/struggle && PAIR_FILE='$PAIR_FILE' scripts/monitor_side_specific_heads_48h.sh"

echo "pair_file=$PAIR_FILE"
echo "balanced_session=$BALANCED_SESSION"
echo "balanced_run=$BALANCED_RUN"
echo "usfocus_session=$USFOCUS_SESSION"
echo "usfocus_run=$USFOCUS_RUN"
echo "monitor_session=$MONITOR_SESSION"
echo "monitor_log=$MONITOR_LOG"
