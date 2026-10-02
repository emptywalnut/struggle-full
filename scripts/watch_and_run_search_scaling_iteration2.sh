#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt2/users/kaile/hantao/game/struggle
PYTHON=/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python
LOG="$ROOT/runs/search_distill/search-scaling-iteration2-phase3u75-u25-gpu4-20260824/orchestrator.log"
mkdir -p "$(dirname "$LOG")"

while true; do
  gpu_state=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '$1 == 4 || $1 == 5 {gsub(/ /, "", $2); gsub(/ /, "", $3); if ($2 > 5000 || $3 > 30) busy=1} END {print busy+0}')
  if [[ "$gpu_state" == "0" ]]; then
    break
  fi
  printf '{"event":"waiting_for_gpu45","time":%s}\n' "$(date +%s)" >> "$LOG"
  sleep 300
done

cd "$ROOT"
exec "$PYTHON" scripts/run_search_scaling_iteration2.py >> "$LOG" 2>&1
