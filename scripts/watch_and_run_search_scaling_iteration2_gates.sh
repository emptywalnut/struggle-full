#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt2/users/kaile/hantao/game/struggle
PYTHON=/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python
CAMPAIGN="$ROOT/runs/search_distill/search-scaling-iteration2-phase3u75-u25-gpu4-20260824"
LOG="$CAMPAIGN/gate_orchestrator.log"
READY="$CAMPAIGN/checkpoints/updates-000025/algorithm_state.pkl"
PRESSURE_STATUS="$ROOT/runs/search_scaling/iteration2-scoring-pressure-probe-20260824/pressure-step66-d8_2048-post-collection.status"
mkdir -p "$CAMPAIGN"

while [[ ! -f "$READY" ]]; do
  printf '{"event":"waiting_for_iteration2_checkpoints","time":%s}\n' "$(date +%s)" >> "$LOG"
  sleep 300
done

while [[ ! -f "$PRESSURE_STATUS" ]]; do
  printf '{"event":"waiting_for_pressure_d8_adjudication","time":%s}\n' "$(date +%s)" >> "$LOG"
  sleep 300
done

cd "$ROOT"
"$PYTHON" scripts/run_search_scaling_iteration2_promotion_gates.py \
  --poll-seconds 30 --seed-pairs 10 --deeper-seed-pairs 4 --workers 20 >> "$LOG" 2>&1
"$PYTHON" scripts/run_search_scaling_iteration2_anomaly_gate.py \
  --poll-seconds 30 >> "$LOG" 2>&1
"$PYTHON" scripts/run_search_scaling_iteration2_cross_mode_gates.py \
  --poll-seconds 30 --seed-pairs 10 --deeper-seed-pairs 4 \
  --confirm-seed-pairs 25 --confirm-deeper-seed-pairs 8 \
  --confirm-d6-seed-pairs 4 >> "$LOG" 2>&1
