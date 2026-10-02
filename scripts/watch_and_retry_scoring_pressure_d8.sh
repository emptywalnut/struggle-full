#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt2/users/kaile/hantao/game/struggle
PYTHON=/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python
COLLECTION="$ROOT/runs/puct_selfplay/search-scaling-iteration2-phase3u75-adaptive-d4d6-32games-gpu45-20260824"
OUT="$ROOT/runs/search_scaling/iteration2-scoring-pressure-probe-20260824"
RESULT="$OUT/pressure-step66-d8_2048-post-collection.json"
LOG="$OUT/pressure-step66-d8_2048-post-collection.log"
STATUS="$OUT/pressure-step66-d8_2048-post-collection.status"

while [[ ! -f "$COLLECTION/collection_summary.json" ]]; do
  sleep 300
done

if [[ -f "$RESULT" ]]; then
  printf 'complete\n' > "$STATUS"
  exit 0
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${SEARCH_SCALING_D8_GPU:-2}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

set +e
"$PYTHON" scripts/sweep_puct_horizon.py \
  --checkpoint runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075 \
  --game-log "$COLLECTION/workers/worker-016/game_logs/games-2417910.jsonl" \
  --position 97600018:66 \
  --config 8:2048 \
  --belief-particles 32 \
  --max-atomic-steps 3072 \
  --max-expanded-nodes 524288 \
  --output "$RESULT" \
  --num-gpus 1 >> "$LOG" 2>&1
rc=$?
set -e

if [[ $rc -eq 0 && -f "$RESULT" ]]; then
  printf 'complete\n' > "$STATUS"
  exit 0
fi
printf 'failed rc=%s\n' "$rc" > "$STATUS"
exit "$rc"
