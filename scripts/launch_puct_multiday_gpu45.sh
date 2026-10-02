#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-$PWD/runs/puct_campaign/multiday-d2-ppo-gpu45-${STAMP}}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$PWD/runs/puct_selfplay/cycle0-student-balanced2games-100updates-gpu4-20260812/checkpoints/distilled}"
DURATION_HOURS="${DURATION_HOURS:-72}"

mkdir -p "$RUN_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set +e
"$PYTHON" scripts/run_puct_multiday_campaign.py \
  --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --output-dir "$RUN_DIR" \
  --duration-hours "$DURATION_HOURS" \
  --games-per-cycle 128 \
  --search-fraction 0.75 \
  --search-workers 32 \
  --gpus 4,5 \
  --learner-gpu 4 \
  --inference-replicas-per-gpu 2 \
  --distill-updates 200 \
  --ppo-env-runners 16 \
  --eval-seed-pairs 10 \
  --large-eval-every-cycles 4 \
  --large-eval-seed-pairs 30 \
  --d4-states 8 \
  --d8-states 1 \
  --wandb-project twilight-struggle-puct-campaign \
  --wandb-run-name "multiday-d2-ppo-gpu45-${STAMP}" \
  --wandb-mode online \
  2>&1 | tee -a "$RUN_DIR/campaign.log"
status=${PIPESTATUS[0]}
set -e
echo "$status" > "$RUN_DIR/exit_status.txt"
exit "$status"
