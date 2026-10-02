#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"
BALANCED_RUN="${BALANCED_RUN:-runs/ppo/sideheads-balanced-rollbackgate-gpu0-38h-20260731-010522}"
USFOCUS_RUN="${USFOCUS_RUN:-runs/ppo/sideheads-usfocus-rollbackgate-gpu1-38h-20260731-010522}"
USFOCUS_ORIGINAL="${USFOCUS_ORIGINAL:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/sideheads-usfocus-rollback-kl-gpu1-43h-20260730-195630/ppo_checkpoints/checkpoints/steps-0000048399-episodes-00000264}"
AUDIT_DIR="${AUDIT_DIR:-runs/ppo/sideheads-final-audit-20260801}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$AUDIT_DIR"

while true; do
  balanced_done=0
  usfocus_done=0
  [[ -f "$BALANCED_RUN/exit_status.txt" ]] && balanced_done=1
  [[ -f "$USFOCUS_RUN/exit_status.txt" ]] && usfocus_done=1
  if ! tmux has-session -t struggle_sideheads_balanced_gpu0 2>/dev/null; then
    balanced_done=1
  fi
  if ! tmux has-session -t struggle_sideheads_usfocus_gpu1 2>/dev/null; then
    usfocus_done=1
  fi
  if [[ "$balanced_done" -eq 1 && "$usfocus_done" -eq 1 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

balanced_status="wrapper_interrupted_before_status"
usfocus_status="wrapper_interrupted_before_status"
[[ -f "$BALANCED_RUN/exit_status.txt" ]] && balanced_status=$(<"$BALANCED_RUN/exit_status.txt")
[[ -f "$USFOCUS_RUN/exit_status.txt" ]] && usfocus_status=$(<"$USFOCUS_RUN/exit_status.txt")
printf 'balanced_exit_status=%s\nusfocus_exit_status=%s\n' "$balanced_status" "$usfocus_status" \
  | tee "$AUDIT_DIR/training_exit_status.txt"

balanced_best=$(jq -r .checkpoint "$BALANCED_RUN/best_checkpoint.json")
usfocus_best=$(jq -r .checkpoint "$USFOCUS_RUN/best_checkpoint.json")
printf 'balanced_best=%s\nusfocus_best=%s\nusfocus_original=%s\n' \
  "$balanced_best" "$usfocus_best" "$USFOCUS_ORIGINAL" | tee "$AUDIT_DIR/checkpoints.txt"

common_args=(
  --eval-only
  --multi-agent
  --policy-sharing unified
  --side-specific-heads
  --model-arch transformer_history
  --hidden 256
  --history-layers 2
  --card-history-layers 2
  --history-attention-heads 4
  --history-dropout 0.05
  --graph-layers 2
  --graph-neighbor-hops 2
  --policy-temperature 1.20
  --num-env-runners 0
  --num-gpus 1
  --eval-games 30
  --eval-min-games-per-side 30
  --eval-max-episode-steps 1200
  --eval-progress-every-games 10
  --terminal-reward-scale 2.0
  --reward-shaping-scale 0.25
  --turn-vp-reward-scale 0.01
  --nuke-death-penalty 1.0
  --scoring-card-held-penalty 1.0
  --high-stability-coup-penalty 0.03
  --low-stability-warzone-coup-reward 0.03
  --headline-opponent-event-penalty 0.03
  --defcon-risk-pick-penalty 0.05
  --defcon-risk-commit-penalty 0.15
  --empty-country-influence-reward 0.01
  --control-battleground-reward 0.02
  --control-non-battleground-reward 0.005
  --defcon-suicide-mode hard_filter
  --max-episode-steps 1200
  --no-force-setup-heuristic
  --wandb-mode disabled
)

run_eval() {
  local gpu=$1
  local name=$2
  local candidate=$3
  local incumbent=$4
  local output_dir="$AUDIT_DIR/$name"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m struggle_ai.train_rllib \
    "${common_args[@]}" \
    --restore-from "$candidate" \
    --eval-initial-from "$incumbent" \
    --checkpoint-dir "$output_dir/checkpoints" \
    --metrics-dir "$output_dir/metrics" \
    --log-games-dir "$output_dir/game_logs" \
    --log-games-every 1 \
    --elo-path "$output_dir/elo.json" \
    --best-checkpoint-manifest "$output_dir/eval_manifest.json" \
    >"$output_dir/eval.log" 2>&1
}

run_eval 0 usfocus_vs_original "$usfocus_best" "$USFOCUS_ORIGINAL" &
pid_usfocus=$!
run_eval 1 balanced_vs_usfocus "$balanced_best" "$usfocus_best" &
pid_balanced=$!

set +e
wait "$pid_usfocus"
status_usfocus=$?
wait "$pid_balanced"
status_balanced=$?
set -e

printf 'usfocus_vs_original_status=%s\nbalanced_vs_usfocus_status=%s\n' \
  "$status_usfocus" "$status_balanced" | tee "$AUDIT_DIR/eval_exit_status.txt"

for name in usfocus_vs_original balanced_vs_usfocus; do
  rg "'event': 'eval_only_done'" "$AUDIT_DIR/$name/eval.log" | tail -n 1 \
    >"$AUDIT_DIR/$name/result.txt" || true
done

if [[ "$status_usfocus" -ne 0 || "$status_balanced" -ne 0 ]]; then
  exit 1
fi
