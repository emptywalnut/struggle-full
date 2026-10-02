#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

GPU="${GPU:-1}"
GPU_LABEL="${GPU//,/}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP=$(date +%Y%m%d-%H%M%S)
BEST_CKPT="${BEST_CKPT:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-noforcesetup-coolshape-mix50202010-gpu01-wide32-hourlywandb-20260719-123716/ppo_checkpoints/checkpoints/steps-0000161922-episodes-00000788}"
RUN_DIR="${RUN_DIR:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-usfocus-from-best00788-gpu${GPU_LABEL}-${STAMP}}"
PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"

NUM_GPUS="${NUM_GPUS:-1}"
NUM_GPUS_PER_ENV_RUNNER="${NUM_GPUS_PER_ENV_RUNNER:-0}"
NUM_ENV_RUNNERS="${NUM_ENV_RUNNERS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1360}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-80}"
ROLLOUT_FRAGMENT_LENGTH="${ROLLOUT_FRAGMENT_LENGTH:-170}"
PPO_EPISODES="${PPO_EPISODES:-30000}"
CHECKPOINT_EVAL_EVERY_EPISODES="${CHECKPOINT_EVAL_EVERY_EPISODES:-1000}"
CHECKPOINT_EVAL_EVERY_SECONDS="${CHECKPOINT_EVAL_EVERY_SECONDS:-1800}"
EVAL_GAMES="${EVAL_GAMES:-20}"
EVAL_MIN_GAMES_PER_SIDE="${EVAL_MIN_GAMES_PER_SIDE:-20}"
EVAL_HISTORY_OPPONENTS="${EVAL_HISTORY_OPPONENTS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-260000}"

LR="${LR:-0.00001}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
KL_COEFF="${KL_COEFF:-0.8}"
KL_TARGET="${KL_TARGET:-0.004}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.008}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-1.20}"
PERSISTENT_RANDOM_PROB="${PERSISTENT_RANDOM_PROB:-0.02}"
HEURISTIC_OVERRIDE_PROB="${HEURISTIC_OVERRIDE_PROB:-0.0}"

WANDB_PROJECT="${WANDB_PROJECT:-twilight-struggle-ppo}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-unified-usfocus-from-best00788-gpu${GPU_LABEL}-${STAMP}}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -d "$BEST_CKPT" ]]; then
  echo "missing best checkpoint: $BEST_CKPT" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"/{ppo_checkpoints,game_logs,metrics}
{
  echo "run_dir=$RUN_DIR"
  echo "best_ckpt=$BEST_CKPT"
  echo "restore_from=$BEST_CKPT"
  echo "focus_side=us"
  echo "fixed_ussr_opponent=$BEST_CKPT"
  echo "gpu=$GPU"
  echo "num_gpus=$NUM_GPUS"
  echo "num_gpus_per_env_runner=$NUM_GPUS_PER_ENV_RUNNER"
  echo "num_env_runners=$NUM_ENV_RUNNERS"
  echo "train_batch_size=$TRAIN_BATCH_SIZE"
  echo "minibatch_size=$MINIBATCH_SIZE"
  echo "rollout_fragment_length=$ROLLOUT_FRAGMENT_LENGTH"
  echo "ppo_episodes=$PPO_EPISODES"
  echo "checkpoint_eval_every_episodes=$CHECKPOINT_EVAL_EVERY_EPISODES"
  echo "checkpoint_eval_every_seconds=$CHECKPOINT_EVAL_EVERY_SECONDS"
  echo "eval_games=$EVAL_GAMES"
  echo "eval_min_games_per_side=$EVAL_MIN_GAMES_PER_SIDE"
  echo "wandb_project=$WANDB_PROJECT"
  echo "wandb_entity=$WANDB_ENTITY"
  echo "wandb_run_name=$WANDB_RUN_NAME"
  echo "wandb_mode=$WANDB_MODE"
  echo "force_setup_heuristic=0"
  echo "history_length=1024"
  echo "graph_neighbor_hops=2"
} | tee "$RUN_DIR/launch_config.txt"

echo "$RUN_DIR" > /tmp/struggle_unified_usfocus_latest.txt

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

timeout "$TIMEOUT_SECONDS" "$PYTHON" -m struggle_ai.train_rllib \
  --multi-agent \
  --policy-sharing unified \
  --unified-train-focus-side us \
  --focus-opponent-policy-from "$BEST_CKPT" \
  --no-adaptive-us-focus \
  --model-arch transformer_history \
  --hidden 256 \
  --history-layers 2 \
  --card-history-layers 2 \
  --history-attention-heads 4 \
  --history-dropout 0.05 \
  --graph-layers 2 \
  --graph-neighbor-hops 2 \
  --policy-temperature "$POLICY_TEMPERATURE" \
  --lr "$LR" \
  --num-epochs "$NUM_EPOCHS" \
  --kl-coeff "$KL_COEFF" \
  --kl-target "$KL_TARGET" \
  --entropy-coeff "$ENTROPY_COEFF" \
  --restore-from "$BEST_CKPT" \
  --partial-warmstart \
  --load-report-path "$RUN_DIR/ppo_load_report.json" \
  --eval-initial-from "$BEST_CKPT" \
  --benchmark-label start_best \
  --benchmark-us-policy-from "$BEST_CKPT" \
  --benchmark-ussr-policy-from "$BEST_CKPT" \
  --stop-episodes "$PPO_EPISODES" \
  --num-env-runners "$NUM_ENV_RUNNERS" \
  --num-gpus "$NUM_GPUS" \
  --num-gpus-per-env-runner "$NUM_GPUS_PER_ENV_RUNNER" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --minibatch-size "$MINIBATCH_SIZE" \
  --rollout-fragment-length "$ROLLOUT_FRAGMENT_LENGTH" \
  --batch-mode complete_episodes \
  --sample-timeout-s 1800 \
  --checkpoint-dir "$RUN_DIR/ppo_checkpoints" \
  --checkpoint-eval-every-steps 0 \
  --checkpoint-eval-every-episodes "$CHECKPOINT_EVAL_EVERY_EPISODES" \
  --checkpoint-eval-every-seconds "$CHECKPOINT_EVAL_EVERY_SECONDS" \
  --eval-games "$EVAL_GAMES" \
  --eval-min-games-per-side "$EVAL_MIN_GAMES_PER_SIDE" \
  --eval-history-opponents "$EVAL_HISTORY_OPPONENTS" \
  --eval-max-episode-steps 1200 \
  --eval-progress-every-games 10 \
  --elo-path "$RUN_DIR/elo_unified.json" \
  --elo-k-factor 32 \
  --best-checkpoint-manifest "$RUN_DIR/best_checkpoint.json" \
  --best-score-side-weight 0.25 \
  --best-score-balance-penalty 0.25 \
  --best-score-nuke-penalty 100 \
  --best-score-scoring-card-held-penalty 100 \
  --best-score-random-side-floor 0.80 \
  --best-score-random-side-penalty 200 \
  --best-score-benchmark-side-floor 0.50 \
  --best-score-benchmark-side-penalty 250 \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --wandb-mode "$WANDB_MODE" \
  --log-games-dir "$RUN_DIR/game_logs" \
  --log-games-every 20 \
  --metrics-dir "$RUN_DIR/metrics" \
  --terminal-reward-scale 2.0 \
  --reward-shaping-scale 0.9 \
  --reward-shaping-final-scale 0.25 \
  --reward-shaping-phaseout-start-episodes 5000 \
  --reward-shaping-phaseout-end-episodes 100000 \
  --turn-vp-reward-scale 0.01 \
  --nuke-death-penalty 1.0 \
  --scoring-card-held-penalty 1.0 \
  --high-stability-coup-penalty 0.03 \
  --low-stability-warzone-coup-reward 0.03 \
  --headline-opponent-event-penalty 0.03 \
  --defcon-risk-pick-penalty 0.05 \
  --defcon-risk-commit-penalty 0.15 \
  --empty-country-influence-reward 0.01 \
  --control-battleground-reward 0.02 \
  --control-non-battleground-reward 0.005 \
  --max-episode-step-penalty 0.25 \
  --defcon-suicide-mode hard_filter \
  --max-episode-steps 1200 \
  --persistent-random-prob "$PERSISTENT_RANDOM_PROB" \
  --heuristic-override-prob "$HEURISTIC_OVERRIDE_PROB" \
  --scripted-side-prob 0.0 \
  --no-force-setup-heuristic \
  2>&1 | tee "$RUN_DIR/ppo_train.log"
