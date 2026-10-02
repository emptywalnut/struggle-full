#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

MODE="${MODE:-balanced}"
GPU="${GPU:-0}"
GPU_LABEL="${GPU//,/}"
PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-172800}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"

START_CKPT="${START_CKPT:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/sideheads-usfocus-rollback-kl-gpu1-43h-20260730-195630/ppo_checkpoints/checkpoints/steps-0000048399-episodes-00000264}"
RESTORE_CKPT="${RESTORE_CKPT:-$START_CKPT}"
INITIAL_CKPT="${INITIAL_CKPT:-$START_CKPT}"
ANCHOR_CKPT="${ANCHOR_CKPT:-$START_CKPT}"
REFERENCE_CKPT="${REFERENCE_CKPT:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/unified-history1024-noforcesetup-coolshape-mix50202010-gpu01-wide32-hourlywandb-20260719-123716/ppo_checkpoints/checkpoints/steps-0000161922-episodes-00000788}"

case "$MODE" in
  balanced)
    NUM_ENV_RUNNERS="${NUM_ENV_RUNNERS:-24}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-6120}"
    MINIBATCH_SIZE="${MINIBATCH_SIZE:-360}"
    ROLLOUT_FRAGMENT_LENGTH="${ROLLOUT_FRAGMENT_LENGTH:-255}"
    EVAL_GAMES="${EVAL_GAMES:-10}"
    EVAL_MIN_GAMES_PER_SIDE="${EVAL_MIN_GAMES_PER_SIDE:-10}"
    if [[ -z "${OPPONENT_MIX:-}" ]]; then
      OPPONENT_MIX='{"current_self":0.50,"teacher":0.20,"anchor":0.20,"random_legal":0.10,"heuristic":0.00}'
    fi
    RUN_TAG="sideheads-balanced"
    ;;
  usfocus)
    NUM_ENV_RUNNERS="${NUM_ENV_RUNNERS:-8}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1360}"
    MINIBATCH_SIZE="${MINIBATCH_SIZE:-80}"
    ROLLOUT_FRAGMENT_LENGTH="${ROLLOUT_FRAGMENT_LENGTH:-170}"
    EVAL_GAMES="${EVAL_GAMES:-10}"
    EVAL_MIN_GAMES_PER_SIDE="${EVAL_MIN_GAMES_PER_SIDE:-10}"
    if [[ -z "${OPPONENT_MIX:-}" ]]; then
      OPPONENT_MIX='{"current_self":0.50,"teacher":0.20,"anchor":0.20,"random_legal":0.10,"heuristic":0.00}'
    fi
    RUN_TAG="sideheads-usfocus-mix"
    ;;
  *)
    echo "unsupported MODE=$MODE; expected balanced or usfocus" >&2
    exit 2
    ;;
esac

RUN_DIR="${RUN_DIR:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/${RUN_TAG}-gpu${GPU_LABEL}-48h-${STAMP}}"
WANDB_PROJECT="${WANDB_PROJECT:-twilight-struggle-ppo}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}-gpu${GPU_LABEL}-48h-${STAMP}}"
WANDB_MODE="${WANDB_MODE:-online}"

LR="${LR:-0.00001}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
KL_COEFF="${KL_COEFF:-0.8}"
KL_TARGET="${KL_TARGET:-0.004}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.008}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-1.20}"
PERSISTENT_RANDOM_PROB="${PERSISTENT_RANDOM_PROB:-0.02}"
FOCUSED_HEAD_ONLY="${FOCUSED_HEAD_ONLY:-0}"
FOCUSED_ACTOR_ANCHOR_PULLBACK="${FOCUSED_ACTOR_ANCHOR_PULLBACK:-0.0}"
REWARD_SHAPING_SCALE="${REWARD_SHAPING_SCALE:-0.9}"
REWARD_SHAPING_FINAL_SCALE="${REWARD_SHAPING_FINAL_SCALE:-0.25}"
UPDATE_ANCHOR_ON_BEST="${UPDATE_ANCHOR_ON_BEST:-1}"
ROLLBACK_ON_ANCHOR_REJECT="${ROLLBACK_ON_ANCHOR_REJECT:-1}"
INITIAL_BEST_MANIFEST="${INITIAL_BEST_MANIFEST:-}"
ELO_SEED_FROM="${ELO_SEED_FROM:-}"
EVAL_HISTORY_OPPONENTS="${EVAL_HISTORY_OPPONENTS:-0}"
CHECKPOINT_EVAL_EVERY_EPISODES="${CHECKPOINT_EVAL_EVERY_EPISODES:-1000}"
CHECKPOINT_EVAL_EVERY_STEPS="${CHECKPOINT_EVAL_EVERY_STEPS:-0}"
CHECKPOINT_EVAL_EVERY_SECONDS="${CHECKPOINT_EVAL_EVERY_SECONDS:-3600}"
PPO_EPISODES="${PPO_EPISODES:-1000000}"

for path in "$RESTORE_CKPT" "$INITIAL_CKPT" "$ANCHOR_CKPT" "$REFERENCE_CKPT"; do
  if [[ ! -d "$path" ]]; then
    echo "missing checkpoint: $path" >&2
    exit 2
  fi
done
if [[ -n "$INITIAL_BEST_MANIFEST" && ! -f "$INITIAL_BEST_MANIFEST" ]]; then
  echo "missing initial best manifest: $INITIAL_BEST_MANIFEST" >&2
  exit 2
fi
if [[ -n "$ELO_SEED_FROM" && ! -f "$ELO_SEED_FROM" ]]; then
  echo "missing Elo seed state: $ELO_SEED_FROM" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"/{ppo_checkpoints,game_logs,metrics}
{
  echo "run_dir=$RUN_DIR"
  echo "mode=$MODE"
  echo "gpu=$GPU"
  echo "timeout_seconds=$TIMEOUT_SECONDS"
  echo "start_ckpt=$START_CKPT"
  echo "restore_ckpt=$RESTORE_CKPT"
  echo "initial_ckpt=$INITIAL_CKPT"
  echo "anchor_ckpt=$ANCHOR_CKPT"
  echo "reference_ckpt=$REFERENCE_CKPT"
  echo "policy_sharing=unified"
  echo "side_specific_heads=1"
  echo "model_arch=transformer_history"
  echo "history_length=1024"
  echo "graph_neighbor_hops=2"
  echo "num_gpus=1"
  echo "num_gpus_per_env_runner=0"
  echo "num_env_runners=$NUM_ENV_RUNNERS"
  echo "train_batch_size=$TRAIN_BATCH_SIZE"
  echo "minibatch_size=$MINIBATCH_SIZE"
  echo "rollout_fragment_length=$ROLLOUT_FRAGMENT_LENGTH"
  echo "lr=$LR"
  echo "num_epochs=$NUM_EPOCHS"
  echo "kl_coeff=$KL_COEFF"
  echo "kl_target=$KL_TARGET"
  echo "entropy_coeff=$ENTROPY_COEFF"
  echo "policy_temperature=$POLICY_TEMPERATURE"
  echo "persistent_random_prob=$PERSISTENT_RANDOM_PROB"
  echo "focused_head_only=$FOCUSED_HEAD_ONLY"
  echo "focused_actor_anchor_pullback=$FOCUSED_ACTOR_ANCHOR_PULLBACK"
  echo "reward_shaping_scale=$REWARD_SHAPING_SCALE"
  echo "reward_shaping_final_scale=$REWARD_SHAPING_FINAL_SCALE"
  echo "update_anchor_on_best=$UPDATE_ANCHOR_ON_BEST"
  echo "rollback_on_anchor_reject=$ROLLBACK_ON_ANCHOR_REJECT"
  echo "initial_best_manifest=$INITIAL_BEST_MANIFEST"
  echo "elo_seed_from=$ELO_SEED_FROM"
  echo "eval_history_opponents=$EVAL_HISTORY_OPPONENTS"
  echo "checkpoint_eval_every_episodes=$CHECKPOINT_EVAL_EVERY_EPISODES"
  echo "checkpoint_eval_every_steps=$CHECKPOINT_EVAL_EVERY_STEPS"
  echo "checkpoint_eval_every_seconds=$CHECKPOINT_EVAL_EVERY_SECONDS"
  echo "eval_games=$EVAL_GAMES"
  echo "eval_min_games_per_side=$EVAL_MIN_GAMES_PER_SIDE"
  echo "opponent_mix=$OPPONENT_MIX"
  echo "wandb_project=$WANDB_PROJECT"
  echo "wandb_entity=$WANDB_ENTITY"
  echo "wandb_run_name=$WANDB_RUN_NAME"
  echo "wandb_mode=$WANDB_MODE"
} | tee "$RUN_DIR/launch_config.txt"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

COMMON_ARGS=(
  --multi-agent
  --policy-sharing unified
  --side-specific-heads
  --no-adaptive-us-focus
  --model-arch transformer_history
  --hidden 256
  --history-layers 2
  --card-history-layers 2
  --history-attention-heads 4
  --history-dropout 0.05
  --graph-layers 2
  --graph-neighbor-hops 2
  --policy-temperature "$POLICY_TEMPERATURE"
  --lr "$LR"
  --num-epochs "$NUM_EPOCHS"
  --kl-coeff "$KL_COEFF"
  --kl-target "$KL_TARGET"
  --entropy-coeff "$ENTROPY_COEFF"
  --restore-from "$RESTORE_CKPT"
  --partial-warmstart
  --load-report-path "$RUN_DIR/ppo_load_report.json"
  --eval-initial-from "$INITIAL_CKPT"
  --benchmark-label pre_sideheads_reference
  --benchmark-us-policy-from "$REFERENCE_CKPT"
  --benchmark-ussr-policy-from "$REFERENCE_CKPT"
  --stop-episodes "$PPO_EPISODES"
  --num-env-runners "$NUM_ENV_RUNNERS"
  --num-gpus 1
  --num-gpus-per-env-runner 0
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --minibatch-size "$MINIBATCH_SIZE"
  --rollout-fragment-length "$ROLLOUT_FRAGMENT_LENGTH"
  --batch-mode complete_episodes
  --sample-timeout-s 1800
  --checkpoint-dir "$RUN_DIR/ppo_checkpoints"
  --checkpoint-eval-every-steps "$CHECKPOINT_EVAL_EVERY_STEPS"
  --checkpoint-eval-every-episodes "$CHECKPOINT_EVAL_EVERY_EPISODES"
  --checkpoint-eval-every-seconds "$CHECKPOINT_EVAL_EVERY_SECONDS"
  --eval-games "$EVAL_GAMES"
  --eval-min-games-per-side "$EVAL_MIN_GAMES_PER_SIDE"
  --eval-history-opponents "$EVAL_HISTORY_OPPONENTS"
  --eval-max-episode-steps 1200
  --eval-progress-every-games 10
  --elo-path "$RUN_DIR/elo_unified.json"
  --elo-k-factor 32
  --best-checkpoint-manifest "$RUN_DIR/best_checkpoint.json"
  --best-score-side-weight 0.25
  --best-score-balance-penalty 0.50
  --best-score-nuke-penalty 100
  --best-score-scoring-card-held-penalty 100
  --best-score-random-side-floor 0.80
  --best-score-random-side-penalty 200
  --best-score-benchmark-side-floor 0.50
  --best-score-benchmark-side-penalty 250
  --best-anchor-confirm-games-per-side 30
  --wandb-project "$WANDB_PROJECT"
  --wandb-entity "$WANDB_ENTITY"
  --wandb-run-name "$WANDB_RUN_NAME"
  --wandb-mode "$WANDB_MODE"
  --log-games-dir "$RUN_DIR/game_logs"
  --log-games-every 20
  --metrics-dir "$RUN_DIR/metrics"
  --terminal-reward-scale 2.0
  --reward-shaping-scale "$REWARD_SHAPING_SCALE"
  --reward-shaping-final-scale "$REWARD_SHAPING_FINAL_SCALE"
  --reward-shaping-phaseout-start-episodes 5000
  --reward-shaping-phaseout-end-episodes 100000
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
  --max-episode-step-penalty 0.25
  --defcon-suicide-mode hard_filter
  --max-episode-steps 1200
  --persistent-random-prob "$PERSISTENT_RANDOM_PROB"
  --focused-actor-anchor-pullback "$FOCUSED_ACTOR_ANCHOR_PULLBACK"
  --heuristic-override-prob 0.0
  --scripted-side-prob 0.0
  --no-force-setup-heuristic
)

if [[ "$UPDATE_ANCHOR_ON_BEST" == "1" ]]; then
  COMMON_ARGS+=(--update-anchor-on-best)
else
  COMMON_ARGS+=(--no-update-anchor-on-best)
fi
if [[ "$ROLLBACK_ON_ANCHOR_REJECT" == "1" ]]; then
  COMMON_ARGS+=(--rollback-on-anchor-reject)
else
  COMMON_ARGS+=(--no-rollback-on-anchor-reject)
fi
if [[ -n "$INITIAL_BEST_MANIFEST" ]]; then
  COMMON_ARGS+=(--initial-best-manifest "$INITIAL_BEST_MANIFEST")
fi
if [[ -n "$ELO_SEED_FROM" ]]; then
  COMMON_ARGS+=(--elo-seed-from "$ELO_SEED_FROM")
fi

MODE_ARGS=()
if [[ "$MODE" == "balanced" ]]; then
  MODE_ARGS=(
    --unified-train-focus-side both
    --unified-opponent-mix-json "$OPPONENT_MIX"
    --unified-anchor-policy-from "[\"$ANCHOR_CKPT\"]"
    --unified-opponent-mix-seed 260730
  )
else
  MODE_ARGS=(
    --unified-train-focus-side us
    --unified-opponent-mix-json "$OPPONENT_MIX"
    --unified-anchor-policy-from "[\"$ANCHOR_CKPT\"]"
    --unified-opponent-mix-seed 260731
  )
  if [[ "$FOCUSED_HEAD_ONLY" == "1" ]]; then
    MODE_ARGS+=(--focused-head-only)
  else
    MODE_ARGS+=(--no-focused-head-only)
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set +e
timeout --signal=INT --kill-after=300 "$((TIMEOUT_SECONDS + 7200))" \
  "$PYTHON" -m struggle_ai.train_rllib \
  "${COMMON_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  --stop-seconds "$TIMEOUT_SECONDS" \
  2>&1 | tee "$RUN_DIR/ppo_train.log"
status=${PIPESTATUS[0]}
set -e

echo "$status" > "$RUN_DIR/exit_status.txt"
exit "$status"
