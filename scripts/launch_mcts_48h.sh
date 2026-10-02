#!/usr/bin/env bash
set -euo pipefail

cd /mnt2/users/kaile/hantao/game/struggle

GPU="${GPU:-0}"
PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-172800}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
RESTORE_CKPT="${RESTORE_CKPT:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/sideheads-usfocus-rollback-kl-gpu1-43h-20260730-195630/ppo_checkpoints/checkpoints/steps-0000048399-episodes-00000264}"
RUN_DIR="${RUN_DIR:-/mnt2/users/kaile/hantao/game/struggle/runs/mcts/belief-chance-sideheads-gpu${GPU}-48h-${STAMP}}"
SIMULATIONS="${SIMULATIONS:-4}"
SEARCH_DEPTH="${SEARCH_DEPTH:-4}"
BELIEF_PARTICLES="${BELIEF_PARTICLES:-4}"
MAX_CHILDREN="${MAX_CHILDREN:-0}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-8192}"
MIN_REPLAY_SIZE="${MIN_REPLAY_SIZE:-2048}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATES_PER_GAME="${UPDATES_PER_GAME:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.000003}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-0.5}"
TRAIN_FOCUS_SIDE="${TRAIN_FOCUS_SIDE:-both}"
FOCUSED_HEAD_ONLY="${FOCUSED_HEAD_ONLY:-0}"
FOCUSED_ACTOR_ANCHOR_PULLBACK="${FOCUSED_ACTOR_ANCHOR_PULLBACK:-0.0}"
RETAINED_ELO_MIN_DELTA="${RETAINED_ELO_MIN_DELTA:-0.0}"
ROLLBACK_ON_EVAL_REJECT="${ROLLBACK_ON_EVAL_REJECT:-1}"
ELO_SEED_FROM="${ELO_SEED_FROM:-/mnt2/users/kaile/hantao/game/struggle/runs/ppo/sideheads-usfocus-rollback-kl-gpu1-43h-20260730-195630/elo_unified.json}"

if [[ ! -d "$RESTORE_CKPT" ]]; then
  echo "missing checkpoint: $RESTORE_CKPT" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
{
  echo "run_dir=$RUN_DIR"
  echo "gpu=$GPU"
  echo "timeout_seconds=$TIMEOUT_SECONDS"
  echo "restore_ckpt=$RESTORE_CKPT"
  echo "algorithm=belief_chance_puct_expert_iteration"
  echo "policy=unified_side_specific_heads"
  echo "critic=two_channel_side_selected"
  echo "simulations=$SIMULATIONS"
  echo "search_depth=$SEARCH_DEPTH"
  echo "belief_particles=$BELIEF_PARTICLES"
  echo "max_children=$MAX_CHILDREN"
  echo "replay_capacity=$REPLAY_CAPACITY"
  echo "min_replay_size=$MIN_REPLAY_SIZE"
  echo "batch_size=$BATCH_SIZE"
  echo "updates_per_game=$UPDATES_PER_GAME"
  echo "learning_rate=$LEARNING_RATE"
  echo "value_loss_weight=$VALUE_LOSS_WEIGHT"
  echo "train_focus_side=$TRAIN_FOCUS_SIDE"
  echo "focused_head_only=$FOCUSED_HEAD_ONLY"
  echo "focused_actor_anchor_pullback=$FOCUSED_ACTOR_ANCHOR_PULLBACK"
  echo "retained_elo_min_delta=$RETAINED_ELO_MIN_DELTA"
  echo "rollback_on_eval_reject=$ROLLBACK_ON_EVAL_REJECT"
  echo "elo_seed_from=$ELO_SEED_FROM"
  echo "belief=root_opponent_hand_redeterminization"
  echo "chance=uniform_forced_d6_samples_through_saito"
} | tee "$RUN_DIR/launch_config.txt"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

focused_head_args=(--no-focused-head-only)
if [[ "$FOCUSED_HEAD_ONLY" == "1" ]]; then
  focused_head_args=(--focused-head-only)
fi

rollback_args=(--no-rollback-on-eval-reject)
if [[ "$ROLLBACK_ON_EVAL_REJECT" == "1" ]]; then
  rollback_args=(--rollback-on-eval-reject)
fi

set +e
timeout --signal=INT --kill-after=300 "$((TIMEOUT_SECONDS + 7200))" \
  "$PYTHON" -m struggle_ai.train_mcts \
    --restore-from "$RESTORE_CKPT" \
    --output-dir "$RUN_DIR" \
    --stop-seconds "$TIMEOUT_SECONDS" \
    --simulations "$SIMULATIONS" \
    --search-depth "$SEARCH_DEPTH" \
    --belief-particles "$BELIEF_PARTICLES" \
    --max-children "$MAX_CHILDREN" \
    --cpuct 1.5 \
    --dirichlet-alpha 0.3 \
    --dirichlet-fraction 0.20 \
    --early-temperature 1.0 \
    --late-temperature 0.25 \
    --temperature-through-turn 3 \
    --replay-capacity "$REPLAY_CAPACITY" \
    --min-replay-size "$MIN_REPLAY_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --updates-per-game "$UPDATES_PER_GAME" \
    --lr "$LEARNING_RATE" \
    --value-loss-weight "$VALUE_LOSS_WEIGHT" \
    --train-focus-side "$TRAIN_FOCUS_SIDE" \
    --focused-actor-anchor-pullback "$FOCUSED_ACTOR_ANCHOR_PULLBACK" \
    "${focused_head_args[@]}" \
    --retained-elo-min-delta "$RETAINED_ELO_MIN_DELTA" \
    "${rollback_args[@]}" \
    --elo-path "$RUN_DIR/elo_unified.json" \
    --elo-seed-from "$ELO_SEED_FROM" \
    --elo-k-factor 32 \
    --checkpoint-every-seconds 1800 \
    --eval-every-seconds 14400 \
    --eval-games-per-side 10 \
    --max-episode-steps 1200 \
    --wandb-project twilight-struggle-mcts \
    --wandb-run-name "belief-chance-sideheads-gpu${GPU}-48h-${STAMP}" \
    --wandb-mode online \
    2>&1 | tee "$RUN_DIR/mcts_train.log"
status=${PIPESTATUS[0]}
set -e

echo "$status" > "$RUN_DIR/exit_status.txt"
exit "$status"
