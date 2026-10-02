#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/mnt2/users/kaile/miniconda3/envs/struggle-ppo/bin/python}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"

cd "$ROOT"

aggregate() {
  local output="$1"
  shift
  mkdir -p "$(dirname "$output")"
  "$PYTHON" scripts/aggregate_paired_eval_shards.py \
    "$@" \
    --output "$output" >/dev/null 2>&1 || true
}

while true; do
  aggregate \
    runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-aggregate-20260827/aggregate.json \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm50-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-shardb22-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-shardc10-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-shardd10-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-sharde10-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-shardf10-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-shardg16-20260827

  aggregate \
    runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-only-aggregate-20260827/aggregate.json \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm50-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-shardb22-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-shardc10-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-shardd10-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-sharde10-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-shardf10-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-confirm-shardg16-20260827

  aggregate \
    runs/eval/search-scaling-phase3u75-d2-128-vs-d2-32-aggregate-20260827/aggregate.json \
    --input runs/eval/search-scaling-phase3u75-d2-128-vs-d2-32-20260827 \
    --input runs/eval/search-scaling-phase3u75-d2-128-vs-d2-32-shardc10-20260827 \
    --input runs/eval/search-scaling-phase3u75-d2-128-vs-d2-32-shardd10-20260827 \
    --input runs/eval/search-scaling-phase3u75-d2-128-vs-d2-32-sharde10-20260827

  aggregate \
    runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-128-aggregate-20260827/aggregate.json \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-128-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-128-shardc10-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-128-shardd10-20260827 \
    --input runs/eval/search-scaling-phase3u75-adaptive-converged-vs-d2-128-sharde16-20260827

  sleep "$INTERVAL_SECONDS"
done
