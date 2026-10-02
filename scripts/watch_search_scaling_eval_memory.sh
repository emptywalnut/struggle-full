#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/mnt2/users/kaile/hantao/game/struggle}"
LOG="${LOG:-$ROOT/runs/eval/search-scaling-phase3u75-amortization-cross-mode-20260821/memory_watch.log}"
MIN_AVAILABLE_KIB="${MIN_AVAILABLE_KIB:-167772160}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOW_SAMPLES_REQUIRED="${LOW_SAMPLES_REQUIRED:-3}"

mkdir -p "$(dirname "$LOG")"
low_samples=0

while true; do
  available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  timestamp=$(date +%s)
  printf '%s available_kib=%s low_samples=%s\n' \
    "$timestamp" "$available_kib" "$low_samples" >> "$LOG"

  if (( available_kib < MIN_AVAILABLE_KIB )); then
    ((low_samples += 1))
  else
    low_samples=0
  fi

  if (( low_samples >= LOW_SAMPLES_REQUIRED )); then
    target=""
    pid=$(pgrep -f 'eval_puct_agent_pair_parallel.py.*confirmatory/d6_512_vs_d4_512' | head -n 1 || true)
    if [[ -n "$pid" ]]; then
      target="confirmatory/d6_512_vs_d4_512"
    else
      pid=$(pgrep -f 'eval_puct_agent_pair_parallel.py.*confirmatory/d4_512_vs_d4_128' | head -n 1 || true)
      if [[ -n "$pid" ]]; then
        target="confirmatory/d4_512_vs_d4_128"
      else
        pid=$(pgrep -f 'eval_puct_agent_pair_parallel.py.*d6_512_vs_d4_512' | head -n 1 || true)
        if [[ -n "$pid" ]]; then
          target="d6_512_vs_d4_512"
        else
          pid=$(pgrep -f 'eval_puct_agent_pair_parallel.py.*u5/d4_512' | head -n 1 || true)
          [[ -n "$pid" ]] && target="u5/d4_512"
        fi
      fi
    fi
    if [[ -n "$target" ]]; then
      printf '%s guard_stop pid=%s available_kib=%s target=%s\n' \
        "$timestamp" "$pid" "$available_kib" "$target" >> "$LOG"
      kill -TERM "$pid"
      low_samples=0
      sleep 180
    fi
  fi

  sleep "$POLL_SECONDS"
done
