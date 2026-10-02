#!/usr/bin/env python3
"""Run the standard promotion gates for search-scaling iteration 2."""

from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_search_scaling_promotion_gates as gates


gates.PARENT = ROOT / "runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/checkpoints/updates-000075"
gates.CAMPAIGN = ROOT / "runs/search_distill/search-scaling-iteration2-phase3u75-u25-gpu4-20260824"
gates.OUTPUT = ROOT / "runs/eval/search-scaling-iteration2-phase3u75-gates-20260824"
PRESSURE_D8_STATUS = ROOT / "runs/search_scaling/iteration2-scoring-pressure-probe-20260824/pressure-step66-d8_2048-post-collection.status"


def wait_for_pressure_d8(*, poll_seconds: float = 30.0) -> None:
    while not PRESSURE_D8_STATUS.is_file():
        time.sleep(max(0.01, float(poll_seconds)))
    status = PRESSURE_D8_STATUS.read_text(encoding="utf-8").strip()
    if status != "complete":
        raise RuntimeError(f"post-collection pressure D8 failed: {status}")


if __name__ == "__main__":
    wait_for_pressure_d8()
    gates.main()
