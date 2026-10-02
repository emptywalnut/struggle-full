#!/usr/bin/env python3
"""Run same-checkpoint search-scaling gates for iteration 2."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_search_scaling_cross_mode_gates as gates


gates.CAMPAIGN = ROOT / "runs/search_distill/search-scaling-iteration2-phase3u75-u25-gpu4-20260824"
gates.ANOMALY_OUTPUT = ROOT / "runs/diagnostics/search-scaling-iteration2-phase3u75-anomaly-gate-20260824"
gates.OUTPUT = ROOT / "runs/eval/search-scaling-iteration2-phase3u75-cross-mode-20260824"


if __name__ == "__main__":
    gates.main()
