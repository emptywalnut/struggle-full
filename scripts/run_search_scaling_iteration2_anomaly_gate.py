#!/usr/bin/env python3
"""Run the standard frozen anomaly gate for search-scaling iteration 2."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_search_scaling_anomaly_gate as gate


gate.GATE_DIR = ROOT / "runs/eval/search-scaling-iteration2-phase3u75-gates-20260824"
gate.CAMPAIGN = ROOT / "runs/search_distill/search-scaling-iteration2-phase3u75-u25-gpu4-20260824"
gate.OUTPUT = ROOT / "runs/diagnostics/search-scaling-iteration2-phase3u75-anomaly-gate-20260824"


if __name__ == "__main__":
    gate.main()
