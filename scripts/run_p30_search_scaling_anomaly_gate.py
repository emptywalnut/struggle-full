#!/usr/bin/env python3
"""Run the frozen anomaly gate for the selected p30/u25 candidate."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_search_scaling_anomaly_gate as gate


GATE_DIR = ROOT / "runs/eval/search-scaling-p30u25-amortization-gates-20260828"
CAMPAIGN = ROOT / "runs/search_distill/search-scaling-p30u25-amortization-u25-gpu4-20260828"
OUTPUT = ROOT / "runs/diagnostics/search-scaling-p30u25-amortization-anomaly-gate-20260828"


def configure() -> None:
    gate.GATE_DIR = GATE_DIR
    gate.CAMPAIGN = CAMPAIGN
    gate.OUTPUT = OUTPUT


if __name__ == "__main__":
    configure()
    gate.main()
