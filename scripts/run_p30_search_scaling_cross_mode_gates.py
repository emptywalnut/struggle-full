#!/usr/bin/env python3
"""Run same-checkpoint search-scaling gates for the p30/u25 candidate."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_search_scaling_cross_mode_gates as gates


CAMPAIGN = ROOT / "runs/search_distill/search-scaling-p30u25-amortization-u25-gpu4-20260828"
ANOMALY_OUTPUT = (
    ROOT / "runs/diagnostics/search-scaling-p30u25-amortization-anomaly-gate-20260828"
)
OUTPUT = ROOT / "runs/eval/search-scaling-p30u25-amortization-cross-mode-20260828"


def configure() -> None:
    gates.CAMPAIGN = CAMPAIGN
    gates.ANOMALY_OUTPUT = ANOMALY_OUTPUT
    gates.OUTPUT = OUTPUT


if __name__ == "__main__":
    configure()
    gates.main()
