#!/usr/bin/env python3
"""Run Raw/D2/D4/deeper promotion gates for the p30/u25 campaign."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_search_scaling_promotion_gates as gates


PARENT = (
    ROOT
    / "runs/search_distill/search-scaling-phase3u75-parentactor-p30u25-composed-20260828/checkpoints/composed"
)
CAMPAIGN = ROOT / "runs/search_distill/search-scaling-p30u25-amortization-u25-gpu4-20260828"
OUTPUT = ROOT / "runs/eval/search-scaling-p30u25-amortization-gates-20260828"


def configure() -> None:
    gates.PARENT = PARENT
    gates.CAMPAIGN = CAMPAIGN
    gates.OUTPUT = OUTPUT


if __name__ == "__main__":
    configure()
    gates.main()
