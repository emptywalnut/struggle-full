from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_selective_puct.py"
SPEC = importlib.util.spec_from_file_location("eval_selective_puct", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_raw_summary_uses_the_paired_evaluation_side() -> None:
    records = [
        {"winner": "ussr", "evaluated_side": "us", "terminal_reason": "final scoring", "steps": 10},
        {"winner": "ussr", "evaluated_side": "ussr", "terminal_reason": "final scoring", "steps": 12},
    ]

    result = MODULE.summarize(records, search_match=False)

    assert result["wins"] == 1
    assert result["win_rate"] == 0.5


def test_paired_strength_counts_improvements_and_regressions() -> None:
    raw = [
        {"winner": "ussr", "evaluated_side": "us"},
        {"winner": "ussr", "evaluated_side": "ussr"},
        {"winner": "us", "evaluated_side": "us"},
    ]
    searched = [
        {"winner": "us", "planned_side": "us"},
        {"winner": "us", "planned_side": "ussr"},
        {"winner": "us", "planned_side": "us"},
    ]

    result = MODULE.summarize_paired_strength(raw, searched)

    assert result == {
        "games": 3,
        "improved": 1,
        "regressed": 1,
        "unchanged_wins": 1,
        "unchanged_losses": 0,
        "net_improvement_rate": 0.0,
    }
