from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_iterative_puct_distill.py"
SPEC = importlib.util.spec_from_file_location("run_iterative_puct_distill", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_replay_cutoff_counts_keeps_safe_examples_from_mixed_shard(monkeypatch) -> None:
    examples = [
        SimpleNamespace(
            metadata=SimpleNamespace(atomic_horizon_cutoffs=0, node_budget_cutoffs=0)
        ),
        SimpleNamespace(
            metadata=SimpleNamespace(atomic_horizon_cutoffs=0, node_budget_cutoffs=3)
        ),
        SimpleNamespace(
            metadata=SimpleNamespace(atomic_horizon_cutoffs=2, node_budget_cutoffs=0)
        ),
    ]
    monkeypatch.setattr(module, "load_search_replay", lambda _paths: examples)

    counts = module.replay_cutoff_counts(Path("unused"))

    assert counts == {
        "total_examples": 3,
        "safe_examples": 1,
        "atomic_cutoff_examples": 1,
        "node_cutoff_examples": 1,
    }


def test_student_acceptance_requires_overall_and_both_side_floors() -> None:
    evaluation = {
        "curves": {
            "raw": {
                "ratings": [
                    {
                        "agent": "student-raw",
                        "score": 0.60,
                        "us_score": 0.70,
                        "ussr_score": 0.30,
                    }
                ]
            }
        }
    }

    result = module.student_acceptance_from_evaluation(
        evaluation,
        min_score=0.50,
        min_side_score=0.40,
    )

    assert result["accepted"] is False
    evaluation["curves"]["raw"]["ratings"][0]["ussr_score"] = 0.50
    result = module.student_acceptance_from_evaluation(
        evaluation,
        min_score=0.50,
        min_side_score=0.40,
    )
    assert result["accepted"] is True
