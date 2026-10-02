import numpy as np

from scripts.summarize_adaptive_anomaly_probe import select_adaptive


def row(depth: int, budget: int, q_values: list[float], visits: list[int]) -> dict:
    return {
        "depth": depth,
        "budget": budget,
        "root_legal_count": len(q_values),
        "root_table": [
            {"action_index": index, "N": visits[index], "Q": q_values[index]}
            for index in range(len(q_values))
        ],
        "choice": str(int(np.argmax(q_values))),
        "choice_Q": max(q_values),
        "atomic_horizon_cutoffs": 0,
        "node_budget_cutoffs": 0,
    }


def test_select_adaptive_accepts_stable_budget() -> None:
    result = select_adaptive(
        [
            row(2, 32, [0.4, 0.1, -0.2], [20, 8, 4]),
            row(2, 128, [0.5, 0.0, -0.3], [90, 28, 10]),
            row(4, 128, [0.0, 0.6, -0.1], [20, 98, 10]),
        ]
    )
    assert result["status"] == "BUDGET_CONVERGENCE"
    assert result["selected"]["depth"] == 2


def test_select_adaptive_accepts_stable_horizon_change() -> None:
    result = select_adaptive(
        [
            row(2, 32, [0.4, 0.1, -0.2], [20, 8, 4]),
            row(2, 128, [0.1, 0.5, -0.3], [20, 98, 10]),
            row(4, 128, [0.0, 0.6, -0.1], [18, 100, 10]),
        ]
    )
    assert result["status"] == "HORIZON_CONVERGENCE"
    assert result["selected"]["depth"] == 4


def test_select_adaptive_falls_back_from_unresolved_depth() -> None:
    result = select_adaptive(
        [
            row(2, 32, [0.4, 0.1, -0.2], [20, 8, 4]),
            row(2, 128, [0.1, 0.5, -0.3], [20, 98, 10]),
            row(4, 128, [0.6, 0.0, -0.1], [100, 18, 10]),
        ]
    )
    assert result["status"] == "UNRESOLVED"
    assert result["fallback"] is True
    assert result["selected"]["depth"] == 2
