from scripts.probe_d4_anomalies import SPECS, parse_matrix
from scripts.validate_d4_anomaly_diagnostic import BASE_MATRIX, D8_MATRIX, validate_search_rows


def test_probe_specs_cover_both_sides_and_failure_types() -> None:
    assert set(SPECS) == {
        "ussr_uk_overcontrol",
        "us_italy_overcontrol",
        "ussr_mexico_realignment",
        "us_venezuela_realignment",
    }
    assert {spec.kind for spec in SPECS.values()} == {"influence", "realignment"}
    assert {spec.suspicious_country for spec in SPECS.values()} == {
        "uk",
        "italy",
        "mexico",
        "venezuela",
    }


def test_parse_matrix_keeps_depth_budget_and_repetition_separate() -> None:
    assert parse_matrix("2:32:2,4:128:1,8:2048:1") == (
        (2, 32, 2),
        (4, 128, 1),
        (8, 2048, 1),
    )


def _search_row(depth: int, budget: int, repeat: int) -> dict:
    return {
        "depth": depth,
        "budget": budget,
        "repeat": repeat,
        "belief_particles": 32,
        "seed": depth * 100_000 + budget + repeat,
        "choice": "example",
        "choice_Q": 0.1,
        "root_Q_gap": 0.01,
        "root_table": [{"action": "example", "action_index": 0, "N": budget, "Q": 0.1, "P": 1.0}],
        "root_legal_count": 1,
        "root_expanded_actions": 1,
        "root_unvisited_actions": 0,
        "prior_order": [
            {"action": "example", "action_index": 0, "P": 1.0, "first_visit_simulation": 1}
        ],
        "child_progressive_widening": [],
        "exact_chance_nodes": 0,
        "sampled_large_chance_nodes": 0,
        "atomic_horizon_cutoffs": 0,
        "node_budget_cutoffs": 0,
        "max_meaningful_depth_reached": depth,
        "latency_seconds": 1.0,
        "simulations_per_second": float(budget),
    }


def test_diagnostic_search_validator_requires_complete_matrix_and_telemetry() -> None:
    expected = BASE_MATRIX | D8_MATRIX
    rows = [_search_row(*key) for key in sorted(expected)]
    summary = validate_search_rows(rows, expected)
    assert summary["runs"] == 7
    assert summary["matrix"] == [list(key) for key in sorted(expected)]
