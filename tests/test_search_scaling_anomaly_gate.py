from scripts.run_search_scaling_anomaly_gate import normalize_action, select_candidate


def test_select_candidate_requires_all_three_search_regimes() -> None:
    payload = {
        "d4_candidates": [5, 10],
        "records": {
            "u5:raw": {"score": 0.55},
            "u5:d2": {"score": 0.60},
            "u5:d4": {"score": 0.50},
            "u10:raw": {"score": 0.50},
            "u10:d2": {"score": 0.80},
            "u10:d4": {"score": 0.40},
        },
    }
    assert select_candidate(payload) == 5


def test_select_candidate_prefers_confirmed_candidate() -> None:
    payload = {
        "confirmatory_candidate": 10,
        "confirmatory_records": {"raw": {}, "d2": {}, "d4": {}},
        "d4_candidates": [5, 10],
        "records": {
            "u5:raw": {"score": 0.80},
            "u5:d2": {"score": 0.80},
            "u5:d4": {"score": 0.80},
            "u10:raw": {"score": 0.50},
            "u10:d2": {"score": 0.55},
            "u10:d4": {"score": 0.50},
        },
    }
    assert select_candidate(payload) == 10


def test_normalize_action_handles_country_ids() -> None:
    assert normalize_action("South Africa [southafrica]") == "southafrica"
    assert normalize_action("UK") == "uk"
