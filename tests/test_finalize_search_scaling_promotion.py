from pathlib import Path

from scripts.finalize_search_scaling_promotion import evaluate_promotion


def _record(score: float, low: float, us: float = 0.6, ussr: float = 0.6):
    return {
        "score": score,
        "paired_bootstrap_ci95_low": low,
        "by_candidate_side": {"us": {"score": us}, "ussr": {"score": ussr}},
    }


def test_promotion_requires_confirmed_parent_and_search_scaling(tmp_path: Path) -> None:
    child = tmp_path / "child"
    gate = {
        "confirmatory_candidate": 10,
        "confirmatory_records": {
            "raw": _record(0.55, 0.45),
            "d2": _record(0.65, 0.52),
            "d4": _record(0.58, 0.46),
        },
    }
    anomaly = {"checkpoint": str(child), "update": 10, "passed": True}
    scaling = {label: _record(0.55, 0.46) for label in (
        "d2_vs_raw", "d4_128_vs_d2_32", "d4_512_vs_d4_128"
    )}
    cross = {
        "checkpoint": str(child),
        "confirmatory_records": scaling,
        "confirmatory_scaling_supported": True,
        "confirmatory_side_balance_supported": True,
    }
    decision = evaluate_promotion(gate, anomaly, cross, parent=tmp_path / "parent")
    assert decision["promoted"] is True
    assert decision["retained_general_parent"] == str(child.resolve())


def test_promotion_fails_closed_on_d2_or_side_regression(tmp_path: Path) -> None:
    child = tmp_path / "child"
    gate = {
        "confirmatory_candidate": 10,
        "confirmatory_records": {
            "raw": _record(0.55, 0.45),
            "d2": _record(0.60, 0.49),
            "d4": _record(0.58, 0.46, us=0.30, ussr=0.70),
        },
    }
    anomaly = {"checkpoint": str(child), "update": 10, "passed": True}
    scaling = {label: _record(0.55, 0.46) for label in (
        "d2_vs_raw", "d4_128_vs_d2_32", "d4_512_vs_d4_128", "d6_512_vs_d4_512"
    )}
    cross = {
        "checkpoint": str(child),
        "confirmatory_records": scaling,
        "confirmatory_scaling_supported": True,
        "confirmatory_side_balance_supported": True,
    }
    parent = tmp_path / "parent"
    decision = evaluate_promotion(gate, anomaly, cross, parent=parent)
    assert decision["promoted"] is False
    assert decision["retained_general_parent"] == str(parent.resolve())
    assert "d2_parent_improvement_not_statistically_supported" in decision["reasons"]
    assert "confirmatory_us_ussr_side_floor_failed" in decision["reasons"]


def test_promotion_requires_forty_percent_side_floor(tmp_path: Path) -> None:
    child = tmp_path / "child"
    gate = {
        "confirmatory_candidate": 10,
        "confirmatory_records": {
            "raw": _record(0.55, 0.45),
            "d2": _record(0.65, 0.52),
            "d4": _record(0.58, 0.46, us=0.38, ussr=0.62),
        },
    }
    anomaly = {"checkpoint": str(child), "update": 10, "passed": True}
    scaling = {
        label: _record(0.55, 0.46)
        for label in (
            "d2_vs_raw",
            "d4_128_vs_d2_32",
            "d4_512_vs_d4_128",
            "d6_512_vs_d4_512",
        )
    }
    cross = {
        "checkpoint": str(child),
        "confirmatory_records": scaling,
        "confirmatory_scaling_supported": True,
        "confirmatory_side_balance_supported": True,
    }

    decision = evaluate_promotion(gate, anomaly, cross, parent=tmp_path / "parent")

    assert decision["promoted"] is False
    assert "confirmatory_us_ussr_side_floor_failed" in decision["reasons"]


def test_promotion_requires_d4_512_but_not_d6(tmp_path: Path) -> None:
    child = tmp_path / "child"
    gate = {
        "confirmatory_candidate": 10,
        "confirmatory_records": {
            "raw": _record(0.55, 0.45),
            "d2": _record(0.65, 0.52),
            "d4": _record(0.58, 0.46),
        },
    }
    anomaly = {"checkpoint": str(child), "update": 10, "passed": True}
    cross = {
        "checkpoint": str(child),
        "confirmatory_records": {
            "d2_vs_raw": _record(0.55, 0.46),
            "d4_128_vs_d2_32": _record(0.55, 0.46),
        },
        "confirmatory_scaling_supported": False,
    }

    decision = evaluate_promotion(gate, anomaly, cross, parent=tmp_path / "parent")

    assert decision["promoted"] is False
    assert "missing_confirmatory_scaling_modes:d4_512_vs_d4_128" in decision["reasons"]
    assert all("d6_512" not in reason for reason in decision["reasons"])


def test_promotion_rejects_side_collapsed_search_scaling(tmp_path: Path) -> None:
    child = tmp_path / "child"
    gate = {
        "confirmatory_candidate": 10,
        "confirmatory_records": {
            "raw": _record(0.55, 0.45),
            "d2": _record(0.65, 0.52),
            "d4": _record(0.58, 0.46),
        },
    }
    anomaly = {"checkpoint": str(child), "update": 10, "passed": True}
    scaling = {
        label: _record(0.55, 0.46)
        for label in ("d2_vs_raw", "d4_128_vs_d2_32", "d4_512_vs_d4_128")
    }
    cross = {
        "checkpoint": str(child),
        "confirmatory_records": scaling,
        "confirmatory_scaling_supported": False,
        "confirmatory_side_balance_supported": False,
    }

    decision = evaluate_promotion(gate, anomaly, cross, parent=tmp_path / "parent")

    assert decision["promoted"] is False
    assert "search_scaling_us_ussr_balance_not_confirmed" in decision["reasons"]
