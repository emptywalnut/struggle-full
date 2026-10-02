from __future__ import annotations

import pytest

from scripts.run_p30_headlinefix_search_evidence import (
    generation_command,
    parse_args,
    validate_root_panel,
)


BALANCED_SIDE_ERA = {
    f"{side}|{era}": 4
    for side in ("us", "ussr")
    for era in ("early", "mid", "late")
}


def test_generation_command_pairs_root_rng_and_preserves_search_semantics(tmp_path) -> None:
    command = generation_command(
        "d4_512",
        manifest=tmp_path / "positions.json",
        gpu="5",
        workers=12,
        depth=4,
        simulations=512,
        max_atomic_steps=8192,
        max_nodes=1_048_576,
        max_search_seconds=600.0,
        max_worker_rss_mib=20_480.0,
    )
    assert command[command.index("--gpus") + 1] == "5"
    assert command[command.index("--workers") + 1] == "12"
    assert command[command.index("--meaningful-depth") + 1] == "4"
    assert command[command.index("--initial-simulations") + 1] == "512"
    assert command[command.index("--belief-particles") + 1] == "32"
    assert command[command.index("--max-atomic-steps") + 1] == "8192"
    assert command[command.index("--max-expanded-nodes") + 1] == "1048576"
    assert command[command.index("--max-search-seconds") + 1] == "600.0"
    assert command[command.index("--max-search-worker-rss-mib") + 1] == "20480.0"
    assert "--root-seeded-search" in command
    assert "--resume" not in command  # Resume is on by default and identity-checked.


def test_fixed_root_worker_parallelism_is_configurable(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_p30_headlinefix_search_evidence.py"])
    assert parse_args().root_workers == 20

    monkeypatch.setattr(
        "sys.argv",
        ["run_p30_headlinefix_search_evidence.py", "--root-workers", "24"],
    )
    assert parse_args().root_workers == 24


def test_root_panel_requires_balanced_diverse_search_states() -> None:
    categories = [
        "influence_allocation",
        "coup_target",
        "realignment_target",
        "scoring_card",
        "opponent_event",
        "other",
    ]
    positions = [
        {
            "side": "us" if index < 12 else "ussr",
            "era": ("early", "mid", "late")[index % 3],
            "branch_bucket": ("low", "medium", "high")[index % 3],
            "category": categories[index % len(categories)],
        }
        for index in range(24)
    ]
    result = validate_root_panel(
        {
            "positions": positions,
            "summary": {
                "by_side": {"us": 12, "ussr": 12},
                "by_era": {"early": 8, "mid": 8, "late": 8},
                "by_side_era": BALANCED_SIDE_ERA,
                "by_branch_bucket": {"low": 8, "medium": 8, "high": 8},
                "by_category": {category: 4 for category in categories},
            },
        },
        24,
    )
    assert result["passed"] is True


def test_root_panel_rejects_missing_realignment_coverage() -> None:
    positions = [{"side": "us"}] * 12 + [{"side": "ussr"}] * 12
    with pytest.raises(RuntimeError, match="tactical_category_coverage"):
        validate_root_panel(
            {
                "positions": positions,
                "summary": {
                    "by_side": {"us": 12, "ussr": 12},
                    "by_era": {"early": 8, "mid": 8, "late": 8},
                    "by_side_era": BALANCED_SIDE_ERA,
                    "by_branch_bucket": {"low": 8, "medium": 8, "high": 8},
                    "by_category": {
                        "influence_allocation": 6,
                        "coup_target": 6,
                        "scoring_card": 4,
                        "opponent_event": 4,
                        "other": 4,
                    },
                },
            },
            24,
        )


def test_root_panel_rejects_era_skew_even_with_minimum_coverage() -> None:
    positions = [{"side": "us"}] * 12 + [{"side": "ussr"}] * 12
    with pytest.raises(RuntimeError, match="era_imbalance"):
        validate_root_panel(
            {
                "positions": positions,
                "summary": {
                    "by_side": {"us": 12, "ussr": 12},
                    "by_era": {"early": 19, "mid": 3, "late": 2},
                    "by_side_era": BALANCED_SIDE_ERA,
                    "by_branch_bucket": {"low": 12, "medium": 6, "high": 6},
                    "by_category": {
                        "influence_allocation": 4,
                        "coup_target": 4,
                        "realignment_target": 4,
                        "scoring_card": 4,
                        "opponent_event": 4,
                        "forced_response": 4,
                    },
                },
            },
            24,
        )


def test_root_panel_rejects_side_era_skew_with_balanced_marginals() -> None:
    positions = [{"side": "us"}] * 12 + [{"side": "ussr"}] * 12
    skewed_side_era = {
        "us|early": 8,
        "us|mid": 4,
        "us|late": 0,
        "ussr|early": 0,
        "ussr|mid": 4,
        "ussr|late": 8,
    }
    with pytest.raises(RuntimeError, match="side_era_imbalance"):
        validate_root_panel(
            {
                "positions": positions,
                "summary": {
                    "by_side": {"us": 12, "ussr": 12},
                    "by_era": {"early": 8, "mid": 8, "late": 8},
                    "by_side_era": skewed_side_era,
                    "by_branch_bucket": {"low": 12, "medium": 6, "high": 6},
                    "by_category": {
                        "influence_allocation": 4,
                        "coup_target": 4,
                        "realignment_target": 4,
                        "scoring_card": 4,
                        "opponent_event": 4,
                        "forced_response": 4,
                    },
                },
            },
            24,
        )


def test_root_panel_rejects_generic_card_timing_without_scoring_pressure() -> None:
    positions = [{"side": "us"}] * 12 + [{"side": "ussr"}] * 12
    with pytest.raises(RuntimeError, match="scoring_pressure_coverage"):
        validate_root_panel(
            {
                "positions": positions,
                "summary": {
                    "by_side": {"us": 12, "ussr": 12},
                    "by_era": {"early": 8, "mid": 8, "late": 8},
                    "by_side_era": BALANCED_SIDE_ERA,
                    "by_branch_bucket": {"low": 8, "medium": 8, "high": 8},
                    "by_category": {
                        "influence_allocation": 4,
                        "coup_target": 4,
                        "realignment_target": 4,
                        "opponent_event": 4,
                        "forced_response": 4,
                        "other": 4,
                    },
                },
            },
            24,
        )
