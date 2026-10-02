from __future__ import annotations

import json

import pytest

from scripts.continue_search_scaling_amortization import (
    validate_collection,
    validate_generation,
)


def test_validate_generation_requires_every_root(tmp_path):
    summary = tmp_path / "generation_summary.json"
    summary.write_text(json.dumps({"roots_completed": 3}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="3/4"):
        validate_generation(summary, 4)


def test_validate_collection_requires_complete_games(tmp_path):
    summary = tmp_path / "collection_summary.json"
    summary.write_text(
        json.dumps({"games_completed": 7, "discarded_games": 1}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="7/8"):
        validate_collection(summary, 8)


def test_validate_collection_returns_valid_payload(tmp_path):
    summary = tmp_path / "collection_summary.json"
    payload = {"games_completed": 8, "discarded_games": 0}
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_collection(summary, 8) == payload
