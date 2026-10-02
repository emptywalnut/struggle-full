from __future__ import annotations

from pathlib import Path

import torch

from scripts.distill_search_replay import (
    discover_search_replay_manifests,
    parameter_hashes,
)


def test_parameter_hashes_track_only_selected_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    selected = {"0.weight", "1.bias"}
    before = parameter_hashes(model, selected)

    with torch.no_grad():
        model[0].bias.add_(1.0)
    assert parameter_hashes(model, selected) == before

    with torch.no_grad():
        model[0].weight.add_(1.0)
    assert parameter_hashes(model, selected) != before


def test_discovers_iterative_and_tournament_replay_layouts(tmp_path: Path) -> None:
    iterative = tmp_path / "iterative" / "shard-000" / "manifest.json"
    tournament = tmp_path / "tournament" / "shard-001" / "search_replay" / "manifest.json"
    iterative.parent.mkdir(parents=True)
    tournament.parent.mkdir(parents=True)
    iterative.write_text("{}\n", encoding="utf-8")
    tournament.write_text("{}\n", encoding="utf-8")

    assert discover_search_replay_manifests([], [tmp_path / "iterative"]) == [iterative.resolve()]
    assert discover_search_replay_manifests([], [tmp_path / "tournament"]) == [tournament.resolve()]


def test_accepts_a_manifest_or_replay_directory_directly(tmp_path: Path) -> None:
    manifest = tmp_path / "replay" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")

    expected = [manifest.resolve()]
    assert discover_search_replay_manifests([], [manifest]) == expected
    assert discover_search_replay_manifests([], [manifest.parent]) == expected
