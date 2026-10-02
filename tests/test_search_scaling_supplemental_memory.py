from scripts.watch_search_scaling_supplemental_memory import OUTPUT


def test_supplemental_memory_guard_targets_only_optional_shard() -> None:
    assert "w4" in OUTPUT.name
    assert "w12" not in OUTPUT.name
