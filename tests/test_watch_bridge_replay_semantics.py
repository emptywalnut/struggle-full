from scripts.watch_bridge_replay_semantics import retain_current_reports


def test_retain_current_reports_removes_quarantined_game() -> None:
    reports = {
        "game-a": {"source_game": "game-a", "passed": True},
        "game-b": {"source_game": "game-b", "passed": True},
    }

    retained, removed = retain_current_reports(reports, {"game-a"})

    assert retained == {"game-a": reports["game-a"]}
    assert removed == {"game-b"}


def test_retain_current_reports_is_stable_without_removals() -> None:
    reports = {"game-a": {"source_game": "game-a", "passed": True}}

    retained, removed = retain_current_reports(reports, {"game-a"})

    assert retained == reports
    assert removed == set()
