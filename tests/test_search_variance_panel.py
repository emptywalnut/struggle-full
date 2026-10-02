from scripts.build_search_variance_panel import select_panel


def test_select_panel_balances_side_and_era_and_avoids_exclusions() -> None:
    positions = []
    evidence = {}
    seed = 1
    for side in ("us", "ussr"):
        for era in ("early", "mid", "late"):
            for index, category in enumerate(("scoring_card", "coup_target", "opponent_event")):
                row = {
                    "seed": seed,
                    "action_index": index,
                    "side": side,
                    "era": era,
                    "category": category,
                    "branch_bucket": "low",
                    "legal_count": index + 2,
                }
                positions.append(row)
                evidence[(seed, index)] = {"status": "UNRESOLVED"}
            seed += 1
    excluded = {(1, 0)}
    selected = select_panel(
        positions,
        evidence,
        per_side_era=2,
        max_legal_count=8,
        excluded=excluded,
    )
    assert len(selected) == 12
    assert all((row["seed"], row["action_index"]) not in excluded for row in selected)
    assert {(row["side"], row["era"]) for row in selected} == {
        (side, era) for side in ("us", "ussr") for era in ("early", "mid", "late")
    }
