from scripts.aggregate_paired_eval_shards import aggregate_payloads


IDENTITY = {
    "candidate_policy": "/candidate",
    "opponent_policy": "/opponent",
    "seed_base": 1,
    "seed_pairs": 1,
    "log_games": True,
}


def record(seed: int, orientation: str, score: float) -> dict:
    side = "us" if orientation == "candidate_us" else "ussr"
    return {
        "seed": seed,
        "orientation": orientation,
        "candidate_side": side,
        "candidate_score": score,
        "candidate_won": score == 1.0,
        "draw": score == 0.5,
        "terminal_reason": "test",
        "wall_clock_seconds": 1.0,
        "searched_decisions": 1,
        "simulations": 32,
        "action_flips": 0,
    }


def test_aggregate_keeps_only_complete_seed_pairs() -> None:
    result = aggregate_payloads(
        [
            {
                "identity": IDENTITY,
                "records": [record(1, "candidate_us", 1.0), record(1, "candidate_ussr", 0.0)],
            },
            {
                "identity": {**IDENTITY, "seed_base": 2},
                "records": [record(2, "candidate_us", 1.0)],
            },
        ],
        complete_pairs_only=True,
        bootstrap_samples=100,
    )
    assert result["summary"]["games"] == 2
    assert result["summary"]["score"] == 0.5
    assert result["excluded_unpaired_records"] == 1


def test_aggregate_rejects_incompatible_agents() -> None:
    try:
        aggregate_payloads(
            [
                {"identity": IDENTITY, "records": [record(1, "candidate_us", 1.0)]},
                {
                    "identity": {**IDENTITY, "candidate_policy": "/different"},
                    "records": [record(2, "candidate_ussr", 0.0)],
                },
            ],
            complete_pairs_only=False,
            bootstrap_samples=100,
        )
    except ValueError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("expected incompatible shard rejection")
