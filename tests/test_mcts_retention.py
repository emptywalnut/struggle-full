from collections import deque

from struggle_ai.train_mcts import (
    mcts_candidate_improves_elo_floor,
    mcts_eval_elo_ratings,
    rollback_mcts_candidate,
)


def test_mcts_candidate_requires_dual_side_floor_improvement():
    retained = {"us": 1742.0, "ussr": 1751.0}
    assert mcts_candidate_improves_elo_floor(
        {"us": 1752.0, "ussr": 1760.0}, retained
    )
    assert not mcts_candidate_improves_elo_floor(
        {"us": 1741.0, "ussr": 1900.0}, retained
    )
    assert not mcts_candidate_improves_elo_floor(
        {"us": 1752.0}, retained
    )


def test_mcts_eval_elo_ratings_extracts_available_channels():
    assert mcts_eval_elo_ratings(
        {
            "elo/us/current_rating": 1752.0,
            "elo/ussr/current_rating": 1760.0,
        }
    ) == {"us": 1752.0, "ussr": 1760.0}


def test_rollback_mcts_candidate_restores_weights_and_clears_state():
    class Policy:
        def __init__(self):
            self.weights = {"w": [9]}

        def set_weights(self, weights):
            self.weights = weights

    class Optimizer:
        state = {"moment": 1}

    policy = Policy()
    optimizer = Optimizer()
    replay = deque([1, 2, 3])
    retained = {"w": [1]}

    report = rollback_mcts_candidate(policy, optimizer, replay, retained)

    assert policy.weights == retained
    assert policy.weights is not retained
    assert optimizer.state == {}
    assert list(replay) == []
    assert report == {
        "optimizer_states_cleared": 1,
        "replay_items_cleared": 3,
    }
