from __future__ import annotations

import numpy as np

from struggle_ai.mcts import ChanceAwareMCTS, SearchNode, calibrate_value, information_state_key
from struggle_ai.train_mcts import SearchSample, sample_side_balanced_replay


class DiceChoiceEnv:
    def __init__(self) -> None:
        self.last_obs = {"side": "us"}
        self.legal_actions = [{"value": "risky"}, {"value": "lose"}]

    @staticmethod
    def observation() -> dict[str, np.ndarray]:
        return {
            "action_mask": np.array([1.0, 1.0, 0.0], dtype=np.float32),
            "state": np.array([1.0], dtype=np.float32),
        }

    def restore(self, snapshot):
        self.last_obs = {"side": "us"}
        return self.observation()

    def redeterminize(self, side: str, seed: int):
        assert side == "us"
        return self.observation()

    def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
        roll = int(float(forced_rolls[0]) * 6) + 1
        if action == 0:
            winner = "us" if roll >= 3 else "ussr"
        else:
            winner = "ussr"
        info = {"winner": winner, "chance_rolls": [roll], "forced_rolls_consumed": forced_rolls[:1]}
        return self.observation(), 0.0, True, False, info


def uniform_evaluator(obs: dict[str, np.ndarray], legal_count: int):
    return np.ones(legal_count, dtype=np.float32), 0.0


def test_search_prefers_higher_expected_dice_action() -> None:
    env = DiceChoiceEnv()
    search = ChanceAwareMCTS(
        uniform_evaluator,
        simulations=256,
        max_depth=2,
        cpuct=1.5,
        dirichlet_fraction=0.0,
        seed=9,
    )
    result = search.search(env, {"root": True}, env.observation(), temperature=0.0)
    assert result.action == 0
    assert result.visits[0] > result.visits[1]
    assert sum(result.chance_roll_histogram.values()) == 256


def test_ussr_node_minimizes_absolute_us_value() -> None:
    node = SearchNode.create("ussr", np.array([0.5, 0.5]))
    node.visits[:] = [4, 4]
    node.value_sum_us[:] = [3.0, -2.0]
    assert node.select(cpuct=0.0) == 1


def test_inactive_actions_are_never_selected() -> None:
    node = SearchNode.create(
        "us",
        np.array([0.7, 0.3, 0.0]),
        np.array([True, True, False]),
    )
    node.visits[:] = [1, 1, 0]
    node.value_sum_us[:] = [-1.0, -2.0, 0.0]
    assert node.select(cpuct=0.0) == 0


def test_policy_guided_expansion_limits_root_children() -> None:
    env = DiceChoiceEnv()

    def ranked_evaluator(obs: dict[str, np.ndarray], legal_count: int):
        return np.array([0.9, 0.1], dtype=np.float32)[:legal_count], 0.0

    search = ChanceAwareMCTS(
        ranked_evaluator,
        simulations=16,
        max_depth=2,
        max_children=1,
        dirichlet_fraction=0.25,
        seed=17,
    )
    result = search.search(env, {"root": True}, env.observation(), temperature=1.0)
    assert result.visits[0] == 16
    assert result.visits[1] == 0
    assert result.action == 0


def test_value_calibration_preserves_order_beyond_terminal_scale() -> None:
    assert -1.0 < calibrate_value(-4.0) < calibrate_value(-2.0) < 0.0
    assert 0.0 < calibrate_value(2.0) < calibrate_value(4.0) < 1.0
    assert calibrate_value(float("nan")) == 0.0


def test_information_state_hash_uses_visible_tensor_content() -> None:
    first = {"x": np.array([1, 2], dtype=np.float32)}
    second = {"x": np.array([1, 3], dtype=np.float32)}
    assert information_state_key(first, "us") == information_state_key(first, "us")
    assert information_state_key(first, "us") != information_state_key(second, "us")
    assert information_state_key(first, "us") != information_state_key(first, "ussr")


def replay_sample(side: str, marker: int) -> SearchSample:
    return SearchSample(
        obs={"marker": np.array([marker], dtype=np.float32)},
        policy=np.array([1.0], dtype=np.float32),
        side=side,
        root_value_us=0.0,
    )


def test_replay_sampling_balances_acting_sides() -> None:
    population = [replay_sample("us", index) for index in range(20)]
    population.extend(replay_sample("ussr", index + 20) for index in range(20))

    batch = sample_side_balanced_replay(population, batch_size=15, rng=np_random_adapter(7))

    assert len(batch) == 15
    assert sum(sample.side == "us" for sample in batch) == 7
    assert sum(sample.side == "ussr" for sample in batch) == 8
    assert len({int(sample.obs["marker"][0]) for sample in batch}) == 15


def test_replay_sampling_fills_from_available_side() -> None:
    population = [replay_sample("us", index) for index in range(2)]
    population.extend(replay_sample("ussr", index + 2) for index in range(20))

    batch = sample_side_balanced_replay(population, batch_size=10, rng=np_random_adapter(11))

    assert len(batch) == 10
    assert sum(sample.side == "us" for sample in batch) == 2
    assert sum(sample.side == "ussr" for sample in batch) == 8


def test_replay_sampling_can_focus_one_acting_side() -> None:
    population = [replay_sample("us", index) for index in range(20)]
    population.extend(replay_sample("ussr", index + 20) for index in range(20))

    batch = sample_side_balanced_replay(
        population,
        batch_size=15,
        rng=np_random_adapter(13),
        focus_side="us",
    )

    assert len(batch) == 15
    assert all(sample.side == "us" for sample in batch)
    assert len({int(sample.obs["marker"][0]) for sample in batch}) == 15


def np_random_adapter(seed: int):
    import random

    return random.Random(seed)
