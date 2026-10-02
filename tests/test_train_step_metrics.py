from __future__ import annotations

from struggle_ai.train_rllib import sampled_steps_for_iteration


def test_sampled_steps_prefers_explicit_per_iteration_counter() -> None:
    result = {
        "num_env_steps_sampled_lifetime": 12_400,
        "num_env_steps_sampled_this_iter": 2_400,
    }

    assert sampled_steps_for_iteration(result, 10_000) == (2_400, 12_400)


def test_sampled_steps_supports_nested_env_runner_metrics() -> None:
    result = {
        "env_runners": {
            "num_env_steps_sampled": 2_048,
            "num_env_steps_sampled_lifetime": 9_048,
        }
    }

    assert sampled_steps_for_iteration(result, 7_000) == (2_048, 9_048)


def test_sampled_steps_falls_back_to_lifetime_delta() -> None:
    result = {"num_env_steps_sampled_lifetime": 7_500}

    assert sampled_steps_for_iteration(result, 7_000) == (500, 7_500)
    assert sampled_steps_for_iteration(result, None) == (0, 7_500)
