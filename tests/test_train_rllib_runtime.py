from __future__ import annotations

import sys

from struggle_ai.train_rllib import linear_annealed_value, parse_args


def test_ray_runtime_limits_are_configurable(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rllib.py",
            "--num-env-runners",
            "16",
            "--ray-num-cpus",
            "20",
            "--torch-num-threads",
            "1",
        ],
    )
    args = parse_args()
    assert args.num_env_runners == 16
    assert args.ray_num_cpus == 20
    assert args.torch_num_threads == 1


def test_linear_annealed_value_clamps_to_endpoints() -> None:
    assert linear_annealed_value(2.0, 1.0, -1, 100) == 2.0
    assert linear_annealed_value(2.0, 1.0, 50, 100) == 1.5
    assert linear_annealed_value(2.0, 1.0, 200, 100) == 1.0
