from __future__ import annotations

import pytest

from struggle_ai.train_rllib import configure_focused_head_only, focused_head_parameter_is_trainable


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("policy_query_us.weight", True),
        ("action_bias_us.bias", True),
        ("value.0.weight", True),
        ("policy_query_ussr.weight", False),
        ("action_bias_ussr.bias", False),
        ("state_encoder.0.weight", False),
        ("action_encoder.0.weight", False),
    ],
)
def test_us_focused_head_parameter_selection(name: str, expected: bool) -> None:
    assert focused_head_parameter_is_trainable(name, "us") is expected


def test_ussr_focused_head_parameter_selection() -> None:
    assert focused_head_parameter_is_trainable("policy_query_ussr.weight", "ussr")
    assert focused_head_parameter_is_trainable("action_bias_ussr.bias", "ussr")
    assert not focused_head_parameter_is_trainable("policy_query_us.weight", "ussr")


def test_focused_head_parameter_selection_rejects_unknown_side() -> None:
    with pytest.raises(ValueError, match="unsupported focused side"):
        focused_head_parameter_is_trainable("value.0.weight", "neutral")


def test_configure_focused_head_only_freezes_actual_shared_encoder() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("ray")

    from struggle_ai.env import TwilightStruggleEnv
    from struggle_ai.rllib_masked_model import TorchMaskedActions

    env = TwilightStruggleEnv({"log_games_dir": None, "metrics_dir": None})
    try:
        model = TorchMaskedActions(
            env.observation_space,
            env.action_space,
            env.action_space.n,
            {"custom_model_config": {"hidden": 32, "graph_layers": 1, "side_specific_heads": True}},
            "focused-head-test",
        )
        report = configure_focused_head_only(model, "us")
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        assert report["trainable_parameters"] > 0
        assert report["frozen_parameters"] > report["trainable_parameters"]
        assert "policy_query_us.weight" in trainable
        assert "action_bias_us.weight" in trainable
        assert "value.0.weight" in trainable
        assert "policy_query_ussr.weight" not in trainable
        assert "global_encoder.0.weight" not in trainable
        assert "action_encoder.0.weight" not in trainable
    finally:
        env.close()
