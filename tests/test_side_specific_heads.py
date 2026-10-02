from __future__ import annotations

import numpy as np
import pytest

from struggle_ai.train_rllib import copy_compatible_or_widened_weights


def test_legacy_shared_heads_migrate_to_both_side_heads() -> None:
    current = {
        "policy_query_us.weight": np.zeros((3, 3), dtype=np.float32),
        "policy_query_ussr.weight": np.zeros((3, 3), dtype=np.float32),
        "action_bias_us.bias": np.zeros((1,), dtype=np.float32),
        "action_bias_ussr.bias": np.zeros((1,), dtype=np.float32),
        "value.2.weight": np.zeros((2, 3), dtype=np.float32),
        "value.2.bias": np.zeros((2,), dtype=np.float32),
    }
    source = {
        "policy_query.weight": np.full((3, 3), 2.0, dtype=np.float32),
        "action_bias.bias": np.array([5.0], dtype=np.float32),
        "value.2.weight": np.full((1, 3), 7.0, dtype=np.float32),
        "value.2.bias": np.array([11.0], dtype=np.float32),
    }

    updated, matched, widened, skipped, missing, unexpected = copy_compatible_or_widened_weights(current, source)

    np.testing.assert_array_equal(updated["policy_query_us.weight"], source["policy_query.weight"])
    np.testing.assert_array_equal(updated["policy_query_ussr.weight"], source["policy_query.weight"])
    np.testing.assert_array_equal(updated["action_bias_us.bias"], source["action_bias.bias"])
    np.testing.assert_array_equal(updated["action_bias_ussr.bias"], source["action_bias.bias"])
    np.testing.assert_array_equal(updated["value.2.weight"], np.repeat(source["value.2.weight"], 2, axis=0))
    np.testing.assert_array_equal(updated["value.2.bias"], np.repeat(source["value.2.bias"], 2, axis=0))
    assert len(matched) == 4
    assert len(widened) == 2
    assert skipped == []
    assert missing == []
    assert unexpected == []


def test_shared_state_encoder_migrates_to_critic_state_tower() -> None:
    current = {
        "state_encoder.0.weight": np.zeros((3, 4), dtype=np.float32),
        "value_state_encoder.0.weight": np.zeros((3, 4), dtype=np.float32),
        "policy_query_us.weight": np.zeros((3, 3), dtype=np.float32),
    }
    source = {
        "state_encoder.0.weight": np.full((3, 4), 2.0, dtype=np.float32),
        "policy_query_us.weight": np.full((3, 3), 5.0, dtype=np.float32),
    }

    updated, matched, widened, skipped, missing, unexpected = copy_compatible_or_widened_weights(
        current, source
    )

    np.testing.assert_array_equal(
        updated["value_state_encoder.0.weight"], source["state_encoder.0.weight"]
    )
    assert sorted(matched) == sorted(current)
    assert widened == []
    assert skipped == []
    assert missing == []
    assert unexpected == []


def test_side_specific_model_routes_actor_and_value_by_observation_side() -> None:
    torch = pytest.importorskip("torch")
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
            "side-head-test",
        )
        for parameter in model.parameters():
            parameter.data.zero_()
        model.action_bias_us.bias.data.fill_(1.0)
        model.action_bias_ussr.bias.data.fill_(-1.0)
        model.value[2].bias.data.copy_(torch.tensor([3.0, -4.0]))

        obs = {
            key: torch.zeros((2,) + space.shape, dtype=torch.float32)
            for key, space in env.observation_space.spaces.items()
        }
        obs["global"][:, 0] = torch.tensor([1.0, -1.0])
        obs["action_mask"].fill_(1.0)

        logits, _ = model({"obs": obs}, [], None)

        assert torch.allclose(logits[0], torch.ones_like(logits[0]))
        assert torch.allclose(logits[1], -torch.ones_like(logits[1]))
        assert torch.allclose(model.value_function(), torch.tensor([3.0, -4.0]))
    finally:
        env.close()


def test_zero_initialized_value_adapter_preserves_actor_and_value() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")

    from struggle_ai.env import TwilightStruggleEnv
    from struggle_ai.rllib_masked_model import TorchMaskedActions

    env = TwilightStruggleEnv({"log_games_dir": None, "metrics_dir": None})
    try:
        base_config = {
            "hidden": 32,
            "graph_layers": 1,
            "side_specific_heads": True,
        }
        torch.manual_seed(17)
        base = TorchMaskedActions(
            env.observation_space,
            env.action_space,
            env.action_space.n,
            {"custom_model_config": base_config},
            "base-value-test",
        )
        torch.manual_seed(17)
        adapted = TorchMaskedActions(
            env.observation_space,
            env.action_space,
            env.action_space.n,
            {"custom_model_config": {**base_config, "value_adapter_hidden": 8}},
            "adapted-value-test",
        )
        missing, unexpected = adapted.load_state_dict(base.state_dict(), strict=False)
        assert missing == [
            "value_adapter.0.weight",
            "value_adapter.0.bias",
            "value_adapter.2.weight",
            "value_adapter.2.bias",
        ]
        assert unexpected == []

        obs = {
            key: torch.zeros((2,) + space.shape, dtype=torch.float32)
            for key, space in env.observation_space.spaces.items()
        }
        obs["global"][:, 0] = torch.tensor([1.0, -1.0])
        obs["action_mask"].fill_(1.0)
        base.eval()
        adapted.eval()
        with torch.no_grad():
            base_logits, _ = base({"obs": obs}, [], None)
            base_value = base.value_function().clone()
            adapted_logits, _ = adapted({"obs": obs}, [], None)
            adapted_value = adapted.value_function().clone()
        assert torch.equal(adapted_logits, base_logits)
        assert torch.equal(adapted_value, base_value)
    finally:
        env.close()


def test_migrated_critic_state_tower_preserves_actor_and_value() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")

    from struggle_ai.env import TwilightStruggleEnv
    from struggle_ai.rllib_masked_model import TorchMaskedActions

    env = TwilightStruggleEnv({"log_games_dir": None, "metrics_dir": None})
    try:
        config = {"hidden": 32, "graph_layers": 1, "side_specific_heads": True}
        torch.manual_seed(23)
        base = TorchMaskedActions(
            env.observation_space,
            env.action_space,
            env.action_space.n,
            {"custom_model_config": config},
            "base-tower-test",
        )
        torch.manual_seed(91)
        tower = TorchMaskedActions(
            env.observation_space,
            env.action_space,
            env.action_space.n,
            {"custom_model_config": {**config, "value_state_tower": True}},
            "critic-tower-test",
        )
        current = {name: value.detach().numpy() for name, value in tower.state_dict().items()}
        source = {name: value.detach().numpy() for name, value in base.state_dict().items()}
        migrated, _matched, _widened, skipped, missing, unexpected = (
            copy_compatible_or_widened_weights(current, source)
        )
        assert skipped == []
        assert missing == []
        assert unexpected == []
        tower.load_state_dict({name: torch.as_tensor(value) for name, value in migrated.items()})

        obs = {
            key: torch.zeros((2,) + space.shape, dtype=torch.float32)
            for key, space in env.observation_space.spaces.items()
        }
        obs["global"][:, 0] = torch.tensor([1.0, -1.0])
        obs["action_mask"].fill_(1.0)
        base.eval()
        tower.eval()
        with torch.no_grad():
            base_logits, _ = base({"obs": obs}, [], None)
            base_value = base.value_function().clone()
            tower_logits, _ = tower({"obs": obs}, [], None)
            tower_value = tower.value_function().clone()
        assert torch.equal(tower_logits, base_logits)
        assert torch.equal(tower_value, base_value)
    finally:
        env.close()
