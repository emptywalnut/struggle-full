from scripts.compose_policy_value_checkpoint import is_value_tensor


def test_value_tensor_scope_includes_critic_state_tower() -> None:
    assert is_value_tensor("value.0.weight")
    assert is_value_tensor("value_state_encoder.0.weight")
    assert not is_value_tensor("state_encoder.0.weight")
    assert not is_value_tensor("policy_query_us.weight")
