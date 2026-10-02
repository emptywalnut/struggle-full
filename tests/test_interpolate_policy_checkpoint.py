from __future__ import annotations

import numpy as np
import pytest

from scripts.interpolate_policy_checkpoint import SCOPES, interpolate_weights


def test_interpolate_weights_changes_only_selected_prefixes() -> None:
    parent = {
        "encoder.weight": np.asarray([1.0, 2.0], dtype=np.float32),
        "policy_query.weight": np.asarray([0.0, 2.0], dtype=np.float32),
        "value.weight": np.asarray([4.0], dtype=np.float32),
    }
    student = {
        "encoder.weight": np.asarray([9.0, 9.0], dtype=np.float32),
        "policy_query.weight": np.asarray([4.0, 6.0], dtype=np.float32),
        "value.weight": np.asarray([0.0], dtype=np.float32),
    }

    blended, changed = interpolate_weights(
        parent,
        student,
        alpha=0.25,
        prefixes=("policy_query", "value"),
    )

    np.testing.assert_array_equal(blended["encoder.weight"], parent["encoder.weight"])
    np.testing.assert_allclose(blended["policy_query.weight"], [1.0, 3.0])
    np.testing.assert_allclose(blended["value.weight"], [3.0])
    assert {entry["name"] for entry in changed} == {
        "policy_query.weight",
        "value.weight",
    }


def test_interpolate_weights_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="alpha"):
        interpolate_weights(
            {"policy.weight": np.ones(1)},
            {"policy.weight": np.ones(1)},
            alpha=1.1,
            prefixes=("policy",),
        )


def test_value_tower_scope_excludes_shared_actor_state_encoder() -> None:
    assert SCOPES["value_tower"] == ("value_state_encoder",)
    assert SCOPES["value_tower_head"] == ("value_state_encoder", "value")
    with pytest.raises(ValueError, match="keys differ"):
        interpolate_weights(
            {"policy.weight": np.ones(1)},
            {"other.weight": np.ones(1)},
            alpha=0.5,
            prefixes=("policy",),
        )


def test_action_head_scope_includes_action_encoder_and_side_heads() -> None:
    assert "action_encoder" in SCOPES["action_head"]
    assert "policy_query_us" in SCOPES["action_head"]
    assert "action_bias_ussr" in SCOPES["action_head"]
    assert "value" not in SCOPES["action_head"]
