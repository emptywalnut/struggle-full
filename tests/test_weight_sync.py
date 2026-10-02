import argparse

import numpy as np

import struggle_ai.train_rllib as train_rllib
from struggle_ai.train_rllib import (
    SHARED_POLICY_ID,
    best_anchor_head_to_head,
    checkpoint_acceptance,
    clear_policy_optimizer_state,
    pull_focused_actor_toward_anchor,
    sync_algorithm_weights,
    update_unified_best_anchor,
)


class SyncGroup:
    def __init__(self):
        self.calls = 0

    def sync_weights(self):
        self.calls += 1


def test_sync_algorithm_weights_prefers_env_runner_group():
    env_runners = SyncGroup()
    workers = SyncGroup()
    algo = type(
        "Algo",
        (),
        {
            "env_runner_group": env_runners,
            "workers": lambda self: workers,
        },
    )()

    assert sync_algorithm_weights(algo) == "env_runner_group"
    assert env_runners.calls == 1
    assert workers.calls == 0


def test_sync_algorithm_weights_supports_callable_workers_fallback():
    workers = SyncGroup()
    algo = type("Algo", (), {"workers": lambda self: workers})()

    assert sync_algorithm_weights(algo) == "workers"
    assert workers.calls == 1


class Policy:
    def __init__(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


def test_pull_focused_actor_toward_anchor_changes_only_requested_actor():
    policy = Policy(
        {
            "policy_query_us.weight": np.array([[4.0, 2.0]], dtype=np.float32),
            "action_bias_us.bias": np.array([2.0], dtype=np.float32),
            "policy_query_ussr.weight": np.array([[7.0]], dtype=np.float32),
            "value.2.bias": np.array([3.0, 5.0], dtype=np.float32),
        }
    )
    anchor = {
        "policy_query_us.weight": np.array([[0.0, 0.0]], dtype=np.float32),
        "action_bias_us.bias": np.array([0.0], dtype=np.float32),
        "policy_query_ussr.weight": np.array([[1.0]], dtype=np.float32),
        "value.2.bias": np.array([1.0, 1.0], dtype=np.float32),
    }

    report = pull_focused_actor_toward_anchor(policy, anchor, "us", 0.25)

    np.testing.assert_allclose(policy.weights["policy_query_us.weight"], [[3.0, 1.5]])
    np.testing.assert_allclose(policy.weights["action_bias_us.bias"], [1.5])
    np.testing.assert_allclose(policy.weights["policy_query_ussr.weight"], [[7.0]])
    np.testing.assert_allclose(policy.weights["value.2.bias"], [3.0, 5.0])
    assert report["tensor_count"] == 2
    assert report["parameter_count"] == 3
    assert report["distance_after_l2"] == report["distance_before_l2"] * 0.75


def test_update_unified_best_anchor_copies_primary_anchor_and_syncs_workers():
    workers = SyncGroup()
    policies = {
        SHARED_POLICY_ID: Policy({"weight": [1, 2, 3]}),
        "anchor_0": Policy({"weight": [0]}),
        "anchor_1": Policy({"weight": [9]}),
    }
    algo = type(
        "Algo",
        (),
        {
            "workers": lambda self: workers,
            "get_policy": lambda self, policy_id: policies[policy_id],
        },
    )()

    assert update_unified_best_anchor(algo, ["anchor_0", "anchor_1"]) == ("anchor_0", "workers")
    assert policies["anchor_0"].weights == {"weight": [1, 2, 3]}
    assert policies["anchor_1"].weights == {"weight": [9]}
    assert policies["anchor_0"].weights is not policies[SHARED_POLICY_ID].weights
    assert workers.calls == 1


def test_best_anchor_head_to_head_requires_overall_win_and_nonlosing_sides():
    metrics = {
        "eval_vs_best_anchor/win_rate": 0.55,
        "eval_vs_best_anchor/us_side_win_rate": 0.50,
        "eval_vs_best_anchor/ussr_side_win_rate": 0.60,
    }
    result = best_anchor_head_to_head(metrics)

    assert result is not None
    assert result["side_floor_win_rate"] == 0.50
    assert result["passed"] is True


def test_best_anchor_head_to_head_rejects_one_sided_candidate():
    metrics = {
        "eval_vs_best_anchor/win_rate": 0.60,
        "eval_vs_best_anchor/us_side_win_rate": 0.20,
        "eval_vs_best_anchor/ussr_side_win_rate": 1.00,
    }

    assert best_anchor_head_to_head(metrics)["passed"] is False


def test_checkpoint_acceptance_uses_anchor_gate_even_when_composite_is_lower():
    metrics = {
        "eval_vs_best_anchor/win_rate": 0.55,
        "eval_vs_best_anchor/us_side_win_rate": 0.50,
        "eval_vs_best_anchor/ussr_side_win_rate": 0.60,
    }
    args = argparse.Namespace(
        best_anchor_min_side_win_rate=0.50,
        best_anchor_min_overall_win_rate=0.50,
        early_stop_min_delta=0.0,
    )

    accepted, min_delta_met, comparison = checkpoint_acceptance(
        metrics,
        {"score": 1400.0},
        1900.0,
        args,
    )

    assert accepted is True
    assert min_delta_met is True
    assert comparison["passed"] is True


def test_checkpoint_acceptance_rejects_one_sided_final_candidate():
    metrics = {
        "eval_vs_best_anchor/win_rate": 0.60,
        "eval_vs_best_anchor/us_side_win_rate": 0.20,
        "eval_vs_best_anchor/ussr_side_win_rate": 1.00,
    }
    args = argparse.Namespace(
        best_anchor_min_side_win_rate=0.50,
        best_anchor_min_overall_win_rate=0.50,
        early_stop_min_delta=0.0,
    )

    accepted, min_delta_met, comparison = checkpoint_acceptance(
        metrics,
        {"score": 2000.0},
        1900.0,
        args,
    )

    assert accepted is False
    assert min_delta_met is False
    assert comparison["passed"] is False


def test_anchor_confirmation_replaces_noisy_preliminary_result(monkeypatch):
    metrics = {
        "eval_vs_best_anchor/games_per_side": 10,
        "eval_vs_best_anchor/win_rate": 0.75,
        "eval_vs_best_anchor/us_side_win_rate": 0.60,
        "eval_vs_best_anchor/ussr_side_win_rate": 0.90,
    }
    args = argparse.Namespace(
        best_anchor_min_side_win_rate=0.50,
        best_anchor_min_overall_win_rate=0.50,
        best_anchor_confirm_games_per_side=30,
        eval_max_episode_steps=1200,
        eval_progress_every_games=10,
        elo_k_factor=32.0,
    )
    calls = []

    def fake_evaluate_policy(**kwargs):
        calls.append(kwargs)
        return {
            "eval_vs_best_anchor/games_per_side": 30,
            "eval_vs_best_anchor/win_rate": 0.5833333333333334,
            "eval_vs_best_anchor/us_side_win_rate": 0.26666666666666666,
            "eval_vs_best_anchor/ussr_side_win_rate": 0.90,
        }

    monkeypatch.setattr(train_rllib, "evaluate_policy", fake_evaluate_policy)
    confirmed = train_rllib.confirm_best_anchor_candidate(
        algo=object(),
        eval_metrics=metrics,
        benchmark_weights={"best_anchor": {"shared_policy": {}}},
        env_config={},
        args=args,
        side_policy_ids={"us": "shared_policy", "ussr": "shared_policy"},
        eval_label="candidate",
        seed_base=123,
    )

    assert confirmed["passed"] is False
    assert metrics["best_anchor/preliminary_passed"] is True
    assert metrics["best_anchor/confirmation_passed"] is False
    assert metrics["eval_vs_best_anchor/games_per_side"] == 30
    assert calls[0]["include_random"] is False


def test_clear_policy_optimizer_state_clears_each_optimizer():
    optimizers = [
        type("Optimizer", (), {"state": {"first": 1}})(),
        type("Optimizer", (), {"state": {"second": 2}})(),
    ]
    policy = type("PolicyWithOptimizers", (), {"_optimizers": optimizers})()

    assert clear_policy_optimizer_state(policy) == 2
    assert all(not optimizer.state for optimizer in optimizers)
