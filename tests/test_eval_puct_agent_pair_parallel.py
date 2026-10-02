from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.eval_puct_agent_pair_parallel import (
    adaptive_search_configs,
    checkpoint_paths,
    load_resumable_records,
    proxy_observation,
    resolve_adaptive_modes,
    resolve_search_modes,
    resolved_search_config,
    summary,
    worker_error_detail,
    write_worker_heartbeat,
    write_progress_snapshot,
    write_failure,
)
from struggle_ai.runtime_limits import required_search_recursion_limit


def test_search_modes_allow_backward_compatible_and_asymmetric_search() -> None:
    assert resolve_search_modes(
        SimpleNamespace(search=True, candidate_search=None, opponent_search=None)
    ) == {"candidate": True, "opponent": True}
    assert resolve_search_modes(
        SimpleNamespace(search=False, candidate_search=True, opponent_search=None)
    ) == {"candidate": True, "opponent": False}
    assert resolve_search_modes(
        SimpleNamespace(search=True, candidate_search=False, opponent_search=True)
    ) == {"candidate": False, "opponent": True}


def test_noncheckpoint_opponent_only_loads_candidate_weights(tmp_path) -> None:
    candidate = tmp_path / "candidate"
    args = SimpleNamespace(
        candidate_checkpoint=candidate,
        candidate_value_checkpoint=None,
        candidate_us_policy_state=None,
        candidate_ussr_policy_state=None,
        opponent_kind="steam_proxy",
        opponent_checkpoint=None,
        opponent_value_checkpoint=None,
    )
    assert checkpoint_paths(args) == {
        "candidate_policy": str(candidate.resolve()),
        "candidate_value": str(candidate.resolve()),
    }
    args.opponent_kind = "checkpoint"
    with pytest.raises(ValueError, match="opponent-checkpoint"):
        checkpoint_paths(args)


def test_side_selected_candidate_loads_policy_state_per_side(tmp_path) -> None:
    us = tmp_path / "us.pkl"
    ussr = tmp_path / "ussr.pkl"
    args = SimpleNamespace(
        candidate_checkpoint=tmp_path / "unused",
        candidate_value_checkpoint=None,
        candidate_us_policy_state=us,
        candidate_ussr_policy_state=ussr,
        opponent_kind="steam_proxy",
        opponent_checkpoint=None,
        opponent_value_checkpoint=None,
    )
    assert checkpoint_paths(args) == {
        "candidate_policy_us": str(us.resolve()),
        "candidate_value_us": str(us.resolve()),
        "candidate_policy_ussr": str(ussr.resolve()),
        "candidate_value_ussr": str(ussr.resolve()),
    }


def test_proxy_observation_uses_safety_filtered_action_catalog() -> None:
    raw = {"side": "us", "legal_actions": [{"value": "unsafe"}]}
    filtered = [{"value": "space"}]
    assert proxy_observation(raw, filtered) == {"side": "us", "legal_actions": filtered}
    assert raw["legal_actions"] == [{"value": "unsafe"}]


def test_deep_eval_worker_recursion_headroom_and_error_context() -> None:
    assert required_search_recursion_limit(1536) == 8192
    assert worker_error_detail(
        {
            "worker_id": 4,
            "error": "RecursionError('boom')",
            "traceback": "Traceback: eval stack",
            "recursion_limit": 8192,
        }
    ) == (
        "worker 4: RecursionError('boom')\n"
        "Traceback: eval stack\nworker_recursion_limit=8192"
    )


def test_adaptive_modes_allow_asymmetric_search_scaling() -> None:
    assert resolve_adaptive_modes(
        SimpleNamespace(
            adaptive_converged_search=False,
            candidate_adaptive_converged_search=True,
            opponent_adaptive_converged_search=None,
        )
    ) == {"candidate": True, "opponent": False}


def test_resolved_search_config_supports_per_agent_depth_and_budget() -> None:
    args = SimpleNamespace(
        initial_simulations=32,
        medium_simulations=32,
        max_simulations=32,
        meaningful_depth=2,
        max_atomic_steps=512,
        max_expanded_nodes=512,
        candidate_simulations=128,
        candidate_meaningful_depth=4,
        candidate_max_atomic_steps=1024,
        candidate_max_expanded_nodes=2048,
        opponent_simulations=None,
        opponent_meaningful_depth=None,
        opponent_max_atomic_steps=None,
        opponent_max_expanded_nodes=None,
    )
    assert resolved_search_config(args, "candidate") == {
        "initial_simulations": 128,
        "medium_simulations": 128,
        "max_simulations": 128,
        "meaningful_depth": 4,
        "max_atomic_steps": 1024,
        "max_expanded_nodes": 2048,
    }
    assert resolved_search_config(args, "opponent")["meaningful_depth"] == 2


def test_adaptive_search_configs_separate_budget_from_depth() -> None:
    args = SimpleNamespace(
        initial_simulations=32,
        medium_simulations=32,
        max_simulations=32,
        meaningful_depth=2,
        max_atomic_steps=1536,
        max_expanded_nodes=32768,
        belief_particles=32,
        candidate_simulations=None,
        candidate_meaningful_depth=None,
        candidate_max_atomic_steps=None,
        candidate_max_expanded_nodes=None,
        adaptive_budget_simulations=128,
        adaptive_deep_depth=4,
        adaptive_deep_simulations=128,
        adaptive_deep_max_expanded_nodes=262144,
    )
    base, budget, deep = adaptive_search_configs(args, "candidate")
    assert (base.meaningful_depth, base.max_simulations) == (2, 32)
    assert (budget.meaningful_depth, budget.max_simulations) == (2, 128)
    assert (deep.meaningful_depth, deep.max_simulations) == (4, 128)
    assert deep.max_expanded_nodes == 262144


def test_summary_separates_candidate_and_opponent_search_cost() -> None:
    records = [
        {
            "seed": 3,
            "candidate_side": "us",
            "candidate_score": 1.0,
            "candidate_won": True,
            "draw": False,
            "terminal_reason": "vp_threshold",
            "wall_clock_seconds": 10.0,
            "searched_decisions": 7,
            "simulations": 224,
            "action_flips": 2,
            "candidate_searched_decisions": 7,
            "candidate_simulations": 224,
            "candidate_action_flips": 2,
            "candidate_atomic_horizon_cutoffs": 1,
            "candidate_node_budget_cutoffs": 0,
            "opponent_searched_decisions": 0,
            "opponent_simulations": 0,
            "opponent_action_flips": 0,
        },
        {
            "seed": 3,
            "candidate_side": "ussr",
            "candidate_score": 0.0,
            "candidate_won": False,
            "draw": False,
            "terminal_reason": "final scoring",
            "wall_clock_seconds": 20.0,
            "searched_decisions": 5,
            "simulations": 160,
            "action_flips": 1,
            "candidate_searched_decisions": 5,
            "candidate_simulations": 160,
            "candidate_action_flips": 1,
            "candidate_atomic_horizon_cutoffs": 0,
            "candidate_node_budget_cutoffs": 2,
            "opponent_searched_decisions": 0,
            "opponent_simulations": 0,
            "opponent_action_flips": 0,
        },
    ]
    result = summary(SimpleNamespace(bootstrap_samples=100, seed_base=3, seed_pairs=1), records)
    assert result["candidate_average_searches_per_game"] == 6.0
    assert result["candidate_average_simulations_per_game"] == 192.0
    assert result["candidate_search_flip_rate"] == 0.25
    assert result["candidate_atomic_horizon_cutoffs"] == 1
    assert result["candidate_node_budget_cutoffs"] == 2
    assert result["opponent_average_searches_per_game"] == 0.0
    assert result["adaptive_fallbacks"] == 0


def test_progress_snapshot_is_explicitly_partial_and_side_split(tmp_path) -> None:
    records = [
        {
            "seed": 7,
            "orientation": "candidate_ussr",
            "candidate_side": "ussr",
            "candidate_score": 1.0,
            "terminal_reason": "vp_threshold",
            "wall_clock_seconds": 4.0,
            "searched_decisions": 10,
            "simulations": 320,
            "action_flips": 2,
            "atomic_horizon_cutoffs": 1,
            "node_budget_cutoffs": 0,
        },
        {
            "seed": 7,
            "orientation": "candidate_us",
            "candidate_side": "us",
            "candidate_score": 0.0,
            "terminal_reason": "thermonuclear war",
            "wall_clock_seconds": 6.0,
            "searched_decisions": 20,
            "simulations": 640,
            "action_flips": 1,
            "atomic_horizon_cutoffs": 0,
            "node_budget_cutoffs": 2,
        },
    ]

    write_progress_snapshot(tmp_path, SimpleNamespace(seed_pairs=4), records)

    payload = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert payload["status"] == "partial"
    assert payload["expected_games"] == 8
    assert payload["completed_games"] == 2
    assert payload["score"] == 0.5
    assert payload["complete_seed_pairs"] == 1
    assert payload["complete_pair_score"] == 0.5
    assert payload["by_candidate_side"]["us"] == {"games": 1, "score": 0.0}
    assert payload["by_candidate_side"]["ussr"] == {"games": 1, "score": 1.0}
    assert payload["terminal_reason_distribution"] == {
        "thermonuclear war": 1,
        "vp_threshold": 1,
    }
    assert payload["average_wall_clock_seconds_per_completed_game"] == 5.0
    assert payload["average_searches_per_completed_game"] == 15.0
    assert payload["average_simulations_per_completed_game"] == 480.0
    assert payload["search_flip_rate"] == 0.1
    assert payload["atomic_horizon_cutoffs"] == 1
    assert payload["node_budget_cutoffs"] == 2


def test_progress_resume_requires_matching_identity_and_unique_tasks(tmp_path) -> None:
    args = SimpleNamespace(seed_pairs=1, seed_base=7, resume=True)
    identity = {"candidate_policy": "/candidate", "seed_pairs": 1}
    records = [
        {
            "seed": 7,
            "orientation": "candidate_us",
            "candidate_side": "us",
            "candidate_score": 1.0,
            "terminal_reason": "final scoring",
            "wall_clock_seconds": 1.0,
            "searched_decisions": 1,
            "simulations": 32,
            "action_flips": 0,
        }
    ]
    write_progress_snapshot(tmp_path, args, records, identity)

    assert load_resumable_records(tmp_path, args, identity) == records
    with pytest.raises(ValueError, match="identity"):
        load_resumable_records(tmp_path, args, {"candidate_policy": "/other"})

    payload = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    payload["records"].append(dict(payload["records"][0]))
    (tmp_path / "progress.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_resumable_records(tmp_path, args, identity)


def test_worker_heartbeat_atomically_records_search_context(tmp_path) -> None:
    write_worker_heartbeat(
        tmp_path,
        3,
        seed=91,
        orientation="candidate_us",
        candidate_side="us",
        side="ussr",
        model="opponent",
        stage="search_complete",
        episode_step=27,
        observation={
            "turn": 4,
            "action_round": 3,
            "phase": "action",
            "prompt": "Select a legal action",
        },
        search_seconds=12.5,
    )

    path = tmp_path / "worker_heartbeats/worker-003.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 91
    assert payload["candidate_side"] == "us"
    assert payload["side"] == "ussr"
    assert payload["turn"] == 4
    assert payload["action_round"] == 3
    assert payload["episode_step"] == 27
    assert payload["stage"] == "search_complete"
    assert payload["search_seconds"] == 12.5
    assert not list(path.parent.glob("*.tmp-*"))


def test_failure_is_durable(tmp_path) -> None:
    write_failure(tmp_path, "worker exploded")
    assert json.loads((tmp_path / "failure.json").read_text())["error"] == "worker exploded"
