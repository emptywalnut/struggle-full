from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from struggle_ai.mcts import HierarchicalSearchTarget, SelectiveSearchDecision
from struggle_ai.search_replay import (
    CompleteGameReplayTransaction,
    DiskBackedCompleteGameReplayTransaction,
    SearchReplayWriter,
    example_from_decision,
    examples_from_decision,
    load_search_replay,
    with_terminal_outcome,
)


def test_search_replay_round_trip_preserves_targets_and_hierarchy(tmp_path) -> None:
    obs = {
        "action_mask": np.array([1.0, 1.0, 0.0], dtype=np.float32),
        "state": np.array([4.0, 5.0], dtype=np.float32),
    }
    decision = SelectiveSearchDecision(
        action=1,
        raw_action=0,
        raw_policy=np.array([0.7, 0.3]),
        search_policy=np.array([0.2, 0.8]),
        visits=np.array([8.0, 24.0]),
        q_root=np.array([-0.4, 0.1]),
        searched=True,
        triggers=("forced_response",),
        simulations=32,
        budgets={},
        root_q_gap=0.5,
        action_flipped=True,
        latency_seconds=0.2,
        expanded_nodes=3,
        exact_chance_nodes=1,
        belief_particles=32,
        meaningful_depth=2,
        hierarchical_path=(
            {"action": "participate", "level": "forced_response"},
            {"action": "Marshall Plan", "level": "card"},
        ),
        max_meaningful_depth_reached=4,
        max_turn_delta=1,
        cross_turn_leaf_evaluations=7,
        atomic_horizon_cutoffs=2,
        node_budget_cutoffs=3,
        sampled_large_chance_nodes=4,
    )
    example = example_from_decision(
        obs,
        [{"value": "boycott"}, {"value": "participate"}],
        "ussr",
        decision,
        search_depth=2,
        source_game="game-1",
        source_step=19,
        convergence_status="BUDGET_CONVERGENCE",
        convergence_configs=("d4_128", "d4_512"),
        convergence_rank_correlation=0.91,
        convergence_q_gap=0.04,
    )

    with SearchReplayWriter(tmp_path, shard_size=1) as writer:
        writer.add(example)

    loaded = load_search_replay([tmp_path])
    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded[0].observation["state"], obs["state"])
    np.testing.assert_allclose(loaded[0].search_policy, [0.2, 0.8, 0.0])
    assert loaded[0].metadata.action_flipped
    assert loaded[0].metadata.hierarchical_path[0]["level"] == "forced_response"
    assert loaded[0].metadata.source_step == 19
    assert loaded[0].metadata.max_meaningful_depth_reached == 4
    assert loaded[0].metadata.max_turn_delta == 1
    assert loaded[0].metadata.cross_turn_leaf_evaluations == 7
    assert loaded[0].metadata.atomic_horizon_cutoffs == 2
    assert loaded[0].metadata.node_budget_cutoffs == 3
    assert loaded[0].metadata.sampled_large_chance_nodes == 4
    assert loaded[0].metadata.convergence_status == "BUDGET_CONVERGENCE"
    assert loaded[0].metadata.convergence_configs == ("d4_128", "d4_512")
    assert loaded[0].metadata.convergence_rank_correlation == 0.91
    assert loaded[0].metadata.convergence_q_gap == 0.04

    with SearchReplayWriter(tmp_path, shard_size=1) as writer:
        writer.add(example)
        writer.add(replace(example, metadata=replace(example.metadata, source_step=20)))

    loaded = load_search_replay([tmp_path])
    assert [item.metadata.source_step for item in loaded] == [19, 20]

    threaded = load_search_replay([tmp_path], workers=2)
    assert [item.metadata.source_step for item in threaded] == [19, 20]
    np.testing.assert_array_equal(threaded[0].search_policy, loaded[0].search_policy)


def test_hierarchical_search_targets_become_distinct_training_examples(tmp_path) -> None:
    root_obs = {
        "action_mask": np.array([1.0, 1.0, 0.0], dtype=np.float32),
        "state": np.array([1.0], dtype=np.float32),
    }
    child_obs = {
        "action_mask": np.array([1.0, 1.0, 0.0], dtype=np.float32),
        "state": np.array([2.0], dtype=np.float32),
    }
    decision = SelectiveSearchDecision(
        action=0,
        raw_action=0,
        raw_policy=np.array([0.8, 0.2]),
        search_policy=np.array([0.7, 0.3]),
        visits=np.array([22.0, 10.0]),
        q_root=np.array([0.2, 0.1]),
        searched=True,
        triggers=("influence_allocation",),
        simulations=32,
        budgets={},
        root_q_gap=0.1,
        action_flipped=False,
        latency_seconds=0.1,
        expanded_nodes=4,
        exact_chance_nodes=0,
        belief_particles=32,
        meaningful_depth=4,
        hierarchical_path=({"side": "us", "action": "Marshall Plan", "level": "card"},),
        node_budget_cutoffs=5,
        hierarchical_targets=(
            HierarchicalSearchTarget(
                observation=child_obs,
                legal_actions=({"value": "france"}, {"value": "italy"}),
                side="us",
                level="influence_allocation",
                path=({"side": "us", "action": "Marshall Plan", "level": "card"},),
                raw_policy=np.array([0.8, 0.2]),
                visits=np.array([4.0, 28.0]),
                q_values=np.array([0.0, 0.3]),
                search_policy=np.array([0.125, 0.875]),
                raw_action=0,
                search_action=1,
            ),
        ),
    )

    examples = examples_from_decision(
        root_obs,
        [{"value": "marshall"}, {"value": "nato"}],
        "us",
        decision,
        search_depth=4,
        source_game="game-2",
        source_step=42,
    )
    assert len(examples) == 2
    assert examples[0].metadata.hierarchy_level == "root"
    assert examples[1].metadata.hierarchy_level == "influence_allocation"
    assert examples[1].metadata.q_advantage_over_raw == 0.3
    assert examples[1].metadata.node_budget_cutoffs == 5
    completed = with_terminal_outcome(examples, winner="us", terminal_reason="final_scoring")
    assert completed[0].metadata.root_value_target == 1.0
    assert completed[1].metadata.root_value_target is None
    assert all(item.metadata.complete_game for item in completed)

    with SearchReplayWriter(tmp_path, shard_size=2) as writer:
        for example in examples:
            writer.add(example)
    loaded = load_search_replay([tmp_path])
    assert [item.metadata.hierarchy_index for item in loaded] == [0, 1]


def test_complete_game_transaction_assigns_side_targets_and_rejects_partial_tree(tmp_path) -> None:
    obs = {"action_mask": np.ones(2, dtype=np.float32), "state": np.ones(1, dtype=np.float32)}
    decision = SelectiveSearchDecision(
        action=1,
        raw_action=0,
        raw_policy=np.array([0.8, 0.2]),
        search_policy=np.array([0.25, 0.75]),
        visits=np.array([8.0, 24.0]),
        q_root=np.array([-0.2, 0.2]),
        searched=True,
        triggers=("opponent_event",),
        simulations=32,
        budgets={},
        root_q_gap=0.4,
        action_flipped=True,
        latency_seconds=0.1,
        expanded_nodes=4,
        exact_chance_nodes=0,
        belief_particles=32,
        meaningful_depth=2,
        hierarchical_path=(),
    )
    writer = SearchReplayWriter(tmp_path / "valid")
    transaction = CompleteGameReplayTransaction(writer)
    for side in ("us", "ussr"):
        transaction.add_decision(
            examples_from_decision(obs, [{"value": "a"}, {"value": "b"}], side, decision, search_depth=2),
            decision,
        )
    assert transaction.commit(winner="us", terminal_reason="final_scoring") == 2
    writer.close()
    loaded = load_search_replay([tmp_path / "valid"])
    assert [example.metadata.root_value_target for example in loaded] == [1.0, -1.0]
    assert all(example.metadata.complete_game for example in loaded)

    partial = replace(decision, node_budget_cutoffs=1)
    rejected_writer = SearchReplayWriter(tmp_path / "rejected")
    rejected = CompleteGameReplayTransaction(rejected_writer)
    rejected.add_decision(
        examples_from_decision(obs, [{"value": "a"}, {"value": "b"}], "us", partial, search_depth=2),
        partial,
    )
    assert rejected.partial_decisions_rejected == 1
    assert rejected.commit(winner="us", terminal_reason="final_scoring") == 0
    rejected_writer.close()
    assert load_search_replay([tmp_path / "rejected"]) == []


def test_disk_backed_complete_game_transaction_is_fail_closed(tmp_path) -> None:
    obs = {"action_mask": np.ones(2, dtype=np.float32), "state": np.ones(1, dtype=np.float32)}
    decision = SelectiveSearchDecision(
        action=1,
        raw_action=0,
        raw_policy=np.array([0.8, 0.2]),
        search_policy=np.array([0.25, 0.75]),
        visits=np.array([8.0, 24.0]),
        q_root=np.array([-0.2, 0.2]),
        searched=True,
        triggers=("opponent_event",),
        simulations=32,
        budgets={},
        root_q_gap=0.4,
        action_flipped=True,
        latency_seconds=0.1,
        expanded_nodes=4,
        exact_chance_nodes=0,
        belief_particles=32,
        meaningful_depth=2,
        hierarchical_path=(),
    )
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    writer = SearchReplayWriter(output, shard_size=1)
    transaction = DiskBackedCompleteGameReplayTransaction(writer, staging, shard_size=1)
    for side in ("us", "ussr"):
        transaction.add_decision(
            examples_from_decision(
                obs,
                [{"value": "a"}, {"value": "b"}],
                side,
                decision,
                search_depth=2,
                source_game="disk-game",
                source_step=0 if side == "us" else 1,
            ),
            decision,
        )
    assert (staging / "manifest.json").is_file()
    assert transaction.commit(winner="ussr", terminal_reason="vp_threshold") == 2
    assert not staging.exists()
    writer.close()
    loaded = load_search_replay([output])
    assert [example.metadata.root_value_target for example in loaded] == [-1.0, 1.0]
    assert all(example.metadata.complete_game for example in loaded)

    rejected_output = tmp_path / "rejected-disk"
    rejected_staging = tmp_path / "rejected-staging"
    rejected_writer = SearchReplayWriter(rejected_output, shard_size=1)
    rejected = DiskBackedCompleteGameReplayTransaction(
        rejected_writer, rejected_staging, shard_size=1
    )
    rejected.add_decision(
        examples_from_decision(
            obs,
            [{"value": "a"}, {"value": "b"}],
            "us",
            decision,
            search_depth=2,
            source_game="discarded-game",
            source_step=0,
        ),
        decision,
    )
    rejected.invalidate("truncated_game")
    assert not rejected_staging.exists()
    assert rejected.commit(winner="us", terminal_reason="unknown") == 0
    rejected_writer.close()
    assert load_search_replay([rejected_output]) == []


@pytest.mark.parametrize("workers", [1, 2])
def test_search_replay_metadata_predicate_filters_before_materialization(
    tmp_path, workers
) -> None:
    obs = {"action_mask": np.ones(2, dtype=np.float32), "state": np.ones(1, dtype=np.float32)}
    decision = SelectiveSearchDecision(
        action=1,
        raw_action=0,
        raw_policy=np.array([0.8, 0.2]),
        search_policy=np.array([0.25, 0.75]),
        visits=np.array([8.0, 24.0]),
        q_root=np.array([-0.2, 0.2]),
        searched=True,
        triggers=("opponent_event",),
        simulations=32,
        budgets={},
        root_q_gap=0.4,
        action_flipped=True,
        latency_seconds=0.1,
        expanded_nodes=4,
        exact_chance_nodes=0,
        belief_particles=32,
        meaningful_depth=2,
        hierarchical_path=(),
    )
    with SearchReplayWriter(tmp_path, shard_size=2) as writer:
        writer.add(
            example_from_decision(
                obs, [{"value": "a"}, {"value": "b"}], "us", decision, search_depth=2
            )
        )
        writer.add(
            example_from_decision(
                obs, [{"value": "a"}, {"value": "b"}], "ussr", decision, search_depth=2
            )
        )

    loaded = load_search_replay(
        [tmp_path],
        workers=workers,
        metadata_predicate=lambda metadata: metadata.side == "us",
    )

    assert len(loaded) == 1
    assert loaded[0].metadata.side == "us"
