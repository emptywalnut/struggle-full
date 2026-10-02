import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.build_converged_teacher_replay import (
    crosses_independent_macro_boundary,
    era_for_example,
    load_excluded_roots,
    q_softened_example,
)
from struggle_ai.search_replay import SearchReplayExample, SearchReplayMetadata


def replay_example(
    *,
    visits: tuple[float, ...],
    q_values: tuple[float, ...],
    triggers: tuple[str, ...] = (),
    convergence_status: str = "BUDGET_CONVERGENCE",
    terminal_evaluations: int = 0,
    search_depth: int = 4,
    turn: int = 1,
) -> SearchReplayExample:
    width = len(visits)
    metadata = SearchReplayMetadata(
        side="us",
        trigger_categories=triggers,
        legal_actions=tuple({"value": str(index)} for index in range(width)),
        hierarchical_path=(),
        raw_action=0,
        search_action=0,
        search_depth=search_depth,
        simulation_budget=8,
        root_q_gap=0.8,
        action_flipped=False,
        complete_game=True,
        convergence_status=convergence_status,
        terminal_evaluations=terminal_evaluations,
    )
    return SearchReplayExample(
        observation={
            "action_mask": np.ones(width, dtype=np.float32),
            "global": np.asarray([0.0, 0.0, turn / 10.0], dtype=np.float32),
        },
        raw_policy=np.full(width, 1.0 / width, dtype=np.float32),
        visit_counts=np.asarray(visits, dtype=np.float32),
        search_policy=np.full(width, 1.0 / width, dtype=np.float32),
        q_values=np.asarray(q_values, dtype=np.float32),
        metadata=metadata,
    )


@pytest.mark.parametrize((("turn", "era")), ((1, "early"), (4, "mid"), (8, "late")))
def test_teacher_example_era_comes_from_masked_global_state(turn: int, era: str) -> None:
    example = replay_example(visits=(6.0, 2.0), q_values=(0.0, 0.8), turn=turn)
    assert era_for_example(example) == era


def test_load_excluded_roots_uses_durable_game_and_step_identity(tmp_path) -> None:
    path = tmp_path / "particle_variance_positions.json"
    path.write_text(
        json.dumps(
            {
                "positions": [
                    {"source_game": "puct-selfplay-000001-seed-7", "action_index": 19},
                    {"source_game": "puct-selfplay-000002-seed-8", "action_index": 23},
                ]
            }
        )
    )
    assert load_excluded_roots([path]) == {
        ("puct-selfplay-000001-seed-7", 19),
        ("puct-selfplay-000002-seed-8", 23),
    }


def test_load_excluded_roots_fails_closed_without_source_identity(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"positions": [{"seed": 7, "action_index": 19}]}))
    with pytest.raises(ValueError, match="source identity"):
        load_excluded_roots([path])


def test_q_teacher_rejects_large_gap_from_too_few_total_visits() -> None:
    example = replay_example(visits=(1.0, 1.0), q_values=(0.8, -0.8))
    assert q_softened_example(
        example,
        temperature=0.05,
        min_q_gap=0.02,
        min_total_visits=8.0,
    ) is None


def test_q_teacher_rejects_unvisited_q_best_action() -> None:
    example = replay_example(visits=(7.0, 1.0), q_values=(0.0, 0.8))
    assert q_softened_example(
        example,
        temperature=0.05,
        min_q_gap=0.02,
        min_total_visits=8.0,
        min_best_action_visits=2.0,
    ) is None


def test_q_teacher_accepts_sufficient_search_evidence() -> None:
    example = replay_example(visits=(6.0, 2.0), q_values=(0.0, 0.8))
    converted = q_softened_example(
        example,
        temperature=0.05,
        min_q_gap=0.02,
        min_total_visits=8.0,
        min_best_action_visits=2.0,
    )
    assert converted is not None
    assert converted.metadata.search_action == 1


def test_q_teacher_rejects_card_target_crossing_from_local_response_root() -> None:
    example = replay_example(visits=(6.0, 2.0), q_values=(0.0, 0.8))
    example.metadata = replace(
        example.metadata,
        hierarchy_index=1,
        hierarchy_level="card",
        hierarchical_path=(
            {"side": "us", "action": "West Germany", "level": "influence_allocation"},
        ),
    )

    assert crosses_independent_macro_boundary(example)
    assert q_softened_example(
        example,
        temperature=0.05,
        min_q_gap=0.02,
    ) is None


def test_q_teacher_keeps_downstream_mode_inside_card_macro() -> None:
    example = replay_example(visits=(6.0, 2.0), q_values=(0.0, 0.8))
    example.metadata = replace(
        example.metadata,
        hierarchy_index=1,
        hierarchy_level="mode",
        hierarchical_path=(
            {"side": "us", "action": "Marshall Plan", "level": "card"},
        ),
    )

    assert not crosses_independent_macro_boundary(example)
    assert q_softened_example(
        example,
        temperature=0.05,
        min_q_gap=0.02,
    ) is not None


def test_q_teacher_requires_deeper_agreement_for_scoring_pressure() -> None:
    shallow = replay_example(
        visits=(6.0, 2.0),
        q_values=(0.0, 0.8),
        triggers=("scoring_card",),
    )
    assert q_softened_example(
        shallow,
        temperature=0.05,
        min_q_gap=0.02,
        require_horizon_triggers=frozenset({"scoring_card"}),
    ) is None

    deep = replay_example(
        visits=(6.0, 2.0),
        q_values=(0.0, 0.8),
        triggers=("scoring_card",),
        convergence_status="HORIZON_CONVERGENCE",
    )
    assert q_softened_example(
        deep,
        temperature=0.05,
        min_q_gap=0.02,
        require_horizon_triggers=frozenset({"scoring_card"}),
    ) is not None


def test_q_teacher_requires_terminal_coverage_for_scoring_pressure() -> None:
    no_terminal = replay_example(
        visits=(6.0, 2.0),
        q_values=(0.0, 0.8),
        triggers=("scoring_pressure",),
        convergence_status="HORIZON_CONVERGENCE",
    )
    assert q_softened_example(
        no_terminal,
        temperature=0.05,
        min_q_gap=0.02,
        require_terminal_triggers=frozenset({"scoring_pressure"}),
    ) is None

    terminal = replay_example(
        visits=(6.0, 2.0),
        q_values=(0.0, 0.8),
        triggers=("scoring_pressure",),
        convergence_status="HORIZON_CONVERGENCE",
        terminal_evaluations=3,
    )
    assert q_softened_example(
        terminal,
        temperature=0.05,
        min_q_gap=0.02,
        require_terminal_triggers=frozenset({"scoring_pressure"}),
    ) is not None


def test_q_teacher_requires_configured_minimum_depth() -> None:
    shallow = replay_example(
        visits=(6.0, 2.0),
        q_values=(0.0, 0.8),
        triggers=("scoring_pressure",),
        convergence_status="HORIZON_CONVERGENCE",
        terminal_evaluations=3,
        search_depth=6,
    )
    assert q_softened_example(
        shallow,
        temperature=0.05,
        min_q_gap=0.02,
        require_min_depth={"scoring_pressure": 8},
    ) is None

    deep = replay_example(
        visits=(6.0, 2.0),
        q_values=(0.0, 0.8),
        triggers=("scoring_pressure",),
        convergence_status="HORIZON_CONVERGENCE",
        terminal_evaluations=3,
        search_depth=8,
    )
    assert q_softened_example(
        deep,
        temperature=0.05,
        min_q_gap=0.02,
        require_min_depth={"scoring_pressure": 8},
    ) is not None
