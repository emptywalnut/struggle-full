from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from struggle_ai.mcts import (
    ExactBeliefPuctSession,
    ExactChanceEnumerator,
    SelectivePuctConfig,
    SelectivePuctPlanner,
    classify_search_triggers,
    continuation_matches_live_boundary,
    executable_continuation_prefix,
    meaningful_decision_kind,
    q_aware_root_action,
    scoring_search_throttle_key,
    search_evidence_gate_rejected,
    validate_information_safe_observation,
)


def test_continuation_never_crosses_into_the_next_card_choice() -> None:
    path = (
        {"side": "us", "action": "West Germany", "level": "influence_allocation"},
        {"side": "us", "action": "Blockade", "level": "card"},
        {"side": "us", "action": "ops", "level": "mode"},
    )

    assert executable_continuation_prefix(path) == ()


def test_continuation_keeps_only_same_local_macro_clicks() -> None:
    path = (
        {"side": "us", "action": "France", "level": "influence_allocation"},
        {"side": "us", "action": "Italy", "level": "influence_allocation"},
        {"side": "us", "action": "Asia Scoring", "level": "card"},
    )

    assert executable_continuation_prefix(path) == (
        {"side": "us", "action": "Italy", "level": "influence_allocation"},
    )


def test_card_macro_keeps_modes_but_stops_before_local_or_second_card() -> None:
    path = (
        {"side": "us", "action": "Marshall Plan", "level": "card"},
        {"side": "us", "action": "ops", "level": "mode"},
        {"side": "us", "action": "place", "level": "ops_action_type"},
        {"side": "us", "action": "France", "level": "influence_allocation"},
        {"side": "us", "action": "Defectors", "level": "card"},
    )

    assert executable_continuation_prefix(path) == (
        {"side": "us", "action": "ops", "level": "mode"},
        {"side": "us", "action": "place", "level": "ops_action_type"},
    )


def test_pending_continuation_never_consumes_new_turn_headline() -> None:
    expected = {"side": "ussr", "action": "Cultural Revolution", "level": "card"}
    obs = {
        "side": "ussr",
        "turn": 5,
        "action_round": 0,
        "phase": "headline_ussr",
        "prompt": "USSR pick headline card",
    }
    actions = [
        {"decision": "list", "value": "culturalrev"},
        {"decision": "list", "value": "missileenvy"},
    ]

    assert not continuation_matches_live_boundary(expected, obs, actions)


def test_pending_continuation_keeps_same_allocation_boundary() -> None:
    expected = {"side": "us", "action": "Italy", "level": "influence_allocation"}
    obs = {
        "side": "us",
        "turn": 2,
        "action_round": 3,
        "phase": "action",
        "prompt": "Place 2 influence",
    }
    actions = [
        {"decision": "country_mouseup", "value": "italy"},
        {"decision": "country_mouseup", "value": "france"},
    ]

    assert continuation_matches_live_boundary(expected, obs, actions)


def test_scoring_search_throttle_resets_each_action_round() -> None:
    ar1 = {"side": "us", "turn": 2, "action_round": 1}
    ar2 = {"side": "us", "turn": 2, "action_round": 2}

    assert scoring_search_throttle_key(7, ar1) != scoring_search_throttle_key(7, ar2)
    assert scoring_search_throttle_key(7, ar1) == (7, "us", 2, 1)


class ExactDiceEnv:
    def __init__(self) -> None:
        self.state = "root"
        self.last_obs = {"side": "us", "prompt": "Choose a roll dice action", "hand": []}
        self.legal_actions = [
            {"type": "saito_choice", "decision": "option", "value": "risky"},
            {"type": "saito_choice", "decision": "option", "value": "safe"},
        ]

    def observation(self) -> dict[str, np.ndarray]:
        return {
            "action_mask": np.array([1.0, 1.0], dtype=np.float32),
            "marker": np.array([0.0], dtype=np.float32),
            "us_hand": np.array([1.0, 0.0], dtype=np.float32),
            "ussr_hand": np.array([0.0, 0.0], dtype=np.float32),
        }

    def snapshot(self, compact: bool = False):
        return {"state": self.state}

    def restore(self, snapshot):
        self.state = snapshot["state"]
        self.last_obs = {"side": "us", "prompt": "Choose a roll dice action", "hand": []}
        self.legal_actions = [
            {"type": "saito_choice", "decision": "option", "value": "risky"},
            {"type": "saito_choice", "decision": "option", "value": "safe"},
        ]
        return self.observation()

    def redeterminize(self, side: str, seed: int):
        assert side == "us"
        return self.observation()

    def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
        self.state = "terminal"
        if action == 0:
            raw = forced_rolls[0]
            roll = int(float(raw) * 6) + 1 if isinstance(raw, float) and raw < 1.0 else int(raw)
            winner = "us" if roll >= 5 else "ussr"
            info = {
                "winner": winner,
                "chance_rolls": [roll],
                "chance_roll_sides": [6],
                "forced_rolls_consumed": [raw],
            }
        else:
            info = {"winner": "", "chance_rolls": [], "chance_roll_sides": [], "forced_rolls_consumed": []}
        return self.observation(), 0.0, True, False, info


class EqualPolicyEvaluator:
    def raw(self, obs, legal_count):
        return {
            "probabilities": np.full(legal_count, 1.0 / legal_count),
            "logits": np.zeros(legal_count),
            "raw_value": 0.0,
            "calibrated_value": 0.0,
        }

    def value_us(self, obs, side):
        return 0.0, 0.0


class LargeAllocationEnv:
    def __init__(self) -> None:
        self.state = -1
        self.last_obs: dict = {}
        self.legal_actions: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.state < 0:
            self.last_obs = {
                "side": "us",
                "current_player": "us",
                "turn": 3,
                "action_round": 2,
                "phase": "action",
                "prompt": "Place 4 influence",
                "hand": [],
            }
            self.legal_actions = [
                {"decision": "country_mouseup", "value": f"country-{index}"}
                for index in range(72)
            ]
        else:
            self.last_obs = {
                "side": "ussr",
                "current_player": "ussr",
                "turn": 3,
                "action_round": 2,
                "phase": "action",
                "prompt": "USSR pick a card",
                "hand": [],
            }
            self.legal_actions = [
                {"decision": "list", "value": "card-a"},
                {"decision": "list", "value": "card-b"},
            ]

    def observation(self) -> dict[str, np.ndarray]:
        count = len(self.legal_actions)
        marker = -1.0 if self.state < 0 else -0.8 + 1.6 * self.state / 71.0
        side = str(self.last_obs["side"])
        return {
            "action_mask": np.ones(count, dtype=np.float32),
            "marker": np.array([marker], dtype=np.float32),
            "us_hand": np.array([1.0, 0.0], dtype=np.float32) if side == "us" else np.zeros(2, dtype=np.float32),
            "ussr_hand": np.array([1.0, 0.0], dtype=np.float32) if side == "ussr" else np.zeros(2, dtype=np.float32),
        }

    def snapshot(self, compact: bool = False):
        return {"state": self.state}

    def restore(self, snapshot):
        self.state = int(snapshot["state"])
        self._load()
        return self.observation()

    def redeterminize(self, side: str, seed: int):
        assert side == "us"
        return self.observation()

    def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
        self.state = int(action)
        self._load()
        return self.observation(), 0.0, False, False, {
            "winner": "",
            "chance_rolls": [],
            "chance_roll_sides": [],
            "forced_rolls_consumed": [],
        }


class LargeAllocationEvaluator:
    def raw(self, obs, legal_count):
        if legal_count == 72:
            logits = np.linspace(4.0, -2.0, legal_count)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
        else:
            logits = np.zeros(legal_count)
            probabilities = np.full(legal_count, 1.0 / legal_count)
        marker = float(np.asarray(obs["marker"])[0])
        return {
            "probabilities": probabilities,
            "logits": logits,
            "raw_value": marker,
            "calibrated_value": marker,
        }

    def value_us(self, obs, side):
        marker = float(np.asarray(obs["marker"])[0])
        # The marker is an absolute-US test value even though the leaf actor is USSR.
        return marker, marker


def test_q_aware_root_selection_never_breaks_visit_ties_by_action_index() -> None:
    selected, detail = q_aware_root_action(
        np.ones(4),
        np.array([-0.4, 0.1, 0.7, 0.2]),
        np.array([0.7, 0.1, 0.1, 0.1]),
        ("a", "b", "c", "d"),
        min_visits=1,
        visit_fraction=1.0,
        min_q_advantage=0.01,
    )

    assert selected == 2
    assert detail["visit_tie_count"] == 4
    assert detail["q_override"] is False  # Q already resolves the visit-leader tie.


def test_production_root_selection_keeps_evaluated_q_candidate_eligible_as_budget_grows() -> None:
    config = SelectivePuctConfig()
    assert config.root_selection_visit_fraction == 0.0

    labels = ("prior-leader", "better-q", "other")
    priors = np.array([0.8, 0.1, 0.1])
    q_values = np.array([0.20, 0.24, 0.10])
    selected_early, early = q_aware_root_action(
        np.array([12.0, 4.0, 3.0]),
        q_values,
        priors,
        labels,
        min_visits=config.root_selection_min_visits,
        visit_fraction=config.root_selection_visit_fraction,
        min_q_advantage=config.root_selection_min_q_advantage,
    )
    selected_late, late = q_aware_root_action(
        np.array([120.0, 4.0, 3.0]),
        q_values,
        priors,
        labels,
        min_visits=config.root_selection_min_visits,
        visit_fraction=config.root_selection_visit_fraction,
        min_q_advantage=config.root_selection_min_q_advantage,
    )

    assert selected_early == selected_late == 1
    assert early["q_override"] is True
    assert late["q_override"] is True
    assert early["required_visits"] == late["required_visits"] == 3.0


def test_large_root_progressive_widening_revisits_and_admits_cheap_q_action() -> None:
    env = LargeAllocationEnv()
    config = SelectivePuctConfig(
        initial_simulations=32,
        medium_simulations=32,
        max_simulations=32,
        belief_particles=1,
        meaningful_depth=1,
        root_initial_children=6,
        root_max_children=32,
        root_q_probe_particles=1,
        root_q_probe_weight=0.6,
        root_q_prior_mix=0.4,
        root_cpuct=0.5,
        enabled_triggers=("influence_allocation",),
    )
    planner = SelectivePuctPlanner(LargeAllocationEvaluator(), env, config, seed=23)

    decision = planner.plan(copy.deepcopy(env.snapshot()), env.observation(), force_search=True)

    assert decision.searched
    assert decision.simulations == 32
    assert np.count_nonzero(decision.visits) < 32
    assert decision.visits.max() > 1
    assert decision.action == 71
    assert decision.root_selection["selected_action"] == "country-71"
    admission = {row["action"]: row for row in decision.root_admission_events}
    assert admission["country-71"]["cheap_Q"] == pytest.approx(0.8)
    assert admission["country-71"]["admitted_at_root_visit"] == 0.0


class OlympicEnv(ExactDiceEnv):
    def __init__(self) -> None:
        super().__init__()
        self.last_obs["prompt"] = "US hosts the Olympic Games:"
        self.legal_actions = [{"type": "saito_choice", "decision": "option", "value": "participate"}]

    def observation(self) -> dict[str, np.ndarray]:
        obs = super().observation()
        obs["action_mask"] = np.array([1.0], dtype=np.float32)
        return obs

    def restore(self, snapshot):
        self.state = snapshot["state"]
        self.last_obs = {"side": "ussr", "prompt": "US hosts the Olympic Games:", "hand": []}
        self.legal_actions = [{"type": "saito_choice", "decision": "option", "value": "participate"}]
        obs = self.observation()
        obs["us_hand"][:] = 0.0
        obs["ussr_hand"][0] = 1.0
        return obs

    def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
        us_roll, ussr_roll = map(int, forced_rolls[:2])
        winner = "us" if us_roll + 2 > ussr_roll else "ussr"
        info = {
            "winner": winner,
            "chance_rolls": [us_roll, ussr_roll],
            "chance_roll_sides": [6, 6],
            "forced_rolls_consumed": list(forced_rolls[:2]),
        }
        return self.restore({"state": "terminal"}), 0.0, True, False, info


class ConditionalDiceEnv(ExactDiceEnv):
    def observation(self) -> dict[str, np.ndarray]:
        marker = 0.0 if self.state == "root" else float(self.state)
        return {
            "action_mask": np.array([1.0], dtype=np.float32),
            "marker": np.array([marker], dtype=np.float32),
            "us_hand": np.array([1.0, 0.0], dtype=np.float32),
            "ussr_hand": np.array([0.0, 0.0], dtype=np.float32),
        }

    def restore(self, snapshot):
        self.state = snapshot["state"]
        self.last_obs = {"side": "us", "prompt": "Resolve conditional dice", "hand": []}
        self.legal_actions = [{"type": "saito_choice", "decision": "option", "value": "roll"}]
        return self.observation()

    @staticmethod
    def _roll(raw: int | float, sides: int) -> int:
        return int(float(raw) * sides) + 1 if isinstance(raw, float) and raw < 1.0 else int(raw)

    def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
        first = self._roll(forced_rolls[0], 2)
        rolls = [first]
        sides = [2]
        consumed = [forced_rolls[0]]
        if first == 2:
            second = self._roll(forced_rolls[1], 3)
            rolls.append(second)
            sides.append(3)
            consumed.append(forced_rolls[1])
            self.state = 10 + second
        else:
            self.state = 1
        info = {
            "winner": "",
            "chance_rolls": rolls,
            "chance_roll_sides": sides,
            "forced_rolls_consumed": consumed,
        }
        return self.observation(), 0.0, True, False, info


def test_selective_puct_uses_exact_dice_expectation_and_reuses_budget() -> None:
    env = ExactDiceEnv()
    config = SelectivePuctConfig(
        initial_simulations=32,
        medium_simulations=64,
        max_simulations=64,
        belief_particles=4,
        meaningful_depth=1,
        enabled_triggers=("chance_heavy",),
        medium_visit_share_threshold=0.99,
    )
    planner = SelectivePuctPlanner(EqualPolicyEvaluator(), env, config, seed=7)

    decision = planner.plan(copy.deepcopy(env.snapshot()), env.observation())

    assert decision.searched
    assert decision.simulations == 64
    assert sorted(decision.budgets) == [32, 64]
    assert sum(decision.visits) == pytest.approx(64.0)
    assert decision.exact_chance_nodes >= 1
    assert decision.raw_action == 0
    assert decision.action == 1
    assert decision.action_flipped
    assert planner.telemetry.metrics()["search/action_flip_rate"] == 1.0


def test_atomic_horizon_cutoff_is_leaf_evaluated_and_reported() -> None:
    class LongDeterministicChainEnv(ExactDiceEnv):
        def __init__(self) -> None:
            super().__init__()
            self.state = 0
            self.last_obs = {"side": "us", "prompt": "Choose next deterministic response", "hand": []}

        def observation(self) -> dict[str, np.ndarray]:
            return {
                "action_mask": np.array([1.0, 1.0], dtype=np.float32),
                "marker": np.array([float(self.state)], dtype=np.float32),
                "us_hand": np.array([1.0, 0.0], dtype=np.float32),
                "ussr_hand": np.array([0.0, 0.0], dtype=np.float32),
            }

        def restore(self, snapshot):
            self.state = snapshot["state"]
            self.last_obs = {"side": "us", "prompt": "Choose next deterministic response", "hand": []}
            self.legal_actions = [
                {"type": "saito_choice", "decision": "option", "value": "left"},
                {"type": "saito_choice", "decision": "option", "value": "right"},
            ]
            return self.observation()

        def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
            self.state += 1 + action
            return self.restore({"state": self.state}), 0.0, False, False, {
                "chance_rolls": [],
                "chance_roll_sides": [],
                "forced_rolls_consumed": [],
            }

    env = LongDeterministicChainEnv()
    config = SelectivePuctConfig(
        initial_simulations=2,
        medium_simulations=2,
        max_simulations=2,
        belief_particles=1,
        meaningful_depth=10,
        forced_response_depth=10,
        max_atomic_steps=1,
    )
    planner = SelectivePuctPlanner(EqualPolicyEvaluator(), env, config, seed=11)

    decision = planner.plan(copy.deepcopy(env.snapshot()), env.observation(), force_search=True)

    assert decision.searched
    assert decision.atomic_horizon_cutoffs == 2
    assert decision.action == decision.raw_action
    assert decision.q_gate_rejected
    assert decision.hierarchical_path == ()
    assert decision.hierarchical_targets == ()
    assert decision.search_policy == pytest.approx(decision.raw_policy)
    assert planner.telemetry.metrics()["search/average_atomic_horizon_cutoffs"] == 2.0


def test_expanded_node_budget_is_leaf_evaluated_and_reported() -> None:
    class BranchingEnv(ExactDiceEnv):
        def __init__(self) -> None:
            super().__init__()
            self.state = 0
            self.last_obs = {"side": "us", "prompt": "Choose next deterministic response", "hand": []}

        def observation(self) -> dict[str, np.ndarray]:
            return {
                "action_mask": np.array([1.0, 1.0], dtype=np.float32),
                "marker": np.array([float(self.state)], dtype=np.float32),
                "us_hand": np.array([1.0, 0.0], dtype=np.float32),
                "ussr_hand": np.array([0.0, 0.0], dtype=np.float32),
            }

        def restore(self, snapshot):
            self.state = snapshot["state"]
            self.last_obs = {"side": "us", "prompt": "Choose next deterministic response", "hand": []}
            self.legal_actions = [
                {"type": "saito_choice", "decision": "option", "value": "left"},
                {"type": "saito_choice", "decision": "option", "value": "right"},
            ]
            return self.observation()

        def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
            self.state += 1 + action
            return self.restore({"state": self.state}), 0.0, False, False, {
                "chance_rolls": [],
                "chance_roll_sides": [],
                "forced_rolls_consumed": [],
            }

    env = BranchingEnv()
    config = SelectivePuctConfig(
        initial_simulations=2,
        medium_simulations=2,
        max_simulations=2,
        belief_particles=1,
        meaningful_depth=10,
        forced_response_depth=10,
        max_atomic_steps=10,
        max_expanded_nodes=1,
    )
    planner = SelectivePuctPlanner(EqualPolicyEvaluator(), env, config, seed=13)

    decision = planner.plan(copy.deepcopy(env.snapshot()), env.observation(), force_search=True)

    assert decision.searched
    assert decision.expanded_nodes == 1
    assert decision.node_budget_cutoffs == 2
    assert decision.action == decision.raw_action
    assert decision.q_gate_rejected
    assert decision.hierarchical_path == ()
    assert decision.hierarchical_targets == ()
    assert decision.search_policy == pytest.approx(decision.raw_policy)
    assert planner.telemetry.metrics()["search/average_node_budget_cutoffs"] == 2.0


def test_process_rss_limit_fails_search_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    env = ExactDiceEnv()
    config = SelectivePuctConfig(
        initial_simulations=2,
        medium_simulations=2,
        max_simulations=2,
        belief_particles=1,
        meaningful_depth=2,
        max_process_rss_mib=1.0,
    )
    monkeypatch.setattr(
        ExactBeliefPuctSession,
        "_resident_memory_mib",
        staticmethod(lambda: 2.0),
    )
    planner = SelectivePuctPlanner(EqualPolicyEvaluator(), env, config, seed=17)

    decision = planner.plan(copy.deepcopy(env.snapshot()), env.observation(), force_search=True)

    assert decision.searched
    assert decision.simulations == 0
    assert decision.node_budget_cutoffs == 1
    assert decision.action == decision.raw_action
    assert decision.q_gate_rejected
    assert decision.hierarchical_targets == ()
    assert decision.search_policy == pytest.approx(decision.raw_policy)


def test_search_wall_clock_limit_fails_search_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    env = ExactDiceEnv()
    config = SelectivePuctConfig(
        initial_simulations=2,
        medium_simulations=2,
        max_simulations=2,
        belief_particles=1,
        meaningful_depth=2,
        max_search_seconds=1.0,
    )
    ticks = iter((10.0, 10.0, 10.0, 12.0, 12.0))
    monkeypatch.setattr("struggle_ai.mcts.time.perf_counter", lambda: next(ticks, 12.0))
    planner = SelectivePuctPlanner(EqualPolicyEvaluator(), env, config, seed=19)

    decision = planner.plan(copy.deepcopy(env.snapshot()), env.observation(), force_search=True)

    assert decision.searched
    assert decision.simulations == 0
    assert decision.node_budget_cutoffs == 1
    assert decision.action == decision.raw_action
    assert decision.q_gate_rejected
    assert decision.hierarchical_targets == ()
    assert decision.search_policy == pytest.approx(decision.raw_policy)


def test_information_safety_rejects_sampled_opponent_hand() -> None:
    obs = {
        "us_hand": np.array([1.0, 0.0], dtype=np.float32),
        "ussr_hand": np.array([0.0, 1.0], dtype=np.float32),
    }
    with pytest.raises(RuntimeError, match="information leak"):
        validate_information_safe_observation(obs, "us")


def test_opponent_owned_card_candidate_triggers_search() -> None:
    env = SimpleNamespace(
        last_obs={"side": "us", "current_player": "us", "prompt": "Select a card", "hand": ["destal"]},
        legal_actions=[{"type": "card", "value": "destal"}, {"type": "card", "value": "us_card"}],
        cards_by_id={
            "destal": {"player": "ussr", "scoring": False},
            "us_card": {"player": "us", "scoring": False},
        },
    )
    config = SelectivePuctConfig(enabled_triggers=("opponent_event",))

    triggers = classify_search_triggers(env, np.array([0.6, 0.4]), config)

    assert triggers == ("opponent_event",)


def test_mandatory_scoring_pressure_is_not_suppressed_by_turn_search_limit() -> None:
    env = SimpleNamespace(
        last_obs={
            "side": "us",
            "current_player": "us",
            "turn": 2,
            "action_round": 4,
            "prompt": "US pick a card",
            "hand": ["asia", "europe", "other"],
        },
        legal_actions=[
            {"type": "card", "decision": "list", "value": "asia"},
            {"type": "card", "decision": "list", "value": "europe"},
            {"type": "card", "decision": "list", "value": "other"},
        ],
        cards_by_id={
            "asia": {"player": "both", "scoring": True},
            "europe": {"player": "both", "scoring": True},
            "other": {"player": "us", "scoring": False},
        },
    )
    config = SelectivePuctConfig(
        enabled_triggers=("scoring_card", "scoring_pressure"),
        low_confidence_sample_rate=0.0,
    )

    assert classify_search_triggers(env, np.array([0.3, 0.3, 0.4]), config) == (
        "scoring_card",
        "scoring_pressure",
    )

    env.last_obs["action_round"] = 3
    assert classify_search_triggers(env, np.array([0.3, 0.3, 0.4]), config) == (
        "scoring_card",
    )


def test_scoring_pressure_search_requires_depth_and_terminal_coverage() -> None:
    config = SelectivePuctConfig()
    assert search_evidence_gate_rejected(
        ("scoring_pressure",), config, meaningful_depth=6, terminal_evaluations=10
    )
    assert search_evidence_gate_rejected(
        ("scoring_pressure",), config, meaningful_depth=8, terminal_evaluations=0
    )
    assert not search_evidence_gate_rejected(
        ("scoring_pressure",), config, meaningful_depth=8, terminal_evaluations=1
    )
    assert not search_evidence_gate_rejected(
        ("scoring_card",), config, meaningful_depth=4, terminal_evaluations=0
    )


def test_opponent_event_context_does_not_retrigger_on_local_allocation() -> None:
    env = SimpleNamespace(
        last_obs={
            "side": "us",
            "current_player": "us",
            "phase": "action",
            "prompt": "Place 3 influence after opponent card Destalinization",
            "hand": [],
        },
        legal_actions=[
            {"decision": "country_mouseup", "value": "thailand", "card": "destal"},
            {"decision": "country_mouseup", "value": "zaire", "card": "destal"},
        ],
        cards_by_id={"destal": {"player": "ussr", "scoring": False}},
    )
    config = SelectivePuctConfig(
        enabled_triggers=("opponent_event", "influence_allocation"),
        low_confidence_sample_rate=0.0,
    )

    assert classify_search_triggers(env, np.array([0.6, 0.4]), config) == (
        "influence_allocation",
    )


def test_allocation_search_is_limited_per_side_action_round() -> None:
    class AllocationEnv(ExactDiceEnv):
        def __init__(self) -> None:
            super().__init__()
            self.last_obs = {
                "seed": 17,
                "turn": 3,
                "action_round": 4,
                "side": "us",
                "current_player": "us",
                "phase": "action",
                "prompt": "Place 3 influence",
                "hand": [],
            }
            self.legal_actions = [
                {"decision": "country_mouseup", "value": "france"},
                {"decision": "country_mouseup", "value": "italy"},
            ]

        def restore(self, snapshot):
            self.state = snapshot["state"]
            self.__init__()
            return self.observation()

    env = AllocationEnv()
    config = SelectivePuctConfig(
        initial_simulations=2,
        medium_simulations=2,
        max_simulations=2,
        meaningful_depth=1,
        belief_particles=1,
        enabled_triggers=("influence_allocation",),
        low_confidence_sample_rate=0.0,
        max_allocation_searches_per_side_round=1,
    )
    planner = SelectivePuctPlanner(EqualPolicyEvaluator(), env, config, seed=17)
    snapshot = copy.deepcopy(env.snapshot())

    first = planner.plan(snapshot, env.observation())
    planner.pending_continuation.clear()
    second = planner.plan(snapshot, env.observation())

    assert first.searched
    assert not second.searched
    assert second.triggers == ()


def test_standard_card_mode_options_are_not_forced_responses() -> None:
    env = SimpleNamespace(
        last_obs={"side": "us", "current_player": "us", "prompt": "Play card as event or ops", "hand": []},
        legal_actions=[
            {"type": "saito_choice", "decision": "option", "value": "event"},
            {"type": "saito_choice", "decision": "option", "value": "ops"},
            {"type": "saito_choice", "decision": "option", "value": "space"},
        ],
        cards_by_id={},
    )
    config = SelectivePuctConfig(enabled_triggers=("forced_response",))

    assert classify_search_triggers(env, np.array([0.5, 0.3, 0.2]), config) == ()


def test_own_event_mode_is_not_misclassified_as_opponent_event() -> None:
    env = SimpleNamespace(
        last_obs={"side": "us", "current_player": "us", "prompt": "US playing US/Japan Defense Pact", "hand": []},
        legal_actions=[
            {"type": "saito_choice", "decision": "option", "value": "event", "card": "usjapan"},
            {"type": "saito_choice", "decision": "option", "value": "ops", "card": "usjapan"},
        ],
        cards_by_id={"usjapan": {"player": "us", "scoring": False}},
    )
    config = SelectivePuctConfig(enabled_triggers=("opponent_event",), low_confidence_sample_rate=0.0)

    assert classify_search_triggers(env, np.array([0.4, 0.6]), config) == ()


def test_ops_menu_with_raw_place_is_not_chance_heavy() -> None:
    env = SimpleNamespace(
        last_obs={"side": "us", "current_player": "us", "prompt": "US plays 4 OPS:", "hand": []},
        legal_actions=[
            {"type": "saito_choice", "decision": "option", "value": "place"},
            {"type": "saito_choice", "decision": "option", "value": "coup"},
            {"type": "saito_choice", "decision": "option", "value": "realign"},
        ],
        cards_by_id={},
    )
    config = SelectivePuctConfig(enabled_triggers=("chance_heavy",), low_confidence_sample_rate=0.0)

    assert classify_search_triggers(env, np.array([0.8, 0.1, 0.1]), config) == ()


def test_controlled_country_prompt_is_not_misclassified_as_dice_roll() -> None:
    env = SimpleNamespace(
        last_obs={
            "side": "us",
            "current_player": "us",
            "phase": "action",
            "prompt": "Place 1 influence in each of 5 non USSR-controlled countries in Western Europe",
            "hand": [],
        },
        legal_actions=[{"decision": "country_click", "value": "france"}],
        cards_by_id={},
    )
    config = SelectivePuctConfig(enabled_triggers=("chance_heavy",), low_confidence_sample_rate=0.0)

    assert classify_search_triggers(env, np.array([1.0]), config) == ()


def test_generic_forced_response_consumes_a_meaningful_layer() -> None:
    last_obs = {"side": "us", "current_player": "us", "prompt": "Blockade: choose response", "hand": []}
    actions = [
        {"type": "saito_choice", "decision": "option", "value": "discard"},
        {"type": "saito_choice", "decision": "option", "value": "remove"},
    ]

    assert meaningful_decision_kind(last_obs, actions) == "forced_response"


@pytest.mark.parametrize(
    ("prompt", "actions", "expected"),
    [
        (
            "US playing Marshall Plan",
            [
                {"decision": "option", "value": "event"},
                {"decision": "option", "value": "ops"},
                {"decision": "option", "value": "space"},
            ],
            "mode",
        ),
        (
            "US plays 4 OPS:",
            [
                {"decision": "option", "value": "place"},
                {"decision": "option", "value": "coup"},
                {"decision": "option", "value": "realign"},
            ],
            "ops_action_type",
        ),
        (
            "Place 4 influence",
            [
                {"decision": "country_mouseup", "value": "france"},
                {"decision": "country_mouseup", "value": "italy"},
            ],
            "influence_allocation",
        ),
        (
            "Pick a country to coup",
            [
                {"decision": "country_mouseup", "value": "iran"},
                {"decision": "country_mouseup", "value": "iraq"},
            ],
            "coup_target",
        ),
    ],
)
def test_hierarchical_choices_are_meaningful_boundaries(prompt, actions, expected) -> None:
    last_obs = {"side": "us", "current_player": "us", "phase": "action", "prompt": prompt, "hand": []}

    assert meaningful_decision_kind(last_obs, actions) == expected


def test_single_legal_action_is_bookkeeping_not_a_meaningful_boundary() -> None:
    last_obs = {"side": "us", "current_player": "us", "prompt": "US playing Marshall Plan", "hand": []}

    assert meaningful_decision_kind(last_obs, [{"decision": "option", "value": "ops"}]) is None


def test_cross_turn_horizon_coalesces_allocation_and_preserves_information_masks() -> None:
    class CrossTurnEnv:
        def __init__(self) -> None:
            self.state = 0
            self.last_obs = {}
            self.legal_actions = []
            self._load()

        def _load(self) -> None:
            states = {
                0: (1, "us", "action", "US pick a card: ", "list", ["marshall", "nato"]),
                1: (1, "us", "action", "US playing Marshall Plan", "option", ["event", "ops"]),
                2: (1, "us", "action", "Place 2 influence", "country_mouseup", ["france", "italy"]),
                3: (1, "us", "action", "Place 1 influence", "country_mouseup", ["france", "italy"]),
                4: (1, "us", "action", "End of Round bookkeeping", "option", ["continue"]),
                5: (2, "ussr", "headline_ussr", "USSR pick headline card", "list", ["warsawpact", "fidel"]),
            }
            if self.state >= 6:
                self.last_obs = {"turn": 2, "side": "ussr", "phase": "terminal", "prompt": "", "hand": []}
                self.legal_actions = []
                return
            turn, side, phase, prompt, decision, values = states[self.state]
            self.last_obs = {
                "turn": turn,
                "action_round": 6 if turn == 1 else 0,
                "side": side,
                "current_player": side,
                "phase": phase,
                "prompt": prompt,
                "hand": values if decision == "list" else [],
            }
            self.legal_actions = [{"decision": decision, "value": value} for value in values]

        def observation(self) -> dict[str, np.ndarray]:
            side = str(self.last_obs.get("side") or "ussr")
            count = max(1, len(self.legal_actions))
            return {
                "action_mask": np.ones(count, dtype=np.float32),
                "marker": np.array([float(self.state)], dtype=np.float32),
                "us_hand": np.array([1.0, 0.0], dtype=np.float32) if side == "us" else np.zeros(2, dtype=np.float32),
                "ussr_hand": np.array([1.0, 0.0], dtype=np.float32) if side == "ussr" else np.zeros(2, dtype=np.float32),
            }

        def snapshot(self, compact: bool = False):
            return {"state": self.state}

        def restore(self, snapshot):
            self.state = int(snapshot["state"])
            self._load()
            return self.observation()

        def redeterminize(self, side: str, seed: int):
            assert side == "us"
            return self.observation()

        def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
            self.state += 1
            self._load()
            terminal = self.state >= 6
            return self.observation(), 0.0, terminal, False, {
                "winner": "us" if terminal and action == 0 else "ussr" if terminal else "",
                "chance_rolls": [],
                "chance_roll_sides": [],
                "forced_rolls_consumed": [],
            }

    env = CrossTurnEnv()
    config = SelectivePuctConfig(
        initial_simulations=2,
        medium_simulations=2,
        max_simulations=2,
        belief_particles=1,
        meaningful_depth=3,
        max_atomic_steps=16,
        max_expanded_nodes=32,
    )
    planner = SelectivePuctPlanner(EqualPolicyEvaluator(), env, config, seed=19)

    decision = planner.plan(copy.deepcopy(env.snapshot()), env.observation(), force_search=True)

    assert decision.max_meaningful_depth_reached == 3
    assert decision.max_turn_delta == 1
    assert decision.cross_turn_leaf_evaluations == 2
    assert decision.atomic_horizon_cutoffs == 0
    assert decision.hierarchical_targets
    assert {target.level for target in decision.hierarchical_targets} <= {
        "mode",
        "influence_allocation",
        "card",
    }
    for target in decision.hierarchical_targets:
        opponent_key = "ussr_hand" if target.side == "us" else "us_hand"
        assert np.asarray(target.observation[opponent_key]).sum() == 0.0


def test_olympic_exact_node_executes_only_observation_distinct_winners() -> None:
    env = OlympicEnv()
    enumerator = ExactChanceEnumerator()
    outcomes = enumerator.outcomes(env, env.snapshot(), 0)
    expected_us = sum(
        1
        for us_roll in range(1, 7)
        for ussr_roll in range(1, 7)
        if us_roll + 2 != ussr_roll and us_roll + 2 > ussr_roll
    )
    non_ties = sum(
        1
        for us_roll in range(1, 7)
        for ussr_roll in range(1, 7)
        if us_roll + 2 != ussr_roll
    )
    by_winner = {outcome.info["winner"]: outcome.probability for outcome in outcomes}

    assert len(outcomes) == 2
    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0)
    assert by_winner["us"] == pytest.approx(expected_us / non_ties)


def test_exact_chance_enumerates_conditional_roll_tree() -> None:
    env = ConditionalDiceEnv()
    outcomes = ExactChanceEnumerator().outcomes(env, env.snapshot(), 0)

    assert len(outcomes) == 4
    assert sorted(outcome.probability for outcome in outcomes) == pytest.approx([1 / 6, 1 / 6, 1 / 6, 1 / 2])
    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0)


def test_two_dice_remains_exact_at_production_bound() -> None:
    class TwoDiceEnv(ExactDiceEnv):
        def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
            rolls = [int(float(value) * 6) + 1 for value in forced_rolls[:2]]
            self.state = "terminal"
            info = {
                "winner": "us" if sum(rolls) >= 7 else "ussr",
                "chance_rolls": rolls,
                "chance_roll_sides": [6, 6],
                "forced_rolls_consumed": list(forced_rolls[:2]),
            }
            return self.observation(), 0.0, True, False, info

    env = TwoDiceEnv()
    enumerator = ExactChanceEnumerator(max_outcomes=36)
    outcomes = enumerator.outcomes(env, env.snapshot(), 0)

    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0)
    assert enumerator.enumerated_nodes == 1
    assert enumerator.sampled_large_nodes == 0


def test_three_dice_uses_audited_per_simulation_sample_at_production_bound() -> None:
    class ThreeDiceEnv(ExactDiceEnv):
        def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
            rolls = [int(float(value) * 6) + 1 for value in forced_rolls[:3]]
            self.state = "terminal"
            info = {
                "winner": "us" if sum(rolls) >= 11 else "ussr",
                "chance_rolls": rolls,
                "chance_roll_sides": [6, 6, 6],
                "forced_rolls_consumed": list(forced_rolls[:3]),
            }
            return self.observation(), 0.0, True, False, info

    env = ThreeDiceEnv()
    enumerator = ExactChanceEnumerator(max_outcomes=36)
    outcomes = enumerator.outcomes(env, env.snapshot(), 0)

    assert len(outcomes) == 1
    assert enumerator.enumerated_nodes == 0
    assert enumerator.sampled_large_nodes == 1


def test_large_chance_node_is_sampled_without_being_reported_exact() -> None:
    class FourDiceEnv(ExactDiceEnv):
        def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]):
            rolls = [int(float(value) * 6) + 1 for value in forced_rolls[:4]]
            self.state = "terminal"
            info = {
                "winner": "us" if sum(rolls) >= 14 else "ussr",
                "chance_rolls": rolls,
                "chance_roll_sides": [6, 6, 6, 6],
                "forced_rolls_consumed": list(forced_rolls[:4]),
            }
            return self.observation(), 0.0, True, False, info

    env = FourDiceEnv()
    enumerator = ExactChanceEnumerator(max_outcomes=216)
    outcomes = enumerator.outcomes(env, env.snapshot(), 0)

    assert len(outcomes) == 1
    assert outcomes[0].probability == 1.0
    assert enumerator.enumerated_nodes == 0
    assert enumerator.sampled_large_nodes == 1
