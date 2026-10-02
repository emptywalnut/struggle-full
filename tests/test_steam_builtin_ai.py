from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from scripts.compare_steam_ai_traces import audit_panel
from scripts.eval_steam_reconstructed_proxy_pair import build_pair_schedule
from struggle_ai.baselines.steam_builtin.bridge_search import BridgeProxySearchRunner
from struggle_ai.baselines.steam_builtin.card_policy import policy_for
from struggle_ai.baselines.steam_builtin.chance import expected_value, proxy_coup_outcomes, proxy_realignment_outcomes, two_dice
from struggle_ai.baselines.steam_builtin.config import STEAM_TUNED_WEIGHTS, SteamProxyConfig
from struggle_ai.baselines.steam_builtin.evaluator import (
    NativePreparedAccumulators,
    evaluate_proxy_leaf,
    extract_proxy_features,
    score_native_base_pair,
    terminal_utility,
)
from struggle_ai.baselines.steam_builtin.filters import filter_and_order
from struggle_ai.baselines.steam_builtin.hidden_hand import estimate_opponent_hand_uniform_proxy
from struggle_ai.baselines.steam_builtin.policy import SteamReconstructedProxy


def state(**updates):
    base = {"side": "us", "current_player": "us", "vp": 2, "defcon": 2, "action_round": 1,
            "countries": [{"id": "iran", "region": "mideast", "us": 2, "ussr": 0,
                           "stability": 2, "battleground": True}], "hand": ["a", "b"]}
    return {**base, **updates}


def test_native_constants_are_not_proxy_feature_weights():
    assert len(STEAM_TUNED_WEIGHTS) == 14
    assert len(extract_proxy_features(state(), "us")) == 5
    assert SteamProxyConfig().proxy_leaf_weights != STEAM_TUNED_WEIGHTS


def test_exact_recovered_score_base_accumulator_formula():
    fields = NativePreparedAccumulators(2, 0.5, (1, 2, 3, 4, 5, 6, 7), -1.0, 10.0, 3.0)
    # ((2 * 2 * .5) + 28) * -1 + 10 + 3 = -17
    assert score_native_base_pair(fields) == (-17.0, 17.0)


def test_proxy_leaf_perspective_and_terminal_timing():
    assert extract_proxy_features(state(), "ussr") == tuple(-x for x in extract_proxy_features(state(), "us"))
    assert evaluate_proxy_leaf(state(), "us") == -evaluate_proxy_leaf(state(), "ussr")
    assert terminal_utility(state(terminal=True, winner="us", terminal_reason_index=1), "us") == 55000
    assert terminal_utility(state(terminal=True, winner="us", terminal_reason_index=1, terminal_timing=3), "us") == 54700


def test_card_policy_defcon_filter_and_confirmed_region_quota():
    assert policy_for("Duck and Cover", "usa")["headline"] == 5
    risky = [{"value": "duckandcover", "card": "Duck and Cover", "prompt": "pick a card"},
             {"value": "space", "card": "Duck and Cover", "prompt": "pick a card"}]
    assert filter_and_order(risky, state(), "us")[0]["value"] == "space"
    countries = [{"id": f"c{i}", "region": "europe", "us": i, "ussr": 0} for i in range(6)]
    actions = [{"value": f"c{i}", "decision": "country_click"} for i in range(6)]
    kept = filter_and_order(actions, state(prompt="place influence", countries=countries), "us", limit=24)
    assert len(kept) == 4


def test_chance_helpers_and_hidden_hand_are_explicit_approximations():
    assert len(two_dice()) == 36
    assert sum(p for p, _ in two_dice()) == pytest.approx(1.0)
    assert expected_value(two_dice(), lambda rolls: rolls[0] - rolls[1]) == pytest.approx(0.0)
    assert expected_value(proxy_coup_outcomes(3, 2)) == pytest.approx(2.5)
    assert expected_value(proxy_realignment_outcomes()) == pytest.approx(0.0)
    first = estimate_opponent_hand_uniform_proxy(["known"], ["a", "b", "c"], 3, random.Random(7))
    second = estimate_opponent_hand_uniform_proxy(["known"], ["a", "b", "c"], 3, random.Random(7))
    assert first == second and first[0] == "known"


def test_recursive_callback_search_uses_opponent_minimum():
    proxy = SteamReconstructedProxy(SteamProxyConfig(search_state_depth=3))
    tree = {"root": [("safe", "opp_safe"), ("trap", "opp_trap")],
            "opp_safe": [("x", "safe_end")], "opp_trap": [("x", "win"), ("y", "loss")]}
    states = {"opp_safe": state(current_player="ussr", node="opp_safe"),
              "opp_trap": state(current_player="ussr", node="opp_trap"),
              "safe_end": state(terminal=True, winner="us", terminal_reason_index=0, node="safe_end"),
              "win": state(terminal=True, winner="us", terminal_reason_index=1, node="win"),
              "loss": state(terminal=True, winner="ussr", terminal_reason_index=1, node="loss")}
    def expand(obs):
        edges = tree.get(obs["node"], [])
        actions = [{"value": label} for label, _ in edges]
        return actions, lambda action: states[next(target for label, target in edges if label == action["value"])]
    assert proxy.search_value(state(node="root"), "us", expand) == 0.0


class FakeBridge:
    def __init__(self):
        self.node = "root"
        self.states = {
            "root": state(node="root", legal_actions=[{"value": "safe"}, {"value": "trap"}]),
            "safe": state(node="safe", current_player="ussr", legal_actions=[{"value": "end"}]),
            "trap": state(node="trap", current_player="ussr", legal_actions=[{"value": "win"}, {"value": "loss"}]),
            "draw": state(node="draw", terminal=True, winner="us", terminal_reason_index=0, legal_actions=[]),
            "win": state(node="win", terminal=True, winner="us", terminal_reason_index=1, legal_actions=[]),
            "loss": state(node="loss", terminal=True, winner="ussr", terminal_reason_index=1, legal_actions=[]),
        }
    def snapshot(self, **_kwargs): return {"node": self.node}
    def restore(self, snapshot): self.node = snapshot["node"]; return self.states[self.node]
    def step(self, action):
        edges = {("root", "safe"): "safe", ("root", "trap"): "trap", ("safe", "end"): "draw",
                 ("trap", "win"): "win", ("trap", "loss"): "loss"}
        self.node = edges[(self.node, action["value"])]
        return {"observation": self.states[self.node]}


def test_bridge_runner_is_really_recursive_and_restores_root():
    bridge = FakeBridge()
    runner = BridgeProxySearchRunner(bridge, SteamProxyConfig(search_state_depth=2, max_candidates=8))
    decision = runner.choose_action(bridge.states["root"])
    assert decision.action_index == 0
    assert decision.expanded_nodes == 5
    assert bridge.node == "root"
    assert "not native expectimax" in decision.chance_mode


def test_paired_schedule_and_trace_gate():
    schedule = build_pair_schedule(100, 2)
    assert schedule == [
        {"pair": 0, "seed": 100, "proxy_side": "us"},
        {"pair": 0, "seed": 100, "proxy_side": "ussr"},
        {"pair": 1, "seed": 101, "proxy_side": "us"},
        {"pair": 1, "seed": 101, "proxy_side": "ussr"},
    ]
    result = audit_panel({"panel_id": "missing", "native_state": {}, "legal_actions": []})
    assert not result["minimum_native_evidence"]
    assert not result["action_comparable"]


def test_native_oracle_spec_matches_supplied_harness_without_execution():
    root = Path(__file__).resolve().parents[1]
    harness = root / "tools/native_oracle_smoke_macos.py"
    spec = json.loads((root / "artifacts/steam_native_oracle_spec.json").read_text())
    digest = hashlib.sha256(harness.read_bytes()).hexdigest()
    assert digest == spec["harness"]["sha256"]
    assert digest == "5e257fedca3b392d981fc46bd604341fcc8fc3f3f8f2f40241940628e3ae6371"
    assert spec["execute_on_linux"] is False
    assert spec["abi"]["AppPlayerData"]["CMediumAI_player_type"] == 2
    assert spec["abi"]["GameOption_callback_record"]["size"] == 64
    assert spec["verified_macos_feasibility"]["bounded_engine_events_observed"] == 104
    assert spec["verified_macos_feasibility"]["CMediumAI_selections_observed"] == 0
    assert spec["verified_macos_feasibility"]["stalled_loop_observation"]["evidentiary_value"].startswith("none")
    capture = spec["experimental_action_capture"]
    assert capture["status"] == (
        "HOST_FIX_LOCALLY_VERIFIED_BUT_ACTION_CAPTURE_BLOCKED_BY_DISABLED_DEVELOPER_MODE"
    )
    assert spec["experimental_host_fix_local_verification"]["temporary_move_buffer_committed"]
    assert spec["experimental_host_fix_local_verification"]["CMediumAI_selections_captured"] == 0
    blocker = spec["action_capture_environment_blocker"]
    assert blocker["security_setting_changed"] is False
    assert "DevToolsSecurity -enable" in blocker["one_time_user_authorized_prerequisite"]


def test_genuine_native_oracle_is_preferred_gate_and_proxy_is_not_authority():
    panel = {
        "panel_id": "oracle-1",
        "native_oracle": {
            "harness_sha256": "ec5d7ad20112e331e4b14d1859b1c296f207a2e42cd570a4fd9bf014c21fa438",
            "observer_sha256": "4f82614f5fabbbfb3869dd682b7673bb1d803e7b8cad6788d4917fd5cd17aba0",
            "validator_sha256": "2dd1f59d6d379d6524c19e2a407e7b09c464ff96469191f551d39e7a2ebeee67",
            "host_smoke_harness_sha256": "5e257fedca3b392d981fc46bd604341fcc8fc3f3f8f2f40241940628e3ae6371",
            "ai_class": "CMediumAI",
            "player_type": 2,
            "headless": True,
            "ai_output_complete": True,
            "selection_provenance": "instrumented_CMediumAI_to_CGameStateOptions_selection",
            "capture_session_id": "test-session-0001",
            "raw_capture_sha256": "d" * 64,
            "capture_validation": {
                "first_genuine_cmedium_selected_action_with_verified_ai_actor_provenance": True,
            },
        },
        "native_state": {
            "world_serialization_sha256": "a" * 64,
            "state_machine_serialization_sha256": "b" * 64,
            "capture_complete_for_mapping": True,
            "private_hands": {"us": [], "ussr": []},
            "known_card_flags": {},
            "deck_order": [],
            "discard": [],
            "removed": [],
            "rng": {"call_sequence_complete": True},
            "estimated_opponent_hand": [],
        },
        "rules": {"deck": "optional", "us_bonus": 2, "optional_cards": [], "variants": {}},
        "mapping_verification": {
            "native_state_to_saito_lossless_verified": True,
            "native_options_to_saito_lossless_verified": True,
            "rules_transition_parity_verified": True,
            "evidence_ref": "manual-fixture",
        },
        "legal_actions": [{
            "native_index": 0,
            "selection_id": 17,
            "selection_hint": 40970,
            "callback_file_address": 0x1234,
            "callback_data_raw": 17,
            "mapping_status": "VERIFIED_LOSSLESS",
            "canonical_key": "country:iran",
            "saito_action": {"value": "iran"},
        }],
        "steam_selection": {
            "native_index": 0,
            "selection_id": 17,
            "selection_hint": 40970,
            "canonical_key": "country:iran",
        },
        "comparison_outputs": [{"participant_id": "candidate", "selected_canonical_key": "country:iran"}],
    }
    result = audit_panel(panel)
    assert result["preferred_parity_gate"] == "genuine_CMediumAI_native_oracle"
    assert result["native_oracle_verified"]
    assert result["action_comparable"]
    assert result["comparison_results"][0]["action_match"]
    assert "python_proxy_diagnostic" not in result
    diagnostic = audit_panel(panel, include_proxy_diagnostic=True)
    assert diagnostic["action_comparable"]
    assert not diagnostic["python_proxy_diagnostic"]["available"]
    assert diagnostic["python_proxy_diagnostic"]["missing"] == ["proxy_observation"]
