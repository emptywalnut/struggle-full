from __future__ import annotations

import random

import pytest

from struggle_ai.browser_ai import BrowserPolicyService
from struggle_ai.bridge_client import TwilightBridgeClient
from struggle_ai.policies import choose_heuristic_action


def test_browser_ai_uses_existing_heuristic_policy_semantics() -> None:
    legal_actions = [
        {"decision": "option", "value": "realign", "label": "Realignment"},
        {"decision": "option", "value": "event", "label": "Event"},
        {"decision": "option", "value": "ops", "label": "Operations"},
    ]
    expected = choose_heuristic_action(legal_actions, rng=random.Random(11), obs={"defcon": 5})
    result = BrowserPolicyService().handle(
        {
            "cmd": "choose",
            "session_id": "same-semantics",
            "seed": 11,
            "policy": "heuristic",
            "observation": {"defcon": 5},
            "legal_actions": legal_actions,
        }
    )
    assert result["action_index"] == expected
    assert result["policy"] == "heuristic"


def test_browser_random_policy_never_leaves_legal_range() -> None:
    service = BrowserPolicyService()
    actions = [{"value": index} for index in range(4)]
    choices = [
        service.handle(
            {
                "cmd": "choose",
                "session_id": "random-session",
                "seed": 7,
                "policy": "random",
                "legal_actions": actions,
            }
        )["action_index"]
        for _ in range(20)
    ]
    assert all(0 <= choice < len(actions) for choice in choices)
    assert len(set(choices)) > 1


def test_unknown_model_id_never_silently_falls_back() -> None:
    class EmptyRegistry:
        @staticmethod
        def contains(_model_id: str) -> bool:
            return False

    service = BrowserPolicyService()
    service._registry = EmptyRegistry()
    with pytest.raises(ValueError, match="unknown registered model ID"):
        service.handle(
            {
                "cmd": "choose",
                "session_id": "no-fallback",
                "seed": 5,
                "policy": "not-a-model",
                "observation": {"side": "us"},
                "legal_actions": [{"decision": "option", "value": "event"}],
            }
        )


def test_browser_history_state_is_isolated_per_session() -> None:
    service = BrowserPolicyService()
    with TwilightBridgeClient() as bridge:
        cards = bridge.cards()
        before = bridge.reset(41)
        legal = before["legal_actions"]
        action = legal[0]
        after = bridge.step(action)["observation"]
    recorded = service.handle(
        {
            "cmd": "record",
            "session_id": "first-game",
            "cards": cards,
            "before": before,
            "after": after,
            "legal_actions": legal,
            "action": action,
            "action_index": 0,
            "side": "ussr",
        }
    )
    assert recorded["history_length"] == 1
    assert recorded["state_scope"] == "session:first-game:ussr"
    assert service.encoders["first-game"].history[0]["side"] == "ussr"
    assert "second-game" not in service.encoders
    service.handle({"cmd": "forget", "session_id": "first-game"})
    assert "first-game" not in service.encoders


def test_browser_model_inference_honors_registered_defcon_filter_without_changing_unfiltered_semantics(monkeypatch) -> None:
    class Registry:
        def __init__(self, mode: str) -> None:
            self.mode = mode

        @staticmethod
        def contains(_model_id: str) -> bool:
            return True

        def environment_for(self, _model_id: str, _side: str) -> dict[str, object]:
            return {"defcon_suicide_mode": self.mode, "force_setup_heuristic": False}

        @staticmethod
        def model_for(_model_id: str, _side: str) -> tuple[object, str]:
            return object(), "cpu"

    seen_legal_counts: list[int] = []

    def choose_first(_model, _encoded, legal_count, *, device):
        seen_legal_counts.append(legal_count)
        return 0, {"raw_value": 0.0, "calibrated_value": 0.0, "device": device}

    monkeypatch.setattr("struggle_ai.batched_inference.choose_legal_action_index", choose_first)
    with TwilightBridgeClient() as bridge:
        cards = bridge.cards()
        observation = bridge.reset(77)
    observation = {
        **observation,
        "side": "ussr",
        "current_player": "ussr",
        "phase": "action",
        "defcon": 2,
        "events": {},
        "prompt": "USSR plays 1 OPS:",
    }
    legal_actions = [
        {"type": "saito_choice", "decision": "option", "value": "coup", "prompt": observation["prompt"]},
        {"type": "saito_choice", "decision": "option", "value": "place", "prompt": observation["prompt"]},
        {"type": "saito_choice", "decision": "option", "value": "realign", "prompt": observation["prompt"]},
    ]

    filtered = BrowserPolicyService()
    filtered._registry = Registry("hard_filter")
    filtered_result = filtered.handle({
        "cmd": "choose", "session_id": "filtered", "seed": 1, "policy": "model",
        "cards": cards, "observation": observation, "legal_actions": legal_actions,
    })
    assert seen_legal_counts[-1] == 2
    assert legal_actions[filtered_result["action_index"]]["value"] == "place"

    unfiltered = BrowserPolicyService()
    unfiltered._registry = Registry("none")
    unfiltered_result = unfiltered.handle({
        "cmd": "choose", "session_id": "unfiltered", "seed": 1, "policy": "model",
        "cards": cards, "observation": observation, "legal_actions": legal_actions,
    })
    assert seen_legal_counts[-1] == 3
    assert legal_actions[unfiltered_result["action_index"]]["value"] == "coup"
