from __future__ import annotations

from struggle_ai.bridge_replay_audit import replay_episode_semantics


ACTION = {
    "type": "saito_choice",
    "decision": "list",
    "value": "duckandcover",
    "prompt": "US pick headline card",
}


class FakeBridge:
    def __init__(self, after: dict):
        self.after = after

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def reset(self, _seed: int) -> dict:
        return {"legal_actions": [ACTION]}

    def step(self, _action: dict) -> dict:
        return {"observation": self.after}


class FakeBridgeWithLog(FakeBridge):
    def __init__(self, after: dict, logs: list[list[str]]):
        super().__init__(after)
        self.logs = iter(logs)

    def log(self) -> dict:
        return {"log": next(self.logs)}


def episode(after: dict) -> dict:
    return {
        "seed": 7,
        "actions": [
            {
                "step": 12,
                "turn": 1,
                "action": dict(ACTION),
                "after": dict(after),
            }
        ],
    }


def test_replay_episode_semantics_accepts_matching_state() -> None:
    after = {"phase": "headline_resolve", "turn": 1, "vp": 0, "defcon": 4}
    result = replay_episode_semantics(episode(after), client_factory=lambda: FakeBridge(after))
    assert result == {"passed": True, "seed": 7, "actions_replayed": 1}


def test_replay_episode_semantics_rejects_stale_headline_queue_semantics() -> None:
    recorded = {"phase": "action", "turn": 1, "vp": 0, "defcon": 5}
    replayed = {"phase": "headline_resolve", "turn": 1, "vp": 0, "defcon": 4}
    result = replay_episode_semantics(
        episode(recorded), client_factory=lambda: FakeBridge(replayed)
    )
    assert result["passed"] is False
    assert result["reason"] == "semantic_observation_mismatch"
    assert result["source_step"] == 12
    assert result["differences"] == {
        "phase": {"recorded": "action", "replayed": "headline_resolve"},
        "defcon": {"recorded": 5, "replayed": 4},
    }


def test_replay_episode_semantics_rejects_event_log_mismatch() -> None:
    after = {"phase": "action", "turn": 1, "vp": 0, "defcon": 4}
    sample = episode(after)
    sample["actions"][0]["saito_log_delta"] = ["expected effect"]
    result = replay_episode_semantics(
        sample,
        client_factory=lambda: FakeBridgeWithLog(
            after, [["reset"], ["reset", "different effect"]]
        ),
    )
    assert result["passed"] is False
    assert result["reason"] == "saito_log_delta_mismatch"
