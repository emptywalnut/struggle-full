"""JSON-lines adapter exposing the repository's existing baseline policies.

The browser server deliberately calls this module instead of maintaining a
JavaScript copy of the heuristic.  That keeps browser play on the same policy
API and semantics as ``struggle_ai.play`` and the training environment.
"""

from __future__ import annotations

import json
import random
import sys
from typing import Any

from struggle_ai.policies import choose_heuristic_action, score_action


class BrowserPolicyService:
    def __init__(self) -> None:
        self.rngs: dict[str, random.Random] = {}
        self.encoders: dict[str, Any] = {}
        self._registry: Any = None

    @property
    def registry(self) -> Any:
        if self._registry is None:
            from struggle_ai.browser_models import BrowserModelRegistry

            self._registry = BrowserModelRegistry()
        return self._registry

    def rng_for(self, session_id: str, seed: int) -> random.Random:
        if session_id not in self.rngs:
            self.rngs[session_id] = random.Random(seed)
        return self.rngs[session_id]

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        command = message.get("cmd")
        if command == "ping":
            return {"service": "struggle_ai.browser_ai", "policies": ["heuristic", "random"]}
        if command == "models":
            return self.registry.catalog()
        if command == "load":
            return self.registry.load(str(message.get("model_id", "")), str(message.get("side", "")))
        if command == "record":
            return self.record_transition(message)
        if command == "forget":
            session_id = str(message.get("session_id", ""))
            self.rngs.pop(session_id, None)
            self.encoders.pop(session_id, None)
            return {"forgotten": True}
        if command != "choose":
            raise ValueError(f"unknown browser policy command: {command}")

        legal_actions = message.get("legal_actions") or []
        if not legal_actions:
            raise ValueError("no legal actions")
        session_id = str(message.get("session_id", "default"))
        rng = self.rng_for(session_id, int(message.get("seed", 1)))
        policy = str(message.get("policy", "heuristic"))
        observation = message.get("observation") or None
        if policy == "random":
            index = rng.randrange(len(legal_actions))
            scores = None
        elif policy == "heuristic":
            index = choose_heuristic_action(legal_actions, rng=rng, obs=observation)
            scores = [score_action(action, obs=observation) for action in legal_actions]
        else:
            index, model_result = self.choose_model(message, policy, legal_actions)
            scores = None
            return {
                "action_index": index,
                "policy": policy,
                "source": f"ai:model:{policy}",
                "device": model_result["device"],
                "raw_value": model_result["raw_value"],
                "calibrated_value": model_result["calibrated_value"],
            }
        return {
            "action_index": index,
            "policy": policy,
            "source": f"ai:{policy}",
            "scores": scores,
        }

    def encoder_for(self, message: dict[str, Any]) -> Any:
        from struggle_ai.env import TwilightStruggleEnv

        session_id = str(message.get("session_id", ""))
        if not session_id:
            raise ValueError("model inference requires a session_id")
        encoder = self.encoders.get(session_id)
        if encoder is None:
            cards = message.get("cards") or []
            observation = message.get("observation") or message.get("before") or {}
            countries = observation.get("countries") or []
            if not cards or not countries:
                raise ValueError("model inference requires the session card and country catalog")
            encoder = TwilightStruggleEnv.encoding_state(cards, countries)
            self.encoders[session_id] = encoder
        return encoder

    def choose_model(
        self,
        message: dict[str, Any],
        model_id: str,
        legal_actions: list[dict[str, Any]],
    ) -> tuple[int, dict[str, Any]]:
        from struggle_ai.batched_inference import choose_legal_action_index

        observation = message.get("observation") or {}
        side = str(observation.get("side") or observation.get("current_player") or "")
        if side not in {"us", "ussr"}:
            raise ValueError(f"model observation has invalid acting side: {side!r}")
        if not self.registry.contains(model_id):
            raise ValueError(f"unknown registered model ID: {model_id}")
        encoder = self.encoder_for(message)
        environment = self.registry.environment_for(model_id, side)
        inference_actions = list(legal_actions)
        raw_indices = list(range(len(legal_actions)))
        if environment.get("defcon_suicide_mode") == "hard_filter":
            encoder.defcon_suicide_mode = "hard_filter"
            encoder.episode_filtered_actions = []
            filtered = encoder.filtered_legal_actions(observation, legal_actions)
            raw_indices = [next(index for index, action in enumerate(legal_actions) if action is candidate) for candidate in filtered]
            inference_actions = filtered
        encoder.last_obs = observation
        encoder.legal_actions = inference_actions
        encoder.last_encoded_obs = None
        encoded = encoder.encode_current_observation(observation)
        model, device = self.registry.model_for(model_id, side)
        index, result = choose_legal_action_index(
            model,
            encoded,
            len(inference_actions),
            device=device,
        )
        return raw_indices[index], {
            "device": device,
            "raw_value": float(result["raw_value"]),
            "calibrated_value": float(result["calibrated_value"]),
        }

    def record_transition(self, message: dict[str, Any]) -> dict[str, Any]:
        before = message.get("before") or {}
        after = message.get("after") or {}
        legal_actions = message.get("legal_actions") or []
        action = message.get("action") or {}
        side = str(message.get("side") or before.get("side") or before.get("current_player") or "")
        action_index = int(message.get("action_index", -1))
        if side not in {"us", "ussr"}:
            raise ValueError(f"transition has invalid acting side: {side!r}")
        if action_index < 0 or action_index >= len(legal_actions):
            raise ValueError(f"transition action index {action_index} is outside {len(legal_actions)} legal actions")
        if legal_actions[action_index] != action:
            raise ValueError("transition action does not match the authoritative legal-action catalog index")
        encoder = self.encoder_for(message)
        encoder.last_obs = before
        encoder.legal_actions = list(legal_actions)
        encoder.last_encoded_obs = None
        selected_features = encoder.selected_action_features(action_index)
        before_context = encoder.action_context(before)
        after_context = encoder.action_context(after)
        encoder.append_history_entry(before_context, side, selected_features)
        encoder.append_or_update_card_history_entry(before, before_context, after_context, side, action)
        encoder.last_obs = after
        encoder.legal_actions = list(after.get("legal_actions") or [])
        encoder.last_encoded_obs = None
        return {
            "recorded": True,
            "history_length": len(encoder.history),
            "card_history_length": len(encoder.card_history),
            "state_scope": f"session:{message.get('session_id')}:{side}",
        }


def main() -> None:
    service = BrowserPolicyService()
    for line in sys.stdin:
        if not line.strip():
            continue
        message: dict[str, Any] = {}
        try:
            message = json.loads(line)
            result = service.handle(message)
            response = {"id": message.get("id"), "ok": True, "result": result}
        except Exception as exc:  # protocol boundary: return errors to Node
            response = {"id": message.get("id"), "ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
