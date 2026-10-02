#!/usr/bin/env python3
"""Project minimized bridge catalogs through the canonical training encoder."""

from __future__ import annotations

import json
from pathlib import Path

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.features import encode_observation


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "bridge_diagnostics.json"
OUTPUT = HERE / "env_mask_diagnostics.json"


def canonical(actions: list[dict]) -> list[dict]:
    return [
        {
            "type": action.get("type", "saito_dom"),
            "decision": action.get("decision") or ("country_click" if action.get("event") == "click" else "option"),
            "value": action.get("value", ""),
            "label": action.get("value", ""),
            "selector": action.get("selector"),
            "event": action.get("event"),
            "prompt": "audit diagnostic",
        }
        for action in actions
    ]


def main() -> None:
    diagnostics = json.loads(SOURCE.read_text())
    env = TwilightStruggleEnv({"seed": 90831, "force_setup_heuristic": False})
    try:
        env.reset(seed=90831)
        base_observation = env.obs_with_history(env.last_obs)
        projections = []
        for item in diagnostics["bounded"]:
            actions = canonical(item["after_first_selection"]["choices"])
            encoded = encode_observation(base_observation, actions, env.feature_spec)
            projections.append({
                "kind": "bounded_country",
                "card": item["card"],
                "expected_repeat_target": item["target"],
                "catalog_contains_expected_target": any(action["value"] == item["target"] for action in actions),
                "catalog_size": len(actions),
                "action_mask_sum": int(encoded["action_mask"].sum()),
                "mask_exactly_mirrors_catalog": int(encoded["action_mask"].sum()) == len(actions),
            })
        for item in diagnostics["regional"]:
            actions = canonical(item["after_printed_ops"]["choices"])
            encoded = encode_observation(base_observation, actions, env.feature_spec)
            projections.append({
                "kind": "regional_bonus",
                "bonus": item["kind"],
                "expected_restricted_country_target": item["target"],
                "catalog_values": [action["value"] for action in actions],
                "catalog_contains_expected_target": any(action["value"] == item["target"] for action in actions),
                "catalog_size": len(actions),
                "action_mask_sum": int(encoded["action_mask"].sum()),
                "mask_exactly_mirrors_catalog": int(encoded["action_mask"].sum()) == len(actions),
            })
        OUTPUT.write_text(json.dumps({
            "encoder": "struggle_ai.features.encode_observation",
            "max_actions": int(env.action_space.n),
            "projections": projections,
        }, indent=2, sort_keys=True) + "\n")
    finally:
        env.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
