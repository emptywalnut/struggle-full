#!/usr/bin/env python3
"""Audit genuine CMediumAI oracle panels and optional mapped outputs.

The preferred parity gate is a captured native ``CMediumAI`` selection versus a
participant output after a lossless, verified native-to-runner translation.  The
Python reconstruction is available only as an explicit shallow diagnostic; it
can never make the preferred native-oracle gate pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from struggle_ai.baselines.steam_builtin import SteamProxyConfig, SteamReconstructedProxy

MINIMUM_NATIVE_KEYS = {
    "world_serialization_sha256", "state_machine_serialization_sha256", "private_hands",
    "known_card_flags", "deck_order", "rng", "estimated_opponent_hand",
}
REQUIRED_MAPPING_FLAGS = {
    "native_state_to_saito_lossless_verified",
    "native_options_to_saito_lossless_verified",
    "rules_transition_parity_verified",
}
HOST_SMOKE_HARNESS_SHA256 = "5e257fedca3b392d981fc46bd604341fcc8fc3f3f8f2f40241940628e3ae6371"
CAPTURE_HOST_SHA256 = "ec5d7ad20112e331e4b14d1859b1c296f207a2e42cd570a4fd9bf014c21fa438"
CAPTURE_OBSERVER_SHA256 = "4f82614f5fabbbfb3869dd682b7673bb1d803e7b8cad6788d4917fd5cd17aba0"
CAPTURE_VALIDATOR_SHA256 = "2dd1f59d6d379d6524c19e2a407e7b09c464ff96469191f551d39e7a2ebeee67"
AI_SELECTION_PROVENANCE = {
    "instrumented_CMediumAI_to_CGameStateOptions_selection",
}


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _complete_native_state(native: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = sorted(MINIMUM_NATIVE_KEYS - set(native))
    invalid = []
    if not _sha256(native.get("world_serialization_sha256")):
        invalid.append("world_serialization_sha256")
    if not _sha256(native.get("state_machine_serialization_sha256")):
        invalid.append("state_machine_serialization_sha256")
    hands = native.get("private_hands")
    if not isinstance(hands, dict) or not all(
        isinstance(hands.get(side), list) for side in ("us", "ussr")
    ):
        invalid.append("private_hands.us_and_ussr")
    for field in ("deck_order", "discard", "removed", "estimated_opponent_hand"):
        if not isinstance(native.get(field), list):
            invalid.append(field)
    if not isinstance(native.get("known_card_flags"), dict):
        invalid.append("known_card_flags")
    rng = native.get("rng")
    if not isinstance(rng, dict) or rng.get("call_sequence_complete") is not True:
        invalid.append("rng.call_sequence_complete")
    if native.get("capture_complete_for_mapping") is not True:
        invalid.append("capture_complete_for_mapping")
    return not missing and not invalid, missing + invalid


def audit_panel(panel: dict[str, Any], include_proxy_diagnostic: bool = False) -> dict[str, Any]:
    native = panel.get("native_state") or {}
    native_complete, missing_native = _complete_native_state(native)
    oracle = panel.get("native_oracle") or {}
    oracle_verified = (
        oracle.get("host_smoke_harness_sha256") == HOST_SMOKE_HARNESS_SHA256
        and oracle.get("harness_sha256") == CAPTURE_HOST_SHA256
        and oracle.get("observer_sha256") == CAPTURE_OBSERVER_SHA256
        and oracle.get("validator_sha256") == CAPTURE_VALIDATOR_SHA256
        and oracle.get("ai_class") == "CMediumAI"
        and oracle.get("player_type") == 2
        and oracle.get("headless") is True
        and oracle.get("ai_output_complete") is True
        and oracle.get("selection_provenance") in AI_SELECTION_PROVENANCE
        and isinstance(oracle.get("capture_session_id"), str)
        and len(oracle["capture_session_id"]) >= 8
        and _sha256(oracle.get("raw_capture_sha256"))
        and (oracle.get("capture_validation") or {}).get(
            "first_genuine_cmedium_selected_action_with_verified_ai_actor_provenance"
        ) is True
    )
    options = panel.get("legal_actions") or []
    mapping_complete = bool(options) and all(
        option.get("mapping_status") == "VERIFIED_LOSSLESS"
        and option.get("canonical_key")
        and isinstance(option.get("native_index"), int)
        and isinstance(option.get("selection_id"), int)
        and isinstance(option.get("selection_hint"), int)
        and isinstance(option.get("callback_file_address"), int)
        and isinstance(option.get("callback_data_raw"), int)
        and isinstance(option.get("saito_action"), dict)
        for option in options
    )
    mapping = panel.get("mapping_verification") or {}
    verified_mapping_flags = {
        key for key in REQUIRED_MAPPING_FLAGS if mapping.get(key) is True
    }
    missing_mapping_flags = sorted(REQUIRED_MAPPING_FLAGS - verified_mapping_flags)
    mapping_evidence_present = bool(mapping.get("evidence_ref"))
    rules = panel.get("rules") or {}
    rules_manifest_complete = bool(
        rules.get("deck") == "optional"
        and rules.get("us_bonus") == 2
        and isinstance(rules.get("optional_cards"), list)
        and isinstance(rules.get("variants"), dict)
    )
    obs = panel.get("proxy_observation")
    comparable = (
        oracle_verified
        and native_complete
        and mapping_complete
        and not missing_mapping_flags
        and mapping_evidence_present
        and rules_manifest_complete
    )
    selection = panel.get("steam_selection") or {}
    selected_option = next(
        (option for option in options if option.get("native_index") == selection.get("native_index")),
        None,
    )
    selection_identity_matches = bool(
        selected_option
        and selected_option.get("selection_id") == selection.get("selection_id")
        and selected_option.get("selection_hint") == selection.get("selection_hint")
        and selected_option.get("canonical_key") == selection.get("canonical_key")
    )
    comparable = comparable and selection_identity_matches
    comparisons = panel.get("comparison_outputs") or []
    result: dict[str, Any] = {
        "panel_id": panel.get("panel_id"),
        "preferred_parity_gate": "genuine_CMediumAI_native_oracle",
        "native_oracle_verified": oracle_verified,
        "minimum_native_evidence": not missing_native,
        "missing_native_evidence": missing_native,
        "action_mapping_complete": mapping_complete,
        "native_selection_identity_matches_mapped_option": selection_identity_matches,
        "missing_mapping_verification": missing_mapping_flags,
        "mapping_evidence_present": mapping_evidence_present,
        "rules_manifest_complete": rules_manifest_complete,
        "action_comparable": comparable,
        "comparison_results": [],
        "behavioral_parity_demonstrated": False,
        "behavioral_parity_limit": "complete-game mapped native traces are required",
    }
    if comparable:
        steam_key = panel.get("steam_selection", {}).get("canonical_key")
        result["comparison_results"] = [
            {
                "participant_id": comparison.get("participant_id"),
                "selected_canonical_key": comparison.get("selected_canonical_key"),
                "native_canonical_key": steam_key,
                "action_match": comparison.get("selected_canonical_key") == steam_key,
            }
            for comparison in comparisons
        ]
    if include_proxy_diagnostic:
        diagnostic_ready = mapping_complete and isinstance(obs, dict)
        result["python_proxy_diagnostic"] = {
            "requested": True,
            "available": diagnostic_ready,
            "mode": "shallow_ordering_only",
            "parity_gate_authority": False,
        }
        if diagnostic_ready:
            legal = [option["saito_action"] for option in options]
            translated_obs = {**obs, "legal_actions": legal}
            proxy = SteamReconstructedProxy(
                SteamProxyConfig(seed=int(panel.get("steam_selection", {}).get("evaluation_seed") or 1))
            )
            proxy_index = proxy.choose_action(legal, translated_obs)
            proxy_key = options[proxy_index]["canonical_key"]
            result["python_proxy_diagnostic"].update({
                "selected_canonical_key": proxy_key,
                "native_canonical_key": panel.get("steam_selection", {}).get("canonical_key"),
                "action_match": proxy_key == panel.get("steam_selection", {}).get("canonical_key"),
            })
        else:
            result["python_proxy_diagnostic"]["missing"] = [
                field for field, present in (
                    ("complete saito_action mapping", mapping_complete),
                    ("proxy_observation", isinstance(obs, dict)),
                ) if not present
            ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panels", type=Path, help="JSONL decision panels conforming to artifacts/steam_ai_trace_schema.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-proxy-diagnostic",
        action="store_true",
        help="also run shallow Python proxy ordering; never counts toward parity",
    )
    args = parser.parse_args()
    panels = [json.loads(line) for line in args.panels.read_text().splitlines() if line.strip()]
    results = [audit_panel(panel, include_proxy_diagnostic=args.include_proxy_diagnostic) for panel in panels]
    comparable = [result for result in results if result["action_comparable"]]
    comparisons = [comparison for result in comparable for comparison in result["comparison_results"]]
    payload = {
        "preferred_parity_gate": "genuine_CMediumAI_native_oracle",
        "python_proxy_is_parity_authority": False,
        "panels": len(results),
        "comparable_panels": len(comparable),
        "comparison_outputs": len(comparisons),
        "matches": sum(bool(comparison.get("action_match")) for comparison in comparisons),
        "all_comparable_match": bool(comparisons) and all(comparison.get("action_match") for comparison in comparisons),
        "results": results,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
