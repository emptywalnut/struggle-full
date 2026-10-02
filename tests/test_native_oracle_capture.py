from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.native_oracle_capture_macos import (
    ai_player_ids,
    count_verified_ai_selections,
    parse_args,
    structure_self_check,
    valid_capture_session_id,
    validate_args,
)
from tools.validate_native_oracle_capture import (
    EXPECTED_HOST_SHA256,
    EXPECTED_OBSERVER_SHA256,
    validate,
)


EXPECTED_SHA = "b45dc29df33f8c11312c2fe64d8b22764d153afb874a8e3f6867465ede0a1fa7"


def test_experimental_host_abi_and_layout_guards():
    assert structure_self_check()["passed"]
    assert ai_player_ids("human-ussr-ai-us") == [1]
    assert ai_player_ids("ai-ussr-human-us") == [0]
    assert ai_player_ids("both-ai") == [0, 1]
    with pytest.raises(ValueError, match="both-ai"):
        validate_args(parse_args(["--layout", "both-ai"]))
    with pytest.raises(ValueError, match="CTutorialAI"):
        validate_args(parse_args(["--ai-difficulty", "3"]))
    assert valid_capture_session_id("session-0001")
    assert not valid_capture_session_id("short")
    assert not valid_capture_session_id("session with spaces")


def test_trace_poll_counts_only_configured_ai_commits(tmp_path: Path):
    trace = tmp_path / "capture.jsonl"
    trace.write_text(
        "\n".join([
            json.dumps({"event": "offline_host_option_stage", "actor_id": 0}),
            "{partial",
            json.dumps({"event": "ai_option_selection", "actor_id": 1,
                        "capture_session_id": "session-one",
                        "ai_actor_provenance_gate": True}),
            json.dumps({"event": "ai_option_selection", "actor_id": 0,
                        "capture_session_id": "session-two",
                        "ai_actor_provenance_gate": True}),
        ]),
        encoding="utf-8",
    )
    assert count_verified_ai_selections(trace, [1]) == 1
    assert count_verified_ai_selections(trace, [0, 1]) == 2
    assert count_verified_ai_selections(trace, [0, 1], "session-one") == 1


def test_validator_requires_native_actor_and_option_identity_match():
    options = {
        "defensive_layout_check_passed": True,
        "identity_fingerprint_sha256": "a" * 64,
        "options": [
            {"index": 0, "selection_id": 17, "selection_hint": 40970, "display_text": "Iran"}
        ],
    }
    common = {
        "schema_version": 2,
        "experimental": True,
        "locally_verified": False,
        "capture_session_id": "test-session-0001",
        "observer_script_sha256": EXPECTED_OBSERVER_SHA256,
    }
    install = {
        **common,
        "event": "instrumentation_installed",
        "sequence": 1,
    }
    prompt = {
        **common,
        "event": "option_list_display",
        "sequence": 2,
        "player_id": 1,
        "prompt": "Place Influence",
        "options_pointer": 0x2000,
        "state_machine_pointer": 0x3000,
        "static_identity_checks_passed": True,
    }
    evaluation = {
        **common,
        "event": "cmedium_evaluation_entry",
        "sequence": 3,
        "evaluation_sequence": 1,
        "actor_id": 1,
        "actor_is_configured_ai": True,
        "state_machine_pointer": 0x3000,
        "module_sha256": EXPECTED_SHA,
        "static_identity_checks_passed": True,
        "native_scalar_state": {"all_reads_passed": True},
        "prompt_context": prompt,
        "option_snapshot": {**options, "options_pointer": 0x2000},
    }
    commit = {
        **common,
        "event": "ai_option_selection",
        "sequence": 4,
        "actor_id": 1,
        "actor_is_configured_ai": True,
        "options_pointer": 0x2000,
        "state_machine_pointer": 0x3000,
        "module_sha256": EXPECTED_SHA,
        "static_identity_checks_passed": True,
        "ai_actor_provenance_gate": True,
        "cmedium_evaluation_active_on_stack": False,
        "matched_evaluation_sequence": 1,
        "matched_evaluation_record_sequence": 3,
        "matched_option_identity_fingerprint_sha256": "a" * 64,
        "selection_option_snapshot": {**options, "options_pointer": 0x2000},
        "selection_snapshot_matches_evaluation": True,
        "prompt_context_matches_selection": True,
        "option_index": 0,
        "selection_id": 17,
        "selection_hint": 40970,
        "selected_option": options["options"][0],
        "start_game_seed": 9,
    }
    host_report = {
        "schema_version": 2,
        "library_sha256": EXPECTED_SHA,
        "capture_session_id": "test-session-0001",
        "host_script_sha256": EXPECTED_HOST_SHA256,
        "expected_observer_script_sha256": EXPECTED_OBSERVER_SHA256,
        "configured_ai_player_ids": [1],
        "start_seed": 9,
        "debugger_ai_selections": 1,
    }
    result = validate([install, prompt, evaluation, commit], [], host_report)
    assert result["first_genuine_cmedium_selected_action_with_verified_ai_actor_provenance"]
    assert not result["steam_behavioral_parity_demonstrated"]
    assert not validate([install, prompt, evaluation, commit], [], None)[
        "first_genuine_cmedium_selected_action_with_verified_ai_actor_provenance"
    ]
    mixed = {
        **install,
        "capture_session_id": "different-session",
        "sequence": 5,
        "event": "rng_call_entry",
    }
    mixed_result = validate([install, prompt, evaluation, commit, mixed], [], host_report)
    assert not mixed_result[
        "first_genuine_cmedium_selected_action_with_verified_ai_actor_provenance"
    ]
    assert "capture_session_count_not_one" in mixed_result[
        "capture_identity_rejection_reasons"
    ]
    active_search = {**commit, "cmedium_evaluation_active_on_stack": True}
    active_result = validate([install, prompt, evaluation, active_search], [], host_report)
    assert "selection_occurred_inside_evaluator" in active_result[
        "rejected_actions"
    ][0]["rejection_reasons"]
    bad = {**commit, "selection_hint": 1}
    rejected = validate([install, prompt, evaluation, bad], [], host_report)
    assert not rejected["first_genuine_cmedium_selected_action_with_verified_ai_actor_provenance"]
    assert rejected["rejected_actions"][0]["rejection_reasons"] == ["selection_hint_mismatch"]


def test_capture_schema_and_static_address_evidence_are_machine_readable():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "artifacts/steam_native_action_capture_schema.json").read_text(encoding="utf-8")
    )
    assert "ai_option_selection" in schema["properties"]["event"]["enum"]
    lldb_source = (root / "tools/native_oracle_capture_lldb.py").read_text(encoding="utf-8")
    for address in (
        "0x1A8F0", "0x6B80", "0x11D98", "0x120A8",
        "0x19A0C", "0x4171", "0x1090E", "0x10BB0"
    ):
        assert address in lldb_source
    mapping = json.loads(
        (root / "artifacts/steam_native_saito_mapping.json").read_text(encoding="utf-8")
    )
    assert mapping["overall_status"] == "UNRESOLVED_NO_LOSSLESS_MAPPING"
    assert mapping["accepted_native_ai_actions"] == 0
    assert mapping["smallest_next_evidence_gap"]["required_trace_events"][-1] == (
        "ai_option_selection"
    )
    assert mapping["external_prior"]["status"] == "NOT_USED"
