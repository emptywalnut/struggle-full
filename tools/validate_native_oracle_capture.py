#!/usr/bin/env python3
"""Validate an experimental native CMediumAI JSONL action capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_LIBRARY_SHA256 = "b45dc29df33f8c11312c2fe64d8b22764d153afb874a8e3f6867465ede0a1fa7"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


EXPECTED_OBSERVER_SHA256 = sha256_file(
    Path(__file__).with_name("native_oracle_capture_lldb.py")
)
EXPECTED_HOST_SHA256 = sha256_file(
    Path(__file__).with_name("native_oracle_capture_macos.py")
)


def read_jsonl(path: Path) -> tuple[list[dict], list[int]]:
    records = []
    invalid_lines = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                invalid_lines.append(line_number)
    return records, invalid_lines


def validate(records: list[dict], invalid_lines: list[int], host_report: dict | None) -> dict:
    capture_reasons = []
    session_ids = {
        record.get("capture_session_id") for record in records
        if isinstance(record.get("capture_session_id"), str)
    }
    if len(session_ids) != 1:
        capture_reasons.append("capture_session_count_not_one")
    capture_session_id = next(iter(session_ids)) if len(session_ids) == 1 else None
    if any(record.get("schema_version") != 2 for record in records):
        capture_reasons.append("trace_schema_version")
    if any(
        record.get("observer_script_sha256") != EXPECTED_OBSERVER_SHA256
        for record in records
    ):
        capture_reasons.append("observer_script_checksum")
    sequences = [record.get("sequence") for record in records]
    sequence_integrity = sequences == list(range(1, len(records) + 1))
    if not sequence_integrity:
        capture_reasons.append("record_sequence_integrity")
    installs = [record for record in records if record.get("event") == "instrumentation_installed"]
    if len(installs) != 1 or (installs and installs[0].get("sequence") != 1):
        capture_reasons.append("instrumentation_install_record")
    if any(record.get("event") == "instrumentation_identity_rejected" for record in records):
        capture_reasons.append("instrumentation_identity_rejection_present")

    evaluation_records = [
        record for record in records if record.get("event") == "cmedium_evaluation_entry"
    ]
    evaluation_keys = [
        (record.get("capture_session_id"), record.get("evaluation_sequence"))
        for record in evaluation_records
    ]
    if len(set(evaluation_keys)) != len(evaluation_keys):
        capture_reasons.append("duplicate_evaluation_sequence")
    evaluations = {
        (record.get("capture_session_id"), record.get("evaluation_sequence")): record
        for record in evaluation_records
    }
    display_records = {
        record.get("sequence"): record
        for record in records if record.get("event") == "option_list_display"
    }
    accepted_actions = []
    rejected_actions = []
    for commit in (record for record in records if record.get("event") == "ai_option_selection"):
        reasons = []
        evaluation = evaluations.get((
            commit.get("capture_session_id"), commit.get("matched_evaluation_sequence")
        ))
        if commit.get("module_sha256") != EXPECTED_LIBRARY_SHA256:
            reasons.append("selection_module_checksum")
        if not commit.get("static_identity_checks_passed"):
            reasons.append("selection_symbol_address")
        if not commit.get("actor_is_configured_ai"):
            reasons.append("actor_not_configured_ai")
        if not commit.get("ai_actor_provenance_gate"):
            reasons.append("instrumentation_provenance_gate")
        if commit.get("cmedium_evaluation_active_on_stack") is not False:
            reasons.append("selection_occurred_inside_evaluator")
        if evaluation is None:
            reasons.append("missing_matched_evaluation")
        else:
            if evaluation.get("actor_id") != commit.get("actor_id"):
                reasons.append("actor_mismatch")
            if evaluation.get("module_sha256") != EXPECTED_LIBRARY_SHA256:
                reasons.append("evaluation_module_checksum")
            if not evaluation.get("static_identity_checks_passed"):
                reasons.append("evaluation_symbol_address")
            if not evaluation.get("native_scalar_state", {}).get("all_reads_passed"):
                reasons.append("native_scalar_state_read")
            if commit.get("matched_evaluation_record_sequence") != evaluation.get("sequence"):
                reasons.append("evaluation_record_sequence_mismatch")
            if evaluation.get("state_machine_pointer") != commit.get("state_machine_pointer"):
                reasons.append("state_machine_pointer_mismatch")
            if (
                evaluation.get("option_snapshot", {}).get("options_pointer")
                != commit.get("options_pointer")
            ):
                reasons.append("options_pointer_mismatch")
            prompt_context = evaluation.get("prompt_context")
            if not isinstance(prompt_context, dict):
                reasons.append("missing_prompt_context")
            else:
                display = display_records.get(prompt_context.get("sequence"))
                if display is None or display != prompt_context:
                    reasons.append("prompt_context_not_trace_record")
                if prompt_context.get("player_id") != commit.get("actor_id"):
                    reasons.append("prompt_actor_mismatch")
                if not prompt_context.get("static_identity_checks_passed"):
                    reasons.append("prompt_symbol_address")
                if prompt_context.get("options_pointer") != commit.get("options_pointer"):
                    reasons.append("prompt_options_pointer_mismatch")
                if (
                    prompt_context.get("state_machine_pointer")
                    != commit.get("state_machine_pointer")
                ):
                    reasons.append("prompt_state_machine_pointer_mismatch")
            snapshot = evaluation.get("option_snapshot", {})
            if not snapshot.get("defensive_layout_check_passed"):
                reasons.append("option_layout_check")
            index = commit.get("option_index")
            options = snapshot.get("options", [])
            if not isinstance(index, int) or not 0 <= index < len(options):
                reasons.append("selected_index_out_of_range")
            else:
                selected = options[index]
                if selected.get("selection_id") != commit.get("selection_id"):
                    reasons.append("selection_id_mismatch")
                if selected.get("selection_hint") != commit.get("selection_hint"):
                    reasons.append("selection_hint_mismatch")
            selection_snapshot = commit.get("selection_option_snapshot", {})
            if not selection_snapshot.get("defensive_layout_check_passed"):
                reasons.append("selection_option_layout_check")
            if (
                snapshot.get("identity_fingerprint_sha256")
                != commit.get("matched_option_identity_fingerprint_sha256")
            ):
                reasons.append("option_fingerprint_mismatch")
            if (
                selection_snapshot.get("identity_fingerprint_sha256")
                != snapshot.get("identity_fingerprint_sha256")
            ):
                reasons.append("selection_snapshot_fingerprint_mismatch")
            if commit.get("selection_snapshot_matches_evaluation") is not True:
                reasons.append("selection_snapshot_gate")
            if commit.get("prompt_context_matches_selection") is not True:
                reasons.append("prompt_context_gate")
        summary = {
            "sequence": commit.get("sequence"),
            "actor_id": commit.get("actor_id"),
            "option_index": commit.get("option_index"),
            "selection_id": commit.get("selection_id"),
            "selection_hint": commit.get("selection_hint"),
            "display_text": (commit.get("selected_option") or {}).get("display_text"),
            "prompt": (evaluation or {}).get("prompt_context", {}).get("prompt")
            if isinstance((evaluation or {}).get("prompt_context"), dict) else None,
            "matched_evaluation_sequence": commit.get("matched_evaluation_sequence"),
            "rejection_reasons": reasons,
        }
        (rejected_actions if reasons else accepted_actions).append(summary)

    host_consistent = None
    host_reasons = []
    if host_report is not None:
        if host_report.get("schema_version") != 2:
            host_reasons.append("host_schema_version")
        if host_report.get("library_sha256") != EXPECTED_LIBRARY_SHA256:
            host_reasons.append("host_module_checksum")
        if host_report.get("capture_session_id") != capture_session_id:
            host_reasons.append("host_capture_session")
        if host_report.get("host_script_sha256") != EXPECTED_HOST_SHA256:
            host_reasons.append("host_script_checksum")
        if host_report.get("expected_observer_script_sha256") != EXPECTED_OBSERVER_SHA256:
            host_reasons.append("host_observer_script_checksum")
        configured = set(host_report.get("configured_ai_player_ids", []))
        if any(action["actor_id"] not in configured for action in accepted_actions):
            host_reasons.append("host_actor_configuration")
        for action in accepted_actions:
            matching_commit = next(
                record for record in records
                if record.get("sequence") == action["sequence"]
            )
            if host_report.get("start_seed") != matching_commit.get("start_game_seed"):
                host_reasons.append("host_seed")
                break
        if host_report.get("debugger_ai_selections") != len(accepted_actions):
            host_reasons.append("host_accepted_action_count")
        host_consistent = not host_reasons

    rng_records = [record for record in records if record.get("event") == "rng_call_entry"]
    first_action_gate = bool(
        accepted_actions
        and not invalid_lines
        and not capture_reasons
        and host_report is not None
        and host_consistent
    )
    return {
        "schema_version": 2,
        "experimental": True,
        "first_genuine_cmedium_selected_action_with_verified_ai_actor_provenance": first_action_gate,
        "steam_behavioral_parity_demonstrated": False,
        "genuine_steam_ai_elo_identifiable": False,
        "accepted_actions": accepted_actions,
        "rejected_actions": rejected_actions,
        "invalid_jsonl_lines": invalid_lines,
        "capture_session_id": capture_session_id,
        "capture_identity_consistent": not capture_reasons,
        "capture_identity_rejection_reasons": capture_reasons,
        "record_sequence_integrity": sequence_integrity,
        "expected_observer_script_sha256": EXPECTED_OBSERVER_SHA256,
        "expected_host_script_sha256": EXPECTED_HOST_SHA256,
        "cmedium_evaluation_entries": len(evaluations),
        "rng_call_entries": len(rng_records),
        "rng_return_values_captured": any(
            record.get("return_value_captured") is True for record in rng_records
        ),
        "host_report_supplied": host_report is not None,
        "host_report_consistent": host_consistent,
        "host_report_rejection_reasons": host_reasons,
        "remaining_limits": [
            "capture is one or more native actions, not a differential parity corpus",
            "RNG return sequence and semantic RNG-state offsets are not captured",
            "lossless native-to-Saito state/action mapping is not demonstrated",
            "rules transition parity is not demonstrated"
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--host-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records, invalid_lines = read_jsonl(args.trace_jsonl.expanduser().resolve())
    host_report = None
    if args.host_report:
        host_report = json.loads(args.host_report.expanduser().resolve().read_text(encoding="utf-8"))
    result = validate(records, invalid_lines, host_report)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["first_genuine_cmedium_selected_action_with_verified_ai_actor_provenance"] else 5


if __name__ == "__main__":
    sys.exit(main())
