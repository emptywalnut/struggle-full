"""EXPERIMENTAL LLDB instrumentation for native CMediumAI action provenance.

Load this file with LLDB's ``command script import``.  It observes native
function entries only; it does not patch code, replace return values, call into
the target, or alter options.  JSONL output is selected by
``TS_NATIVE_CAPTURE_PATH``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import lldb


EXPECTED_LIBRARY_SHA256 = "b45dc29df33f8c11312c2fe64d8b22764d153afb874a8e3f6867465ede0a1fa7"
MODULE_BASENAME = "TwilightStruggleLib-OSX"
WORLD_OBJECT_BYTES = 7040
STATE_MACHINE_PREFIX_BYTES = 64
MAX_OPTIONS = 1024
MAX_OPTION_TEXT_BYTES = 512
MAX_RNG_RECORDS = 100000
SETUP_COUNTRY_HINT = 41010
SETUP_COUNTRY_CALLBACK_FILE_ADDRESSES = {"arm64": 0x493F8, "x86_64": 0x48782}

# File addresses are from the checksum-pinned 1.4.11/build-167 universal binary.
EXPECTED_FILE_ADDRESSES = {
    "arm64": {
        "cmedium_evaluate": 0x1A8F0,
        "offline_option_select": 0x6B80,
        "game_state_select_option": 0x11D98,
        "display_option_list": 0x120A8,
        "world_random_bounded": 0x124C0,
        "world_random_raw": 0x12584,
        "world_random_range": 0x125F8,
        "generator_set_seed": 0x13AD4,
        "generator_random_bounded": 0x13C6C,
        "generator_random_range": 0x13D24,
    },
    "x86_64": {
        "cmedium_evaluate": 0x19A0C,
        "offline_option_select": 0x4171,
        "game_state_select_option": 0x1090E,
        "display_option_list": 0x10BB0,
        "world_random_bounded": 0x10F92,
        "world_random_raw": 0x11038,
        "world_random_range": 0x110AA,
        "generator_set_seed": 0x124EC,
        "generator_random_bounded": 0x12692,
        "generator_random_range": 0x12736,
    },
}

_sequence = 0
_evaluation_sequence = 0
_latest_evaluation_by_context = {}
_actor_by_thread = {}
_prompt_by_options_pointer = {}
_module_sha_cache = {}
_address_error_once = set()
_rng_record_count = 0
_observer_sha_cache = None


def _capture_session_id() -> str:
    value = os.environ.get("TS_NATIVE_CAPTURE_SESSION_ID", "")
    if not 8 <= len(value) <= 128 or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in value
    ):
        raise RuntimeError(
            "TS_NATIVE_CAPTURE_SESSION_ID must be 8-128 ASCII letters, digits, '-' or '_'"
        )
    return value


def _observer_script_sha256() -> str:
    global _observer_sha_cache
    if _observer_sha_cache is None:
        _observer_sha_cache = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    return _observer_sha_cache


def _capture_path() -> Path:
    raw = os.environ.get("TS_NATIVE_CAPTURE_PATH")
    if not raw:
        raise RuntimeError("TS_NATIVE_CAPTURE_PATH is required")
    return Path(raw).expanduser().resolve()


def _ai_ids() -> set[int]:
    raw = os.environ.get("TS_NATIVE_AI_PLAYER_IDS", "")
    return {int(value, 0) for value in raw.split(",") if value.strip()}


def _write(record: dict) -> dict | None:
    global _sequence
    _sequence += 1
    enriched = {
        "schema_version": 2,
        "experimental": True,
        "locally_verified": False,
        "capture_session_id": _capture_session_id(),
        "observer_script_sha256": _observer_script_sha256(),
        "sequence": _sequence,
        "observer_unix_ns": time.time_ns(),
        **record,
    }
    try:
        path = _capture_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, sort_keys=True) + "\n")
            handle.flush()
        return enriched
    except Exception as exc:  # LLDB callbacks must never stop or modify the target.
        print(f"native oracle capture write failed: {exc}")
        return None


def _architecture(frame) -> str:
    triple = frame.GetThread().GetProcess().GetTarget().GetTriple().lower()
    if "arm64" in triple or "aarch64" in triple:
        return "arm64"
    if "x86_64" in triple:
        return "x86_64"
    return "unsupported"


def _register(frame, name: str) -> int:
    value = frame.FindRegister(name)
    return value.GetValueAsUnsigned(0) if value.IsValid() else 0


def _read(process, address: int, size: int) -> tuple[bytes | None, str | None]:
    if not address or size < 0:
        return None, "null_or_invalid_address"
    error = lldb.SBError()
    data = process.ReadMemory(address, size, error)
    if error.Fail() or data is None or len(data) != size:
        return None, error.GetCString() or "short_read"
    return bytes(data), None


def _read_integer(process, address: int, size: int) -> tuple[int | None, str | None]:
    data, error = _read(process, address, size)
    if data is None:
        return None, error
    return int.from_bytes(data, "little", signed=False), None


def _read_signed_integer(process, address: int, size: int) -> tuple[int | None, str | None]:
    data, error = _read(process, address, size)
    if data is None:
        return None, error
    return int.from_bytes(data, "little", signed=True), None


def _snapshot_world_scalars(process, world: int) -> dict:
    fields = {
        "victory_point_track_raw": 5528,
        "defcon_level_raw": 5532,
        "military_operations_player_index_0_raw": 5544,
        "military_operations_player_index_1_raw": 5548,
        "space_race_player_index_0_raw": 5552,
        "space_race_player_index_1_raw": 5556,
        "turn_number_raw": 5584,
        "action_round_number_raw": 5588,
    }
    values = {}
    errors = {}
    for name, offset in fields.items():
        value, error = _read_signed_integer(process, world + offset, 4)
        values[name] = value
        if error:
            errors[name] = error
    return {
        **values,
        "read_errors": errors,
        "all_reads_passed": not errors,
        "mapping_status": "RAW_NATIVE_SCALARS_ONLY",
        "player_index_to_side": "UNRESOLVED",
        "vp_sign_to_side": "UNRESOLVED",
        "offset_evidence": {
            "victory_point_track_raw": {
                "offset": 5528,
                "setter_arm64": "0x21de0",
                "setter_x86_64": "0x21c3e",
            },
            "defcon_level_raw": {
                "offset": 5532,
                "setter_arm64": "0x21e6c",
                "setter_x86_64": "0x21cc0",
            },
            "military_operations_pair_raw": {
                "offset": 5544,
                "setter_arm64": "0x21fc4",
                "setter_x86_64": "0x21e0c",
            },
            "space_race_pair_raw": {
                "offset": 5552,
                "setter_arm64": "0x2207c",
                "setter_x86_64": "0x21ebe",
            },
            "turn_number_raw": {
                "offset": 5584,
                "setter_arm64": "0x22354",
                "setter_x86_64": "0x22178",
            },
            "action_round_number_raw": {
                "offset": 5588,
                "setter_arm64": "0x22398",
                "setter_x86_64": "0x221b0",
            },
        },
    }


def _read_cstring(process, address: int) -> tuple[str | None, str | None]:
    if not address:
        return None, "null_text_pointer"
    error = lldb.SBError()
    value = process.ReadCStringFromMemory(address, MAX_OPTION_TEXT_BYTES, error)
    if error.Fail():
        return None, error.GetCString() or "cstring_read_failed"
    return value, None


def _module_evidence(frame, fact: str) -> tuple[dict, bool]:
    module = frame.GetModule()
    file_spec = module.GetFileSpec()
    module_path = Path(str(file_spec)).expanduser()
    if not module_path.is_absolute():
        directory = file_spec.GetDirectory() or ""
        module_path = Path(directory) / file_spec.GetFilename()
    module_path = module_path.resolve()
    digest = _module_sha_cache.get(str(module_path))
    if digest is None:
        try:
            digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
        except OSError as exc:
            digest = f"read_error:{exc}"
        _module_sha_cache[str(module_path)] = digest
    pc_address = frame.GetPCAddress()
    file_address = pc_address.GetFileAddress()
    architecture = _architecture(frame)
    expected_address = EXPECTED_FILE_ADDRESSES.get(architecture, {}).get(fact)
    evidence = {
        "architecture": architecture,
        "module_path": str(module_path),
        "module_basename": module_path.name,
        "module_sha256": digest,
        "symbol": frame.GetFunctionName(),
        "function_file_address": file_address,
        "expected_function_file_address": expected_address,
    }
    accepted = (
        module_path.name == MODULE_BASENAME
        and digest == EXPECTED_LIBRARY_SHA256
        and expected_address is not None
        and file_address == expected_address
    )
    evidence["static_identity_checks_passed"] = accepted
    if not accepted:
        key = (fact, architecture, file_address, digest)
        if key not in _address_error_once:
            _address_error_once.add(key)
            _write({"event": "instrumentation_identity_rejected", "fact": fact, **evidence})
    return evidence, accepted


def _thread_evidence(frame) -> dict:
    thread = frame.GetThread()
    return {
        "process_id": thread.GetProcess().GetProcessID(),
        "thread_id": thread.GetThreadID(),
        "thread_index_id": thread.GetIndexID(),
        "thread_name": thread.GetName(),
        "queue_name": thread.GetQueueName(),
    }


def _argument_registers(frame) -> tuple[int, int, int, int, int]:
    architecture = _architecture(frame)
    names = ("x0", "x1", "x2", "x3", "x4") if architecture == "arm64" else (
        "rdi", "rsi", "rdx", "rcx", "r8"
    )
    return tuple(_register(frame, name) for name in names)


def _stack_function_names(frame, maximum: int = 64) -> list[str]:
    thread = frame.GetThread()
    return [
        thread.GetFrameAtIndex(index).GetFunctionName() or ""
        for index in range(min(thread.GetNumFrames(), maximum))
    ]


def _snapshot_options(process, options_pointer: int) -> dict:
    count, count_error = _read_integer(process, options_pointer + 88, 4)
    storage, storage_error = _read_integer(process, options_pointer + 96, 8)
    if count is not None and count >= 0x80000000:
        count -= 0x100000000
    result = {
        "options_pointer": options_pointer,
        "option_count": count,
        "options_storage_pointer": storage,
        "count_read_error": count_error,
        "storage_read_error": storage_error,
        "layout_evidence": {
            "count_offset": 88,
            "storage_offset": 96,
            "entry_size": 32,
            "selection_id_offset": 0,
            "selection_hint_offset": 2,
            "display_text_pointer_offset": 8,
        },
        "options": [],
    }
    if count is None or storage is None or not 0 <= count <= MAX_OPTIONS or not storage:
        result["defensive_layout_check_passed"] = False
        return result
    identities = []
    for index in range(count):
        entry_address = storage + index * 32
        raw, raw_error = _read(process, entry_address, 32)
        if raw is None:
            result["options"].append({"index": index, "read_error": raw_error})
            continue
        selection_id, selection_hint = struct.unpack_from("<HH", raw, 0)
        text_pointer = struct.unpack_from("<Q", raw, 8)[0]
        callback_pointer = struct.unpack_from("<Q", raw, 16)[0]
        callback_data = struct.unpack_from("<Q", raw, 24)[0]
        callback_address = process.GetTarget().ResolveLoadAddress(callback_pointer)
        callback_file_address = (
            callback_address.GetFileAddress() if callback_address.IsValid() else None
        )
        if callback_file_address == 0xFFFFFFFFFFFFFFFF:
            callback_file_address = None
        callback_module = callback_address.GetModule() if callback_address.IsValid() else None
        callback_module_name = None
        if callback_module and callback_module.IsValid():
            callback_module_name = callback_module.GetFileSpec().GetFilename()
        text, text_error = _read_cstring(process, text_pointer)
        target_triple = process.GetTarget().GetTriple().lower()
        option_architecture = (
            "arm64" if "arm64" in target_triple or "aarch64" in target_triple
            else "x86_64" if "x86_64" in target_triple else "unsupported"
        )
        setup_callback_expected = SETUP_COUNTRY_CALLBACK_FILE_ADDRESSES.get(
            option_architecture
        )
        setup_country_mapping = bool(
            selection_hint == SETUP_COUNTRY_HINT
            and callback_data == selection_id
            and callback_module_name == MODULE_BASENAME
            and callback_file_address == setup_callback_expected
        )
        option = {
            "index": index,
            "entry_address": entry_address,
            "selection_id": selection_id,
            "selection_hint": selection_hint,
            "display_text_pointer": text_pointer,
            "display_text": text,
            "display_text_read_error": text_error,
            "callback_pointer": callback_pointer,
            "callback_module_basename": callback_module_name,
            "callback_file_address": callback_file_address,
            "callback_data_raw": callback_data,
            "static_semantic_mapping": {
                "status": (
                    "CONFIRMED_NATIVE_SETUP_COUNTRY_INDEX"
                    if setup_country_mapping else "UNRESOLVED_CONTEXT_DEPENDENT"
                ),
                "native_action_family": "country_click" if setup_country_mapping else None,
                "native_country_index": selection_id if setup_country_mapping else None,
                "saito_country_id": None,
                "saito_country_crosswalk_verified": False,
            },
            "raw_entry_hex": raw.hex(),
        }
        result["options"].append(option)
        identities.append([
            index,
            selection_id,
            selection_hint,
            text,
            callback_module_name,
            callback_file_address,
            callback_data,
        ])
    identity_json = json.dumps(identities, ensure_ascii=False, separators=(",", ":"))
    result["identity_fingerprint_sha256"] = hashlib.sha256(identity_json.encode()).hexdigest()
    result["defensive_layout_check_passed"] = (
        len(result["options"]) == count
        and all(
            option.get("selection_id") is not None
            and option.get("selection_hint") is not None
            and option.get("display_text_read_error") is None
            for option in result["options"]
        )
    )
    return result


def cmedium_evaluate_callback(frame, _bp_loc, _dict) -> bool:
    global _evaluation_sequence
    module_evidence, accepted = _module_evidence(frame, "cmedium_evaluate")
    if not accepted:
        return False
    process = frame.GetThread().GetProcess()
    evaluator, world, state_machine, options, out_value = _argument_registers(frame)
    actor_id, actor_error = _read_integer(process, evaluator + 8, 4)
    option_snapshot = _snapshot_options(process, options)
    world_raw, world_error = _read(process, world, WORLD_OBJECT_BYTES)
    state_machine_raw, state_machine_error = _read(
        process, state_machine, STATE_MACHINE_PREFIX_BYTES
    )
    _evaluation_sequence += 1
    record = {
        "event": "cmedium_evaluation_entry",
        "evaluation_sequence": _evaluation_sequence,
        "actor_id": actor_id,
        "actor_read_error": actor_error,
        "actor_is_configured_ai": actor_id in _ai_ids(),
        "evaluator_pointer": evaluator,
        "world_pointer": world,
        "state_machine_pointer": state_machine,
        "out_value_pointer": out_value,
        "start_game_seed": int(os.environ.get("TS_NATIVE_START_SEED", "0"), 0),
        "world_snapshot_bytes": WORLD_OBJECT_BYTES if world_raw is not None else 0,
        "world_snapshot_sha256": hashlib.sha256(world_raw).hexdigest() if world_raw else None,
        "world_snapshot_read_error": world_error,
        "native_scalar_state": _snapshot_world_scalars(process, world),
        "state_machine_prefix_bytes": len(state_machine_raw) if state_machine_raw else 0,
        "state_machine_prefix_sha256": (
            hashlib.sha256(state_machine_raw).hexdigest() if state_machine_raw else None
        ),
        "state_machine_prefix_read_error": state_machine_error,
        "option_snapshot": option_snapshot,
        "prompt_context": _prompt_by_options_pointer.get(options),
        "rng_context_limit": (
            "start seed and pre-evaluation world hash are captured; individual RNG call "
            "entries are optional and return values/RNG field offsets remain unresolved"
        ),
        **_thread_evidence(frame),
        **module_evidence,
    }
    if os.environ.get("TS_NATIVE_CAPTURE_RAW_WORLD") == "1" and world_raw:
        record["world_snapshot_base64"] = base64.b64encode(world_raw).decode("ascii")
    enriched = _write(record)
    if actor_id is not None and enriched is not None:
        context_key = (
            frame.GetThread().GetThreadID(), actor_id, state_machine, options
        )
        _latest_evaluation_by_context[context_key] = enriched
        _actor_by_thread[frame.GetThread().GetThreadID()] = actor_id
    return False


def display_option_list_callback(frame, _bp_loc, _dict) -> bool:
    module_evidence, accepted = _module_evidence(frame, "display_option_list")
    if not accepted:
        return False
    process = frame.GetThread().GetProcess()
    architecture = _architecture(frame)
    names = ("x0", "x1", "x2", "x3") if architecture == "arm64" else (
        "rdi", "rsi", "rdx", "rcx"
    )
    options, state_machine, player, prompt_pointer = (
        _register(frame, name) for name in names
    )
    player_id, player_id_error = _read_integer(process, player + 24, 4)
    prompt, prompt_error = _read_cstring(process, prompt_pointer)
    record = {
        "event": "option_list_display",
        "options_pointer": options,
        "state_machine_pointer": state_machine,
        "player_pointer": player,
        "player_id": player_id,
        "player_id_read_error": player_id_error,
        "prompt_pointer": prompt_pointer,
        "prompt": prompt,
        "prompt_read_error": prompt_error,
        **_thread_evidence(frame),
        **module_evidence,
    }
    enriched = _write(record)
    if enriched is not None:
        _prompt_by_options_pointer[options] = enriched
    return False


def game_state_select_option_callback(frame, _bp_loc, _dict) -> bool:
    """Observe the selection function called immediately after evaluator return.

    CGameStateOptions::UpdateState invokes the configured evaluator and, after a
    bounds check, calls this function with the returned index.  Evaluator search
    may itself drive cloned states, so an observation is accepted only when no
    CMediumAI::Evaluate frame remains on the stack and the exact thread, actor,
    state-machine and options tuple has an unconsumed evaluation entry.
    """
    module_evidence, accepted = _module_evidence(frame, "game_state_select_option")
    if not accepted:
        return False
    process = frame.GetThread().GetProcess()
    architecture = _architecture(frame)
    if architecture == "arm64":
        options_pointer = _register(frame, "x0")
        state_machine_pointer = _register(frame, "x1")
        option_index_raw = _register(frame, "x2") & 0xFFFFFFFF
        optional_data_pointer = _register(frame, "x3")
    else:
        options_pointer = _register(frame, "rdi")
        state_machine_pointer = _register(frame, "rsi")
        option_index_raw = _register(frame, "rdx") & 0xFFFFFFFF
        optional_data_pointer = _register(frame, "rcx")
    option_index = (
        option_index_raw - 0x100000000
        if option_index_raw >= 0x80000000 else option_index_raw
    )
    stack_functions = _stack_function_names(frame)
    evaluation_active_on_stack = any(
        "twilight_ai::CMediumAI::Evaluate(" in name for name in stack_functions[1:]
    )
    thread_id = frame.GetThread().GetThreadID()
    candidates = [
        (key, value) for key, value in _latest_evaluation_by_context.items()
        if key[0] == thread_id
        and key[2] == state_machine_pointer
        and key[3] == options_pointer
    ]
    candidates.sort(key=lambda item: item[1].get("sequence", 0), reverse=True)
    context_key, evaluation = candidates[0] if candidates else (None, None)
    actor = evaluation.get("actor_id") if evaluation else None
    selection_snapshot = _snapshot_options(process, options_pointer)
    selected = None
    selected_identity_valid = False
    if 0 <= option_index < len(selection_snapshot.get("options", [])):
        selected = selection_snapshot["options"][option_index]
        selected_identity_valid = (
            selected.get("selection_id") is not None
            and selected.get("selection_hint") is not None
        )
    fingerprints_match = bool(
        evaluation
        and selection_snapshot.get("identity_fingerprint_sha256")
        == evaluation.get("option_snapshot", {}).get("identity_fingerprint_sha256")
    )
    prompt = evaluation.get("prompt_context") if evaluation else None
    prompt_matches = bool(
        isinstance(prompt, dict)
        and prompt.get("options_pointer") == options_pointer
        and prompt.get("state_machine_pointer") == state_machine_pointer
        and prompt.get("player_id") == actor
        and prompt.get("static_identity_checks_passed") is True
    )
    is_ai = actor in _ai_ids() if actor is not None else False
    provenance_gate = bool(
        is_ai
        and evaluation
        and not evaluation_active_on_stack
        and evaluation.get("static_identity_checks_passed") is True
        and evaluation.get("actor_is_configured_ai") is True
        and selection_snapshot.get("defensive_layout_check_passed") is True
        and selected_identity_valid
        and fingerprints_match
        and prompt_matches
    )
    record = {
        "event": "ai_option_selection" if is_ai else "unattributed_option_selection",
        "actor_id": actor,
        "actor_is_configured_ai": is_ai,
        "configured_ai_player_ids": sorted(_ai_ids()),
        "options_pointer": options_pointer,
        "state_machine_pointer": state_machine_pointer,
        "option_index": option_index,
        "selection_id": selected.get("selection_id") if selected else None,
        "selection_hint": selected.get("selection_hint") if selected else None,
        "selected_option": selected,
        "selection_option_snapshot": selection_snapshot,
        "optional_data_pointer": optional_data_pointer,
        "matched_evaluation_sequence": (
            evaluation.get("evaluation_sequence") if evaluation else None
        ),
        "matched_evaluation_record_sequence": evaluation.get("sequence") if evaluation else None,
        "matched_option_identity_fingerprint_sha256": (
            evaluation.get("option_snapshot", {}).get("identity_fingerprint_sha256")
            if evaluation else None
        ),
        "selection_snapshot_matches_evaluation": fingerprints_match,
        "prompt_context_matches_selection": prompt_matches,
        "cmedium_evaluation_active_on_stack": evaluation_active_on_stack,
        "stack_function_names": stack_functions,
        "ai_actor_provenance_gate": provenance_gate,
        "selection_abi": {
            "options_this_argument": 0,
            "state_machine_argument": 1,
            "option_index_argument": 2,
            "optional_data_pointer_argument": 3,
            "arm64_file_address": 0x11D98,
            "x86_64_file_address": 0x1090E,
        },
        "start_game_seed": int(os.environ.get("TS_NATIVE_START_SEED", "0"), 0),
        **_thread_evidence(frame),
        **module_evidence,
    }
    _write(record)
    if provenance_gate and context_key is not None:
        # A native evaluator result may prove at most one live selection.
        _latest_evaluation_by_context.pop(context_key, None)
    return False


def offline_option_select_callback(frame, _bp_loc, _dict) -> bool:
    module_evidence, accepted = _module_evidence(frame, "offline_option_select")
    if not accepted:
        return False
    process = frame.GetThread().GetProcess()
    architecture = _architecture(frame)
    if architecture == "arm64":
        actor = _register(frame, "x2") & 0xFFFFFFFF
        option_index_raw = _register(frame, "x3") & 0xFFFFFFFF
        selection_id = _register(frame, "x4") & 0xFFFF
        selection_hint = _register(frame, "x5") & 0xFFFF
        optional_data_pointer = _register(frame, "x6")
    else:
        actor = _register(frame, "rdx") & 0xFFFFFFFF
        option_index_raw = _register(frame, "rcx") & 0xFFFFFFFF
        selection_id = _register(frame, "r8") & 0xFFFF
        selection_hint = _register(frame, "r9") & 0xFFFF
        stack_pointer = _register(frame, "rsp")
        optional_data_pointer, _ = _read_integer(process, stack_pointer + 8, 8)
        optional_data_pointer = optional_data_pointer or 0
    option_index = (
        option_index_raw - 0x100000000 if option_index_raw >= 0x80000000 else option_index_raw
    )
    optional_data, optional_data_error = (None, None)
    if optional_data_pointer:
        optional_data, optional_data_error = _read_integer(process, optional_data_pointer, 4)

    is_ai = actor in _ai_ids()
    _write({
        "event": "offline_host_option_stage",
        "actor_id": actor,
        "actor_is_configured_ai": is_ai,
        "configured_ai_player_ids": sorted(_ai_ids()),
        "option_index": option_index,
        "selection_id": selection_id,
        "selection_hint": selection_hint,
        "optional_data_pointer": optional_data_pointer,
        "optional_data": optional_data,
        "optional_data_read_error": optional_data_error,
        "ai_actor_provenance_gate": False,
        "provenance_limit": (
            "OfflineOptionSelectFunc is the exported host/network staging path; "
            "CMediumAI returns through CGameStateOptions::UpdateState and is observed "
            "at CGameStateOptions::SelectOption instead"
        ),
        "commit_abi": {
            "actor_argument": 2,
            "option_index_argument": 3,
            "selection_id_argument": 4,
            "selection_hint_argument": 5,
            "optional_data_pointer_argument": 6,
        },
        "start_game_seed": int(os.environ.get("TS_NATIVE_START_SEED", "0"), 0),
        **_thread_evidence(frame),
        **module_evidence,
    })
    return False


def rng_entry_callback(frame, _bp_loc, internal_dict) -> bool:
    global _rng_record_count
    if _rng_record_count >= MAX_RNG_RECORDS:
        return False
    function_name = frame.GetFunctionName() or ""
    fact = internal_dict.get("fact_by_breakpoint", {}).get(frame.GetThread().GetStopReasonDataAtIndex(0))
    if not fact:
        for candidate in EXPECTED_FILE_ADDRESSES.get(_architecture(frame), {}):
            if candidate.startswith(("world_random", "generator_random", "generator_set_seed")):
                expected = EXPECTED_FILE_ADDRESSES[_architecture(frame)][candidate]
                if frame.GetPCAddress().GetFileAddress() == expected:
                    fact = candidate
                    break
    if not fact:
        return False
    module_evidence, accepted = _module_evidence(frame, fact)
    if not accepted:
        return False
    architecture = _architecture(frame)
    arg_names = ("x0", "x1", "x2") if architecture == "arm64" else ("rdi", "rsi", "rdx")
    receiver, argument_1, argument_2 = (_register(frame, name) for name in arg_names)
    process = frame.GetThread().GetProcess()
    receiver_prefix, receiver_error = _read(process, receiver, 64)
    thread_id = frame.GetThread().GetThreadID()
    actor = _actor_by_thread.get(thread_id)
    _rng_record_count += 1
    _write({
        "event": "rng_call_entry",
        "rng_fact": fact,
        "function_name": function_name,
        "actor_id_from_cmedium_thread": actor,
        "actor_is_configured_ai": actor in _ai_ids() if actor is not None else False,
        "receiver_pointer": receiver,
        "argument_1_raw": argument_1,
        "argument_2_raw": argument_2,
        "receiver_prefix_sha256": (
            hashlib.sha256(receiver_prefix).hexdigest() if receiver_prefix else None
        ),
        "receiver_prefix_read_error": receiver_error,
        "return_value_captured": False,
        **_thread_evidence(frame),
        **module_evidence,
    })
    return False


def _install_breakpoint(target, regex: str, callback: str, label: str):
    breakpoint = target.BreakpointCreateByRegex(regex, MODULE_BASENAME)
    breakpoint.SetScriptCallbackFunction(f"{__name__}.{callback}")
    breakpoint.SetAutoContinue(True)
    return {
        "label": label,
        "breakpoint_id": breakpoint.GetID(),
        "pending_locations_at_install": breakpoint.GetNumLocations(),
        "regex": regex,
    }


def __lldb_init_module(debugger, internal_dict):
    try:
        _capture_path()
        _capture_session_id()
        if not _ai_ids():
            raise RuntimeError("TS_NATIVE_AI_PLAYER_IDS is required")
        int(os.environ.get("TS_NATIVE_START_SEED", ""), 0)
    except Exception as exc:
        print(f"native oracle instrumentation not installed: {exc}")
        return
    target = debugger.GetSelectedTarget()
    installed = [
        _install_breakpoint(
            target,
            r"^twilight_ai::CMediumAI::Evaluate\(core::CWorldBase\*, CStateMachine const&",
            "cmedium_evaluate_callback",
            "CMediumAI evaluation entry",
        ),
        _install_breakpoint(
            target,
            r"^CGameStateOptions::SelectOption\(CStateMachine&",
            "game_state_select_option_callback",
            "post-evaluator native option selection",
        ),
        _install_breakpoint(
            target,
            r"^OfflineOptionSelectFunc\(",
            "offline_option_select_callback",
            "offline host option stage (not AI provenance)",
        ),
        _install_breakpoint(
            target,
            r"^CGameStateOptions::DisplayOptionList\(",
            "display_option_list_callback",
            "native prompt and option-list owner",
        ),
    ]
    if os.environ.get("TS_NATIVE_CAPTURE_RNG", "1") == "1":
        installed.append(_install_breakpoint(
            target,
            r"^(core::CWorldBase|CRandomGenerator)::(GetRandomInt|SetSeedValue)\(",
            "rng_entry_callback",
            "RNG call entry",
        ))
    internal_dict["installed_breakpoints"] = installed
    _write({
        "event": "instrumentation_installed",
        "configured_ai_player_ids": sorted(_ai_ids()),
        "start_game_seed": int(os.environ["TS_NATIVE_START_SEED"], 0),
        "target_triple_at_install": target.GetTriple(),
        "breakpoints": installed,
        "observation_method": "LLDB auto-continue entry breakpoints; no target calls or writes",
    })
    print("experimental native CMediumAI capture breakpoints installed")
