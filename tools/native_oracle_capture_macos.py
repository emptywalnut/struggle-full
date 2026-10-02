#!/usr/bin/env python3
"""EXPERIMENTAL bounded host for debugger-captured native CMediumAI actions.

This host does not claim Steam parity.  The selected-action evidence is produced
by ``native_oracle_capture_lldb.py`` at the native option-commit function, not by
the public option callback or GetGamePlayerAIState.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Iterable


EXPECTED_LIBRARY_SHA256 = "b45dc29df33f8c11312c2fe64d8b22764d153afb874a8e3f6867465ede0a1fa7"
DEFAULT_APP_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common/Twilight Struggle/"
    "TwilightStruggle.app/Contents"
)
DEFAULT_START_SEED = 0x5A17C0DE
INITIALIZE_MODE_WITH_AI_HANDLER = 2


class GameParameters(ctypes.Structure):
    _fields_ = [
        ("additional_card_flags", ctypes.c_uint32),
        ("scenario", ctypes.c_int32),
        ("choose_sides_method", ctypes.c_int32),
        ("additional_influence", ctypes.c_int32),
    ]


class AppPlayerData(ctypes.Structure):
    _fields_ = [
        ("player_id", ctypes.c_int32),
        ("user_avatars", ctypes.c_uint16 * 2),
        ("user_ratings", ctypes.c_uint16 * 2),
        ("player_type", ctypes.c_int8),
        ("ai_difficulty", ctypes.c_int8),
        ("network_player_state", ctypes.c_uint16),
        ("network_player_timer", ctypes.c_uint32),
        ("name", ctypes.c_char * 32),
    ]


class GameOption(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("option_index", ctypes.c_int32),
        ("selection_id", ctypes.c_uint16),
        ("selection_hint", ctypes.c_uint16),
        ("is_hidden", ctypes.c_uint8),
        ("option_text", ctypes.c_char * 55),
    ]


LAYOUTS = {
    "human-ussr-ai-us": ((0, 0, 0, "OracleHumanUSSR"), (1, 2, 2, "NativeCMediumAIUS")),
    "ai-ussr-human-us": ((0, 2, 2, "NativeCMediumAIUSSR"), (1, 0, 0, "OracleHumanUS")),
    "both-ai": ((0, 2, 2, "NativeCMediumAIUSSR"), (1, 2, 2, "NativeCMediumAIUS")),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ai_player_ids(layout: str) -> list[int]:
    return [record[0] for record in LAYOUTS[layout] if record[1] == 2]


def valid_capture_session_id(value: str | None) -> bool:
    return bool(
        value
        and 8 <= len(value) <= 128
        and all(character.isascii() and (character.isalnum() or character in "-_") for character in value)
    )


def structure_self_check() -> dict[str, object]:
    host_path = Path(__file__).resolve()
    observer_path = host_path.with_name("native_oracle_capture_lldb.py")
    result = {
        "experimental": True,
        "pointer_size": ctypes.sizeof(ctypes.c_void_p),
        "GameParameters_size": ctypes.sizeof(GameParameters),
        "AppPlayerData_size": ctypes.sizeof(AppPlayerData),
        "AppPlayerData_player_type_offset": AppPlayerData.player_type.offset,
        "AppPlayerData_ai_difficulty_offset": AppPlayerData.ai_difficulty.offset,
        "GameOption_size": ctypes.sizeof(GameOption),
        "initialize_mode": INITIALIZE_MODE_WITH_AI_HANDLER,
        "host_script_sha256": sha256_file(host_path),
        "observer_script_sha256": (
            sha256_file(observer_path) if observer_path.is_file() else None
        ),
    }
    result["passed"] = (
        result["pointer_size"] == 8
        and result["GameParameters_size"] == 16
        and result["AppPlayerData_size"] == 52
        and result["AppPlayerData_player_type_offset"] == 12
        and result["AppPlayerData_ai_difficulty_offset"] == 13
        and result["GameOption_size"] == 64
    )
    return result


def count_verified_ai_selections(
    trace_path: Path | None,
    configured_ai_ids: Iterable[int],
    capture_session_id: str | None = None,
) -> int:
    if trace_path is None or not trace_path.exists():
        return 0
    wanted = set(configured_ai_ids)
    count = 0
    try:
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    record.get("event") == "ai_option_selection"
                    and record.get("actor_id") in wanted
                    and record.get("ai_actor_provenance_gate") is True
                    and (
                        capture_session_id is None
                        or record.get("capture_session_id") == capture_session_id
                    )
                ):
                    count += 1
    except OSError:
        return 0
    return count


def _player_array(layout: str, ai_difficulty: int) -> ctypes.Array[AppPlayerData]:
    players = (AppPlayerData * 2)()
    for slot, (player_id, player_type, raw_difficulty, name) in enumerate(LAYOUTS[layout]):
        players[slot].player_id = player_id
        players[slot].player_type = player_type
        players[slot].ai_difficulty = ai_difficulty if player_type == 2 else raw_difficulty
        players[slot].name = name.encode("ascii")
    return players


def _require_export(lib: ctypes.CDLL, name: str):
    try:
        return getattr(lib, name)
    except AttributeError as exc:
        raise RuntimeError(f"required native export is absent: {name}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--app-contents", type=Path, default=DEFAULT_APP_ROOT)
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="human-ussr-ai-us")
    parser.add_argument("--allow-both-ai-experimental", action="store_true")
    parser.add_argument("--ai-difficulty", type=int, default=2)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=DEFAULT_START_SEED)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--max-wall-seconds", type=float, default=60.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.01)
    parser.add_argument("--max-auto-human-selections", type=int, default=6)
    parser.add_argument("--commit-after-auto-selections", type=int, default=6)
    parser.add_argument("--stop-after-ai-selections", type=int, default=1)
    parser.add_argument("--event-buffer-bytes", type=int, default=65536)
    parser.add_argument("--event-preview-bytes", type=int, default=128)
    parser.add_argument("--trace-jsonl", type=Path)
    parser.add_argument("--report", type=Path, required=False)
    parser.add_argument("--allow-unverified-binary", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.layout == "both-ai" and not args.allow_both_ai_experimental:
        raise ValueError("both-ai requires --allow-both-ai-experimental")
    if args.ai_difficulty == 3:
        raise ValueError("ai_difficulty 3 constructs CTutorialAI, not CMediumAI")
    if not -128 <= args.ai_difficulty <= 127:
        raise ValueError("ai_difficulty must fit the confirmed int8 field")
    if args.max_iterations < 1 or args.max_wall_seconds <= 0 or args.sleep_seconds < 0:
        raise ValueError("iteration/time bounds must be positive and sleep non-negative")
    if args.event_buffer_bytes < 64 or args.event_preview_bytes < 0:
        raise ValueError("event buffer must be >=64 bytes and preview non-negative")
    if args.max_auto_human_selections < 0 or args.commit_after_auto_selections < 0:
        raise ValueError("human selection bounds must be non-negative")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    self_check = structure_self_check()
    if args.self_check:
        print(json.dumps(self_check, sort_keys=True))
        return 0 if self_check["passed"] else 2
    validate_args(args)
    if sys.platform != "darwin":
        raise RuntimeError("this experimental native host must only execute on macOS")
    if not self_check["passed"]:
        raise RuntimeError(f"native ABI self-check failed: {self_check}")

    app_root = args.app_contents.expanduser().resolve()
    lib_path = (
        app_root
        / "PlugIns/TwilightStruggleLib-OSX.bundle/Contents/MacOS/"
        "TwilightStruggleLib-OSX"
    )
    lua_path = app_root / "Resources/Data/StreamingAssets/Lua"
    if not lib_path.is_file() or not lua_path.is_dir():
        raise FileNotFoundError(f"installed native library or Lua path missing under {app_root}")
    library_sha256 = sha256_file(lib_path)
    if library_sha256 != EXPECTED_LIBRARY_SHA256 and not args.allow_unverified_binary:
        raise RuntimeError(
            "native library checksum mismatch; refusing instrumentation without "
            "--allow-unverified-binary"
        )

    expected_ai_ids = ai_player_ids(args.layout)
    trace_path = args.trace_jsonl.expanduser().resolve() if args.trace_jsonl else None
    env_ai_ids = os.environ.get("TS_NATIVE_AI_PLAYER_IDS")
    env_seed = os.environ.get("TS_NATIVE_START_SEED")
    capture_session_id = os.environ.get("TS_NATIVE_CAPTURE_SESSION_ID")
    observer_path = Path(__file__).with_name("native_oracle_capture_lldb.py")
    observer_script_sha256 = sha256_file(observer_path) if observer_path.is_file() else None
    debugger_contract = {
        "trace_path_supplied": trace_path is not None,
        "env_ai_ids": env_ai_ids,
        "env_seed": env_seed,
        "ai_ids_match": env_ai_ids == ",".join(str(value) for value in expected_ai_ids),
        "seed_matches": env_seed is not None and int(env_seed, 0) == args.seed,
        "capture_session_id": capture_session_id,
        "capture_session_id_valid": valid_capture_session_id(capture_session_id),
        "expected_observer_script_sha256": observer_script_sha256,
    }
    if trace_path and not debugger_contract["ai_ids_match"]:
        raise RuntimeError("TS_NATIVE_AI_PLAYER_IDS must exactly match the selected layout")
    if trace_path and not debugger_contract["seed_matches"]:
        raise RuntimeError("TS_NATIVE_START_SEED must exactly match --seed")
    if trace_path and not debugger_contract["capture_session_id_valid"]:
        raise RuntimeError(
            "TS_NATIVE_CAPTURE_SESSION_ID must be 8-128 ASCII letters, digits, '-' or '_'"
        )

    lib = ctypes.CDLL(str(lib_path))
    initialize = _require_export(lib, "Initialize")
    shutdown = _require_export(lib, "Shutdown")
    set_listener = _require_export(lib, "SetGameOptionsListener")
    start_game = _require_export(lib, "StartGame")
    update_game = _require_export(lib, "UpdateGame")
    select_option = _require_export(lib, "SelectGameOption")
    has_temporary = _require_export(lib, "HasTemporaryMoveBuffer")
    commit_temporary = _require_export(lib, "CommitTemporaryMoveBuffer")

    initialize.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    initialize.restype = None
    shutdown.argtypes = []
    shutdown.restype = None
    set_listener.argtypes = [ctypes.c_void_p]
    set_listener.restype = None
    start_game.argtypes = [
        ctypes.POINTER(GameParameters), ctypes.c_int32,
        ctypes.POINTER(AppPlayerData), ctypes.c_uint32,
    ]
    start_game.restype = None
    update_game.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    update_game.restype = ctypes.c_int32
    select_option.argtypes = [ctypes.c_int32]
    select_option.restype = None
    has_temporary.argtypes = []
    has_temporary.restype = ctypes.c_bool
    commit_temporary.argtypes = []
    commit_temporary.restype = None

    prompts: list[dict[str, object]] = []
    pending: list[int] = []
    callback_type = ctypes.CFUNCTYPE(
        None, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_void_p
    )

    @callback_type
    def on_options(player_id, prompt, option_count, options_ptr):
        safe_count = max(0, min(int(option_count), 1024))
        if safe_count != int(option_count):
            prompts.append({
                "player_id": int(player_id),
                "prompt": prompt.decode("utf-8", "replace") if prompt else "",
                "option_count": int(option_count),
                "defensive_layout_check_passed": False,
                "options": [],
            })
            return
        option_array = ctypes.cast(options_ptr, ctypes.POINTER(GameOption))
        prompts.append({
            "player_id": int(player_id),
            "prompt": prompt.decode("utf-8", "replace") if prompt else "",
            "option_count": safe_count,
            "options": [
                {
                    "ordinal": ordinal,
                    "option_index": int(option_array[ordinal].option_index),
                    "selection_id": int(option_array[ordinal].selection_id),
                    "selection_hint": int(option_array[ordinal].selection_hint),
                    "is_hidden": bool(option_array[ordinal].is_hidden),
                    "text": bytes(option_array[ordinal].option_text)
                    .split(b"\0", 1)[0].decode("utf-8", "replace"),
                }
                for ordinal in range(safe_count)
            ],
        })
        if safe_count and len(pending) + auto_selections[0] < args.max_auto_human_selections:
            pending.append(0)

    players = _player_array(args.layout, args.ai_difficulty)
    parameters = GameParameters(0, 0, 0, 0)
    event_buffer = (ctypes.c_ubyte * args.event_buffer_bytes)()
    auto_selections = [0]
    committed_temporary = False
    logical_event_count = 0
    nonempty_updates = 0
    event_previews: list[dict[str, object]] = []
    started_at = time.monotonic()
    stop_reason = "max_iterations"

    initialize(str(lua_path).encode("utf-8"), INITIALIZE_MODE_WITH_AI_HANDLER)
    try:
        set_listener(ctypes.cast(on_options, ctypes.c_void_p))
        start_game(ctypes.byref(parameters), 2, players, args.seed)
        for iteration in range(args.max_iterations):
            ctypes.memset(event_buffer, 0, ctypes.sizeof(event_buffer))
            returned_count = max(
                0, int(update_game(event_buffer, ctypes.sizeof(event_buffer)))
            )
            logical_event_count += returned_count
            if returned_count:
                nonempty_updates += 1
                if len(event_previews) < 16:
                    preview_length = min(args.event_preview_bytes, ctypes.sizeof(event_buffer))
                    event_previews.append({
                        "iteration": iteration,
                        "returned_logical_event_count": returned_count,
                        "raw_prefix_hex": bytes(event_buffer[:preview_length]).hex(),
                    })
            if pending:
                select_option(pending.pop(0))
                auto_selections[0] += 1
            if (
                not committed_temporary
                and args.commit_after_auto_selections
                and auto_selections[0] >= args.commit_after_auto_selections
                and bool(has_temporary())
            ):
                commit_temporary()
                committed_temporary = True
            observed_selections = count_verified_ai_selections(
                trace_path, expected_ai_ids, capture_session_id
            )
            if (
                args.stop_after_ai_selections
                and observed_selections >= args.stop_after_ai_selections
            ):
                stop_reason = "debugger_observed_ai_selection"
                break
            if time.monotonic() - started_at >= args.max_wall_seconds:
                stop_reason = "max_wall_seconds"
                break
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)
    finally:
        shutdown()

    report = {
        "schema_version": 2,
        "experimental": True,
        "verified_locally": False,
        "fidelity_claim": "none",
        "platform": platform.platform(),
        "library_path": str(lib_path),
        "library_sha256": library_sha256,
        "lua_path": str(lua_path),
        "layout": args.layout,
        "players": [
            {
                "player_id": int(players[index].player_id),
                "player_type": int(players[index].player_type),
                "ai_difficulty": int(players[index].ai_difficulty),
            }
            for index in range(2)
        ],
        "configured_ai_player_ids": expected_ai_ids,
        "start_seed": args.seed,
        "initialize_mode": INITIALIZE_MODE_WITH_AI_HANDLER,
        "debugger_contract": debugger_contract,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started_at,
        "human_callbacks": prompts,
        "auto_human_selections": auto_selections[0],
        "temporary_move_buffer_committed": committed_temporary,
        "update_game": {
            "buffer_byte_capacity": ctypes.sizeof(event_buffer),
            "logical_event_count": logical_event_count,
            "nonempty_updates": nonempty_updates,
            "raw_previews_not_semantically_decoded": event_previews,
        },
        "capture_session_id": capture_session_id,
        "host_script_sha256": sha256_file(Path(__file__).resolve()),
        "expected_observer_script_sha256": observer_script_sha256,
        "debugger_ai_selections": count_verified_ai_selections(
            trace_path, expected_ai_ids, capture_session_id
        ),
        "trace_jsonl": str(trace_path) if trace_path else None,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["debugger_ai_selections"] else 4


if __name__ == "__main__":
    sys.exit(main())
