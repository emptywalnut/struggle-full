#!/usr/bin/env python3
"""Isolated smoke test for the installed macOS Twilight Struggle native engine."""

from __future__ import annotations

import ctypes
import argparse
import json
import sys
import time
from pathlib import Path


DEFAULT_APP_ROOT = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common/Twilight Struggle/"
    "TwilightStruggle.app/Contents"
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-contents", type=Path, default=DEFAULT_APP_ROOT)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()
    app_root = args.app_contents.expanduser().resolve()
    lib_path = (
        app_root
        / "PlugIns/TwilightStruggleLib-OSX.bundle/Contents/MacOS/"
        "TwilightStruggleLib-OSX"
    )
    lua_path = app_root / "Resources/Data/StreamingAssets/Lua"
    if (
        ctypes.sizeof(GameParameters) != 16
        or ctypes.sizeof(AppPlayerData) != 52
        or ctypes.sizeof(GameOption) != 64
    ):
        raise RuntimeError("native structure size mismatch")

    lib = ctypes.CDLL(str(lib_path))
    lib.Initialize.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.Initialize.restype = None
    lib.Shutdown.argtypes = []
    lib.Shutdown.restype = None
    lib.SetGameOptionsListener.argtypes = [ctypes.c_void_p]
    lib.SetGameOptionsListener.restype = None
    lib.StartGame.argtypes = [
        ctypes.POINTER(GameParameters),
        ctypes.c_int32,
        ctypes.POINTER(AppPlayerData),
        ctypes.c_uint32,
    ]
    lib.StartGame.restype = None
    lib.UpdateGame.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.UpdateGame.restype = ctypes.c_int32
    lib.SelectGameOption.argtypes = [ctypes.c_int32]
    lib.SelectGameOption.restype = None

    pending: list[int] = []
    prompts: list[dict[str, object]] = []
    callback_type = ctypes.CFUNCTYPE(
        None, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_void_p
    )

    @callback_type
    def on_options(player_id, prompt, option_count, options_ptr):
        option_array = ctypes.cast(options_ptr, ctypes.POINTER(GameOption))
        prompts.append(
            {
                "player_id": int(player_id),
                "prompt": prompt.decode("utf-8", "replace") if prompt else "",
                "option_count": int(option_count),
                "options": [
                    {
                        "index": int(option_array[index].option_index),
                        "selection_id": int(option_array[index].selection_id),
                        "selection_hint": int(option_array[index].selection_hint),
                        "hidden": bool(option_array[index].is_hidden),
                        "text": bytes(option_array[index].option_text)
                        .split(b"\0", 1)[0]
                        .decode("utf-8", "replace"),
                    }
                    for index in range(max(0, option_count))
                ],
            }
        )
        if option_count > 0:
            pending.append(0)

    players = (AppPlayerData * 2)()
    players[0].player_id = 0
    players[0].player_type = 0
    players[0].ai_difficulty = 0
    players[0].name = b"NativeOracleLocal"
    players[1].player_id = 1
    players[1].player_type = 2
    players[1].ai_difficulty = 2
    players[1].name = b"SteamCMediumAI"
    parameters = GameParameters(0, 0, 0, 0)
    events = (ctypes.c_uint64 * 1024)()

    lib.Initialize(str(lua_path).encode(), 1)
    try:
        lib.SetGameOptionsListener(ctypes.cast(on_options, ctypes.c_void_p))
        lib.StartGame(ctypes.byref(parameters), 2, players, 0x5A17C0DE)
        event_count = 0
        for _ in range(args.max_iterations):
            event_count += max(0, int(lib.UpdateGame(events, len(events))))
            if pending:
                lib.SelectGameOption(pending.pop(0))
            if len(prompts) >= args.max_prompts:
                break
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)
        print(json.dumps({"prompts": prompts, "event_count": event_count}, ensure_ascii=False))
    finally:
        lib.Shutdown()
    return 0 if prompts else 3


if __name__ == "__main__":
    sys.exit(main())
