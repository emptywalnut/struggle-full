#!/usr/bin/env python3
"""Extract the small, non-copyrightable AI annotation subset from Steam Lua."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIELDS = {
    "headline", "resolve_opponent_event_first", "resolve_opponent_event_last",
    "defcon_drop", "defcon_ops", "defcon_event", "defcon_draw", "defcon_discard",
    "do_not_predict_headline", "play_after_action_round", "play_event_before_action_round",
}


def matching_brace(text: str, start: int) -> int:
    depth, quote, escape = 0, None, False
    for pos in range(start, len(text)):
        char = text[pos]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'": quote = char
        elif char == "{": depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0: return pos
    raise ValueError(f"unclosed table at {start}")


def scalar_fields(block: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, raw in re.findall(r"\b([a-z_]+)\s*=\s*(-?\d+(?:\.\d+)?|true|false)\s*[,;]", block):
        if key not in FIELDS: continue
        out[key] = raw == "true" if raw in {"true", "false"} else float(raw) if "." in raw else int(raw)
    return out


def extract(path: Path) -> dict[str, dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    cards: dict[str, dict[str, object]] = {}
    for match in re.finditer(r'g_twilight_cards\["([^"\\]+)"\]\s*=\s*\{', text):
        name, start = match.group(1), match.end() - 1
        block = text[start:matching_brace(text, start) + 1]
        ai_match = re.search(r"\bai\s*=\s*\{", block)
        if not ai_match: continue
        ai_start = ai_match.end() - 1
        ai = block[ai_start:matching_brace(block, ai_start) + 1]
        sides: dict[str, object] = {}
        spans = []
        for side in ("usa", "ussr"):
            sm = re.search(rf"\b{side}\s*=\s*\{{", ai)
            if sm:
                ss = sm.end() - 1; se = matching_brace(ai, ss) + 1
                sides[side] = scalar_fields(ai[ss:se]); spans.append((ss, se))
        common = ai
        for ss, se in sorted(spans, reverse=True): common = common[:ss] + common[se:]
        cards[name] = {"common": scalar_fields(common), **sides, "source": path.name}
    return cards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lua_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cards: dict[str, dict[str, object]] = {}
    for name in ("twilight_cards.lua", "twilight_promos.lua", "twilight_turnzero.lua"):
        cards.update(extract(args.lua_dir / name))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": 1, "cards": cards}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cards)} annotated cards to {args.output}")


if __name__ == "__main__": main()
