#!/usr/bin/env python3
"""Read-only scan of saved self-play actions for exposure to audited prompts."""

from __future__ import annotations

import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("training_exposure.json")
TOKENS = {
    "socgov": "socialist governments",
    "liberation": "liberation theology",
    "reformer": "the reformer",
    "warsawpact_add": "add five influence in eastern europe",
    "vietnam_bonus": "extra 1 op available for southeast asia",
    "china_bonus": "extra 1 op available for asia",
}


def main() -> None:
    candidates = set((ROOT / "runs").glob("**/game_logs/*.jsonl"))
    candidates.update((ROOT / "runs").glob("games-*.jsonl"))
    prompt_actions = collections.Counter()
    selected_cards = collections.Counter()
    regional_followups = collections.Counter()
    games = 0
    actions_seen = 0
    parse_errors = []
    for path in sorted(candidates):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except Exception as error:
                    parse_errors.append({"file": str(path.relative_to(ROOT)), "line": line_number, "error": str(error)})
                    continue
                games += 1
                actions = record.get("actions") or []
                actions_seen += len(actions)
                for index, row in enumerate(actions):
                    action = row.get("action") or row.get("choice") or {}
                    prompt = str(action.get("prompt") or "").lower()
                    value = str(action.get("value") or "")
                    if action.get("decision") == "list" and value in {"socgov", "liberation", "reformer", "warsawpact", "vietnamrevolts", "china", "oas"}:
                        selected_cards[value] += 1
                    for key, token in TOKENS.items():
                        if token in prompt:
                            prompt_actions[key] += 1
                            if key.endswith("_bonus") and index + 1 < len(actions):
                                next_action = actions[index + 1].get("action") or actions[index + 1].get("choice") or {}
                                next_prompt = str(next_action.get("prompt") or "").lower()
                                if "plays " in next_prompt and " ops:" in next_prompt:
                                    regional_followups[f"{key}_reentered_ops"] += 1
                                else:
                                    regional_followups[f"{key}_other_next_prompt"] += 1
    result = {
        "scope": "saved repository self-play JSONL only; checkpoint training may include additional unsaved samples",
        "files_scanned": len(candidates),
        "games_scanned": games,
        "actions_scanned": actions_seen,
        "selected_card_actions": dict(sorted(selected_cards.items())),
        "prompt_actions": dict(sorted(prompt_actions.items())),
        "regional_bonus_followups": dict(sorted(regional_followups.items())),
        "parse_errors": parse_errors,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
