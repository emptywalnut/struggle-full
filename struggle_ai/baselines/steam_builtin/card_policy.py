from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


def normalize_card_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


@lru_cache(maxsize=1)
def load_card_policy() -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(__file__).with_name("card_policy.json").read_text())
    return {normalize_card_id(name): value for name, value in raw["cards"].items()}


def policy_for(card: Any, side: str) -> dict[str, Any]:
    item = load_card_policy().get(normalize_card_id(card), {})
    return {**item.get("common", {}), **item.get(side.lower(), {})}
