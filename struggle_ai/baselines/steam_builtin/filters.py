from __future__ import annotations

from typing import Any

from .card_policy import policy_for


def card_id(action: dict[str, Any]) -> str:
    return str(action.get("card") or action.get("value") or action.get("label") or "")


def action_score(action: dict[str, Any], obs: dict[str, Any], side: str) -> float:
    value = str(action.get("value") or "").lower()
    prompt = str(action.get("prompt") or obs.get("prompt") or "").lower()
    data = policy_for(card_id(action), side)
    score = 0.0
    if "headline" in prompt: score += 10.0 * float(data.get("headline", 0))
    ar = int(obs.get("action_round", 0) or 0)
    if data.get("play_event_before_action_round") is not None:
        score += 25.0 if ar <= int(data["play_event_before_action_round"]) else -10.0
    if data.get("play_after_action_round") is not None:
        score += 25.0 if ar > int(data["play_after_action_round"]) else -10.0
    if value in {"event", "playevent"}: score += 8.0
    if value == "space": score += 3.0
    if value in {"cancel", "skip", "nope"}: score -= 100.0
    if int(obs.get("defcon", 5) or 5) <= 2:
        risky = bool(data.get("defcon_drop"))
        if value == "ops": risky = risky or bool(data.get("defcon_ops"))
        if value in {"event", "playevent"}: risky = risky or bool(data.get("defcon_event"))
        if risky: score -= 1000.0
    country = next((c for c in obs.get("countries", []) if c.get("id") == action.get("value")), None)
    if country:
        score += 12.0 if country.get("battleground") or country.get("bg") else 2.0
        if "coup" in prompt: score += max(0.0, 5.0 - float(country.get("stability", 3) or 3))
    return score


def filter_and_order(actions: list[dict[str, Any]], obs: dict[str, Any], side: str,
                     limit: int = 24) -> list[dict[str, Any]]:
    """Proxy ordering plus the confirmed four-per-region placement quota."""
    ranked = sorted(enumerate(actions), key=lambda pair: (-action_score(pair[1], obs, side), pair[0]))
    safe = [pair for pair in ranked if action_score(pair[1], obs, side) > -900]
    selected = safe or ranked
    prompt = str(obs.get("prompt") or "").lower()
    decisions = {str(action.get("decision") or "").lower() for _, action in selected}
    if "place" in prompt and decisions & {"country_click", "country_mouseup"}:
        by_country = {str(country.get("id")): country for country in obs.get("countries", []) or []}
        per_region: dict[str, int] = {}
        quota_selected = []
        for pair in selected:
            country = by_country.get(str(pair[1].get("value") or ""), {})
            region = str(country.get("region") or "unknown").lower()
            if per_region.get(region, 0) >= 4:
                continue
            per_region[region] = per_region.get(region, 0) + 1
            quota_selected.append(pair)
        selected = quota_selected or selected
    return [action for _, action in selected[:max(1, limit)]]
