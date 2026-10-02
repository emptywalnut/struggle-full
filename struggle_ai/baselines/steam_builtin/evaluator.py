from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import PROXY_LEAF_WEIGHTS

TERMINAL_UTILITIES = (0.0, 55000.0, 60000.0, 50000.0, 70000.0, 100000.0, 80000.0, 60000.0)


@dataclass(frozen=True)
class NativePreparedAccumulators:
    """Inputs consumed after the native 14 weights have already been applied.

    A raw Saito observation cannot populate these fields faithfully without a
    complete recovery of ``PrepareEvaluation`` or a native instrumentation dump.
    """

    vp_counter: int
    weight_0: float
    region_accumulators: tuple[float, float, float, float, float, float, float]
    perspective_sign: float
    player_accumulator_7000: float
    player_accumulator_7032: float


def score_native_base_pair(fields: NativePreparedAccumulators) -> tuple[float, float]:
    """Exact ``ScoreBaseState`` data flow at arm64 0x1a4a8-0x1a518."""
    score = fields.weight_0 * (2.0 * fields.vp_counter)
    score += sum(fields.region_accumulators)
    score *= fields.perspective_sign
    score += fields.player_accumulator_7000 + fields.player_accumulator_7032
    return score, -score


def terminal_utility(obs: dict[str, Any], side: str) -> float | None:
    if not obs.get("terminal"):
        return None
    winner = str(obs.get("winner") or "")
    if winner not in {"us", "ussr"}:
        return 0.0
    raw_reason = obs.get("terminal_reason_index", 5)
    reason = 5 if raw_reason is None else int(raw_reason)
    utility = TERMINAL_UTILITIES[reason] if 0 <= reason < len(TERMINAL_UTILITIES) else 100000.0
    utility -= 100.0 * float(obs.get("terminal_timing", 0) or 0)
    return utility if winner == side else -utility


def extract_proxy_features(obs: dict[str, Any], side: str) -> tuple[float, ...]:
    """Observable Saito proxy features; none is asserted to be a native slot."""
    sign = 1.0 if side.lower() == "us" else -1.0
    influence_balance = battleground_balance = controlled_balance = 0.0
    for country in obs.get("countries", []) or []:
        balance = float(country.get("us", 0) or 0) - float(country.get("ussr", 0) or 0)
        influence_balance += balance
        if country.get("battleground") or country.get("bg"):
            battleground_balance += balance
        stability = max(1.0, float(country.get("stability", 1) or 1))
        controlled_balance += max(-1.0, min(1.0, balance / stability))
    vp = float(obs.get("vp", 0) or 0)
    space_state = obs.get("space") or {}
    space = float(space_state.get("us", obs.get("us_space", 0)) or 0)
    space -= float(space_state.get("ussr", obs.get("ussr_space", 0)) or 0)
    return tuple(sign * value for value in (vp, influence_balance, battleground_balance, controlled_balance, space))


def evaluate_proxy_leaf(obs: dict[str, Any], side: str,
                        weights: tuple[float, ...] = PROXY_LEAF_WEIGHTS) -> float:
    terminal = terminal_utility(obs, side)
    if terminal is not None:
        return terminal
    if len(weights) != 5:
        raise ValueError("proxy leaf evaluator requires exactly five weights")
    return sum(weight * feature for weight, feature in zip(weights, extract_proxy_features(obs, side)))
