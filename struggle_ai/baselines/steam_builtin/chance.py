from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def expected_value(outcomes: Iterable[tuple[float, T]], value: Callable[[T], float] = float) -> float:
    items = list(outcomes)
    total = sum(probability for probability, _ in items)
    if total <= 0:
        raise ValueError("chance probabilities must have positive mass")
    return sum(probability * value(outcome) for probability, outcome in items) / total


def one_die() -> tuple[tuple[float, int], ...]:
    return tuple((1 / 6, roll) for roll in range(1, 7))


def two_dice() -> tuple[tuple[float, tuple[int, int]], ...]:
    return tuple((1 / 36, (a, b)) for a in range(1, 7) for b in range(1, 7))


def proxy_coup_outcomes(ops: int, stability: int, modifier: int = 0) -> tuple[tuple[float, int], ...]:
    """Rules proxy before native queries/clamping; not recovered calculator parity."""
    return tuple((p, max(0, die + ops + modifier - 2 * stability)) for p, die in one_die())


def proxy_realignment_outcomes(us_mod: int = 0, ussr_mod: int = 0) -> tuple[tuple[float, int], ...]:
    """Rules proxy; native adjustment queries and influence clamping are external."""
    return tuple((p, (a + us_mod) - (b + ussr_mod)) for p, (a, b) in two_dice())
