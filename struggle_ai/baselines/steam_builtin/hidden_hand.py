from __future__ import annotations

import random
from collections.abc import Iterable


def estimate_opponent_hand_uniform_proxy(known: Iterable[str], unknown_pool: Iterable[str], hand_size: int,
                                         rng: random.Random) -> tuple[str, ...]:
    """Uniform proxy: native sampling bias and RNG sequence remain unresolved."""
    fixed = tuple(dict.fromkeys(str(card) for card in known))
    slots = max(0, hand_size - len(fixed))
    pool = [str(card) for card in unknown_pool if str(card) not in fixed]
    if slots > len(pool):
        raise ValueError("not enough unknown cards to estimate opponent hand")
    return fixed + tuple(rng.sample(pool, slots))
