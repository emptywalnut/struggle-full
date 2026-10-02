from __future__ import annotations

import sys


def required_search_recursion_limit(max_atomic_steps: int) -> int:
    """Return stack headroom for recursive atomic search and serialization."""

    return max(4096, 4 * int(max_atomic_steps) + 2048)


def configure_search_recursion_limit(max_atomic_steps: int) -> int:
    """Raise, but never lower, the process recursion limit for search work."""

    required = required_search_recursion_limit(max_atomic_steps)
    if sys.getrecursionlimit() < required:
        sys.setrecursionlimit(required)
    return sys.getrecursionlimit()
