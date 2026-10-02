from __future__ import annotations

from dataclasses import dataclass

STEAM_TUNED_WEIGHTS = (0.741206, 0.0, 0.229500, 0.375208, 0.0, 0.2, 0.384150,
                       0.395302, 0.051567, 0.0, 0.013033, 0.126330, 0.2, 0.15)
STEAM_FALLBACK_WEIGHTS = (0.5, 0.1, 1.0, 0.6, 0.2, 0.3, 0.1, 0.2, 0.2, 0.2,
                          0.2, 0.4, 0.2, 0.15)

# Deliberately independent of the native 14 constants. These score observable
# Saito fields for the runnable reconstruction proxy only.
PROXY_LEAF_WEIGHTS = (100.0, 8.0, 16.0, 5.0, 4.0)


@dataclass(frozen=True)
class SteamProxyConfig:
    proxy_leaf_weights: tuple[float, ...] = PROXY_LEAF_WEIGHTS
    search_state_depth: int = 2
    max_candidates: int = 8
    seed: int = 1
    native_state_depth_limit: int = 40

    def __post_init__(self) -> None:
        if len(self.proxy_leaf_weights) != 5:
            raise ValueError("proxy leaf evaluator requires exactly five weights")
        if self.search_state_depth < 0:
            raise ValueError("search_state_depth must be non-negative")
        if self.native_state_depth_limit != 40:
            raise ValueError("native_state_depth_limit is evidence, not a tuning parameter")
