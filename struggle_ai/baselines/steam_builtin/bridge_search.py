from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from struggle_ai.bridge_client import TwilightBridgeClient

from .config import SteamProxyConfig
from .evaluator import evaluate_proxy_leaf
from .filters import filter_and_order


@dataclass(frozen=True)
class ProxySearchDecision:
    action_index: int
    action_value: float
    root_values: dict[int, float]
    expanded_nodes: int
    depth_cutoffs: int
    search_state_depth: int
    chance_mode: str = "single restored Saito RNG sequence; not native expectimax"


class BridgeProxySearchRunner:
    """Bounded recursive minimax connected to the Saito bridge.

    This is stronger than the shallow action ordering proxy, but remains neither
    native Steam search nor exact chance aggregation.
    """

    def __init__(self, bridge: TwilightBridgeClient, config: SteamProxyConfig | None = None):
        self.bridge = bridge
        self.config = config or SteamProxyConfig()
        self.expanded_nodes = 0
        self.depth_cutoffs = 0

    def choose_action(self, obs: dict[str, Any]) -> ProxySearchDecision:
        legal = list(obs.get("legal_actions") or [])
        if not legal:
            raise ValueError("no legal actions")
        perspective = str(obs.get("current_player") or obs.get("side") or "us")
        root = self.bridge.snapshot(runtime_checkpoint=False)
        ordered = filter_and_order(legal, obs, perspective, self.config.max_candidates)
        self.expanded_nodes = 0
        self.depth_cutoffs = 0
        values: dict[int, float] = {}
        for action in ordered:
            self.bridge.restore(root)
            child = self.bridge.step(action)["observation"]
            child_snapshot = self.bridge.snapshot(runtime_checkpoint=False)
            index = next((idx for idx, candidate in enumerate(legal) if candidate is action), legal.index(action))
            values[index] = self._value(child, child_snapshot, perspective, depth=1)
        self.bridge.restore(root)
        best = max(values, key=lambda index: (values[index], -index))
        return ProxySearchDecision(
            action_index=best,
            action_value=values[best],
            root_values=values,
            expanded_nodes=self.expanded_nodes,
            depth_cutoffs=self.depth_cutoffs,
            search_state_depth=self.config.search_state_depth,
        )

    def _value(self, obs: dict[str, Any], snapshot: dict[str, Any], perspective: str, depth: int) -> float:
        self.expanded_nodes += 1
        if obs.get("terminal"):
            return evaluate_proxy_leaf(obs, perspective, self.config.proxy_leaf_weights)
        if depth >= self.config.search_state_depth:
            self.depth_cutoffs += 1
            return evaluate_proxy_leaf(obs, perspective, self.config.proxy_leaf_weights)
        legal = list(obs.get("legal_actions") or [])
        if not legal:
            return evaluate_proxy_leaf(obs, perspective, self.config.proxy_leaf_weights)
        actor = str(obs.get("current_player") or obs.get("side") or perspective)
        ordered = filter_and_order(legal, obs, actor, self.config.max_candidates)
        values = []
        for action in ordered:
            self.bridge.restore(snapshot)
            child = self.bridge.step(action)["observation"]
            child_snapshot = self.bridge.snapshot(runtime_checkpoint=False)
            values.append(self._value(child, child_snapshot, perspective, depth + 1))
        self.bridge.restore(snapshot)
        return (max if actor == perspective else min)(values)
