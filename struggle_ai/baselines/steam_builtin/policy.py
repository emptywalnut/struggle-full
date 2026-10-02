from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np

from struggle_ai.baseline_policies import RandomLegalPolicy
from .config import SteamProxyConfig
from .evaluator import evaluate_proxy_leaf
from .filters import action_score, filter_and_order
from .hidden_hand import estimate_opponent_hand_uniform_proxy


class SteamReconstructedProxy:
    """Evidence-informed proxy, not a validated implementation of Steam AI."""

    display_name = "Steam-reconstructed proxy"

    def __init__(self, config: SteamProxyConfig | None = None):
        self.config = config or SteamProxyConfig()
        self.rng = random.Random(self.config.seed)

    def estimate_hand_uniform_proxy(self, known, unknown_pool, hand_size: int) -> tuple[str, ...]:
        return estimate_opponent_hand_uniform_proxy(known, unknown_pool, hand_size, self.rng)

    def search_value(self, obs: dict[str, Any], perspective: str,
                     expand: Callable[[dict[str, Any]], tuple[list[dict[str, Any]], Callable[[dict[str, Any]], Any]]],
                     depth: int = 0) -> float:
        if obs.get("terminal") or depth >= self.config.search_state_depth:
            return evaluate_proxy_leaf(obs, perspective, self.config.proxy_leaf_weights)
        actions, transition = expand(obs)
        if not actions:
            return evaluate_proxy_leaf(obs, perspective, self.config.proxy_leaf_weights)
        actor = str(obs.get("current_player") or obs.get("side") or perspective)
        ordered = filter_and_order(actions, obs, actor, self.config.max_candidates)
        values = []
        for action in ordered:
            outcome = transition(action)
            branches = outcome if isinstance(outcome, list) else [(1.0, outcome)]
            mass = sum(float(probability) for probability, _ in branches)
            if mass <= 0:
                raise ValueError("chance probabilities must have positive mass")
            values.append(sum(float(probability) * self.search_value(child, perspective, expand, depth + 1)
                              for probability, child in branches) / mass)
        return (max if actor == perspective else min)(values)

    def choose_action(self, legal_actions: list[dict[str, Any]], obs: dict[str, Any],
                      transition: Callable[[dict[str, Any]], Any] | None = None) -> int:
        if not legal_actions:
            raise ValueError("no legal actions")
        side = str(obs.get("current_player") or obs.get("side") or "us")
        ordered = filter_and_order(legal_actions, obs, side, self.config.max_candidates)
        values: dict[int, float] = {}
        for action in ordered:
            idx = next(i for i, candidate in enumerate(legal_actions) if candidate is action)
            if transition is None:
                values[idx] = action_score(action, obs, side)
                continue
            outcome = transition(action)
            branches = outcome if isinstance(outcome, list) else [(1.0, outcome)]
            mass = sum(float(probability) for probability, _ in branches)
            values[idx] = sum(float(probability) * evaluate_proxy_leaf(state, side, self.config.proxy_leaf_weights)
                              for probability, state in branches) / mass
        return max(values, key=lambda idx: (values[idx], -idx))


class SteamReconstructedProxyPolicy(RandomLegalPolicy):
    """RLlib-shaped adapter for the proxy, not the recovered Steam policy."""

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                        prev_reward_batch=None, **kwargs):
        actions = []
        for obs in self._iter_obs(obs_batch):
            mask, features = self._mask_and_features(obs)
            legal = np.flatnonzero(mask > 0.5)
            if not len(legal):
                actions.append(0)
            elif features.ndim == 2 and features.shape[1]:
                actions.append(int(legal[int(np.argmax(features[legal, -1]))]))
            else:
                actions.append(int(legal[0]))
        return np.asarray(actions, dtype=np.int64), [], {}
