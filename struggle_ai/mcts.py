from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol

import numpy as np

from struggle_ai.search_convergence import compare_q_rankings
from struggle_ai.runtime_limits import configure_search_recursion_limit


class SearchEnvironment(Protocol):
    last_obs: dict[str, Any] | None
    legal_actions: list[dict[str, Any]]

    def restore(self, snapshot: dict[str, Any]) -> dict[str, np.ndarray]: ...
    def redeterminize(self, side: str, seed: int) -> dict[str, np.ndarray]: ...
    def snapshot(self, compact: bool = False) -> dict[str, Any]: ...
    def step_with_forced_rolls(self, action: int, forced_rolls: list[int | float]): ...


def local_search_snapshot(env: SearchEnvironment) -> dict[str, Any]:
    snapshot_local = getattr(env, "snapshot_local", None)
    if callable(snapshot_local):
        return snapshot_local(compact=True)
    return env.snapshot(compact=True)


def restore_search_state(env: SearchEnvironment, snapshot: dict[str, Any]) -> None:
    restore_local = getattr(env, "restore_local", None)
    if snapshot.get("local_bridge") and callable(restore_local):
        restore_local(snapshot, encode=False)
        return
    env.restore(snapshot)


PolicyValueFn = Callable[[dict[str, np.ndarray], int], tuple[np.ndarray, float]]


@dataclass
class SearchNode:
    side: str
    priors: np.ndarray
    active: np.ndarray
    visits: np.ndarray
    value_sum_us: np.ndarray

    @classmethod
    def create(cls, side: str, priors: np.ndarray, active: np.ndarray | None = None) -> "SearchNode":
        priors = np.asarray(priors, dtype=np.float64)
        active = np.asarray(active if active is not None else priors > 0.0, dtype=np.bool_)
        if active.shape != priors.shape or not bool(active.any()):
            raise ValueError("MCTS node requires at least one active action")
        return cls(
            side=side,
            priors=priors,
            active=active,
            visits=np.zeros_like(priors, dtype=np.int64),
            value_sum_us=np.zeros_like(priors, dtype=np.float64),
        )

    def select(self, cpuct: float) -> int:
        total = int(self.visits.sum())
        q_us = np.divide(
            self.value_sum_us,
            self.visits,
            out=np.zeros_like(self.value_sum_us),
            where=self.visits > 0,
        )
        perspective_q = q_us if self.side == "us" else -q_us
        exploration = cpuct * self.priors * math.sqrt(total + 1.0) / (1.0 + self.visits)
        score = perspective_q + exploration
        score = np.where(self.active, score, -np.inf)
        return int(np.argmax(score))


@dataclass
class SearchResult:
    action: int
    policy: np.ndarray
    visits: np.ndarray
    root_value_us: float
    simulations: int
    expanded_nodes: int
    max_depth_reached: int
    belief_particles: int
    chance_roll_histogram: dict[str, int] = field(default_factory=dict)


def information_state_key(obs: dict[str, np.ndarray], side: str) -> str:
    """Hash only model-visible tensors, preventing hidden-state tree leakage."""
    digest = hashlib.blake2b(digest_size=20)
    digest.update(side.encode("ascii"))
    for key in sorted(obs):
        value = np.asarray(obs[key])
        digest.update(key.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def normalize_priors(priors: np.ndarray, legal_count: int) -> np.ndarray:
    priors = np.asarray(priors[:legal_count], dtype=np.float64)
    priors = np.nan_to_num(priors, nan=0.0, posinf=0.0, neginf=0.0)
    priors = np.maximum(priors, 0.0)
    total = float(priors.sum())
    if total <= 0.0:
        return np.full(legal_count, 1.0 / legal_count, dtype=np.float64)
    return priors / total


def temperature_scale_priors(priors: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrate policy priors without changing legal-action support."""

    priors = normalize_priors(priors, len(priors))
    if temperature <= 0.0:
        raise ValueError("prior temperature must be positive")
    if math.isclose(temperature, 1.0):
        return priors
    scaled = np.power(np.maximum(priors, 1e-30), 1.0 / temperature)
    return scaled / max(float(scaled.sum()), 1e-30)


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    """Return deterministic [1, 0] rank scores, averaging exact ties."""

    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if count <= 1:
        return np.ones(count, dtype=np.float64)
    order = sorted(range(count), key=lambda index: (-float(values[index]), index))
    ranks = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        end = start + 1
        while end < count and math.isclose(
            float(values[order[start]]),
            float(values[order[end]]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            end += 1
        average_rank = 0.5 * (start + end - 1)
        score = 1.0 - average_rank / (count - 1)
        for offset in range(start, end):
            ranks[order[offset]] = score
        start = end
    return ranks


def q_aware_root_action(
    visits: np.ndarray,
    q_root: np.ndarray,
    priors: np.ndarray,
    labels: tuple[str, ...],
    *,
    min_visits: int,
    visit_fraction: float,
    min_q_advantage: float,
) -> tuple[int, dict[str, Any]]:
    """Select a robust root child without index-based tie breaking.

    Visit count remains the primary reliability filter. Among sufficiently
    explored near-leaders, a materially better Q estimate may override the
    visit leader. All residual ties use Q, prior, then label text.
    """

    visits = np.asarray(visits, dtype=np.float64)
    q_root = np.asarray(q_root, dtype=np.float64)
    priors = np.asarray(priors, dtype=np.float64)
    if not (len(visits) == len(q_root) == len(priors) == len(labels)) or not len(visits):
        raise ValueError("root selection arrays must be non-empty and aligned")
    maximum_visits = float(visits.max())
    visit_leaders = [index for index in range(len(visits)) if math.isclose(float(visits[index]), maximum_visits)]
    robust = max(
        visit_leaders,
        key=lambda index: (float(q_root[index]), float(priors[index]), labels[index]),
    )
    required_visits = max(float(min_visits), maximum_visits * max(0.0, float(visit_fraction)))
    eligible = [
        index
        for index in range(len(visits))
        if float(visits[index]) + 1e-12 >= required_visits
    ]
    q_candidate = max(
        eligible or [robust],
        key=lambda index: (float(q_root[index]), float(visits[index]), float(priors[index]), labels[index]),
    )
    override = bool(
        q_candidate != robust
        and float(q_root[q_candidate]) >= float(q_root[robust]) + float(min_q_advantage)
    )
    selected = q_candidate if override else robust
    return selected, {
        "visit_leader": int(robust),
        "q_candidate": int(q_candidate),
        "selected": int(selected),
        "q_override": override,
        "required_visits": required_visits,
        "visit_tie_count": len(visit_leaders),
        "q_advantage": float(q_root[q_candidate] - q_root[robust]),
    }


def calibrate_value(value: float) -> float:
    """Map an unbounded PPO critic output onto the MCTS terminal-value scale."""
    value = float(np.nan_to_num(value, nan=0.0, posinf=20.0, neginf=-20.0))
    return float(np.tanh(value))


class ChanceAwareMCTS:
    """PUCT over information states with root belief particles and dice samples.

    Values are stored in absolute US perspective. Each simulation restores the
    exact public history, redeterminizes the hidden opponent hand, and supplies
    explicit uniformly sampled d6 outcomes to the Saito rules engine.
    """

    def __init__(
        self,
        evaluator: PolicyValueFn,
        *,
        simulations: int = 32,
        max_depth: int = 16,
        cpuct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25,
        discount: float = 1.0,
        belief_particles: int = 4,
        max_children: int = 0,
        seed: int = 1,
    ) -> None:
        if simulations < 2:
            raise ValueError("MCTS requires at least two simulations")
        self.evaluator = evaluator
        self.simulations = int(simulations)
        self.max_depth = int(max_depth)
        self.cpuct = float(cpuct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_fraction = float(dirichlet_fraction)
        self.discount = float(discount)
        self.belief_particles = max(1, int(belief_particles))
        self.max_children = max(0, int(max_children))
        self.rng = np.random.default_rng(seed)

    def _expand(
        self,
        nodes: dict[str, SearchNode],
        obs: dict[str, np.ndarray],
        side: str,
        legal_count: int,
        *,
        root: bool = False,
    ) -> tuple[SearchNode, float]:
        priors, value = self.evaluator(obs, legal_count)
        priors = normalize_priors(priors, legal_count)
        active = np.ones(legal_count, dtype=np.bool_)
        if self.max_children and legal_count > self.max_children:
            active = np.zeros(legal_count, dtype=np.bool_)
            top_indices = np.argpartition(priors, -self.max_children)[-self.max_children :]
            active[top_indices] = True
            priors = np.where(active, priors, 0.0)
            priors /= priors.sum()
        if root and int(active.sum()) > 1 and self.dirichlet_fraction > 0.0:
            active_indices = np.flatnonzero(active)
            noise = self.rng.dirichlet(np.full(len(active_indices), self.dirichlet_alpha))
            priors[active_indices] = (
                (1.0 - self.dirichlet_fraction) * priors[active_indices]
                + self.dirichlet_fraction * noise
            )
        node = SearchNode.create(side, priors, active)
        nodes[information_state_key(obs, side)] = node
        value = calibrate_value(value)
        return node, value if side == "us" else -value

    def search(
        self,
        env: SearchEnvironment,
        root_snapshot: dict[str, Any],
        root_obs: dict[str, np.ndarray],
        *,
        temperature: float = 1.0,
    ) -> SearchResult:
        env.restore(root_snapshot)
        root_side = str(env.last_obs["side"] if env.last_obs else "ussr")
        root_legal_count = int(np.asarray(root_obs["action_mask"]).sum())
        if root_legal_count <= 0:
            raise RuntimeError("MCTS root has no legal actions")
        nodes: dict[str, SearchNode] = {}
        root, _root_prior_value_us = self._expand(
            nodes,
            root_obs,
            root_side,
            root_legal_count,
            root=True,
        )
        chance_histogram: dict[str, int] = {}
        max_depth_reached = 0
        belief_seeds = [
            int(self.rng.integers(1, 2**31 - 1))
            for _ in range(min(self.simulations, self.belief_particles))
        ]

        for simulation in range(self.simulations):
            obs = env.restore(root_snapshot)
            belief_seed = belief_seeds[simulation % len(belief_seeds)]
            obs = env.redeterminize(root_side, belief_seed)
            node = root
            path: list[tuple[SearchNode, int]] = []
            leaf_value_us = 0.0

            for depth in range(self.max_depth):
                max_depth_reached = max(max_depth_reached, depth + 1)
                action = node.select(self.cpuct)
                path.append((node, action))
                # Quantiles, rather than fixed d6 faces, remain uniform when a
                # card asks Saito to roll over a dynamic number of choices.
                forced_rolls = [float(value) for value in self.rng.random(16)]
                next_obs, _reward, terminated, truncated, info = env.step_with_forced_rolls(action, forced_rolls)
                consumed = tuple(int(roll) for roll in info.get("chance_rolls", []))
                if consumed:
                    label = ",".join(map(str, consumed))
                    chance_histogram[label] = chance_histogram.get(label, 0) + 1
                if terminated or truncated:
                    winner = str(info.get("winner") or "")
                    leaf_value_us = 1.0 if winner == "us" else -1.0 if winner == "ussr" else 0.0
                    break

                side = str(env.last_obs["side"] if env.last_obs else "ussr")
                legal_count = int(np.asarray(next_obs["action_mask"]).sum())
                if legal_count <= 0:
                    leaf_value_us = 0.0
                    break
                key = information_state_key(next_obs, side)
                child = nodes.get(key)
                if child is None:
                    child, leaf_value_us = self._expand(nodes, next_obs, side, legal_count)
                    break
                node = child
            else:
                _priors, value = self.evaluator(next_obs, int(np.asarray(next_obs["action_mask"]).sum()))
                leaf_value_us = calibrate_value(value)
                if str(env.last_obs["side"] if env.last_obs else "ussr") == "ussr":
                    leaf_value_us = -leaf_value_us

            backed_value = leaf_value_us
            for visited_node, action in reversed(path):
                visited_node.visits[action] += 1
                visited_node.value_sum_us[action] += backed_value
                backed_value *= self.discount

        visits = root.visits.astype(np.float64)
        policy = self._visit_policy(visits, root.priors, temperature)
        action = int(self.rng.choice(len(policy), p=policy))
        root_total = int(root.visits.sum())
        root_value_us = float(root.value_sum_us.sum() / root_total) if root_total else 0.0
        return SearchResult(
            action=action,
            policy=policy.astype(np.float32),
            visits=root.visits.copy(),
            root_value_us=root_value_us,
            simulations=self.simulations,
            expanded_nodes=len(nodes),
            max_depth_reached=max_depth_reached,
            belief_particles=len(belief_seeds),
            chance_roll_histogram=chance_histogram,
        )

    @staticmethod
    def _visit_policy(
        visits: np.ndarray,
        priors: np.ndarray,
        temperature: float,
        *,
        preferred_action: int | None = None,
    ) -> np.ndarray:
        if float(visits.sum()) <= 0.0:
            return normalize_priors(priors, len(priors))
        if temperature <= 1e-6:
            policy = np.zeros_like(visits, dtype=np.float64)
            policy[int(np.argmax(visits))] = 1.0
            return policy
        scaled = np.power(visits, 1.0 / temperature, dtype=np.float64)
        if float(scaled.sum()) <= 0.0:
            return normalize_priors(priors, len(priors))
        return scaled / scaled.sum()


class TorchPolicyValueEvaluator:
    """Adapter from the existing masked side-specific model to MCTS."""

    def __init__(self, policy: Any) -> None:
        import torch

        self.torch = torch
        self.policy = policy
        self.model = policy.model
        self.device = next(self.model.parameters()).device
        self.cache: dict[tuple[str, int], dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def __call__(self, obs: dict[str, np.ndarray], legal_count: int) -> tuple[np.ndarray, float]:
        raw = self.raw(obs, legal_count)
        return raw["probabilities"], float(raw["raw_value"])

    def raw(self, obs: dict[str, np.ndarray], legal_count: int) -> dict[str, Any]:
        key = (information_state_key(obs, "network"), int(legal_count))
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        torch = self.torch
        batch = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
            for key, value in obs.items()
        }
        self.model.eval()
        with torch.no_grad():
            logits, _state = self.model({"obs": batch}, [], None)
            value = self.model.value_function()
            legal_logits = logits[0, :legal_count]
            priors = torch.softmax(legal_logits, dim=-1)
        result = {
            "logits": legal_logits.detach().cpu().numpy().astype(np.float64),
            "probabilities": priors.detach().cpu().numpy().astype(np.float64),
            "raw_value": float(value[0].detach().cpu()),
            "calibrated_value": calibrate_value(float(value[0].detach().cpu())),
        }
        self.cache[key] = result
        return result

    def value_us(self, obs: dict[str, np.ndarray], side: str) -> tuple[float, float]:
        raw = self.raw(obs, int(np.asarray(obs["action_mask"]).sum()))
        sign = 1.0 if side == "us" else -1.0
        return sign * float(raw["raw_value"]), sign * float(raw["calibrated_value"])


def action_text(action: dict[str, Any]) -> str:
    label = str(action.get("label") or action.get("value") or action.get("type") or "?")
    value = str(action.get("value") or "")
    return label if not value or value.lower() == label.lower() else f"{label} [{value}]"


def root_side(last_obs: dict[str, Any] | None) -> str:
    return str((last_obs or {}).get("side") or "ussr")


def scoring_search_throttle_key(
    episode_seed: int | None,
    last_obs: dict[str, Any] | None,
) -> tuple[int | None, str, int, int]:
    state = last_obs or {}
    return (
        episode_seed,
        root_side(state),
        int(state.get("turn") or 0),
        int(state.get("action_round") or 0),
    )


def is_forced_response_decision(last_obs: dict[str, Any] | None, legal_actions: list[dict[str, Any]]) -> bool:
    if not last_obs:
        return False
    prompt = str(last_obs.get("prompt") or "").lower()
    decisions = {str(action.get("decision") or "").lower() for action in legal_actions}
    values = {str(action.get("value") or "").lower() for action in legal_actions}
    standard_mode_values = {
        "event",
        "ops",
        "space",
        "before_ops",
        "after_ops",
        "place",
        "coup",
        "realign",
    }
    return "hosts the" in prompt or (
        "option" in decisions
        and len(legal_actions) <= 6
        and values.isdisjoint(standard_mode_values)
    )


@dataclass(frozen=True)
class MeaningfulDecisionBoundary:
    kind: str
    identity: str
    coalesce: bool = False


def _normalized_decision_prompt(prompt: str) -> str:
    normalized = re.sub(r"<[^>]+>", " ", prompt.lower())
    normalized = re.sub(r"\b\d+\b", "#", normalized)
    return " ".join(normalized.split())


def meaningful_decision_boundary(
    last_obs: dict[str, Any] | None,
    legal_actions: list[dict[str, Any]],
) -> MeaningfulDecisionBoundary | None:
    """Classify a strategic choice while coalescing multi-click allocations.

    A state with only one legal action is bookkeeping, even when its UI prompt
    resembles a strategic choice. Influence placement is the only coalesced
    boundary: all consecutive country clicks in one allocation consume one
    meaningful layer, and search continues until that allocation is complete.
    """

    if not last_obs or len(legal_actions) <= 1:
        return None
    prompt = str(last_obs.get("prompt") or "")
    prompt_lower = prompt.lower()
    phase = str(last_obs.get("phase") or "").lower()
    side = root_side(last_obs)
    turn = int(last_obs.get("turn") or 0)
    action_round = int(last_obs.get("action_round") or 0)
    normalized_prompt = _normalized_decision_prompt(prompt)
    levels = {action_level(action, prompt) for action in legal_actions}

    if phase.startswith("headline_") or "headline" in prompt_lower:
        return MeaningfulDecisionBoundary("headline", f"headline:{turn}:{side}")
    if is_forced_response_decision(last_obs, legal_actions):
        return MeaningfulDecisionBoundary(
            "forced_response",
            f"forced_response:{turn}:{action_round}:{side}:{normalized_prompt}",
        )
    if levels == {"card"} or "pick a card" in prompt_lower or "choose a card" in prompt_lower or "select a card" in prompt_lower:
        return MeaningfulDecisionBoundary(
            "card_choice",
            f"card_choice:{turn}:{action_round}:{side}:{normalized_prompt}",
        )
    if levels == {"mode"}:
        return MeaningfulDecisionBoundary(
            "mode",
            f"mode:{turn}:{action_round}:{side}:{normalized_prompt}",
        )
    if levels == {"ops_action_type"}:
        return MeaningfulDecisionBoundary(
            "ops_action_type",
            f"ops_action_type:{turn}:{action_round}:{side}:{normalized_prompt}",
        )
    if levels == {"influence_allocation"}:
        return MeaningfulDecisionBoundary(
            "influence_allocation",
            f"influence_allocation:{turn}:{action_round}:{side}:{normalized_prompt}",
            coalesce=True,
        )
    if levels == {"coup_target"}:
        return MeaningfulDecisionBoundary(
            "coup_target",
            f"coup_target:{turn}:{action_round}:{side}:{normalized_prompt}",
        )
    if levels == {"realignment_target"}:
        return MeaningfulDecisionBoundary(
            "realignment_target",
            f"realignment_target:{turn}:{action_round}:{side}:{normalized_prompt}",
        )
    return None


def meaningful_decision_kind(last_obs: dict[str, Any] | None, legal_actions: list[dict[str, Any]]) -> str | None:
    boundary = meaningful_decision_boundary(last_obs, legal_actions)
    return boundary.kind if boundary else None


def action_level(action: dict[str, Any], prompt: str = "") -> str:
    decision = str(action.get("decision") or "").lower()
    value = str(action.get("value") or "").lower()
    context = f"{prompt} {action.get('prompt') or ''}".lower()
    if "headline" in context:
        return "card"
    if decision in {"country_click", "country_mouseup"}:
        if "realign" in context:
            return "realignment_target"
        if "coup" in context:
            return "coup_target"
        return "influence_allocation"
    if value in {"event", "ops", "before_ops", "after_ops", "space"}:
        return "mode"
    if value in {"place", "coup", "realign"}:
        return "ops_action_type"
    if decision == "option":
        return "forced_response"
    return "card" if decision == "list" else "other"


LOCAL_DECISION_LEVELS = frozenset(
    {
        "influence_allocation",
        "coup_target",
        "realignment_target",
        "forced_response",
    }
)


def executable_continuation_prefix(
    behavior_path: tuple[dict[str, str], ...],
) -> tuple[dict[str, str], ...]:
    """Keep only actions belonging to the root's current strategic macro.

    A search trace may span several meaningful decisions. It is useful as
    planning evidence, but only the current macro may be executed without
    asking the planner again. In particular, an event response or allocation
    must never pre-commit the following Action Round's card choice.
    """

    if len(behavior_path) <= 1:
        return ()
    root_side = str(behavior_path[0].get("side") or "")
    root_level = str(behavior_path[0].get("level") or "")
    continuation: list[dict[str, str]] = []
    for entry in behavior_path[1:]:
        level = str(entry.get("level") or "")
        if str(entry.get("side") or "") != root_side:
            break
        if level == "card":
            break
        if root_level in LOCAL_DECISION_LEVELS:
            if level != root_level:
                break
        elif level in LOCAL_DECISION_LEVELS:
            break
        continuation.append(dict(entry))
    return tuple(continuation)


def continuation_matches_live_boundary(
    expected: dict[str, str],
    last_obs: dict[str, Any] | None,
    legal_actions: list[dict[str, Any]],
) -> bool:
    """Return whether a stored macro continuation belongs to the live root."""

    if len(legal_actions) <= 1:
        return True
    boundary = meaningful_decision_boundary(last_obs, legal_actions)
    # Headlines and ordinary card choices are always independent planning roots.
    if boundary is not None and boundary.kind in {"headline", "card_choice"}:
        return False
    expected_level = str(expected.get("level") or "")
    if boundary is not None:
        return boundary.kind == expected_level
    prompt = str((last_obs or {}).get("prompt") or "")
    return expected_level in {action_level(action, prompt) for action in legal_actions}


def validate_information_safe_observation(obs: dict[str, np.ndarray], side: str) -> None:
    """Reject an encoded observation that exposes the sampled opponent hand."""

    opponent_key = "ussr_hand" if side == "us" else "us_hand"
    if opponent_key in obs and float(np.asarray(obs[opponent_key]).sum()) > 1e-6:
        raise RuntimeError(f"information leak: {opponent_key} is non-empty for {side} actor")


@dataclass
class ExactOutcome:
    probability: float
    obs: dict[str, np.ndarray]
    snapshot: dict[str, Any]
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    chance_rolls: tuple[int, ...]
    current_in_env: bool = False


class ExactChanceEnumerator:
    """Enumerate auditable small chance transitions through the Saito bridge."""

    def __init__(self, *, max_outcomes: int = 4096) -> None:
        self.max_allowed_outcomes = int(max_outcomes)
        self.enumerated_nodes = 0
        self.deterministic_nodes = 0
        self.max_outcomes = 0
        self.unknown_nodes: list[dict[str, Any]] = []
        self.chance_cache: dict[tuple[str, int], list[ExactOutcome]] = {}
        self.chance_cache_hits = 0
        self.sampled_large_nodes = 0
        self.sample_index = 0

    @staticmethod
    def _cache_key(snapshot: dict[str, Any], action_index: int) -> tuple[str, int]:
        if snapshot.get("local_bridge"):
            bridge = snapshot.get("bridge") or {}
            namespace = bridge.get("checkpoint_namespace")
            token = bridge.get("checkpoint_token")
            if namespace and token:
                return f"runtime:{namespace}:{token}", int(action_index)
        payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.blake2b(payload.encode("ascii"), digest_size=16).hexdigest(), int(action_index)

    @staticmethod
    def _observation_digest(obs: dict[str, np.ndarray]) -> str:
        digest = hashlib.blake2b(digest_size=16)
        for key in sorted(obs):
            digest.update(key.encode("ascii"))
            digest.update(np.asarray(obs[key]).tobytes())
        return digest.hexdigest()

    @classmethod
    def _collapse_equivalent(cls, branches: list[ExactOutcome]) -> list[ExactOutcome]:
        collapsed: dict[tuple[str, bool, bool, str], ExactOutcome] = {}
        for branch in branches:
            key = (
                cls._observation_digest(branch.obs),
                branch.terminated,
                branch.truncated,
                str(branch.info.get("winner") or ""),
            )
            existing = collapsed.get(key)
            if existing is None:
                collapsed[key] = branch
            else:
                existing.probability += branch.probability
        return list(collapsed.values())

    @staticmethod
    def _run(env: SearchEnvironment, snapshot: dict[str, Any], action_index: int, rolls: list[int | float]) -> ExactOutcome:
        restore_search_state(env, snapshot)
        return ExactChanceEnumerator._run_loaded(env, action_index, rolls)

    @staticmethod
    def _run_loaded(env: SearchEnvironment, action_index: int, rolls: list[int | float]) -> ExactOutcome:
        obs, _reward, terminated, truncated, info = env.step_with_forced_rolls(action_index, rolls)
        return ExactOutcome(
            probability=1.0,
            obs=obs,
            snapshot=local_search_snapshot(env),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
            chance_rolls=tuple(int(value) for value in info.get("chance_rolls", [])),
        )

    def outcomes(
        self,
        env: SearchEnvironment,
        snapshot: dict[str, Any],
        action_index: int,
        *,
        state_loaded: bool = False,
    ) -> list[ExactOutcome]:
        # A conditional exact-chance expansion can create thousands of
        # temporary branch checkpoints. Keep its ancestor alive until the
        # planner clears the whole process-local checkpoint generation.
        pin_snapshot = getattr(env, "pin_local_snapshot", None)
        if snapshot.get("local_bridge") and callable(pin_snapshot):
            pin_snapshot(snapshot, True)
        cache_key = self._cache_key(snapshot, action_index)
        cached = self.chance_cache.get(cache_key)
        if cached is not None:
            self.chance_cache_hits += 1
            # A cached transition carries snapshots, not the live env position.
            result = [copy.copy(outcome) for outcome in cached]
            for outcome in result:
                outcome.current_in_env = False
            return result

        if not state_loaded:
            restore_search_state(env, snapshot)
        prompt = str((env.last_obs or {}).get("prompt") or "")
        action = env.legal_actions[action_index]
        value = str(action.get("value") or "").lower()

        # Olympic ties reroll indefinitely. Conditioning on the first non-tie
        # pair is the exact finite distribution over the resulting states.
        if "hosts the" in prompt.lower() and value == "participate":
            host = "us" if prompt.strip().lower().startswith("us hosts") else "ussr"
            grouped_pairs: dict[str, list[tuple[int, int]]] = {"us": [], "ussr": []}
            for us_roll in range(1, 7):
                for ussr_roll in range(1, 7):
                    us_total = us_roll + (2 if host == "us" else 0)
                    ussr_total = ussr_roll + (2 if host == "ussr" else 0)
                    if us_total != ussr_total:
                        grouped_pairs["us" if us_total > ussr_total else "ussr"].append((us_roll, ussr_roll))
            total_pairs = sum(len(pairs) for pairs in grouped_pairs.values())
            branches = []
            for group_index, pairs in enumerate(grouped_pairs.values()):
                if not pairs:
                    continue
                us_roll, ussr_roll = pairs[0]
                branch = (
                    self._run_loaded(env, action_index, [us_roll, ussr_roll])
                    if state_loaded and group_index == 0
                    else self._run(env, snapshot, action_index, [us_roll, ussr_roll])
                )
                branch.probability = len(pairs) / total_pairs
                branches.append(branch)
            for branch in branches:
                branch.current_in_env = False
            if branches:
                live_branch = branches.pop()
                live_branch.current_in_env = True
                branches.insert(0, live_branch)
            return self._record(env, cache_key, branches)

        probe_rolls = [0.071, 0.213, 0.357, 0.499, 0.643, 0.787, 0.929] * 8
        probe = (
            self._run_loaded(env, action_index, probe_rolls)
            if state_loaded
            else self._run(env, snapshot, action_index, probe_rolls)
        )
        consumed = list(probe.info.get("forced_rolls_consumed", []))
        if not consumed:
            probe.current_in_env = True
            self.deterministic_nodes += 1
            pin_snapshot = getattr(env, "pin_local_snapshot", None)
            if probe.snapshot.get("local_bridge") and callable(pin_snapshot):
                pin_snapshot(probe.snapshot, True)
            self.chance_cache[cache_key] = [probe]
            return [probe]
        sides = [int(value) for value in probe.info.get("chance_roll_sides", [])]
        if len(sides) != len(consumed) or any(side < 1 for side in sides):
            return self._unknown(prompt, action, consumed, sides, "chance-node arity is not auditable")
        outcome_count = math.prod(sides)
        if outcome_count > self.max_allowed_outcomes:
            return self._sample_large_node(env, snapshot, action_index, cache_key)
        branches: list[ExactOutcome] = []
        calls = 0

        class EnumerationLimit(Exception):
            pass

        def enumerate_prefix(prefix: list[int], probability: float) -> None:
            nonlocal calls
            calls += 1
            if calls > self.max_allowed_outcomes * 4:
                raise EnumerationLimit
            branch = self._run(env, snapshot, action_index, [*prefix, *probe_rolls])
            observed_sides = [int(item) for item in branch.info.get("chance_roll_sides", [])]
            if any(item < 1 for item in observed_sides):
                self._unknown(prompt, action, prefix, observed_sides, "conditional chance tree has invalid die")
            if len(observed_sides) <= len(prefix):
                branch.current_in_env = False
                branch.probability = probability
                branches.append(branch)
                if len(branches) > self.max_allowed_outcomes:
                    raise EnumerationLimit
                return
            if len(prefix) >= 16:
                self._unknown(prompt, action, prefix, observed_sides, "conditional chance depth exceeds bound")
            next_side = observed_sides[len(prefix)]
            for roll in range(1, next_side + 1):
                enumerate_prefix([*prefix, roll], probability / next_side)

        # The initial probe above establishes that this is a chance node. The
        # prefix tree then discovers conditional extra rolls one at a time and
        # exactly weights every finite leaf, including mixed 2-roll/3-roll paths.
        try:
            enumerate_prefix([], 1.0)
        except EnumerationLimit:
            return self._sample_large_node(env, snapshot, action_index, cache_key)
        total_probability = sum(branch.probability for branch in branches)
        if not math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            self._unknown(prompt, action, [], sides, f"chance probabilities sum to {total_probability}")
        if branches:
            live_branch = branches.pop()
            live_branch.current_in_env = True
            branches.insert(0, live_branch)
        return self._record(env, cache_key, branches)

    def _sample_large_node(
        self,
        env: SearchEnvironment,
        snapshot: dict[str, Any],
        action_index: int,
        cache_key: tuple[str, int],
    ) -> list[ExactOutcome]:
        self.sample_index += 1
        digest = hashlib.blake2b(
            f"{cache_key[0]}:{cache_key[1]}:{self.sample_index}".encode("ascii"),
            digest_size=64,
        ).digest()
        rolls = [((byte + 0.5) / 256.0) for byte in digest]
        branch = self._run(env, snapshot, action_index, rolls)
        branch.probability = 1.0
        branch.current_in_env = True
        self.sampled_large_nodes += 1
        return [branch]

    def _record(
        self,
        env: SearchEnvironment,
        cache_key: tuple[str, int],
        branches: list[ExactOutcome],
    ) -> list[ExactOutcome]:
        self.enumerated_nodes += 1
        self.max_outcomes = max(self.max_outcomes, len(branches))
        collapsed = self._collapse_equivalent(branches)
        pin_snapshot = getattr(env, "pin_local_snapshot", None)
        if callable(pin_snapshot):
            for outcome in collapsed:
                if outcome.snapshot.get("local_bridge"):
                    pin_snapshot(outcome.snapshot, True)
        self.chance_cache[cache_key] = collapsed
        return collapsed

    def _unknown(self, prompt: str, action: dict[str, Any], consumed: list[Any], sides: list[int], message: str):
        detail = {
            "prompt": prompt,
            "action": copy.deepcopy(action),
            "consumed": consumed,
            "chance_roll_sides": sides,
        }
        self.unknown_nodes.append(detail)
        raise RuntimeError(f"{message}: {detail}")


@dataclass
class ExactPuctNode:
    side: str
    raw_priors: np.ndarray
    priors: np.ndarray
    labels: tuple[str, ...]
    levels: tuple[str, ...]
    visits: np.ndarray
    value_sum_us: np.ndarray
    active: np.ndarray
    ordered_actions: np.ndarray
    initial_children: int
    max_children: int
    widening_c: float
    widening_alpha: float
    admission_q: np.ndarray
    admission_score: np.ndarray
    admitted_at_visit: np.ndarray
    root: bool = False

    def widen(self) -> tuple[int, ...]:
        before = self.active.copy()
        if self.max_children <= 0 or self.max_children >= len(self.priors):
            self.active[:] = True
        else:
            desired = max(
                self.initial_children,
                int(math.ceil(self.widening_c * (float(self.visits.sum()) + 1.0) ** self.widening_alpha)),
            )
            desired = min(self.max_children, len(self.priors), desired)
            self.active[self.ordered_actions[:desired]] = True
        added = np.flatnonzero(self.active & ~before)
        if len(added):
            self.admitted_at_visit[added] = float(self.visits.sum())
        return tuple(int(index) for index in added)

    def reset_admission(self, ordered_actions: np.ndarray, admission_score: np.ndarray) -> None:
        self.ordered_actions = np.asarray(ordered_actions, dtype=np.int64)
        self.admission_score = np.asarray(admission_score, dtype=np.float64)
        self.active[:] = False
        initial = len(self.priors) if self.max_children <= 0 else min(self.initial_children, self.max_children)
        initial = min(len(self.priors), max(1, initial))
        self.active[self.ordered_actions[:initial]] = True
        self.admitted_at_visit[:] = -1.0
        self.admitted_at_visit[self.active] = 0.0

    def select(self, cpuct: float, *, force_unvisited: bool = False) -> int:
        self.widen()
        if force_unvisited:
            unvisited = np.flatnonzero(self.active & (self.visits <= 1e-12))
            if len(unvisited):
                return int(unvisited[np.argmax(self.priors[unvisited])])
        q_us = np.divide(self.value_sum_us, self.visits, out=np.zeros_like(self.value_sum_us), where=self.visits > 0)
        perspective_q = q_us if self.side == "us" else -q_us
        exploration = cpuct * self.priors * math.sqrt(float(self.visits.sum()) + 1.0) / (1.0 + self.visits)
        return int(np.argmax(np.where(self.active, perspective_q + exploration, -np.inf)))


@dataclass(frozen=True)
class SelectivePuctConfig:
    initial_simulations: int = 32
    medium_simulations: int = 128
    max_simulations: int = 512
    cpuct: float = 1.5
    root_cpuct: float = 0.9
    root_prior_temperature: float = 2.0
    root_initial_children: int = 6
    root_max_children: int = 32
    root_progressive_widening_c: float = 1.5
    root_progressive_widening_alpha: float = 0.5
    root_q_probe_weight: float = 0.5
    root_q_prior_mix: float = 0.25
    root_q_prior_temperature: float = 0.02
    root_q_probe_particles: int = 1
    root_q_probe_max_actions: int = 96
    root_q_probe_min_branching: int = 6
    root_q_probe_levels: tuple[str, ...] = (
        "influence_allocation",
        "realignment_target",
        "coup_target",
    )
    root_prior_protected_children: int = 2
    root_selection_min_visits: int = 3
    # A relative-to-leader visit floor is non-monotonic: as the leader gains
    # visits, an already evaluated high-Q alternative can become ineligible.
    # Keep the hook for explicit ablations, but production selection uses the
    # fixed minimum-visit reliability floor.
    root_selection_visit_fraction: float = 0.0
    root_selection_min_q_advantage: float = 0.002
    belief_particles: int = 32
    meaningful_depth: int = 2
    forced_response_depth: int = 1
    max_atomic_steps: int = 96
    max_expanded_nodes: int = 512
    # Operational fail-safe for branch-local simulator snapshots. A positive
    # limit aborts the current tree fail-closed before one pathological root
    # can exhaust host memory. Cutoff trees are never emitted as teachers.
    max_process_rss_mib: float = 0.0
    # A positive wall-clock limit provides the same fail-closed behavior for
    # pathological roots whose exact branches are computationally expensive
    # without consuming enough memory to trip the RSS guard.
    max_search_seconds: float = 0.0
    child_initial: int = 4
    child_max: int = 24
    progressive_widening_c: float = 2.0
    progressive_widening_alpha: float = 0.5
    temperature: float = 0.0
    behavior_temperature: float = 0.0
    low_confidence_threshold: float = 0.55
    policy_gap_threshold: float = 0.12
    low_confidence_sample_rate: float = 0.05
    allocation_min_remaining: int = 3
    search_setup_allocations: bool = False
    max_scoring_searches_per_side_turn: int = 1
    max_chance_searches_per_side_round: int = 1
    max_allocation_searches_per_side_round: int = 1
    min_q_advantage_to_flip: float = 0.0
    medium_visit_share_threshold: float = 0.62
    medium_q_gap_threshold: float = 0.05
    max_visit_share_threshold: float = 0.55
    max_q_gap_threshold: float = 0.03
    escalate_on_disagreement: bool = True
    enabled_triggers: tuple[str, ...] = (
        "forced_response",
        "opponent_event",
        "influence_allocation",
        "scoring_card",
        "scoring_pressure",
        "chance_heavy",
        "low_confidence",
    )
    max_exact_chance_outcomes: int = 36
    fail_closed_min_depths: tuple[tuple[str, int], ...] = (("scoring_pressure", 8),)
    fail_closed_terminal_triggers: tuple[str, ...] = ("scoring_pressure",)


@dataclass(frozen=True)
class ConvergedAdaptivePuctConfig:
    min_rank_correlation: float = 0.70
    meaningful_q_gap: float = 0.01
    escalation_q_gap: float = 0.03
    always_compare: bool = False
    escalate_on_flip: bool = True
    escalation_triggers: tuple[str, ...] = (
        "scoring_card",
        "scoring_pressure",
        "forced_response",
        "opponent_event",
        "chance_heavy",
    )


@dataclass
class HierarchicalSearchTarget:
    observation: dict[str, np.ndarray]
    legal_actions: tuple[dict[str, Any], ...]
    side: str
    level: str
    path: tuple[dict[str, str], ...]
    raw_policy: np.ndarray
    visits: np.ndarray
    q_values: np.ndarray
    search_policy: np.ndarray
    raw_action: int
    search_action: int


@dataclass
class SelectiveSearchDecision:
    action: int
    raw_action: int
    raw_policy: np.ndarray
    search_policy: np.ndarray
    visits: np.ndarray
    q_root: np.ndarray
    searched: bool
    triggers: tuple[str, ...]
    simulations: int
    budgets: dict[int, list[dict[str, Any]]]
    root_q_gap: float
    action_flipped: bool
    latency_seconds: float
    expanded_nodes: int
    exact_chance_nodes: int
    belief_particles: int
    meaningful_depth: int
    hierarchical_path: tuple[dict[str, str], ...]
    q_advantage_over_raw: float = 0.0
    sampled_large_chance_nodes: int = 0
    q_gate_rejected: bool = False
    atomic_horizon_cutoffs: int = 0
    node_budget_cutoffs: int = 0
    max_meaningful_depth_reached: int = 0
    max_turn_delta: int = 0
    cross_turn_leaf_evaluations: int = 0
    terminal_evaluations: int = 0
    evidence_gate_rejected: bool = False
    hierarchical_targets: tuple[HierarchicalSearchTarget, ...] = ()
    teacher_action: int | None = None
    behavior_path: tuple[dict[str, str], ...] = ()
    root_selection: dict[str, Any] = field(default_factory=dict)
    root_admission_events: tuple[dict[str, Any], ...] = ()
    root_probe_exact_chance_nodes: int = 0
    root_probe_sampled_large_chance_nodes: int = 0
    convergence_status: str | None = None
    convergence_configs: tuple[str, ...] = ()
    convergence_comparisons: tuple[dict[str, Any], ...] = ()
    adaptive_fallback: bool = False


@dataclass
class PlannerTelemetry:
    decisions: int = 0
    searched_decisions: int = 0
    simulations: int = 0
    latency_seconds: float = 0.0
    disagreements: int = 0
    action_flips: int = 0
    q_gap_sum: float = 0.0
    q_advantage_sum: float = 0.0
    expanded_nodes: int = 0
    exact_chance_nodes: int = 0
    sampled_large_chance_nodes: int = 0
    q_gate_rejections: int = 0
    atomic_horizon_cutoffs: int = 0
    node_budget_cutoffs: int = 0
    max_meaningful_depth_reached: int = 0
    max_turn_delta: int = 0
    cross_turn_leaf_evaluations: int = 0
    terminal_evaluations: int = 0
    evidence_gate_rejections: int = 0
    trigger_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, decision: SelectiveSearchDecision) -> None:
        self.decisions += 1
        if decision.searched:
            self.searched_decisions += 1
            self.simulations += decision.simulations
            self.latency_seconds += decision.latency_seconds
            self.q_gap_sum += decision.root_q_gap
            self.q_advantage_sum += decision.q_advantage_over_raw
            self.expanded_nodes += decision.expanded_nodes
            self.exact_chance_nodes += decision.exact_chance_nodes
            self.sampled_large_chance_nodes += decision.sampled_large_chance_nodes
            self.q_gate_rejections += int(decision.q_gate_rejected)
            self.atomic_horizon_cutoffs += decision.atomic_horizon_cutoffs
            self.node_budget_cutoffs += decision.node_budget_cutoffs
            self.max_meaningful_depth_reached = max(
                self.max_meaningful_depth_reached,
                decision.max_meaningful_depth_reached,
            )
            self.max_turn_delta = max(self.max_turn_delta, decision.max_turn_delta)
            self.cross_turn_leaf_evaluations += decision.cross_turn_leaf_evaluations
            self.terminal_evaluations += decision.terminal_evaluations
            self.evidence_gate_rejections += int(decision.evidence_gate_rejected)
            self.disagreements += int(decision.raw_action != decision.action)
            self.action_flips += int(decision.action_flipped)
            for trigger in decision.triggers:
                self.trigger_counts[trigger] += 1

    def metrics(self) -> dict[str, float]:
        searched = max(1, self.searched_decisions)
        decisions = max(1, self.decisions)
        return {
            "search/fraction_decisions_searched": self.searched_decisions / decisions,
            "search/average_simulations_per_real_action": self.simulations / decisions,
            "search/average_latency_seconds": self.latency_seconds / searched,
            "search/average_latency_seconds_per_real_action": self.latency_seconds / decisions,
            "search/raw_search_disagreement_rate": self.disagreements / searched,
            "search/action_flip_rate": self.action_flips / searched,
            "search/average_root_q_gap": self.q_gap_sum / searched,
            "search/average_q_advantage_over_raw": self.q_advantage_sum / searched,
            "search/average_expanded_nodes": self.expanded_nodes / searched,
            "search/average_exact_chance_nodes": self.exact_chance_nodes / searched,
            "search/average_sampled_large_chance_nodes": self.sampled_large_chance_nodes / searched,
            "search/q_gate_rejection_rate": self.q_gate_rejections / searched,
            "search/average_atomic_horizon_cutoffs": self.atomic_horizon_cutoffs / searched,
            "search/average_node_budget_cutoffs": self.node_budget_cutoffs / searched,
            "search/max_meaningful_depth_reached": float(self.max_meaningful_depth_reached),
            "search/max_turn_delta": float(self.max_turn_delta),
            "search/average_cross_turn_leaf_evaluations": self.cross_turn_leaf_evaluations / searched,
            "search/average_terminal_evaluations": self.terminal_evaluations / searched,
            "search/evidence_gate_rejection_rate": self.evidence_gate_rejections / searched,
            **{f"search/trigger_{key}": float(value) for key, value in self.trigger_counts.items()},
        }


class ExactBeliefPuctSession:
    """Incremental exact-chance PUCT session over information-set nodes."""

    def __init__(
        self,
        evaluator: Any,
        config: SelectivePuctConfig,
        *,
        seed: int,
    ) -> None:
        self.evaluator = evaluator
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.nodes: dict[str, ExactPuctNode] = {}
        self.stepper = ExactChanceEnumerator(max_outcomes=config.max_exact_chance_outcomes)
        self.root_probe_stepper = ExactChanceEnumerator(max_outcomes=config.max_exact_chance_outcomes)
        self.trace_stats: dict[tuple[tuple[str, str, str], ...], list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.completed = 0
        self.root_node: ExactPuctNode | None = None
        self.root_player = "ussr"
        self.root_snapshot: dict[str, Any] | None = None
        self.root_obs: dict[str, np.ndarray] | None = None
        self.belief_seeds = [self.seed + 104729 * index for index in range(config.belief_particles)]
        self.boundary_evaluations = 0
        self.terminal_evaluations = 0
        self.atomic_horizon_cutoffs = 0
        self.node_budget_cutoffs = 0
        self.root_turn = 0
        self.max_meaningful_depth_reached = 0
        self.max_turn_delta = 0
        self.cross_turn_leaf_evaluations = 0
        self.node_observations: dict[str, dict[str, np.ndarray]] = {}
        self.node_legal_actions: dict[str, tuple[dict[str, Any], ...]] = {}
        self.trace_node_keys: dict[tuple[tuple[str, str, str], ...], set[str]] = defaultdict(set)
        self.root_admission_events: list[dict[str, Any]] = []
        self.root_selection: dict[str, Any] = {}
        self.resource_limit_exceeded = False
        self.resource_limit_reason = ""
        self._resource_check_counter = 0
        self._search_started_at = time.perf_counter()

    @staticmethod
    def _resident_memory_mib() -> float:
        try:
            with open("/proc/self/statm", encoding="ascii") as handle:
                resident_pages = int(handle.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE") / 2**20
        except (OSError, ValueError, IndexError):
            return -1.0

    def _resource_limit_reached(self, *, force: bool = False) -> bool:
        self._resource_check_counter += 1
        if not force and self._resource_check_counter % 64:
            return self.resource_limit_exceeded
        time_limit = float(self.config.max_search_seconds)
        if time_limit > 0.0 and time.perf_counter() - self._search_started_at >= time_limit:
            if not self.resource_limit_exceeded:
                self.node_budget_cutoffs += 1
            self.resource_limit_exceeded = True
            self.resource_limit_reason = "wall_clock"
            return True
        memory_limit = float(self.config.max_process_rss_mib)
        resident = self._resident_memory_mib() if memory_limit > 0.0 else -1.0
        if resident >= memory_limit > 0.0:
            if not self.resource_limit_exceeded:
                self.node_budget_cutoffs += 1
            self.resource_limit_exceeded = True
            self.resource_limit_reason = "rss"
        return self.resource_limit_exceeded

    def release(self) -> None:
        """Drop branch-local search state after result tensors are extracted."""
        self.nodes.clear()
        self.stepper.chance_cache.clear()
        self.root_probe_stepper.chance_cache.clear()
        self.stepper.unknown_nodes.clear()
        self.root_probe_stepper.unknown_nodes.clear()
        self.trace_stats.clear()
        self.node_observations.clear()
        self.node_legal_actions.clear()
        self.trace_node_keys.clear()
        self.root_admission_events.clear()
        self.root_node = None
        self.root_snapshot = None
        self.root_obs = None

    def begin(self, env: SearchEnvironment, root_snapshot: dict[str, Any], root_obs: dict[str, np.ndarray]) -> None:
        self._search_started_at = time.perf_counter()
        env.restore(root_snapshot)
        self.root_player = root_side(env.last_obs)
        self.root_turn = int((env.last_obs or {}).get("turn") or 0)
        validate_information_safe_observation(root_obs, self.root_player)
        self.root_snapshot = local_search_snapshot(env)
        pin_snapshot = getattr(env, "pin_local_snapshot", None)
        if self.root_snapshot.get("local_bridge") and callable(pin_snapshot):
            pin_snapshot(self.root_snapshot, True)
        self.root_obs = root_obs
        self.root_node = self._node(env, root_obs, root=True)
        self._configure_root_admission(env)

    def _root_probe_enabled(self, env: SearchEnvironment) -> bool:
        assert self.root_node is not None
        levels = set(self.root_node.levels)
        return bool(
            self.config.root_q_probe_weight > 0.0
            and len(self.root_node.priors) >= self.config.root_q_probe_min_branching
            and len(self.root_node.priors) <= self.config.root_q_probe_max_actions
            and levels & set(self.config.root_q_probe_levels)
        )

    def _probe_root_q(self, env: SearchEnvironment) -> np.ndarray:
        if self.root_node is None or self.root_snapshot is None:
            raise RuntimeError("root probe requires an initialized session")
        legal_count = len(self.root_node.priors)
        values = np.zeros(legal_count, dtype=np.float64)
        particles = max(1, min(self.config.root_q_probe_particles, len(self.belief_seeds)))
        for particle_index in range(particles):
            restore_search_state(env, self.root_snapshot)
            obs = env.redeterminize(self.root_player, self.belief_seeds[particle_index])
            validate_information_safe_observation(obs, self.root_player)
            particle_snapshot = local_search_snapshot(env)
            for action_index in range(legal_count):
                if self._resource_limit_reached(force=True):
                    break
                outcomes = self.root_probe_stepper.outcomes(env, particle_snapshot, action_index)
                expected_us = 0.0
                for outcome in outcomes:
                    if self._resource_limit_reached(force=True):
                        break
                    if outcome.terminated or outcome.truncated:
                        value_us = self._terminal_value_us(outcome.info)
                    else:
                        restore_search_state(env, outcome.snapshot)
                        outcome_side = root_side(env.last_obs)
                        validate_information_safe_observation(outcome.obs, outcome_side)
                        value_us = float(self.evaluator.value_us(outcome.obs, outcome_side)[1])
                    expected_us += float(outcome.probability) * value_us
                if self.resource_limit_exceeded:
                    break
                values[action_index] += expected_us / particles
            if self.resource_limit_exceeded:
                break
        restore_search_state(env, self.root_snapshot)
        return values if self.root_player == "us" else -values

    def _configure_root_admission(self, env: SearchEnvironment) -> None:
        assert self.root_node is not None
        node = self.root_node
        prior_scores = rank_percentiles(node.raw_priors)
        admission_q = np.full(len(node.priors), np.nan, dtype=np.float64)
        admission_scores = prior_scores.copy()
        if self._root_probe_enabled(env):
            admission_q = self._probe_root_q(env)
            q_scores = rank_percentiles(admission_q)
            weight = min(1.0, max(0.0, float(self.config.root_q_probe_weight)))
            admission_scores = (1.0 - weight) * prior_scores + weight * q_scores
            if self.config.root_q_prior_mix > 0.0:
                temperature = max(float(self.config.root_q_prior_temperature), 1e-6)
                shifted = (admission_q - float(np.max(admission_q))) / temperature
                q_priors = np.exp(np.clip(shifted, -60.0, 0.0))
                q_priors /= max(float(q_priors.sum()), 1e-30)
                mix = min(1.0, max(0.0, float(self.config.root_q_prior_mix)))
                node.priors = normalize_priors((1.0 - mix) * node.priors + mix * q_priors, len(node.priors))
        node.admission_q = admission_q
        admission_order = sorted(
            range(len(node.priors)),
            key=lambda index: (
                -float(admission_scores[index]),
                -float(node.priors[index]),
                node.labels[index],
            ),
        )
        protected = sorted(
            range(len(node.raw_priors)),
            key=lambda index: (-float(node.raw_priors[index]), node.labels[index]),
        )[: max(0, int(self.config.root_prior_protected_children))]
        protected_set = set(protected)
        ordered = np.asarray(
            [*protected, *(index for index in admission_order if index not in protected_set)],
            dtype=np.int64,
        )
        node.reset_admission(ordered, admission_scores)
        rank_by_action = {
            int(index): rank
            for rank, index in enumerate(node.ordered_actions, 1)
        }
        self.root_admission_events = [
            {
                "action_index": index,
                "action": node.labels[index],
                "admission_rank": rank_by_action[index],
                "raw_P": float(node.raw_priors[index]),
                "search_P": float(node.priors[index]),
                "cheap_Q": (
                    None
                    if not math.isfinite(float(node.admission_q[index]))
                    else float(node.admission_q[index])
                ),
                "admission_score": float(node.admission_score[index]),
                "admitted_at_root_visit": (
                    0.0 if bool(node.active[index]) else None
                ),
            }
            for index in range(len(node.priors))
        ]

    def _node(self, env: SearchEnvironment, obs: dict[str, np.ndarray], *, root: bool) -> ExactPuctNode:
        side = root_side(env.last_obs)
        validate_information_safe_observation(obs, side)
        key = information_state_key(obs, side)
        existing = self.nodes.get(key)
        if existing is not None:
            return existing
        legal_count = len(env.legal_actions)
        raw = self.evaluator.raw(obs, legal_count)
        raw_priors = normalize_priors(raw["probabilities"], legal_count)
        priors = (
            temperature_scale_priors(raw_priors, self.config.root_prior_temperature)
            if root
            else raw_priors.copy()
        )
        ordered = np.asarray(
            sorted(range(legal_count), key=lambda index: (-float(priors[index]), action_text(env.legal_actions[index]))),
            dtype=np.int64,
        )
        max_children = self.config.root_max_children if root else self.config.child_max
        initial_children = self.config.root_initial_children if root else self.config.child_initial
        widening_c = self.config.root_progressive_widening_c if root else self.config.progressive_widening_c
        widening_alpha = (
            self.config.root_progressive_widening_alpha if root else self.config.progressive_widening_alpha
        )
        active = np.ones(legal_count, dtype=np.bool_) if max_children <= 0 or max_children >= legal_count else np.zeros(legal_count, dtype=np.bool_)
        if max_children > 0 and max_children < legal_count:
            active[ordered[: min(initial_children, max_children, legal_count)]] = True
        prompt = str((env.last_obs or {}).get("prompt") or "")
        node = ExactPuctNode(
            side=side,
            raw_priors=raw_priors,
            priors=priors,
            labels=tuple(action_text(action) for action in env.legal_actions),
            levels=tuple(action_level(action, prompt) for action in env.legal_actions),
            visits=np.zeros(legal_count, dtype=np.float64),
            value_sum_us=np.zeros(legal_count, dtype=np.float64),
            active=active,
            ordered_actions=ordered,
            initial_children=initial_children,
            max_children=max_children,
            widening_c=widening_c,
            widening_alpha=widening_alpha,
            admission_q=np.full(legal_count, np.nan, dtype=np.float64),
            admission_score=rank_percentiles(priors),
            admitted_at_visit=np.where(active, 0.0, -1.0),
            root=root,
        )
        self.nodes[key] = node
        self.node_observations[key] = {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in obs.items()
        }
        self.node_legal_actions[key] = tuple(copy.deepcopy(env.legal_actions))
        return node

    @staticmethod
    def _terminal_value_us(info: dict[str, Any]) -> float:
        winner = str(info.get("winner") or "")
        return 1.0 if winner == "us" else -1.0 if winner == "ussr" else 0.0

    def _leaf_value(self, env: SearchEnvironment, obs: dict[str, np.ndarray]) -> float:
        self.boundary_evaluations += 1
        return float(self.evaluator.value_us(obs, root_side(env.last_obs))[1])

    def _simulate(
        self,
        env: SearchEnvironment,
        snapshot: dict[str, Any],
        obs: dict[str, np.ndarray],
        *,
        decisions_used: int,
        atomic_steps: int,
        update_weight: float,
        trace: tuple[tuple[str, str, str], ...],
        at_root: bool,
        active_macro_identity: str | None,
        state_loaded: bool = False,
    ) -> float:
        if not state_loaded:
            restore_search_state(env, snapshot)
        if self._resource_limit_reached():
            value = self._leaf_value(env, obs)
            stat = self.trace_stats[trace]
            stat[0] += update_weight
            stat[1] += update_weight * value
            return value
        current_turn = int((env.last_obs or {}).get("turn") or self.root_turn)
        self.max_turn_delta = max(self.max_turn_delta, max(0, current_turn - self.root_turn))
        boundary = meaningful_decision_boundary(env.last_obs, env.legal_actions)
        new_boundary = bool(
            boundary
            and (
                not boundary.coalesce
                or boundary.identity != active_macro_identity
            )
        )
        if atomic_steps > 0 and new_boundary and decisions_used >= self.config.meaningful_depth:
            value = self._leaf_value(env, obs)
            if current_turn > self.root_turn:
                self.cross_turn_leaf_evaluations += 1
            stat = self.trace_stats[trace]
            stat[0] += update_weight
            stat[1] += update_weight * value
            return value
        if atomic_steps >= self.config.max_atomic_steps:
            # Some events contain long deterministic choice chains before the
            # next macro decision. This is a normal finite search horizon, not
            # an environment failure; expose every cutoff in telemetry.
            self.atomic_horizon_cutoffs += 1
            value = self._leaf_value(env, obs)
            stat = self.trace_stats[trace]
            stat[0] += update_weight
            stat[1] += update_weight * value
            return value

        node_key = information_state_key(obs, root_side(env.last_obs))
        if (
            self.config.max_expanded_nodes > 0
            and node_key not in self.nodes
            and len(self.nodes) >= self.config.max_expanded_nodes
        ):
            self.node_budget_cutoffs += 1
            value = self._leaf_value(env, obs)
            stat = self.trace_stats[trace]
            stat[0] += update_weight
            stat[1] += update_weight * value
            return value

        node = self._node(env, obs, root=at_root)
        self.trace_node_keys[trace].add(node_key)
        cpuct = self.config.root_cpuct if at_root else self.config.cpuct
        previously_active = node.active.copy()
        action_index = node.select(cpuct, force_unvisited=at_root)
        if at_root:
            newly_active = np.flatnonzero(node.active & ~previously_active)
            for index in newly_active:
                self.root_admission_events[int(index)]["admitted_at_root_visit"] = float(
                    node.admitted_at_visit[int(index)]
                )
        step = (node.side, node.labels[action_index], node.levels[action_index])
        next_decisions = decisions_used + int(new_boundary)
        self.max_meaningful_depth_reached = max(self.max_meaningful_depth_reached, next_decisions)
        next_macro_identity = boundary.identity if boundary and boundary.coalesce else None
        outcomes = self.stepper.outcomes(env, snapshot, action_index, state_loaded=True)
        expected = 0.0
        for outcome in outcomes:
            next_trace = trace + (step,)
            if outcome.terminated or outcome.truncated:
                value = self._terminal_value_us(outcome.info)
                self.terminal_evaluations += 1
                stat = self.trace_stats[next_trace]
                stat[0] += update_weight * outcome.probability
                stat[1] += update_weight * outcome.probability * value
            else:
                value = self._simulate(
                    env,
                    outcome.snapshot,
                    outcome.obs,
                    decisions_used=next_decisions,
                    atomic_steps=atomic_steps + 1,
                    update_weight=update_weight * outcome.probability,
                    trace=next_trace,
                    at_root=False,
                    active_macro_identity=next_macro_identity,
                    state_loaded=outcome.current_in_env,
                )
            expected += outcome.probability * value
        node.visits[action_index] += update_weight
        node.value_sum_us[action_index] += update_weight * expected
        return expected

    def advance_to(self, env: SearchEnvironment, budget: int) -> None:
        if self.root_node is None or self.root_snapshot is None or self.root_obs is None:
            raise RuntimeError("begin() must be called before advance_to()")
        while self.completed < int(budget):
            if self._resource_limit_reached(force=True):
                break
            restore_search_state(env, self.root_snapshot)
            particle_seed = self.belief_seeds[self.completed % len(self.belief_seeds)]
            obs = env.redeterminize(self.root_player, particle_seed)
            validate_information_safe_observation(obs, self.root_player)
            particle_snapshot = local_search_snapshot(env)
            self._simulate(
                env,
                particle_snapshot,
                obs,
                decisions_used=0,
                atomic_steps=0,
                update_weight=1.0,
                trace=(),
                at_root=True,
                active_macro_identity=None,
                state_loaded=True,
            )
            self.completed += 1

    def root_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.root_node is None:
            raise RuntimeError("search has not begun")
        q_us = np.divide(
            self.root_node.value_sum_us,
            self.root_node.visits,
            out=np.zeros_like(self.root_node.value_sum_us),
            where=self.root_node.visits > 0,
        )
        q_root = q_us if self.root_player == "us" else -q_us
        return self.root_node.visits.copy(), q_root

    def select_root_action(self) -> tuple[int, dict[str, Any]]:
        visits, q_root = self.root_arrays()
        assert self.root_node is not None
        selected, detail = q_aware_root_action(
            visits,
            q_root,
            self.root_node.priors,
            self.root_node.labels,
            min_visits=self.config.root_selection_min_visits,
            visit_fraction=self.config.root_selection_visit_fraction,
            min_q_advantage=self.config.root_selection_min_q_advantage,
        )
        detail = {
            **detail,
            "selected_action": self.root_node.labels[selected],
            "visit_leader_action": self.root_node.labels[int(detail["visit_leader"])],
            "q_candidate_action": self.root_node.labels[int(detail["q_candidate"])],
        }
        self.root_selection = detail
        return selected, detail

    def root_table(self) -> list[dict[str, Any]]:
        visits, q_root = self.root_arrays()
        assert self.root_node is not None
        rows = [
            {
                "action_index": index,
                "action": self.root_node.labels[index],
                "level": self.root_node.levels[index],
                "N": float(visits[index]),
                "Q": float(q_root[index]),
                "P": float(self.root_node.priors[index]),
                "raw_P": float(self.root_node.raw_priors[index]),
                "cheap_Q": (
                    None
                    if not math.isfinite(float(self.root_node.admission_q[index]))
                    else float(self.root_node.admission_q[index])
                ),
                "admission_score": float(self.root_node.admission_score[index]),
                "admission_rank": int(np.where(self.root_node.ordered_actions == index)[0][0]) + 1,
                "active": bool(self.root_node.active[index]),
                "admitted_at_root_visit": (
                    None
                    if self.root_node.admitted_at_visit[index] < 0
                    else float(self.root_node.admitted_at_visit[index])
                ),
            }
            for index in range(len(visits))
        ]
        return sorted(rows, key=lambda row: (-row["N"], -row["Q"], -row["P"]))

    def best_trace(self, action_index: int) -> tuple[dict[str, str], ...]:
        assert self.root_node is not None
        root_label = self.root_node.labels[action_index]
        candidates = [
            (trace, stats)
            for trace, stats in self.trace_stats.items()
            if trace and trace[0][1] == root_label
        ]
        if not candidates:
            return ({"side": self.root_node.side, "action": root_label, "level": self.root_node.levels[action_index]},)
        trace, _stats = max(candidates, key=lambda item: (item[1][0], item[1][1]))
        return tuple({"side": side, "action": label, "level": level} for side, label, level in trace)

    def hierarchical_targets(
        self,
        trace: tuple[dict[str, str], ...],
        *,
        temperature: float,
    ) -> tuple[HierarchicalSearchTarget, ...]:
        targets: list[HierarchicalSearchTarget] = []
        trainable_levels = {
            "card",
            "mode",
            "ops_action_type",
            "influence_allocation",
            "coup_target",
            "realignment_target",
            "forced_response",
        }
        trace_tuples = tuple(
            (str(entry["side"]), str(entry["action"]), str(entry["level"]))
            for entry in trace
        )
        for depth in range(1, len(trace_tuples)):
            prefix = trace_tuples[:depth]
            selected_side, selected_label, selected_level = trace_tuples[depth]
            if selected_level not in trainable_levels:
                continue
            for node_key in sorted(self.trace_node_keys.get(prefix, ())):
                node = self.nodes[node_key]
                if len(node.labels) <= 1 or node.side != selected_side or selected_label not in node.labels:
                    continue
                visits = node.visits.copy()
                if float(visits.sum()) <= 0.0:
                    continue
                q_us = np.divide(
                    node.value_sum_us,
                    visits,
                    out=np.zeros_like(node.value_sum_us),
                    where=visits > 0,
                )
                q_values = q_us if node.side == "us" else -q_us
                search_action, _selection = q_aware_root_action(
                    visits,
                    q_values,
                    node.priors,
                    node.labels,
                    min_visits=self.config.root_selection_min_visits,
                    visit_fraction=self.config.root_selection_visit_fraction,
                    min_q_advantage=self.config.root_selection_min_q_advantage,
                )
                if temperature <= 1e-6:
                    search_policy = np.zeros_like(visits, dtype=np.float64)
                    search_policy[search_action] = 1.0
                else:
                    scaled = np.power(visits, 1.0 / temperature)
                    search_policy = scaled / max(float(scaled.sum()), 1e-12)
                targets.append(
                    HierarchicalSearchTarget(
                        observation=self.node_observations[node_key],
                        legal_actions=self.node_legal_actions[node_key],
                        side=node.side,
                        level=selected_level,
                        path=tuple(
                            {"side": side, "action": label, "level": level}
                            for side, label, level in prefix
                        ),
                        raw_policy=node.raw_priors.copy(),
                        visits=visits,
                        q_values=q_values.copy(),
                        search_policy=search_policy,
                        raw_action=int(np.argmax(node.raw_priors)),
                        search_action=search_action,
                    )
                )
        return tuple(targets)


def classify_search_triggers(
    env: SearchEnvironment,
    raw_policy: np.ndarray,
    config: SelectivePuctConfig,
) -> tuple[str, ...]:
    last_obs = env.last_obs or {}
    prompt = str(last_obs.get("prompt") or "").lower()
    phase = str(last_obs.get("phase") or "").lower()
    actions = env.legal_actions
    values = {str(action.get("value") or "").lower() for action in actions}
    decisions = {str(action.get("decision") or "").lower() for action in actions}
    raw_action = int(np.argmax(raw_policy)) if len(raw_policy) else 0
    raw_value = str(actions[raw_action].get("value") or "").lower() if actions else ""
    raw_level = action_level(actions[raw_action], prompt) if actions else "other"
    if meaningful_decision_kind(last_obs, actions) in {"card_choice", "headline"}:
        raw_level = "card"
    triggers: set[str] = set()

    if is_forced_response_decision(last_obs, actions):
        triggers.add("forced_response")
    if raw_level in {"card", "mode"} and (
        values & {"before_ops", "after_ops"} or "opponent card" in prompt
    ):
        triggers.add("opponent_event")
    if (
        decisions & {"country_click", "country_mouseup"}
        and ("place" in prompt or "influence" in prompt)
        and (config.search_setup_allocations or not phase.startswith("setup_"))
    ):
        remaining_match = re.search(r"(?:place|add)\s+(\d+)\s+(?:influence|ussr|us)", prompt)
        if remaining_match and int(remaining_match.group(1)) >= config.allocation_min_remaining:
            triggers.add("influence_allocation")
    hand = set(last_obs.get("hand") or [])
    cards_by_id = getattr(env, "cards_by_id", {})
    side = root_side(last_obs)
    card_resolver = getattr(env, "action_card_id", None)
    resolved_card_id = card_resolver(last_obs, actions[raw_action]) if actions and callable(card_resolver) else None
    raw_card_id = str(resolved_card_id or actions[raw_action].get("card") or actions[raw_action].get("value") or "") if actions else ""
    raw_card_side = str(
        cards_by_id.get(raw_card_id, {}).get("player")
        or cards_by_id.get(raw_card_id, {}).get("side")
        or ""
    ).lower()
    if (
        raw_level in {"card", "mode"}
        and raw_card_id in cards_by_id
        and raw_card_side not in {"", side, "both", "neutral"}
    ):
        triggers.add("opponent_event")
    card_selection = "pick a card" in prompt or "select a card" in prompt or "headline" in prompt
    scoring_cards_in_hand = sum(
        bool(cards_by_id.get(card_id, {}).get("scoring")) for card_id in hand
    )
    if "scoring" in prompt or (
        card_selection
        and "headline" not in prompt
        and scoring_cards_in_hand
    ):
        triggers.add("scoring_card")
    if card_selection and "headline" not in prompt and scoring_cards_in_hand:
        turn = int(last_obs.get("turn") or 1)
        action_round = int(last_obs.get("action_round") or 0)
        normal_rounds = 6 if turn <= 3 else 7
        # Saito exposes AR1 as action_round=0, AR2 as 1, and so on.
        remaining_rounds = max(1, normal_rounds - max(0, action_round))
        if scoring_cards_in_hand >= remaining_rounds:
            triggers.add("scoring_pressure")
    chance_pattern = r"\b(?:coup|realign(?:ment)?|war|space|olympic|dice|rolls?)\b"
    if re.search(chance_pattern, prompt) or raw_value in {"coup", "realign", "participate"}:
        triggers.add("chance_heavy")
    ordered = np.sort(np.asarray(raw_policy, dtype=np.float64))[::-1]
    top = float(ordered[0]) if len(ordered) else 0.0
    gap = top - float(ordered[1]) if len(ordered) > 1 else 1.0
    if top < config.low_confidence_threshold or gap < config.policy_gap_threshold:
        payload = "|".join(
            [side, phase, prompt, *(str(action.get("value") or "") for action in actions)]
        ).encode("utf-8")
        bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / float(2**64)
        if bucket < config.low_confidence_sample_rate:
            triggers.add("low_confidence")
    if phase.startswith("headline_"):
        triggers.add("opponent_event")
    return tuple(sorted(triggers & set(config.enabled_triggers)))


def search_evidence_gate_rejected(
    triggers: tuple[str, ...],
    config: SelectivePuctConfig,
    *,
    meaningful_depth: int,
    terminal_evaluations: int,
) -> bool:
    trigger_set = set(triggers)
    if any(
        trigger in trigger_set and int(meaningful_depth) < int(minimum_depth)
        for trigger, minimum_depth in config.fail_closed_min_depths
    ):
        return True
    return bool(
        trigger_set.intersection(config.fail_closed_terminal_triggers)
        and int(terminal_evaluations) <= 0
    )


class SelectivePuctPlanner:
    """Adaptive inference wrapper around exact-chance information-set PUCT."""

    def __init__(
        self,
        evaluator: Any,
        search_env: SearchEnvironment,
        config: SelectivePuctConfig | None = None,
        *,
        seed: int = 1,
    ) -> None:
        self.evaluator = evaluator
        self.search_env = search_env
        self.config = config or SelectivePuctConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.search_index = 0
        self.telemetry = PlannerTelemetry()
        self.pending_continuation: list[dict[str, str]] = []
        self.pending_continuation_seed: int | None = None
        self.scoring_search_counts: dict[tuple[int | None, str, int, int], int] = defaultdict(int)
        self.chance_search_counts: dict[tuple[int | None, str, int, int, str], int] = defaultdict(int)
        self.allocation_search_counts: dict[tuple[int | None, str, int, int], int] = defaultdict(int)

    def _clear_runtime_checkpoints(self) -> None:
        clear_checkpoints = getattr(self.search_env, "clear_runtime_checkpoints", None)
        if callable(clear_checkpoints):
            clear_checkpoints()

    @staticmethod
    def _visit_policy(
        visits: np.ndarray,
        priors: np.ndarray,
        temperature: float,
        *,
        preferred_action: int | None = None,
    ) -> np.ndarray:
        if float(visits.sum()) <= 0.0:
            return normalize_priors(priors, len(priors))
        if temperature <= 1e-6:
            result = np.zeros_like(visits, dtype=np.float64)
            selected = int(preferred_action) if preferred_action is not None else int(np.argmax(visits))
            result[selected] = 1.0
            return result
        scaled = np.power(visits, 1.0 / temperature)
        return scaled / max(float(scaled.sum()), 1e-12)

    @staticmethod
    def _uncertainty(visits: np.ndarray, q_root: np.ndarray) -> tuple[float, float]:
        order = np.argsort(-visits)
        total = max(float(visits.sum()), 1e-12)
        visit_share = float(visits[order[0]] / total)
        q_gap = float(q_root[order[0]] - q_root[order[1]]) if len(order) > 1 else float("inf")
        return visit_share, q_gap

    def plan(
        self,
        root_snapshot: dict[str, Any],
        root_obs: dict[str, np.ndarray],
        *,
        force_search: bool = False,
    ) -> SelectiveSearchDecision:
        started = time.perf_counter()
        configure_search_recursion_limit(self.config.max_atomic_steps)
        if hasattr(self.evaluator, "cache"):
            self.evaluator.cache.clear()
        self.search_env.restore(root_snapshot)
        side = root_side(self.search_env.last_obs)
        validate_information_safe_observation(root_obs, side)
        legal_count = len(self.search_env.legal_actions)
        raw = self.evaluator.raw(root_obs, legal_count)
        raw_policy = normalize_priors(raw["probabilities"], legal_count)
        raw_action = int(np.argmax(raw_policy))
        episode_seed = getattr(self.search_env, "episode_seed", None)
        if self.pending_continuation_seed != episode_seed:
            self.pending_continuation.clear()
            self.pending_continuation_seed = episode_seed
        if self.pending_continuation:
            expected = self.pending_continuation[0]
            # A principal variation may cross several strategic boundaries, but
            # only clicks belonging to the current macro are executable without
            # rerooting. Fail closed if the restored live boundary differs from
            # the stored continuation level. This is especially important at a
            # new Turn's headline/card choice.
            if not continuation_matches_live_boundary(
                expected, self.search_env.last_obs, self.search_env.legal_actions
            ):
                self.pending_continuation.clear()
            matching = [
                index
                for index, action in enumerate(self.search_env.legal_actions)
                if action_text(action) == expected.get("action") and side == expected.get("side")
            ] if self.pending_continuation else []
            if matching:
                action = matching[0]
                self.pending_continuation.pop(0)
                decision = SelectiveSearchDecision(
                    action=action,
                    raw_action=raw_action,
                    raw_policy=raw_policy,
                    search_policy=raw_policy.copy(),
                    visits=np.zeros(legal_count, dtype=np.float64),
                    q_root=np.zeros(legal_count, dtype=np.float64),
                    searched=False,
                    triggers=("search_continuation",),
                    simulations=0,
                    budgets={},
                    root_q_gap=0.0,
                    action_flipped=action != raw_action,
                    latency_seconds=time.perf_counter() - started,
                    expanded_nodes=0,
                    exact_chance_nodes=0,
                    belief_particles=0,
                    meaningful_depth=0,
                    hierarchical_path=(expected,),
                )
                self.telemetry.record(decision)
                return decision
            self.pending_continuation.clear()
        triggers = () if legal_count <= 1 else classify_search_triggers(self.search_env, raw_policy, self.config)
        # A scoring card can remain in hand after an earlier AR's search chose
        # another card. Throttle repeated UI decisions within one AR, not every
        # subsequent card choice for the rest of the Turn.
        scoring_key = scoring_search_throttle_key(
            episode_seed, self.search_env.last_obs
        )
        raw_level = action_level(
            self.search_env.legal_actions[raw_action],
            str((self.search_env.last_obs or {}).get("prompt") or ""),
        )
        chance_stage = "target" if raw_level in {"coup_target", "realignment_target", "forced_response"} else "root"
        chance_key = (
            episode_seed,
            side,
            int((self.search_env.last_obs or {}).get("turn") or 0),
            int((self.search_env.last_obs or {}).get("action_round") or 0),
            chance_stage,
        )
        allocation_key = (
            episode_seed,
            side,
            int((self.search_env.last_obs or {}).get("turn") or 0),
            int((self.search_env.last_obs or {}).get("action_round") or 0),
        )
        if (
            "scoring_card" in triggers
            and self.scoring_search_counts[scoring_key] >= self.config.max_scoring_searches_per_side_turn
        ):
            triggers = tuple(trigger for trigger in triggers if trigger != "scoring_card")
        if (
            "chance_heavy" in triggers
            and self.chance_search_counts[chance_key] >= self.config.max_chance_searches_per_side_round
        ):
            triggers = tuple(trigger for trigger in triggers if trigger != "chance_heavy")
        if (
            "influence_allocation" in triggers
            and self.allocation_search_counts[allocation_key]
            >= self.config.max_allocation_searches_per_side_round
        ):
            triggers = tuple(trigger for trigger in triggers if trigger != "influence_allocation")
        if legal_count <= 1 or (not triggers and not force_search):
            decision = SelectiveSearchDecision(
                action=raw_action,
                raw_action=raw_action,
                raw_policy=raw_policy,
                search_policy=raw_policy.copy(),
                visits=np.zeros(legal_count, dtype=np.float64),
                q_root=np.zeros(legal_count, dtype=np.float64),
                searched=False,
                triggers=(),
                simulations=0,
                budgets={},
                root_q_gap=0.0,
                action_flipped=False,
                latency_seconds=time.perf_counter() - started,
                expanded_nodes=0,
                exact_chance_nodes=0,
                belief_particles=0,
                meaningful_depth=0,
                hierarchical_path=(),
            )
            self.telemetry.record(decision)
            return decision

        self.search_index += 1
        session_config = self.config
        if (
            "forced_response" in triggers
            and not set(triggers) & {"opponent_event", "scoring_card", "influence_allocation"}
        ):
            session_config = replace(
                self.config,
                meaningful_depth=max(1, int(self.config.forced_response_depth)),
            )
        session = ExactBeliefPuctSession(
            self.evaluator,
            session_config,
            seed=self.seed + 1_000_003 * self.search_index,
        )
        try:
            session.begin(self.search_env, root_snapshot, root_obs)
        except Exception:
            self._clear_runtime_checkpoints()
            raise
        tables: dict[int, list[dict[str, Any]]] = {}
        try:
            session.advance_to(self.search_env, self.config.initial_simulations)
        except Exception:
            self._clear_runtime_checkpoints()
            raise
        tables[self.config.initial_simulations] = session.root_table()
        visits, q_root = session.root_arrays()
        search_action, _selection = session.select_root_action()
        visit_share, q_gap = self._uncertainty(visits, q_root)
        disagreement = search_action != raw_action
        escalate_medium = (
            self.config.medium_simulations > self.config.initial_simulations
            and (
                (self.config.escalate_on_disagreement and disagreement)
                or visit_share < self.config.medium_visit_share_threshold
                or abs(q_gap) < self.config.medium_q_gap_threshold
            )
        )
        if escalate_medium:
            try:
                session.advance_to(self.search_env, self.config.medium_simulations)
            except Exception:
                self._clear_runtime_checkpoints()
                raise
            tables[self.config.medium_simulations] = session.root_table()
            visits, q_root = session.root_arrays()
            search_action, _selection = session.select_root_action()
            visit_share, q_gap = self._uncertainty(visits, q_root)
            disagreement = search_action != raw_action
        high_value = bool(
            set(triggers)
            & {
                "forced_response",
                "opponent_event",
                "scoring_card",
                "scoring_pressure",
                "chance_heavy",
            }
        )
        escalate_max = (
            self.config.max_simulations > session.completed
            and high_value
            and (
                visit_share < self.config.max_visit_share_threshold
                or abs(q_gap) < self.config.max_q_gap_threshold
                or (self.config.escalate_on_disagreement and disagreement)
            )
        )
        if escalate_max:
            try:
                session.advance_to(self.search_env, self.config.max_simulations)
            except Exception:
                self._clear_runtime_checkpoints()
                raise
            tables[self.config.max_simulations] = session.root_table()
            visits, q_root = session.root_arrays()
            search_action, _selection = session.select_root_action()
            _visit_share, q_gap = self._uncertainty(visits, q_root)
        visit_action = search_action
        q_advantage = float(q_root[visit_action] - q_root[raw_action])
        evidence_gate_rejected = search_evidence_gate_rejected(
            triggers,
            session_config,
            meaningful_depth=session_config.meaningful_depth,
            terminal_evaluations=session.terminal_evaluations,
        )
        search_incomplete = bool(
            session.atomic_horizon_cutoffs > 0 or session.node_budget_cutoffs > 0
            or evidence_gate_rejected
        )
        q_gate_rejected = bool(
            search_incomplete
            or (
                visit_action != raw_action
                and q_advantage <= self.config.min_q_advantage_to_flip
            )
        )
        if q_gate_rejected:
            search_action = raw_action
            q_advantage = 0.0
        assert session.root_node is not None
        search_policy = (
            raw_policy.copy()
            if search_incomplete
            else self._visit_policy(
                visits,
                session.root_node.priors,
                self.config.temperature,
                preferred_action=search_action,
            )
        )
        teacher_action = search_action
        if not q_gate_rejected and not search_incomplete and self.config.behavior_temperature > 1e-6:
            behavior_policy = self._visit_policy(
                visits,
                session.root_node.priors,
                self.config.behavior_temperature,
            )
            search_action = int(self.rng.choice(len(behavior_policy), p=behavior_policy))
        hierarchical_path = () if search_incomplete else session.best_trace(teacher_action)
        behavior_path = () if search_incomplete else session.best_trace(search_action)
        decision = SelectiveSearchDecision(
            action=search_action,
            raw_action=raw_action,
            raw_policy=raw_policy,
            search_policy=search_policy,
            visits=visits,
            q_root=q_root,
            searched=True,
            triggers=triggers,
            simulations=session.completed,
            budgets=tables,
            root_q_gap=float(q_gap),
            action_flipped=teacher_action != raw_action,
            latency_seconds=time.perf_counter() - started,
            expanded_nodes=len(session.nodes),
            exact_chance_nodes=session.stepper.enumerated_nodes,
            belief_particles=self.config.belief_particles,
            meaningful_depth=session_config.meaningful_depth,
            hierarchical_path=hierarchical_path,
            q_advantage_over_raw=q_advantage,
            sampled_large_chance_nodes=session.stepper.sampled_large_nodes,
            q_gate_rejected=q_gate_rejected,
            atomic_horizon_cutoffs=session.atomic_horizon_cutoffs,
            node_budget_cutoffs=session.node_budget_cutoffs,
            max_meaningful_depth_reached=session.max_meaningful_depth_reached,
            max_turn_delta=session.max_turn_delta,
            cross_turn_leaf_evaluations=session.cross_turn_leaf_evaluations,
            terminal_evaluations=session.terminal_evaluations,
            evidence_gate_rejected=evidence_gate_rejected,
            hierarchical_targets=(
                ()
                if search_incomplete
                else session.hierarchical_targets(
                    hierarchical_path,
                    temperature=self.config.temperature,
                )
            ),
            teacher_action=teacher_action,
            behavior_path=behavior_path,
            root_selection=dict(session.root_selection),
            root_admission_events=tuple(copy.deepcopy(session.root_admission_events)),
            root_probe_exact_chance_nodes=session.root_probe_stepper.enumerated_nodes,
            root_probe_sampled_large_chance_nodes=session.root_probe_stepper.sampled_large_nodes,
        )
        self.pending_continuation = list(
            executable_continuation_prefix(decision.behavior_path)
        )
        self.pending_continuation_seed = episode_seed
        if "scoring_card" in triggers:
            self.scoring_search_counts[scoring_key] += 1
        if "chance_heavy" in triggers:
            self.chance_search_counts[chance_key] += 1
        if "influence_allocation" in triggers:
            self.allocation_search_counts[allocation_key] += 1
        self.telemetry.record(decision)
        session.release()
        self._clear_runtime_checkpoints()
        return decision


class ConvergedAdaptivePuctPlanner:
    """Escalate search depth only when Q rankings converge, otherwise fail shallow."""

    def __init__(
        self,
        base: SelectivePuctPlanner,
        budget: SelectivePuctPlanner,
        deep: SelectivePuctPlanner,
        config: ConvergedAdaptivePuctConfig | None = None,
        *,
        seed: int = 1,
    ) -> None:
        self.base = base
        self.budget = budget
        self.deep = deep
        self.config = config or ConvergedAdaptivePuctConfig()
        self.seed = int(seed)

    @property
    def planners(self) -> tuple[SelectivePuctPlanner, ...]:
        return self.base, self.budget, self.deep

    def reset(self) -> None:
        for planner in self.planners:
            planner.pending_continuation.clear()
            planner.pending_continuation_seed = None

    def _root_seed(self, snapshot: dict[str, Any]) -> int:
        episode_seed = int(snapshot.get("episode_seed") or 0)
        episode_step = int(snapshot.get("episode_step") or 0)
        return self.seed + episode_seed * 1_000_003 + episode_step * 10_007

    def _pair_root_rng(self, snapshot: dict[str, Any]) -> None:
        root_seed = self._root_seed(snapshot)
        for planner in self.planners:
            planner.seed = root_seed
            planner.rng = np.random.default_rng(root_seed)
            planner.search_index = 0

    def _sync_continuation(self, selected: SelectivePuctPlanner) -> None:
        continuation = [dict(item) for item in selected.pending_continuation]
        continuation_seed = selected.pending_continuation_seed
        for planner in self.planners:
            planner.pending_continuation = [dict(item) for item in continuation]
            planner.pending_continuation_seed = continuation_seed

    def _should_escalate(self, decision: SelectiveSearchDecision) -> bool:
        return bool(
            self.config.always_compare
            or abs(float(decision.root_q_gap)) <= float(self.config.escalation_q_gap)
            or (self.config.escalate_on_flip and decision.action_flipped)
            or set(decision.triggers).intersection(self.config.escalation_triggers)
        )

    def _compare(
        self,
        left: SelectiveSearchDecision,
        right: SelectiveSearchDecision,
    ) -> dict[str, Any]:
        if len(left.visits) != len(right.visits):
            return {
                "comparable": False,
                "stable": False,
                "near_tie": False,
                "reason": "legal_action_count_changed",
            }
        if (
            left.atomic_horizon_cutoffs
            or left.node_budget_cutoffs
            or left.evidence_gate_rejected
            or right.atomic_horizon_cutoffs
            or right.node_budget_cutoffs
            or right.evidence_gate_rejected
        ):
            return {
                "comparable": False,
                "stable": False,
                "near_tie": False,
                "reason": "incomplete_search_evidence",
            }
        return compare_q_rankings(
            np.asarray(left.visits),
            np.asarray(left.q_root),
            np.asarray(right.visits),
            np.asarray(right.q_root),
            min_rank_correlation=float(self.config.min_rank_correlation),
            meaningful_q_gap=float(self.config.meaningful_q_gap),
        )

    @staticmethod
    def _config_label(decision: SelectiveSearchDecision) -> str:
        return f"d{int(decision.meaningful_depth)}_{int(decision.simulations)}"

    def _combined(
        self,
        selected: SelectiveSearchDecision,
        decisions: list[SelectiveSearchDecision],
        *,
        status: str,
        comparisons: list[dict[str, Any]],
        fallback: bool,
    ) -> SelectiveSearchDecision:
        return replace(
            selected,
            simulations=sum(int(decision.simulations) for decision in decisions),
            latency_seconds=sum(float(decision.latency_seconds) for decision in decisions),
            expanded_nodes=sum(int(decision.expanded_nodes) for decision in decisions),
            exact_chance_nodes=sum(int(decision.exact_chance_nodes) for decision in decisions),
            sampled_large_chance_nodes=sum(
                int(decision.sampled_large_chance_nodes) for decision in decisions
            ),
            atomic_horizon_cutoffs=sum(
                int(decision.atomic_horizon_cutoffs) for decision in decisions
            ),
            node_budget_cutoffs=sum(int(decision.node_budget_cutoffs) for decision in decisions),
            max_meaningful_depth_reached=max(
                int(decision.max_meaningful_depth_reached) for decision in decisions
            ),
            max_turn_delta=max(int(decision.max_turn_delta) for decision in decisions),
            cross_turn_leaf_evaluations=sum(
                int(decision.cross_turn_leaf_evaluations) for decision in decisions
            ),
            terminal_evaluations=sum(
                int(decision.terminal_evaluations) for decision in decisions
            ),
            convergence_status=status,
            convergence_configs=tuple(self._config_label(decision) for decision in decisions),
            convergence_comparisons=tuple(copy.deepcopy(comparisons)),
            adaptive_fallback=bool(fallback),
        )

    def plan(
        self,
        root_snapshot: dict[str, Any],
        root_obs: dict[str, np.ndarray],
        *,
        force_search: bool = False,
    ) -> SelectiveSearchDecision:
        self._pair_root_rng(root_snapshot)
        base = self.base.plan(root_snapshot, root_obs, force_search=force_search)
        if not base.searched or not self._should_escalate(base):
            self._sync_continuation(self.base)
            return replace(
                base,
                convergence_status="UNVALIDATED" if base.searched else None,
                convergence_configs=(self._config_label(base),) if base.searched else (),
            )

        budget = self.budget.plan(root_snapshot, root_obs, force_search=True)
        decisions = [base, budget]
        budget_comparison = self._compare(base, budget)
        comparisons = [
            {
                "left": self._config_label(base),
                "right": self._config_label(budget),
                **budget_comparison,
            }
        ]
        if budget_comparison.get("stable"):
            self._sync_continuation(self.budget)
            return self._combined(
                budget,
                decisions,
                status="BUDGET_CONVERGENCE",
                comparisons=comparisons,
                fallback=False,
            )
        if budget_comparison.get("near_tie"):
            self._sync_continuation(self.budget)
            return self._combined(
                budget,
                decisions,
                status="NEAR_TIE",
                comparisons=comparisons,
                fallback=True,
            )

        deep = self.deep.plan(root_snapshot, root_obs, force_search=True)
        decisions.append(deep)
        deep_comparison = self._compare(budget, deep)
        comparisons.append(
            {
                "left": self._config_label(budget),
                "right": self._config_label(deep),
                **deep_comparison,
            }
        )
        if deep_comparison.get("stable"):
            self._sync_continuation(self.deep)
            return self._combined(
                deep,
                decisions,
                status="HORIZON_CONVERGENCE",
                comparisons=comparisons,
                fallback=False,
            )

        # A deeper but unresolved tree is evidence, not a safe behavior policy.
        self._sync_continuation(self.budget)
        status = "NEAR_TIE" if deep_comparison.get("near_tie") else "UNRESOLVED"
        return self._combined(
            budget,
            decisions,
            status=status,
            comparisons=comparisons,
            fallback=True,
        )
