from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from struggle_ai.env import TwilightStruggleEnv
from struggle_ai.mcts import calibrate_value, information_state_key


ACTION_MATCH_KEYS = ("type", "decision", "event", "selector", "value", "label")


def action_text(action: dict[str, Any]) -> str:
    label = str(action.get("label") or action.get("value") or action.get("type") or "?")
    value = str(action.get("value") or "")
    return label if not value or value.lower() == label.lower() else f"{label} [{value}]"


def find_action_index(legal_actions: list[dict[str, Any]], wanted: dict[str, Any]) -> int:
    exact: list[int] = []
    loose: list[int] = []
    for index, action in enumerate(legal_actions):
        shared = [key for key in ACTION_MATCH_KEYS if wanted.get(key) not in (None, "")]
        if shared and all(str(action.get(key) or "") == str(wanted.get(key) or "") for key in shared):
            exact.append(index)
        if (
            str(action.get("value") or "") == str(wanted.get("value") or "")
            and str(action.get("label") or "") == str(wanted.get("label") or "")
        ):
            loose.append(index)
    candidates = exact or loose
    if len(candidates) != 1:
        raise RuntimeError(
            f"cannot uniquely match logged action {wanted!r}; candidates="
            f"{[(i, legal_actions[i]) for i in candidates]}"
        )
    return candidates[0]


def is_meaningful_root(last_obs: dict[str, Any] | None) -> bool:
    if not last_obs:
        return False
    prompt = str(last_obs.get("prompt") or "").lower()
    phase = str(last_obs.get("phase") or "").lower()
    return "pick a card" in prompt or "hosts the" in prompt or phase.startswith("headline_")


def root_side(last_obs: dict[str, Any] | None) -> str:
    return str((last_obs or {}).get("side") or "ussr")


def terminal_value_us(info: dict[str, Any]) -> float:
    winner = str(info.get("winner") or "")
    return 1.0 if winner == "us" else -1.0 if winner == "ussr" else 0.0


class FrozenNetwork:
    def __init__(self, policy: Any, value_policy: Any | None = None) -> None:
        import torch

        self.torch = torch
        self.policy = policy
        self.value_policy = value_policy or policy
        self.model = getattr(policy, "model", policy)
        self.value_model = getattr(self.value_policy, "model", self.value_policy)
        self.device = next(self.model.parameters()).device
        self.value_device = next(self.value_model.parameters()).device
        self.model.eval()
        self.value_model.eval()
        self.cache: dict[tuple[str, int], dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def raw(self, obs: dict[str, np.ndarray], legal_count: int) -> dict[str, Any]:
        key = (observation_digest(obs), legal_count)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        policy_batch = {
            key: self.torch.as_tensor(value, dtype=self.torch.float32, device=self.device).unsqueeze(0)
            for key, value in obs.items()
        }
        with self.torch.no_grad():
            logits, _state = self.model({"obs": policy_batch}, [], None)
            if self.value_model is self.model:
                raw_value = float(self.model.value_function()[0].detach().cpu())
            else:
                value_batch = {
                    key: self.torch.as_tensor(
                        value,
                        dtype=self.torch.float32,
                        device=self.value_device,
                    ).unsqueeze(0)
                    for key, value in obs.items()
                }
                self.value_model({"obs": value_batch}, [], None)
                raw_value = float(self.value_model.value_function()[0].detach().cpu())
            legal_logits = logits[0, :legal_count]
            probabilities = self.torch.softmax(legal_logits, dim=-1)
        result = {
            "logits": legal_logits.detach().cpu().numpy().astype(np.float64),
            "probabilities": probabilities.detach().cpu().numpy().astype(np.float64),
            "raw_value": raw_value,
            "calibrated_value": calibrate_value(raw_value),
        }
        self.cache[key] = result
        return result

    def value_us(self, obs: dict[str, np.ndarray], side: str) -> tuple[float, float]:
        result = self.raw(obs, int(np.asarray(obs["action_mask"]).sum()))
        raw = float(result["raw_value"])
        calibrated = float(result["calibrated_value"])
        return (raw if side == "us" else -raw, calibrated if side == "us" else -calibrated)


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


class ExactChanceStepper:
    """Enumerate small Saito chance transitions and reject opaque large nodes."""

    def __init__(self) -> None:
        self.enumerated_nodes = 0
        self.deterministic_nodes = 0
        self.max_outcomes = 0
        self.unknown_nodes: list[dict[str, Any]] = []
        self.chance_cache: dict[tuple[str, int], list[ExactOutcome]] = {}
        self.chance_cache_hits = 0

    @staticmethod
    def _cache_key(snapshot: dict[str, Any], action_index: int) -> tuple[str, int]:
        payload = json.dumps(snapshot["bridge"], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.blake2b(payload.encode("ascii"), digest_size=16).hexdigest(), int(action_index)

    @staticmethod
    def _collapse_observation_equivalent(branches: list[ExactOutcome]) -> list[ExactOutcome]:
        collapsed: dict[tuple[str, bool, bool, str], ExactOutcome] = {}
        for branch in branches:
            key = (
                observation_digest(branch.obs),
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
    def _run(
        env: TwilightStruggleEnv,
        snapshot: dict[str, Any],
        action_index: int,
        forced_rolls: list[int | float],
    ) -> ExactOutcome:
        env.restore(snapshot)
        obs, _reward, terminated, truncated, info = env.step_with_forced_rolls(action_index, forced_rolls)
        return ExactOutcome(
            probability=1.0,
            obs=obs,
            snapshot=env.snapshot(compact=True),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
            chance_rolls=tuple(int(value) for value in info.get("chance_rolls", [])),
        )

    def outcomes(
        self,
        env: TwilightStruggleEnv,
        snapshot: dict[str, Any],
        action_index: int,
    ) -> list[ExactOutcome]:
        cache_key = self._cache_key(snapshot, action_index)
        cached = self.chance_cache.get(cache_key)
        if cached is not None:
            self.chance_cache_hits += 1
            return cached
        env.restore(snapshot)
        prompt = str((env.last_obs or {}).get("prompt") or "")
        action = env.legal_actions[action_index]
        value = str(action.get("value") or "").lower()

        if "hosts the" in prompt.lower() and value == "participate":
            host = "us" if prompt.strip().lower().startswith("us hosts") else "ussr"
            pairs = []
            for us_roll in range(1, 7):
                for ussr_roll in range(1, 7):
                    us_total = us_roll + (2 if host == "us" else 0)
                    ussr_total = ussr_roll + (2 if host == "ussr" else 0)
                    if us_total != ussr_total:
                        pairs.append((us_roll, ussr_roll))
            branches = [self._run(env, snapshot, action_index, [a, b]) for a, b in pairs]
            for branch in branches:
                branch.probability = 1.0 / len(branches)
            self.enumerated_nodes += 1
            self.max_outcomes = max(self.max_outcomes, len(branches))
            collapsed = self._collapse_observation_equivalent(branches)
            self.chance_cache[cache_key] = collapsed
            return collapsed

        probe = self._run(
            env,
            snapshot,
            action_index,
            [0.071, 0.213, 0.357, 0.499, 0.643, 0.787, 0.929] * 3,
        )
        consumed = list(probe.info.get("forced_rolls_consumed", []))
        if not consumed:
            probe.current_in_env = True
            self.deterministic_nodes += 1
            return [probe]
        sides = [int(value) for value in probe.info.get("chance_roll_sides", [])]
        if len(sides) != len(consumed) or any(side < 1 for side in sides):
            detail = {
                "prompt": prompt,
                "action": copy.deepcopy(action),
                "consumed": consumed,
                "chance_roll_sides": sides,
            }
            self.unknown_nodes.append(detail)
            raise RuntimeError(f"chance-node arity is not auditable: {detail}")
        outcome_count = math.prod(sides)
        if outcome_count > 4096:
            detail = {
                "prompt": prompt,
                "action": copy.deepcopy(action),
                "chance_roll_sides": sides,
                "outcome_count": outcome_count,
            }
            self.unknown_nodes.append(detail)
            raise RuntimeError(f"chance node exceeds exact-enumeration bound: {detail}")
        forced_outcomes = itertools.product(*(range(1, side + 1) for side in sides))
        branches = [self._run(env, snapshot, action_index, list(forced)) for forced in forced_outcomes]
        for branch in branches:
            observed_sides = list(branch.info.get("chance_roll_sides", []))
            if observed_sides != sides:
                detail = {
                    "prompt": prompt,
                    "action": copy.deepcopy(action),
                    "probe_sides": sides,
                    "branch_sides": observed_sides,
                    "chance_rolls": branch.chance_rolls,
                }
                self.unknown_nodes.append(detail)
                raise RuntimeError(f"variable-length chance node requires a dedicated enumerator: {detail}")
            branch.probability = 1.0 / outcome_count
        self.enumerated_nodes += 1
        self.max_outcomes = max(self.max_outcomes, len(branches))
        collapsed = self._collapse_observation_equivalent(branches)
        self.chance_cache[cache_key] = collapsed
        return collapsed


@dataclass
class PuctNode:
    side: str
    priors: np.ndarray
    active: np.ndarray
    labels: tuple[str, ...]
    visits: np.ndarray
    value_sum_us: np.ndarray

    def select(self, cpuct: float, force_unvisited: bool = False) -> int:
        if force_unvisited:
            unvisited = np.flatnonzero(self.active & (self.visits <= 1e-12))
            if len(unvisited):
                return int(unvisited[np.argmax(self.priors[unvisited])])
        q_us = np.divide(
            self.value_sum_us,
            self.visits,
            out=np.zeros_like(self.value_sum_us),
            where=self.visits > 0.0,
        )
        perspective_q = q_us if self.side == "us" else -q_us
        exploration = cpuct * self.priors * math.sqrt(float(self.visits.sum()) + 1.0) / (1.0 + self.visits)
        return int(np.argmax(np.where(self.active, perspective_q + exploration, -np.inf)))


@dataclass
class PuctRun:
    budgets: dict[int, list[dict[str, Any]]]
    traces: list[dict[str, Any]]
    expanded_nodes: int
    exact_chance_nodes: int
    max_exact_outcomes: int
    boundary_evaluations: int
    terminal_evaluations: int
    chance_cache_hits: int


class ExactBeliefPuct:
    def __init__(
        self,
        network: FrozenNetwork,
        *,
        cpuct: float = 1.5,
        belief_particles: int = 32,
        max_meaningful_decisions: int = 2,
        child_top_k: int = 16,
        max_atomic_steps: int = 96,
        seed: int = 1,
    ) -> None:
        self.network = network
        self.cpuct = cpuct
        self.belief_particles = belief_particles
        self.max_meaningful_decisions = max_meaningful_decisions
        self.child_top_k = child_top_k
        self.max_atomic_steps = max_atomic_steps
        self.seed = seed
        self.nodes: dict[str, PuctNode] = {}
        self.stepper = ExactChanceStepper()
        self.boundary_evaluations = 0
        self.terminal_evaluations = 0
        self.trace_stats: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])

    def _node(self, env: TwilightStruggleEnv, obs: dict[str, np.ndarray], *, root: bool) -> PuctNode:
        side = root_side(env.last_obs)
        key = information_state_key(obs, side)
        existing = self.nodes.get(key)
        if existing is not None:
            return existing
        legal_count = len(env.legal_actions)
        raw = self.network.raw(obs, legal_count)
        priors = raw["probabilities"]
        active = np.ones(legal_count, dtype=np.bool_)
        if not root and self.child_top_k > 0 and legal_count > self.child_top_k:
            active[:] = False
            indices = np.argpartition(priors, -self.child_top_k)[-self.child_top_k :]
            active[indices] = True
            priors = np.where(active, priors, 0.0)
            priors = priors / priors.sum()
        node = PuctNode(
            side=side,
            priors=priors,
            active=active,
            labels=tuple(action_text(action) for action in env.legal_actions),
            visits=np.zeros(legal_count, dtype=np.float64),
            value_sum_us=np.zeros(legal_count, dtype=np.float64),
        )
        self.nodes[key] = node
        return node

    def _leaf_value(self, env: TwilightStruggleEnv, obs: dict[str, np.ndarray]) -> float:
        self.boundary_evaluations += 1
        return self.network.value_us(obs, root_side(env.last_obs))[1]

    def _simulate(
        self,
        env: TwilightStruggleEnv,
        snapshot: dict[str, Any],
        obs: dict[str, np.ndarray],
        *,
        decisions_used: int,
        atomic_steps: int,
        update_weight: float,
        trace: tuple[str, ...],
        at_root: bool,
        state_loaded: bool = False,
    ) -> float:
        if not state_loaded:
            env.restore(snapshot)
        if atomic_steps > 0 and is_meaningful_root(env.last_obs) and decisions_used >= self.max_meaningful_decisions:
            value = self._leaf_value(env, obs)
            stat = self.trace_stats[trace]
            stat[0] += update_weight
            stat[1] += update_weight * value
            return value
        if atomic_steps >= self.max_atomic_steps:
            raise RuntimeError(
                f"search exceeded {self.max_atomic_steps} atomic actions before a meaningful boundary: "
                f"{(env.last_obs or {}).get('prompt')}"
            )

        node = self._node(env, obs, root=at_root)
        action_index = node.select(self.cpuct, force_unvisited=at_root)
        action_label = node.labels[action_index]
        next_decisions = decisions_used + (1 if is_meaningful_root(env.last_obs) else 0)
        outcomes = self.stepper.outcomes(env, snapshot, action_index)
        expected = 0.0
        for outcome in outcomes:
            next_trace = trace + (action_label,)
            if outcome.terminated or outcome.truncated:
                value = terminal_value_us(outcome.info)
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
                    state_loaded=outcome.current_in_env,
                )
            expected += outcome.probability * value
        node.visits[action_index] += update_weight
        node.value_sum_us[action_index] += update_weight * expected
        return expected

    @staticmethod
    def _root_table(node: PuctNode, root_player: str) -> list[dict[str, Any]]:
        q_us = np.divide(
            node.value_sum_us,
            node.visits,
            out=np.zeros_like(node.value_sum_us),
            where=node.visits > 0.0,
        )
        q_root = q_us if root_player == "us" else -q_us
        rows = [
            {
                "action_index": index,
                "action": node.labels[index],
                "N": float(node.visits[index]),
                "Q": float(q_root[index]),
                "P": float(node.priors[index]),
            }
            for index in range(len(node.labels))
        ]
        return sorted(rows, key=lambda row: (-row["N"], -row["Q"], -row["P"]))

    def run(
        self,
        env: TwilightStruggleEnv,
        root_snapshot: dict[str, Any],
        root_obs: dict[str, np.ndarray],
        budgets: tuple[int, ...] = (32, 128, 512),
    ) -> PuctRun:
        env.restore(root_snapshot)
        root_player = root_side(env.last_obs)
        root_node = self._node(env, root_obs, root=True)
        belief_seeds = [self.seed + 104729 * index for index in range(self.belief_particles)]
        tables: dict[int, list[dict[str, Any]]] = {}
        completed = 0
        for budget in sorted(set(budgets)):
            while completed < budget:
                env.restore(root_snapshot)
                obs = env.redeterminize(root_player, belief_seeds[completed % len(belief_seeds)])
                particle_snapshot = env.snapshot(compact=True)
                self._simulate(
                    env,
                    particle_snapshot,
                    obs,
                    decisions_used=0,
                    atomic_steps=0,
                    update_weight=1.0,
                    trace=(),
                    at_root=True,
                    state_loaded=True,
                )
                completed += 1
            tables[budget] = self._root_table(root_node, root_player)
        traces = []
        for sequence, (weight, value_sum) in self.trace_stats.items():
            traces.append(
                {
                    "sequence": list(sequence),
                    "weight": float(weight),
                    "Q_us": float(value_sum / weight) if weight else 0.0,
                    "Q_root": float((value_sum / weight) * (1.0 if root_player == "us" else -1.0)) if weight else 0.0,
                }
            )
        traces.sort(key=lambda row: (-row["weight"], -row["Q_root"]))
        return PuctRun(
            budgets=tables,
            traces=traces,
            expanded_nodes=len(self.nodes),
            exact_chance_nodes=self.stepper.enumerated_nodes,
            max_exact_outcomes=self.stepper.max_outcomes,
            boundary_evaluations=self.boundary_evaluations,
            terminal_evaluations=self.terminal_evaluations,
            chance_cache_hits=self.stepper.chance_cache_hits,
        )


def observation_digest(obs: dict[str, np.ndarray]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for key in sorted(obs):
        digest.update(key.encode("ascii"))
        digest.update(np.asarray(obs[key]).tobytes())
    return digest.hexdigest()
