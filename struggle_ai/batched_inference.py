from __future__ import annotations

import os
import pickle
import queue
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.mcts import calibrate_value, information_state_key


STAT_REQUESTS = 0
STAT_BATCHES = 1
STAT_BATCH_ITEMS = 2
STAT_CACHE_HITS = 3
STAT_INFERENCE_SECONDS = 4
STAT_QUEUE_WAIT_SECONDS = 5
STAT_MAX_BATCH = 6
STAT_ERRORS = 7
STAT_COUNT = 8


def current_model_spaces() -> tuple[Any, Any]:
    from gymnasium import spaces

    from struggle_ai.bridge_client import TwilightBridgeClient
    from struggle_ai.env import observation_space_for_spec
    from struggle_ai.features import MAX_ACTIONS, make_spec

    with TwilightBridgeClient() as bridge:
        spec = make_spec(bridge.cards(), bridge.countries())
    return observation_space_for_spec(spec), spaces.Discrete(MAX_ACTIONS)


def load_checkpoint_model(checkpoint: Path | str, device: str):
    """Load the RLlib custom model without starting Ray or an Algorithm."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.is_file():
        observation_space, action_space = current_model_spaces()
        model, _metadata = load_policy_state_model(
            checkpoint,
            device=device,
            observation_space=observation_space,
            action_space=action_space,
        )
        return model
    policy_state_path = checkpoint / "policies" / "shared_policy" / "policy_state.pkl"
    if not policy_state_path.exists():
        raise FileNotFoundError(f"shared policy state not found: {policy_state_path}")
    model, _metadata = load_policy_state_model(policy_state_path, device=device)
    return model


def _original_space(space: Any) -> Any:
    return getattr(space, "original_space", space)


def load_policy_state_model(
    policy_state_path: Path | str,
    *,
    device: str,
    observation_space: Any | None = None,
    action_space: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load one RLlib policy state through the canonical masked-model stack.

    Older selected policies are widened with the same repository helper used by
    training restores. New feature columns are therefore zeroed instead of
    being reinterpreted, while the current observation/action catalog remains
    authoritative.
    """

    import torch
    from ray.rllib.utils.serialization import space_from_dict

    from struggle_ai.rllib_masked_model import TorchMaskedActions
    from struggle_ai.train_rllib import copy_compatible_or_widened_weights

    policy_state_path = Path(policy_state_path).expanduser().resolve()
    if not policy_state_path.is_file():
        raise FileNotFoundError(f"policy state not found: {policy_state_path}")
    with policy_state_path.open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, dict) or not isinstance(state.get("policy_spec"), dict):
        raise RuntimeError("invalid RLlib policy state: missing policy_spec")
    if not isinstance(state.get("weights"), dict):
        raise RuntimeError("invalid RLlib policy state: missing weights")
    spec = state["policy_spec"]
    source_observation_space = space_from_dict(spec["observation_space"])
    source_action_space = space_from_dict(spec["action_space"])
    target_observation_space = observation_space or source_observation_space
    target_action_space = action_space or source_action_space
    if not hasattr(source_action_space, "n") or not hasattr(target_action_space, "n"):
        raise RuntimeError("checkpoint action space is not discrete")
    if int(source_action_space.n) != int(target_action_space.n):
        raise RuntimeError(
            f"action-space mismatch: checkpoint={source_action_space.n}, current={target_action_space.n}"
        )
    model_config = spec["config"]["model"]
    if model_config.get("custom_model") not in {None, "twilight_masked_model"}:
        raise RuntimeError(f"unsupported custom model: {model_config.get('custom_model')}")
    model = TorchMaskedActions(
        target_observation_space,
        target_action_space,
        int(target_action_space.n),
        model_config,
        "batched_puct_inference",
    )
    source_weights = state["weights"]
    exact_schema = _original_space(source_observation_space) == _original_space(target_observation_space)
    current_weights = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    if exact_schema:
        migrated_weights = source_weights
        migration = {
            "mode": "native",
            "matched": len(source_weights),
            "widened": [],
            "missing_source": [],
        }
    else:
        migrated_weights, matched, widened, skipped, missing_source, unexpected_source = (
            copy_compatible_or_widened_weights(current_weights, source_weights)
        )
        allowed_missing_prefixes = (
            "card_history_encoder.",
            "card_history_transformer.",
        )
        disallowed_missing = [
            name for name in missing_source if not name.startswith(allowed_missing_prefixes)
        ]
        if skipped or unexpected_source or disallowed_missing:
            raise RuntimeError(
                "checkpoint/model migration mismatch: "
                f"skipped={skipped}, unexpected={sorted(unexpected_source)}, "
                f"missing={sorted(disallowed_missing)}"
            )
        migration = {
            "mode": "canonical_widening",
            "matched": len(matched),
            "widened": widened,
            "missing_source": sorted(missing_source),
        }
    weights = {name: torch.as_tensor(value) for name, value in migrated_weights.items()}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    model.to(torch.device(device))
    model.eval()
    custom = model_config.get("custom_model_config", {})
    environment_config = spec.get("config", {}).get("env_config", {}) or {}
    return model, {
        "architecture": str(custom.get("model_arch", "feedforward")),
        "hidden": int(custom.get("hidden", 256)),
        "side_specific_heads": bool(custom.get("side_specific_heads", False)),
        "value_state_tower": bool(custom.get("value_state_tower", False)),
        "action_space_n": int(target_action_space.n),
        "environment": {
            "defcon_suicide_mode": str(environment_config.get("defcon_suicide_mode", "none")),
            "force_setup_heuristic": bool(environment_config.get("force_setup_heuristic", True)),
        },
        "migration": migration,
    }


def choose_legal_action_index(
    model: Any,
    observation: dict[str, np.ndarray],
    legal_count: int,
    *,
    device: str,
) -> tuple[int, dict[str, Any]]:
    """Return RLlib's deterministic masked-logit choice for the legal catalog."""

    legal_count = int(legal_count)
    mask = np.asarray(observation.get("action_mask"), dtype=np.float32).reshape(-1)
    if legal_count <= 0 or legal_count > len(mask):
        raise ValueError(f"invalid legal action count: {legal_count}")
    active = np.flatnonzero(mask > 0.5)
    expected = np.arange(legal_count)
    if not np.array_equal(active, expected):
        raise ValueError(
            "action mask/catalog mismatch: legal actions must occupy indices 0..legal_count-1"
        )
    result = evaluate_observation_batch(model, [observation], [legal_count], device=device)[0]
    logits = np.asarray(result["logits"], dtype=np.float64)
    if logits.shape != (legal_count,) or not np.all(np.isfinite(logits)):
        raise RuntimeError(f"model returned invalid legal logits with shape {logits.shape}")
    return int(np.argmax(logits)), result


def observation_signature(observation: dict[str, np.ndarray]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(sorted((name, tuple(np.asarray(value).shape)) for name, value in observation.items()))


def evaluate_observation_batch(
    model: Any,
    observations: list[dict[str, np.ndarray]],
    legal_counts: list[int],
    *,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    if not observations:
        return []
    keys = tuple(sorted(observations[0]))
    if any(tuple(sorted(observation)) != keys for observation in observations):
        raise ValueError("batched inference observations have inconsistent keys")
    batch = {
        name: torch.as_tensor(
            np.stack([observation[name] for observation in observations]),
            dtype=torch.float32,
            device=torch.device(device),
        )
        for name in keys
    }
    model.eval()
    with torch.inference_mode():
        logits, _state = model({"obs": batch}, [], None)
        values = model.value_function().reshape(-1)
    results: list[dict[str, Any]] = []
    for index, legal_count in enumerate(legal_counts):
        legal_logits = logits[index, : int(legal_count)]
        probabilities = torch.softmax(legal_logits, dim=-1)
        raw_value = float(values[index].detach().cpu())
        results.append(
            {
                "logits": legal_logits.detach().cpu().numpy().astype(np.float64),
                "probabilities": probabilities.detach().cpu().numpy().astype(np.float64),
                "raw_value": raw_value,
                "calibrated_value": calibrate_value(raw_value),
            }
        )
    return results


def _lru_put(cache: OrderedDict[str, dict[str, Any]], key: str, value: dict[str, Any], capacity: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max(0, int(capacity)):
        cache.popitem(last=False)


def inference_server_main(
    *,
    server_id: int,
    gpu: str,
    checkpoint: str,
    request_queue: Any,
    response_queues: list[Any],
    control_queue: Any,
    stats: Any,
    max_batch_size: int,
    max_batch_delay_ms: float,
    cache_capacity: int,
) -> None:
    """Resident process serving dynamically balanced batched NN inference."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable in inference server for physical GPU {gpu}")
        torch.cuda.set_device(0)
        model = load_checkpoint_model(checkpoint, "cuda:0")
        control_queue.put({"event": "inference_server_ready", "server_id": server_id, "gpu": gpu})
        cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        stopping = False
        while not stopping:
            request = request_queue.get()
            if request is None:
                break
            pending = [request]
            deadline = time.perf_counter() + max(0.0, float(max_batch_delay_ms)) / 1000.0
            while len(pending) < max(1, int(max_batch_size)):
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    item = request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    stopping = True
                    break
                pending.append(item)

            now_ns = time.time_ns()
            misses: list[dict[str, Any]] = []
            grouped: dict[tuple[tuple[str, tuple[int, ...]], ...], list[dict[str, Any]]] = {}
            for item in pending:
                stats[STAT_REQUESTS] += 1.0
                stats[STAT_QUEUE_WAIT_SECONDS] += max(
                    0.0,
                    (now_ns - int(item["enqueued_ns"])) / 1_000_000_000.0,
                )
                key = str(item["cache_key"])
                cached = cache.get(key)
                if cached is not None:
                    stats[STAT_CACHE_HITS] += 1.0
                    cache.move_to_end(key)
                    response_queues[int(item["worker_id"])].put(
                        {"request_id": int(item["request_id"]), "result": cached}
                    )
                    continue
                misses.append(item)
                grouped.setdefault(observation_signature(item["observation"]), []).append(item)

            for group in grouped.values():
                started = time.perf_counter()
                try:
                    results = evaluate_observation_batch(
                        model,
                        [item["observation"] for item in group],
                        [int(item["legal_count"]) for item in group],
                        device="cuda:0",
                    )
                    elapsed = time.perf_counter() - started
                    stats[STAT_BATCHES] += 1.0
                    stats[STAT_BATCH_ITEMS] += float(len(group))
                    stats[STAT_INFERENCE_SECONDS] += elapsed
                    stats[STAT_MAX_BATCH] = max(float(stats[STAT_MAX_BATCH]), float(len(group)))
                    for item, result in zip(group, results, strict=True):
                        _lru_put(cache, str(item["cache_key"]), result, cache_capacity)
                        response_queues[int(item["worker_id"])].put(
                            {"request_id": int(item["request_id"]), "result": result}
                        )
                except Exception as exc:
                    stats[STAT_ERRORS] += float(len(group))
                    for item in group:
                        response_queues[int(item["worker_id"])].put(
                            {"request_id": int(item["request_id"]), "error": repr(exc)}
                        )
            if stopping:
                break
        control_queue.put({"event": "inference_server_stopped", "server_id": server_id, "gpu": gpu})
    except BaseException as exc:
        stats[STAT_ERRORS] += 1.0
        control_queue.put(
            {"event": "inference_server_error", "server_id": server_id, "gpu": gpu, "error": repr(exc)}
        )
        raise


class BatchedInferenceClient:
    """Synchronous evaluator facade backed by a shared asynchronous GPU queue."""

    def __init__(
        self,
        *,
        worker_id: int,
        request_queue: Any,
        response_queue: Any,
        timeout_seconds: float = 120.0,
        cache_capacity: int = 4096,
    ) -> None:
        self.worker_id = int(worker_id)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout_seconds = float(timeout_seconds)
        self.cache_capacity = int(cache_capacity)
        self.request_id = 0
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _cache_key(observation: dict[str, np.ndarray], legal_count: int) -> str:
        return f"{information_state_key(observation, 'network')}:{int(legal_count)}"

    def raw(self, obs: dict[str, np.ndarray], legal_count: int) -> dict[str, Any]:
        key = self._cache_key(obs, legal_count)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return cached
        self.cache_misses += 1
        self.request_id += 1
        request_id = self.request_id
        self.request_queue.put(
            {
                "worker_id": self.worker_id,
                "request_id": request_id,
                "cache_key": key,
                "legal_count": int(legal_count),
                "observation": {
                    name: np.asarray(value, dtype=np.float32)
                    for name, value in obs.items()
                },
                "enqueued_ns": time.time_ns(),
            }
        )
        try:
            response = self.response_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(
                f"batched inference timed out after {self.timeout_seconds}s for worker {self.worker_id}"
            ) from exc
        if int(response.get("request_id", -1)) != request_id:
            raise RuntimeError(
                f"inference response mismatch: expected {request_id}, got {response.get('request_id')}"
            )
        if "error" in response:
            raise RuntimeError(f"inference server failed: {response['error']}")
        result = response["result"]
        _lru_put(self.cache, key, result, self.cache_capacity)
        return result

    def __call__(self, obs: dict[str, np.ndarray], legal_count: int) -> tuple[np.ndarray, float]:
        raw = self.raw(obs, legal_count)
        return raw["probabilities"], float(raw["raw_value"])

    def value_us(self, obs: dict[str, np.ndarray], side: str) -> tuple[float, float]:
        raw = self.raw(obs, int(np.asarray(obs["action_mask"]).sum()))
        sign = 1.0 if side == "us" else -1.0
        return sign * float(raw["raw_value"]), sign * float(raw["calibrated_value"])


def inference_stats_payload(server_id: int, gpu: str, stats: Any) -> dict[str, float | int | str]:
    requests = float(stats[STAT_REQUESTS])
    batches = float(stats[STAT_BATCHES])
    items = float(stats[STAT_BATCH_ITEMS])
    misses = max(0.0, requests - float(stats[STAT_CACHE_HITS]))
    return {
        "server_id": int(server_id),
        "gpu": str(gpu),
        "requests": int(requests),
        "batches": int(batches),
        "batch_items": int(items),
        "cache_hits": int(stats[STAT_CACHE_HITS]),
        "cache_hit_rate": float(stats[STAT_CACHE_HITS]) / max(1.0, requests),
        "average_batch_size": items / max(1.0, batches),
        "max_batch_size": int(stats[STAT_MAX_BATCH]),
        "inference_seconds": float(stats[STAT_INFERENCE_SECONDS]),
        "requests_per_inference_second": misses / max(1e-9, float(stats[STAT_INFERENCE_SECONDS])),
        "average_queue_wait_seconds": float(stats[STAT_QUEUE_WAIT_SECONDS]) / max(1.0, requests),
        "errors": int(stats[STAT_ERRORS]),
    }
