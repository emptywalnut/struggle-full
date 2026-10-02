from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

from struggle_ai import batched_inference
from struggle_ai.batched_inference import BatchedInferenceClient, evaluate_observation_batch


def test_load_checkpoint_model_accepts_direct_policy_state(monkeypatch, tmp_path) -> None:
    policy_state = tmp_path / "policy_state.pkl"
    policy_state.touch()
    expected = object()
    monkeypatch.setattr(batched_inference, "current_model_spaces", lambda: ("obs", "action"))
    monkeypatch.setattr(
        batched_inference,
        "load_policy_state_model",
        lambda path, **kwargs: (expected, {"path": str(path), **kwargs}),
    )
    assert batched_inference.load_checkpoint_model(policy_state, "cpu") is expected


def test_evaluate_observation_batch_preserves_per_request_legal_counts() -> None:
    torch = pytest.importorskip("torch")

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, input_dict, state, seq_lens):
            del seq_lens
            marker = input_dict["obs"]["marker"][:, 0]
            logits = marker[:, None] + torch.arange(4, dtype=torch.float32)[None, :]
            self.values = marker * 0.5
            return logits, state

        def value_function(self):
            return self.values

    observations = [
        {"marker": np.array([2.0], dtype=np.float32)},
        {"marker": np.array([-2.0], dtype=np.float32)},
    ]
    results = evaluate_observation_batch(FakeModel(), observations, [2, 3], device="cpu")

    assert len(results[0]["logits"]) == 2
    assert len(results[1]["logits"]) == 3
    assert results[0]["raw_value"] == pytest.approx(1.0)
    assert results[1]["raw_value"] == pytest.approx(-1.0)
    assert results[0]["probabilities"].sum() == pytest.approx(1.0)


def test_batched_inference_client_caches_identical_information_state() -> None:
    requests: queue.Queue = queue.Queue()
    responses: queue.Queue = queue.Queue()
    client = BatchedInferenceClient(
        worker_id=0,
        request_queue=requests,
        response_queue=responses,
        timeout_seconds=1.0,
        cache_capacity=4,
    )
    observation = {
        "action_mask": np.array([1.0, 1.0, 0.0], dtype=np.float32),
        "marker": np.array([3.0], dtype=np.float32),
    }

    def respond() -> None:
        request = requests.get(timeout=1.0)
        responses.put(
            {
                "request_id": request["request_id"],
                "result": {
                    "logits": np.array([0.0, 1.0]),
                    "probabilities": np.array([0.25, 0.75]),
                    "raw_value": 0.4,
                    "calibrated_value": 0.2,
                },
            }
        )

    thread = threading.Thread(target=respond)
    thread.start()
    first = client.raw(observation, 2)
    thread.join(timeout=1.0)
    second = client.raw(observation, 2)

    assert first is second
    assert client.cache_misses == 1
    assert client.cache_hits == 1
    assert requests.empty()
