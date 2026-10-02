from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ray")

from struggle_ai.batched_inference import choose_legal_action_index
from struggle_ai.browser_models import BrowserModelRegistry


@pytest.fixture(scope="module")
def registry_cpu() -> BrowserModelRegistry:
    return BrowserModelRegistry(device_preference="cpu")


def test_registry_discovers_only_validated_repository_models(registry_cpu: BrowserModelRegistry) -> None:
    catalog = registry_cpu.catalog()
    models = {entry["id"]: entry for entry in catalog["policies"] if entry["kind"] == "model"}
    assert set(models) == {
        "phase3-u75-parent",
        "phase3-composed-p30u25",
        "legacy-mixed-side-best",
        "transformer-riskshape10-selected",
        "transformer-defconfilter5-selected",
    }
    assert models["legacy-mixed-side-best"]["architecture"] == "feedforward"
    assert models["phase3-u75-parent"]["architecture"] == "transformer_history"
    assert models["phase3-u75-parent"]["compatibility"] == {"us": "native", "ussr": "native"}
    assert models["transformer-riskshape10-selected"]["compatibility"]["us"] == "canonical_widening"
    assert models["transformer-defconfilter5-selected"]["inference_environment"]["us"]["defcon_suicide_mode"] == "hard_filter"
    assert models["phase3-u75-parent"]["inference_environment"]["us"]["defcon_suicide_mode"] == "none"
    assert all(not Path(entry["source"]).is_absolute() for entry in models.values())
    rejected = {entry["id"]: entry["reason"] for entry in catalog["rejected"]}
    assert "mcts-best-topk4" in rejected
    assert "absent" in rejected["mcts-best-topk4"]
    assert "p30u25-value-only" in rejected


def test_registry_rejects_invalid_or_incompatible_checkpoint(tmp_path: Path) -> None:
    invalid = tmp_path / "bad" / "policy_state.pkl"
    invalid.parent.mkdir(parents=True)
    with invalid.open("wb") as handle:
        pickle.dump({"not": "an RLlib policy"}, handle)
    candidate = ({
        "id": "invalid-test-model",
        "label": "Invalid test model",
        "source": "bad",
        "paths": {"shared": "bad/policy_state.pkl"},
        "provenance": "test fixture",
    },)
    registry = BrowserModelRegistry(root=tmp_path, candidates=candidate, device_preference="cpu")
    assert registry.entries == {}
    reason = next(item["reason"] for item in registry.rejected if item["id"] == "invalid-test-model")
    assert "incompatible checkpoint" in reason
    assert "policy_spec" in reason


def test_action_mask_and_catalog_indices_are_not_reinterpreted() -> None:
    class FakeModel:
        def eval(self):
            return self

        def __call__(self, input_dict, state, seq_lens):
            del seq_lens
            batch = input_dict["obs"]["action_mask"].shape[0]
            logits = torch.tensor([[0.2, 4.0, 1.1, 99.0]], dtype=torch.float32).repeat(batch, 1)
            self.values = torch.zeros(batch)
            return logits, state

        def value_function(self):
            return self.values

    observation = {"action_mask": np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)}
    index, result = choose_legal_action_index(FakeModel(), observation, 3, device="cpu")
    assert index == 1
    assert result["logits"].tolist() == pytest.approx([0.2, 4.0, 1.1])
    with pytest.raises(ValueError, match="mask/catalog mismatch"):
        choose_legal_action_index(
            FakeModel(),
            {"action_mask": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)},
            2,
            device="cpu",
        )


def test_model_cache_reuses_shared_checkpoint_and_cpu_path(registry_cpu: BrowserModelRegistry) -> None:
    first = registry_cpu.load("phase3-u75-parent", "us")
    second = registry_cpu.load("phase3-u75-parent", "ussr")
    source = (
        "runs/search_distill/root-mechanics-actor-correction-actionhead-u100-gpu4-20260819/"
        "checkpoints/updates-000075/policies/shared_policy/policy_state.pkl"
    )
    assert first["readiness"]["us"] == {"status": "ready", "device": "cpu", "error": None}
    assert second["readiness"]["ussr"] == {"status": "ready", "device": "cpu", "error": None}
    assert registry_cpu.load_counts[source] == 1
    assert registry_cpu.model_for("phase3-u75-parent", "us")[0] is registry_cpu.model_for(
        "phase3-u75-parent", "ussr"
    )[0]


def test_model_errors_never_fall_back_to_a_baseline(registry_cpu: BrowserModelRegistry) -> None:
    with pytest.raises(ValueError, match="unknown registered model ID"):
        registry_cpu.load("../../arbitrary-checkpoint", "us")
    with pytest.raises(RuntimeError, match="is not loaded"):
        registry_cpu.model_for("legacy-mixed-side-best", "us")
