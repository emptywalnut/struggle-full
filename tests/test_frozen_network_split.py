import numpy as np
import pytest


torch = pytest.importorskip("torch")

from struggle_ai.frozen_diagnostic import FrozenNetwork


class FakeModel(torch.nn.Module):
    def __init__(self, logits, value):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.logits = torch.as_tensor(logits, dtype=torch.float32)
        self.value = float(value)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        batch = next(iter(input_dict["obs"].values())).shape[0]
        self._value_out = torch.full((batch,), self.value, device=self.anchor.device)
        return self.logits.to(self.anchor.device).repeat(batch, 1), state

    def value_function(self):
        return self._value_out


def test_frozen_network_can_split_policy_logits_and_value_model():
    policy = FakeModel([3.0, 1.0], -0.75)
    value = FakeModel([0.0, 0.0], 0.5)
    network = FrozenNetwork(policy, value)
    result = network.raw(
        {"action_mask": np.asarray([1.0, 1.0], dtype=np.float32)},
        2,
    )
    assert result["logits"].tolist() == [3.0, 1.0]
    assert result["raw_value"] == pytest.approx(0.5)
    assert result["probabilities"][0] > result["probabilities"][1]
