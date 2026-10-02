from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from struggle_ai.search_distill import (
    GameOutcomeBalancedValueSampler,
    SearchReplaySampler,
    actor_parameters,
    build_search_optimizer,
    distill_search_batch,
    limit_examples_per_source_root,
    run_search_distillation_updates,
    restore_search_optimizer_state,
    save_search_optimizer_state,
    search_training_parameters,
)
from struggle_ai.search_replay import SearchReplayExample, SearchReplayMetadata


def test_search_distillation_updates_actor_but_not_critic() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.policy_head = torch.nn.Linear(2, 2, bias=False)
            self.value = torch.nn.Linear(2, 1, bias=False)

        def forward(self, input_dict, state, seq_lens):
            hidden = self.encoder(input_dict["obs"]["x"])
            return self.policy_head(hidden), state

    model = TinyModel()
    policy = SimpleNamespace(model=model)
    optimizer = build_search_optimizer(policy, lr=0.1)
    critic_before = model.value.weight.detach().clone()
    actor_before = model.policy_head.weight.detach().clone()
    metadata = SearchReplayMetadata(
        side="us",
        trigger_categories=("forced_response",),
        legal_actions=({"value": "a"}, {"value": "b"}),
        hierarchical_path=({"action": "b", "level": "forced_response"},),
        raw_action=0,
        search_action=1,
        search_depth=2,
        simulation_budget=32,
        root_q_gap=0.3,
        action_flipped=True,
    )
    examples = [
        SearchReplayExample(
            observation={"x": np.array([1.0, -1.0], dtype=np.float32)},
            raw_policy=np.array([0.8, 0.2], dtype=np.float32),
            visit_counts=np.array([4.0, 28.0], dtype=np.float32),
            search_policy=np.array([0.125, 0.875], dtype=np.float32),
            q_values=np.array([-0.2, 0.1], dtype=np.float32),
            metadata=metadata,
        )
    ]

    metrics = distill_search_batch(policy, optimizer, examples, lambda_search=0.5)

    assert metrics["search_distill/loss"] > 0.0
    assert not torch.equal(model.policy_head.weight.detach(), actor_before)
    assert torch.equal(model.value.weight.detach(), critic_before)


def test_complete_game_search_training_updates_actor_and_critic() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.policy_head = torch.nn.Linear(2, 2, bias=False)
            self.value = torch.nn.Linear(2, 1, bias=False)
            self._hidden = None

        def forward(self, input_dict, state, seq_lens):
            self._hidden = self.encoder(input_dict["obs"]["x"])
            return self.policy_head(self._hidden), state

        def value_function(self):
            return self.value(self._hidden).squeeze(-1)

    model = TinyModel()
    policy = SimpleNamespace(model=model)
    optimizer = build_search_optimizer(policy, lr=0.05, parameter_scope="actor_value")
    actor_before = model.policy_head.weight.detach().clone()
    critic_before = model.value.weight.detach().clone()
    value_anchor_model = TinyModel()
    value_anchor_model.load_state_dict(model.state_dict())
    for parameter in value_anchor_model.parameters():
        parameter.requires_grad_(False)
    example = SearchReplayExample(
        observation={"x": np.array([1.0, -1.0], dtype=np.float32)},
        raw_policy=np.array([0.8, 0.2], dtype=np.float32),
        visit_counts=np.array([4.0, 28.0], dtype=np.float32),
        search_policy=np.array([0.125, 0.875], dtype=np.float32),
        q_values=np.array([-0.2, 0.3], dtype=np.float32),
        metadata=SearchReplayMetadata(
            side="us",
            trigger_categories=("forced_response",),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=(),
            raw_action=0,
            search_action=1,
            search_depth=2,
            simulation_budget=32,
            root_q_gap=0.5,
            action_flipped=True,
            root_value_target=1.0,
            complete_game=True,
        ),
    )

    metrics = distill_search_batch(
        policy,
        optimizer,
        [example],
        lambda_search=1.0,
        lambda_value=1.0,
        lambda_search_q=0.1,
        value_anchor_model=value_anchor_model,
        lambda_value_anchor=0.2,
        parameter_scope="actor_value",
    )

    assert metrics["search_distill/value_examples"] == 1.0
    assert metrics["search_distill/value_mse"] >= 0.0
    assert metrics["search_distill/value_anchor_mse"] >= 0.0
    assert not torch.equal(model.policy_head.weight.detach(), actor_before)
    assert not torch.equal(model.value.weight.detach(), critic_before)


def test_actor_value_heads_updates_heads_but_freezes_shared_encoder() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = torch.nn.Linear(2, 2, bias=False)
            self.action_encoder = torch.nn.Linear(2, 2, bias=False)
            self.policy_query = torch.nn.Linear(2, 2, bias=False)
            self.action_bias = torch.nn.Linear(2, 1, bias=False)
            self.value = torch.nn.Linear(2, 1, bias=False)
            self._hidden = None

        def forward(self, input_dict, state, seq_lens):
            obs = input_dict["obs"]
            self._hidden = self.state_encoder(obs["x"])
            action_hidden = self.action_encoder(obs["x"])
            return self.policy_query(self._hidden) + action_hidden, state

        def value_function(self):
            return self.value(self._hidden).squeeze(-1)

    model = TinyModel()
    policy = SimpleNamespace(model=model)
    optimizer = build_search_optimizer(policy, lr=0.05, parameter_scope="actor_value_heads")
    encoder_before = model.state_encoder.weight.detach().clone()
    action_encoder_before = model.action_encoder.weight.detach().clone()
    value_before = model.value.weight.detach().clone()
    example = SearchReplayExample(
        observation={"x": np.array([1.0, -1.0], dtype=np.float32)},
        raw_policy=np.array([0.8, 0.2], dtype=np.float32),
        visit_counts=np.array([4.0, 28.0], dtype=np.float32),
        search_policy=np.array([0.125, 0.875], dtype=np.float32),
        q_values=np.array([-0.2, 0.3], dtype=np.float32),
        metadata=SearchReplayMetadata(
            side="us",
            trigger_categories=("forced_response",),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=(),
            raw_action=0,
            search_action=1,
            search_depth=2,
            simulation_budget=32,
            root_q_gap=0.5,
            action_flipped=True,
            root_value_target=1.0,
            complete_game=True,
        ),
    )

    distill_search_batch(
        policy,
        optimizer,
        [example],
        lambda_search=1.0,
        lambda_value=1.0,
        lambda_search_q=0.1,
        parameter_scope="actor_value_heads",
    )

    assert torch.equal(model.state_encoder.weight.detach(), encoder_before)
    assert not torch.equal(model.action_encoder.weight.detach(), action_encoder_before)
    assert not torch.equal(model.value.weight.detach(), value_before)


def test_policy_value_heads_also_freezes_action_encoder() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = torch.nn.Linear(2, 2, bias=False)
            self.action_encoder = torch.nn.Linear(2, 2, bias=False)
            self.policy_query = torch.nn.Linear(2, 2, bias=False)
            self.action_bias = torch.nn.Linear(2, 1, bias=False)
            self.value = torch.nn.Linear(2, 1, bias=False)
            self._hidden = None

        def forward(self, input_dict, state, seq_lens):
            obs = input_dict["obs"]
            self._hidden = self.state_encoder(obs["x"])
            action_hidden = self.action_encoder(obs["x"])
            return self.policy_query(self._hidden) + action_hidden, state

        def value_function(self):
            return self.value(self._hidden).squeeze(-1)

    model = TinyModel()
    policy = SimpleNamespace(model=model)
    optimizer = build_search_optimizer(policy, lr=0.05, parameter_scope="policy_value_heads")
    state_before = model.state_encoder.weight.detach().clone()
    action_before = model.action_encoder.weight.detach().clone()
    query_before = model.policy_query.weight.detach().clone()
    value_before = model.value.weight.detach().clone()
    example = SearchReplayExample(
        observation={"x": np.array([1.0, -1.0], dtype=np.float32)},
        raw_policy=np.array([0.8, 0.2], dtype=np.float32),
        visit_counts=np.array([4.0, 28.0], dtype=np.float32),
        search_policy=np.array([0.125, 0.875], dtype=np.float32),
        q_values=np.array([-0.2, 0.3], dtype=np.float32),
        metadata=SearchReplayMetadata(
            side="us",
            trigger_categories=("forced_response",),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=(),
            raw_action=0,
            search_action=1,
            search_depth=2,
            simulation_budget=32,
            root_q_gap=0.5,
            action_flipped=True,
            root_value_target=1.0,
            complete_game=True,
        ),
    )

    distill_search_batch(
        policy,
        optimizer,
        [example],
        lambda_search=1.0,
        lambda_value=1.0,
        lambda_search_q=0.1,
        parameter_scope="policy_value_heads",
    )

    assert torch.equal(model.state_encoder.weight.detach(), state_before)
    assert torch.equal(model.action_encoder.weight.detach(), action_before)
    assert not torch.equal(model.policy_query.weight.detach(), query_before)
    assert not torch.equal(model.value.weight.detach(), value_before)


def test_replay_sampler_enforces_game_diversity_when_available() -> None:
    examples = []
    for game in ("game-a", "game-b", "game-c"):
        for step in range(4):
            examples.append(
                SearchReplayExample(
                    observation={
                        "x": np.array([1.0, -1.0], dtype=np.float32),
                        "global": np.zeros((24,), dtype=np.float32),
                    },
                    raw_policy=np.array([0.5, 0.5], dtype=np.float32),
                    visit_counts=np.array([16.0, 16.0], dtype=np.float32),
                    search_policy=np.array([0.5, 0.5], dtype=np.float32),
                    q_values=np.array([0.0, 0.0], dtype=np.float32),
                    metadata=SearchReplayMetadata(
                        side="us" if step % 2 == 0 else "ussr",
                        trigger_categories=("opponent_event",),
                        legal_actions=({"value": "a"}, {"value": "b"}),
                        hierarchical_path=(),
                        raw_action=0,
                        search_action=1,
                        search_depth=2,
                        simulation_budget=32,
                        root_q_gap=0.0,
                        action_flipped=True,
                        source_game=game,
                        source_step=step,
                    ),
                )
            )

    sampled = SearchReplaySampler(examples, seed=7).sample(
        6,
        diversity_caps={"game": 2},
    )

    counts = {
        game: sum(example.metadata.source_game == game for example in sampled)
        for game in ("game-a", "game-b", "game-c")
    }
    assert counts == {"game-a": 2, "game-b": 2, "game-c": 2}


def test_replay_sampler_enforces_era_and_side_era_minima() -> None:
    examples = []
    for side_index, side in enumerate(("us", "ussr")):
        for era, turn in (("early", 2), ("mid", 5), ("late", 9)):
            for step in range(12):
                global_features = np.zeros((24,), dtype=np.float32)
                global_features[2] = turn / 10.0
                examples.append(
                    SearchReplayExample(
                        observation={
                            "x": np.array([1.0, -1.0], dtype=np.float32),
                            "global": global_features,
                        },
                        raw_policy=np.array([0.5, 0.5], dtype=np.float32),
                        visit_counts=np.array([16.0, 16.0], dtype=np.float32),
                        search_policy=np.array([0.5, 0.5], dtype=np.float32),
                        q_values=np.array([0.0, 0.0], dtype=np.float32),
                        metadata=SearchReplayMetadata(
                            side=side,
                            trigger_categories=("opponent_event",),
                            legal_actions=({"value": "a"}, {"value": "b"}),
                            hierarchical_path=(),
                            raw_action=0,
                            search_action=1,
                            search_depth=4,
                            simulation_budget=128,
                            root_q_gap=0.1,
                            action_flipped=True,
                            source_game=f"{side}-{era}-{step}",
                            source_step=side_index * 100 + step,
                        ),
                    )
                )

    sampled = SearchReplaySampler(examples, seed=7).sample(
        48,
        min_examples_per_era=10,
        min_examples_per_side_era=4,
    )

    sampler = SearchReplaySampler(examples, seed=8)
    assert all(sum(sampler.era(example) == era for example in sampled) >= 10 for era in ("early", "mid", "late"))
    assert all(
        sum(example.metadata.side == side and sampler.era(example) == era for example in sampled) >= 4
        for side in ("us", "ussr")
        for era in ("early", "mid", "late")
    )


def test_replay_sampler_fails_closed_when_required_era_is_missing() -> None:
    examples = []
    for step in range(8):
        global_features = np.zeros((24,), dtype=np.float32)
        global_features[2] = 0.2
        examples.append(
            SearchReplayExample(
                observation={"global": global_features},
                raw_policy=np.array([1.0], dtype=np.float32),
                visit_counts=np.array([1.0], dtype=np.float32),
                search_policy=np.array([1.0], dtype=np.float32),
                q_values=np.array([0.0], dtype=np.float32),
                metadata=SearchReplayMetadata(
                    side="us",
                    trigger_categories=(),
                    legal_actions=({"value": "a"},),
                    hierarchical_path=(),
                    raw_action=0,
                    search_action=0,
                    search_depth=4,
                    simulation_budget=128,
                    root_q_gap=0.0,
                    action_flipped=False,
                ),
            )
        )

    with pytest.raises(ValueError, match="replay stratum mid"):
        SearchReplaySampler(examples, seed=7).sample(6, min_examples_per_era=1)


def test_optimizer_restore_fails_closed_for_incompatible_scope(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    narrow = torch.nn.Linear(2, 1)
    broad = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    narrow_optimizer = torch.optim.AdamW(narrow.parameters())
    broad_optimizer = torch.optim.AdamW(broad.parameters())
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_search_optimizer_state(checkpoint, {"shared_policy": narrow_optimizer})

    assert not restore_search_optimizer_state(
        checkpoint,
        {"shared_policy": broad_optimizer},
    )


def test_value_anchor_requires_frozen_reference_model() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_head = torch.nn.Linear(2, 2)
            self.value = torch.nn.Linear(2, 1)
            self.hidden = None

        def forward(self, input_dict, state, seq_lens):
            self.hidden = input_dict["obs"]["x"]
            return self.policy_head(self.hidden), state

        def value_function(self):
            return self.value(self.hidden).squeeze(-1)

    policy = SimpleNamespace(model=TinyModel())
    optimizer = build_search_optimizer(policy, lr=0.01, parameter_scope="actor_value")
    example = SearchReplayExample(
        observation={"x": np.asarray([1.0, 0.0], dtype=np.float32)},
        raw_policy=np.asarray([0.5, 0.5], dtype=np.float32),
        visit_counts=np.asarray([16.0, 16.0], dtype=np.float32),
        search_policy=np.asarray([0.5, 0.5], dtype=np.float32),
        q_values=np.asarray([0.0, 0.0], dtype=np.float32),
        metadata=SearchReplayMetadata(
            side="us",
            trigger_categories=("low_confidence",),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=(),
            raw_action=0,
            search_action=0,
            search_depth=2,
            simulation_budget=32,
            root_q_gap=0.0,
            action_flipped=False,
            root_value_target=1.0,
            complete_game=True,
        ),
    )

    with pytest.raises(ValueError, match="value_anchor_model"):
        distill_search_batch(
            policy,
            optimizer,
            [example],
            lambda_search=1.0,
            lambda_value=0.1,
            lambda_value_anchor=0.2,
            parameter_scope="actor_value",
        )


def test_head_only_distillation_freezes_shared_encoder() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.policy_query = torch.nn.Linear(2, 2, bias=False)
            self.action_bias = torch.nn.Linear(2, 1, bias=False)
            self.value = torch.nn.Linear(2, 1, bias=False)

        def forward(self, input_dict, state, seq_lens):
            hidden = self.encoder(input_dict["obs"]["x"])
            return self.policy_query(hidden), state

    model = TinyModel()
    policy = SimpleNamespace(model=model)
    optimizer = build_search_optimizer(policy, lr=0.1, parameter_scope="head")
    encoder_before = model.encoder.weight.detach().clone()
    query_before = model.policy_query.weight.detach().clone()
    metadata = SearchReplayMetadata(
        side="us",
        trigger_categories=("opponent_event",),
        legal_actions=({"value": "a"}, {"value": "b"}),
        hierarchical_path=({"action": "b", "level": "card"},),
        raw_action=0,
        search_action=1,
        search_depth=2,
        simulation_budget=32,
        root_q_gap=0.3,
        action_flipped=True,
    )
    examples = [
        SearchReplayExample(
            observation={"x": np.array([1.0, -1.0], dtype=np.float32)},
            raw_policy=np.array([0.8, 0.2], dtype=np.float32),
            visit_counts=np.array([4.0, 28.0], dtype=np.float32),
            search_policy=np.array([0.125, 0.875], dtype=np.float32),
            q_values=np.array([-0.2, 0.1], dtype=np.float32),
            metadata=metadata,
        )
    ]

    metrics = distill_search_batch(
        policy,
        optimizer,
        examples,
        lambda_search=1.0,
        lambda_anchor=0.5,
        parameter_scope="head",
    )

    assert metrics["search_distill/anchor_cross_entropy"] > 0.0
    assert torch.equal(model.encoder.weight.detach(), encoder_before)
    assert not torch.equal(model.policy_query.weight.detach(), query_before)


def test_distillation_supports_separate_general_anchor_batch() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_query = torch.nn.Linear(2, 2, bias=False)

        def forward(self, input_dict, state, seq_lens):
            return self.policy_query(input_dict["obs"]["x"]), state

    def example(x, raw, search) -> SearchReplayExample:
        metadata = SearchReplayMetadata(
            side="us",
            trigger_categories=("opponent_event",),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=({"action": "b", "level": "card"},),
            raw_action=int(np.argmax(raw)),
            search_action=int(np.argmax(search)),
            search_depth=2,
            simulation_budget=32,
            root_q_gap=0.1,
            action_flipped=True,
        )
        return SearchReplayExample(
            observation={"x": np.asarray(x, dtype=np.float32)},
            raw_policy=np.asarray(raw, dtype=np.float32),
            visit_counts=np.asarray(search, dtype=np.float32) * 32,
            search_policy=np.asarray(search, dtype=np.float32),
            q_values=np.zeros(2, dtype=np.float32),
            metadata=metadata,
        )

    policy = SimpleNamespace(model=TinyModel())
    optimizer = build_search_optimizer(policy, lr=0.01, parameter_scope="head")
    teacher = example([1, 0], [0.9, 0.1], [0.1, 0.9])
    anchor = example([0, 1], [0.8, 0.2], [0.8, 0.2])

    metrics = distill_search_batch(
        policy,
        optimizer,
        [teacher],
        anchor_examples=[anchor],
        lambda_search=1.0,
        lambda_anchor=1.0,
        parameter_scope="head",
    )

    assert metrics["search_distill/batch_examples"] == 1.0
    assert metrics["search_distill/anchor_batch_examples"] == 1.0
    assert metrics["search_distill/anchor_cross_entropy"] > 0.0


def test_distillation_supports_separate_complete_root_value_batch() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_query = torch.nn.Linear(2, 2, bias=False)
            self.value = torch.nn.Linear(2, 1, bias=False)
            self.hidden = None

        def forward(self, input_dict, state, seq_lens):
            self.hidden = input_dict["obs"]["x"]
            return self.policy_query(self.hidden), state

        def value_function(self):
            return self.value(self.hidden).squeeze(-1)

    def example(x, *, flipped, value_target) -> SearchReplayExample:
        metadata = SearchReplayMetadata(
            side="us",
            trigger_categories=("opponent_event",),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=(),
            raw_action=0,
            search_action=1 if flipped else 0,
            search_depth=2,
            simulation_budget=32,
            root_q_gap=0.2 if flipped else 0.0,
            action_flipped=flipped,
            root_value_target=value_target,
            complete_game=value_target is not None,
        )
        return SearchReplayExample(
            observation={"x": np.asarray(x, dtype=np.float32)},
            raw_policy=np.asarray([0.8, 0.2], dtype=np.float32),
            visit_counts=np.asarray([4.0, 28.0] if flipped else [28.0, 4.0], dtype=np.float32),
            search_policy=np.asarray([0.125, 0.875] if flipped else [0.875, 0.125], dtype=np.float32),
            q_values=np.asarray([-0.2, 0.2], dtype=np.float32),
            metadata=metadata,
        )

    policy = SimpleNamespace(model=TinyModel())
    optimizer = build_search_optimizer(policy, lr=0.01, parameter_scope="policy_value_heads")
    policy_example = example([1.0, 0.0], flipped=True, value_target=None)
    value_example = example([0.0, 1.0], flipped=False, value_target=-1.0)

    metrics = distill_search_batch(
        policy,
        optimizer,
        [policy_example],
        value_examples=[value_example],
        lambda_search=1.0,
        lambda_value=0.1,
        parameter_scope="policy_value_heads",
    )

    assert metrics["search_distill/batch_examples"] == 1.0
    assert metrics["search_distill/value_examples"] == 1.0
    assert metrics["search_distill/separate_value_batch"] == 1.0


def test_value_anchor_without_terminal_loss_uses_policy_batch_values() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_query = torch.nn.Linear(2, 2, bias=False)
            self.value = torch.nn.Linear(2, 1, bias=False)
            self.hidden = None

        def forward(self, input_dict, state, seq_lens):
            self.hidden = input_dict["obs"]["x"]
            return self.policy_query(self.hidden), state

        def value_function(self):
            return self.value(self.hidden).squeeze(-1)

    model = TinyModel()
    anchor_model = TinyModel()
    anchor_model.load_state_dict(model.state_dict())
    policy = SimpleNamespace(model=model)
    optimizer = build_search_optimizer(policy, lr=0.01, parameter_scope="policy_value_heads")
    example = SearchReplayExample(
        observation={"x": np.asarray([1.0, 0.0], dtype=np.float32)},
        raw_policy=np.asarray([0.8, 0.2], dtype=np.float32),
        visit_counts=np.asarray([4.0, 28.0], dtype=np.float32),
        search_policy=np.asarray([0.125, 0.875], dtype=np.float32),
        q_values=np.asarray([-0.2, 0.2], dtype=np.float32),
        metadata=SearchReplayMetadata(
            side="us",
            trigger_categories=("opponent_event",),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=(),
            raw_action=0,
            search_action=1,
            search_depth=2,
            simulation_budget=32,
            root_q_gap=0.4,
            action_flipped=True,
        ),
    )

    metrics = distill_search_batch(
        policy,
        optimizer,
        [example],
        lambda_search=1.0,
        lambda_value_anchor=1.0,
        value_anchor_model=anchor_model,
        parameter_scope="policy_value_heads",
    )

    assert metrics["search_distill/value_anchor_mse"] == 0.0


def test_search_policy_mix_forms_local_parent_teacher_target() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_head = torch.nn.Linear(2, 2, bias=False)

        def forward(self, input_dict, state, seq_lens):
            return self.policy_head(input_dict["obs"]["x"]), state

    example = SearchReplayExample(
        observation={"x": np.asarray([1.0, 0.0], dtype=np.float32)},
        raw_policy=np.asarray([0.9, 0.1], dtype=np.float32),
        visit_counts=np.asarray([3.2, 28.8], dtype=np.float32),
        search_policy=np.asarray([0.1, 0.9], dtype=np.float32),
        q_values=np.asarray([-0.1, 0.2], dtype=np.float32),
        metadata=SearchReplayMetadata(
            side="us",
            trigger_categories=("opponent_event",),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=(),
            raw_action=0,
            search_action=1,
            search_depth=2,
            simulation_budget=32,
            root_q_gap=0.3,
            action_flipped=True,
        ),
    )
    model = TinyModel()
    with torch.no_grad():
        model.policy_head.weight.zero_()
    policy = SimpleNamespace(model=model)
    optimizer = build_search_optimizer(policy, lr=0.01)

    metrics = distill_search_batch(
        policy,
        optimizer,
        [example],
        lambda_search=1.0,
        search_policy_mix=0.25,
    )

    # The mixed target is [0.7, 0.3], whose CE against uniform is log(2).
    assert metrics["search_distill/policy_cross_entropy"] == pytest.approx(np.log(2.0))
    assert metrics["search_distill/search_policy_mix"] == 0.25

    with pytest.raises(ValueError, match="search_policy_mix"):
        distill_search_batch(
            policy,
            optimizer,
            [example],
            lambda_search=1.0,
            search_policy_mix=1.1,
        )


def test_search_replay_priority_rewards_q_disagreement_and_allocation() -> None:
    def example(q_advantage: float, level: str) -> SearchReplayExample:
        metadata = SearchReplayMetadata(
            side="us",
            trigger_categories=("opponent_event",),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=({"action": "b", "level": level},),
            raw_action=0,
            search_action=1,
            search_depth=2,
            simulation_budget=32,
            root_q_gap=q_advantage,
            action_flipped=True,
            q_advantage_over_raw=q_advantage,
        )
        return SearchReplayExample(
            observation={"action_mask": np.ones(2, dtype=np.float32)},
            raw_policy=np.array([0.8, 0.2], dtype=np.float32),
            visit_counts=np.array([4.0, 28.0], dtype=np.float32),
            search_policy=np.array([0.125, 0.875], dtype=np.float32),
            q_values=np.array([-0.2, 0.1], dtype=np.float32),
            metadata=metadata,
        )

    low = example(0.01, "card")
    high_allocation = example(0.20, "influence_allocation")
    sampler = SearchReplaySampler([low, high_allocation])

    assert sampler.priority(high_allocation) > sampler.priority(low)


def test_side_policy_weight_applies_after_side_balancing() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_query = torch.nn.Linear(2, 2, bias=False)

        def forward(self, input_dict, state, seq_lens):
            return self.policy_query(input_dict["obs"]["x"]), state

    examples = []
    for side in ("us", "ussr"):
        examples.append(
            SearchReplayExample(
                observation={"x": np.asarray([1.0, -1.0], dtype=np.float32)},
                raw_policy=np.asarray([0.8, 0.2], dtype=np.float32),
                visit_counts=np.asarray([4.0, 28.0], dtype=np.float32),
                search_policy=np.asarray([0.125, 0.875], dtype=np.float32),
                q_values=np.asarray([-0.2, 0.3], dtype=np.float32),
                metadata=SearchReplayMetadata(
                    side=side,
                    trigger_categories=("forced_response",),
                    legal_actions=({"value": "a"}, {"value": "b"}),
                    hierarchical_path=(),
                    raw_action=0,
                    search_action=1,
                    search_depth=2,
                    simulation_budget=32,
                    root_q_gap=0.5,
                    action_flipped=True,
                ),
            )
        )
    policy = SimpleNamespace(model=TinyModel())
    optimizer = build_search_optimizer(policy, lr=0.01, parameter_scope="head")
    metrics = distill_search_batch(
        policy,
        optimizer,
        examples,
        lambda_search=1.0,
        balance_sides=True,
        us_policy_weight=1.0,
        ussr_policy_weight=1.5,
        parameter_scope="head",
    )

    assert metrics["search_distill/us_sample_weight_fraction"] == pytest.approx(0.4)
    assert metrics["search_distill/ussr_sample_weight_fraction"] == pytest.approx(0.6)


def test_joint_sampler_guarantees_side_balanced_terminal_value_roots() -> None:
    examples = []
    for index in range(20):
        side = "us" if index % 2 == 0 else "ussr"
        examples.append(
            SearchReplayExample(
                observation={"action_mask": np.ones(2, dtype=np.float32)},
                raw_policy=np.array([0.5, 0.5], dtype=np.float32),
                visit_counts=np.array([16.0, 16.0], dtype=np.float32),
                search_policy=np.array([0.5, 0.5], dtype=np.float32),
                q_values=np.array([0.0, 0.0], dtype=np.float32),
                metadata=SearchReplayMetadata(
                    side=side,
                    trigger_categories=("scoring_card",),
                    legal_actions=({"value": "a"}, {"value": "b"}),
                    hierarchical_path=(),
                    raw_action=0,
                    search_action=0,
                    search_depth=2,
                    simulation_budget=32,
                    root_q_gap=0.0,
                    action_flipped=False,
                    root_value_target=(1.0 if side == "us" else -1.0) if index < 8 else None,
                    complete_game=True,
                ),
            )
        )
    batch = SearchReplaySampler(examples, seed=7).sample(12, min_value_examples=6)
    value_examples = [example for example in batch if example.metadata.root_value_target is not None]

    assert len(value_examples) >= 6
    assert {example.metadata.side for example in value_examples} == {"us", "ussr"}


def test_in_loop_distillation_uses_independent_complete_value_pool() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_query = torch.nn.Linear(2, 2, bias=False)
            self.value = torch.nn.Linear(2, 1, bias=False)
            self.hidden = None

        def forward(self, input_dict, state, seq_lens):
            self.hidden = input_dict["obs"]["x"]
            return self.policy_query(self.hidden), state

        def value_function(self):
            return self.value(self.hidden).squeeze(-1)

    def example(*, flipped: bool, terminal: float | None) -> SearchReplayExample:
        return SearchReplayExample(
            observation={"x": np.asarray([1.0, -1.0], dtype=np.float32)},
            raw_policy=np.asarray([0.8, 0.2], dtype=np.float32),
            visit_counts=np.asarray([4.0, 28.0], dtype=np.float32),
            search_policy=np.asarray([0.125, 0.875], dtype=np.float32),
            q_values=np.asarray([-0.2, 0.3], dtype=np.float32),
            metadata=SearchReplayMetadata(
                side="us",
                trigger_categories=("forced_response",),
                legal_actions=({"value": "a"}, {"value": "b"}),
                hierarchical_path=(),
                raw_action=0,
                search_action=1,
                search_depth=2,
                simulation_budget=32,
                root_q_gap=0.5,
                action_flipped=flipped,
                root_value_target=terminal,
                complete_game=terminal is not None,
            ),
        )

    policy = SimpleNamespace(model=TinyModel())
    algo = SimpleNamespace(get_policy=lambda _policy_id: policy)
    optimizer = build_search_optimizer(policy, lr=0.01, parameter_scope="policy_value_heads")
    metrics = run_search_distillation_updates(
        algo,
        SearchReplaySampler([example(flipped=True, terminal=None)]),
        SearchReplaySampler([example(flipped=False, terminal=1.0)]),
        {"shared_policy": optimizer},
        {"shared_policy": "us"},
        updates=1,
        batch_size=1,
        lambda_search=1.0,
        lambda_value=0.1,
        max_grad_norm=1.0,
        parameter_scope="policy_value_heads",
        min_value_examples_per_batch=1,
    )

    assert metrics["shared_policy/search_distill/separate_value_batch"] == 1.0
    assert metrics["shared_policy/search_distill/value_examples"] == 1.0


def test_action_head_scope_includes_action_encoder_but_not_state_encoder() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_encoder = torch.nn.Linear(2, 2)
            self.state_encoder = torch.nn.Linear(2, 2)
            self.policy_query = torch.nn.Linear(2, 2)
            self.action_bias = torch.nn.Linear(2, 1)
            self.value = torch.nn.Linear(2, 1)

    model = TinyModel()
    parameters = set(actor_parameters(model, scope="action_head"))

    assert model.action_encoder.weight in parameters
    assert model.policy_query.weight in parameters
    assert model.action_bias.weight in parameters
    assert model.state_encoder.weight not in parameters
    assert model.value.weight not in parameters


def test_source_root_cap_preserves_hierarchy_coverage() -> None:
    def example(step: int, level: str, q: float) -> SearchReplayExample:
        metadata = SearchReplayMetadata(
            source_game="game",
            source_step=step,
            side="us",
            hierarchy_level=level,
            trigger_categories=("opponent_event",),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=({"action": "b", "level": level},),
            raw_action=0,
            search_action=1,
            search_depth=4,
            simulation_budget=128,
            root_q_gap=q,
            action_flipped=True,
            q_advantage_over_raw=q,
            teacher_action_visits=8,
            teacher_visit_share=0.75,
        )
        return SearchReplayExample(
            observation={"action_mask": np.ones(2, dtype=np.float32)},
            raw_policy=np.array([0.8, 0.2], dtype=np.float32),
            visit_counts=np.array([4.0, 12.0], dtype=np.float32),
            search_policy=np.array([0.25, 0.75], dtype=np.float32),
            q_values=np.array([0.0, q], dtype=np.float32),
            metadata=metadata,
        )

    examples = [
        example(10, "card", 0.4),
        example(10, "card", 0.3),
        example(10, "mode", 0.2),
        example(10, "forced_response", 0.1),
        example(11, "card", 0.05),
    ]

    selected = limit_examples_per_source_root(examples, 3)

    assert len(selected) == 4
    root_ten = [item for item in selected if item.metadata.source_step == 10]
    assert {item.metadata.hierarchy_level for item in root_ten} == {
        "card",
        "mode",
        "forced_response",
    }


def test_search_distillation_can_equalize_total_side_weight() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_head = torch.nn.Linear(2, 2, bias=False)

        def forward(self, input_dict, state, seq_lens):
            return self.policy_head(input_dict["obs"]["x"]), state

    def example(side: str, x: tuple[float, float]) -> SearchReplayExample:
        return SearchReplayExample(
            observation={"x": np.asarray(x, dtype=np.float32)},
            raw_policy=np.asarray([0.8, 0.2], dtype=np.float32),
            visit_counts=np.asarray([4.0, 28.0], dtype=np.float32),
            search_policy=np.asarray([0.125, 0.875], dtype=np.float32),
            q_values=np.asarray([-0.2, 0.1], dtype=np.float32),
            metadata=SearchReplayMetadata(
                side=side,
                trigger_categories=("forced_response",),
                legal_actions=({"value": "a"}, {"value": "b"}),
                hierarchical_path=({"action": "b", "level": "forced_response"},),
                raw_action=0,
                search_action=1,
                search_depth=2,
                simulation_budget=32,
                root_q_gap=0.3,
                action_flipped=True,
            ),
        )

    policy = SimpleNamespace(model=TinyModel())
    optimizer = build_search_optimizer(policy, lr=0.01)
    metrics = distill_search_batch(
        policy,
        optimizer,
        [
            example("us", (1.0, 0.0)),
            example("ussr", (0.0, 1.0)),
            example("ussr", (1.0, 1.0)),
        ],
        lambda_search=1.0,
        balance_sides=True,
    )

    assert metrics["search_distill/side_balanced"] == 1.0
    assert metrics["search_distill/us_sample_weight_fraction"] == pytest.approx(0.5)
    assert metrics["search_distill/ussr_sample_weight_fraction"] == pytest.approx(0.5)


def test_pairwise_q_ranking_moves_actor_toward_better_visited_action() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_head = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.zeros_(self.policy_head.weight)

        def forward(self, input_dict, state, seq_lens):
            return self.policy_head(input_dict["obs"]["x"]), state

    model = TinyModel()
    policy = SimpleNamespace(model=model)
    optimizer = build_search_optimizer(policy, lr=0.1)
    example = SearchReplayExample(
        observation={
            "x": np.asarray([1.0, 0.0], dtype=np.float32),
            "action_mask": np.ones(2, dtype=np.float32),
        },
        raw_policy=np.asarray([0.8, 0.2], dtype=np.float32),
        visit_counts=np.asarray([12.0, 8.0], dtype=np.float32),
        search_policy=np.asarray([0.6, 0.4], dtype=np.float32),
        q_values=np.asarray([-0.2, 0.2], dtype=np.float32),
        metadata=SearchReplayMetadata(
            side="us",
            trigger_categories=("influence_allocation",),
            legal_actions=({"value": "bad"}, {"value": "better"}),
            hierarchical_path=({"action": "better", "level": "influence_allocation"},),
            raw_action=0,
            search_action=1,
            search_depth=4,
            simulation_budget=128,
            root_q_gap=0.4,
            action_flipped=True,
        ),
    )

    metrics = distill_search_batch(
        policy,
        optimizer,
        [example],
        lambda_search=0.0,
        lambda_pairwise=1.0,
    )

    logits, _state = model(
        {"obs": {key: torch.as_tensor(value).unsqueeze(0) for key, value in example.observation.items()}},
        [],
        None,
    )
    assert metrics["search_distill/pairwise_pairs"] == 1.0
    assert metrics["search_distill/pairwise_ranking_loss"] > 0.0
    assert logits[0, 1] > logits[0, 0]


def test_value_head_scope_updates_only_value_parameters() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.policy_head = torch.nn.Linear(2, 2, bias=False)
            self.value = torch.nn.Linear(2, 1, bias=False)
            self._hidden = None

        def forward(self, input_dict, state, seq_lens):
            self._hidden = self.encoder(input_dict["obs"]["x"])
            return self.policy_head(self._hidden), state

        def value_function(self):
            return self.value(self._hidden).squeeze(-1)

    model = TinyModel()
    policy = SimpleNamespace(model=model)
    optimizer = build_search_optimizer(policy, lr=0.1, parameter_scope="value_head")
    encoder_before = model.encoder.weight.detach().clone()
    actor_before = model.policy_head.weight.detach().clone()
    value_before = model.value.weight.detach().clone()
    example = SearchReplayExample(
        observation={"x": np.asarray([1.0, -1.0], dtype=np.float32)},
        raw_policy=np.asarray([0.5, 0.5], dtype=np.float32),
        visit_counts=np.asarray([1.0, 1.0], dtype=np.float32),
        search_policy=np.asarray([0.5, 0.5], dtype=np.float32),
        q_values=np.asarray([0.0, 0.0], dtype=np.float32),
        metadata=SearchReplayMetadata(
            side="us",
            trigger_categories=(),
            legal_actions=({"value": "a"}, {"value": "b"}),
            hierarchical_path=(),
            raw_action=0,
            search_action=0,
            search_depth=2,
            simulation_budget=32,
            root_q_gap=0.0,
            action_flipped=False,
            root_value_target=1.0,
            source_game="game-us-win",
            complete_game=True,
            game_winner="us",
        ),
    )

    distill_search_batch(
        policy,
        optimizer,
        [example],
        value_examples=[example],
        lambda_search=0.0,
        lambda_value=1.0,
        parameter_scope="value_head",
    )

    assert torch.equal(model.encoder.weight.detach(), encoder_before)
    assert torch.equal(model.policy_head.weight.detach(), actor_before)
    assert not torch.equal(model.value.weight.detach(), value_before)


def test_value_adapter_scope_selects_only_residual_adapter() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.value = torch.nn.Linear(2, 1)
            self.value_adapter = torch.nn.Sequential(
                torch.nn.Linear(2, 2),
                torch.nn.ReLU(),
                torch.nn.Linear(2, 1),
            )

    model = TinyModel()
    selected = set(search_training_parameters(model, scope="value_adapter"))
    assert selected == set(model.value_adapter.parameters())
    assert selected.isdisjoint(set(model.encoder.parameters()))
    assert selected.isdisjoint(set(model.value.parameters()))


def test_value_tower_scopes_do_not_select_actor_or_shared_encoder() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = torch.nn.Linear(2, 2)
            self.value_state_encoder = torch.nn.Linear(2, 2)
            self.policy_query = torch.nn.Linear(2, 2)
            self.value = torch.nn.Linear(2, 1)

    model = TinyModel()
    tower = set(search_training_parameters(model, scope="value_tower"))
    tower_head = set(search_training_parameters(model, scope="value_tower_head"))
    assert tower == set(model.value_state_encoder.parameters())
    assert tower_head == tower | set(model.value.parameters())
    assert tower_head.isdisjoint(set(model.state_encoder.parameters()))
    assert tower_head.isdisjoint(set(model.policy_query.parameters()))


def test_game_outcome_value_sampler_balances_four_strata() -> None:
    examples = []
    for side in ("us", "ussr"):
        for outcome, target in (("win", 1.0), ("loss", -1.0)):
            for game_index in range(2):
                examples.append(
                    SearchReplayExample(
                        observation={"x": np.asarray([float(game_index)], dtype=np.float32)},
                        raw_policy=np.asarray([1.0], dtype=np.float32),
                        visit_counts=np.asarray([1.0], dtype=np.float32),
                        search_policy=np.asarray([1.0], dtype=np.float32),
                        q_values=np.asarray([0.0], dtype=np.float32),
                        metadata=SearchReplayMetadata(
                            side=side,
                            trigger_categories=(),
                            legal_actions=({"value": "a"},),
                            hierarchical_path=(),
                            raw_action=0,
                            search_action=0,
                            search_depth=2,
                            simulation_budget=32,
                            root_q_gap=0.0,
                            action_flipped=False,
                            root_value_target=target,
                            source_game=f"{side}-{outcome}-{game_index}",
                            complete_game=True,
                            game_winner=side if outcome == "win" else ("ussr" if side == "us" else "us"),
                        ),
                    )
                )
    sampled = GameOutcomeBalancedValueSampler(examples, seed=7).sample(400)
    counts: dict[tuple[str, str], int] = {}
    for example in sampled:
        outcome = "win" if example.metadata.root_value_target > 0 else "loss"
        key = (example.metadata.side, outcome)
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        ("us", "loss"): 100,
        ("us", "win"): 100,
        ("ussr", "loss"): 100,
        ("ussr", "win"): 100,
    }


def test_game_outcome_value_sampler_supports_stratum_weights() -> None:
    examples = []
    for side in ("us", "ussr"):
        for outcome, target in (("win", 1.0), ("loss", -1.0)):
            examples.append(
                SearchReplayExample(
                    observation={"x": np.asarray([0.0], dtype=np.float32)},
                    raw_policy=np.asarray([1.0], dtype=np.float32),
                    visit_counts=np.asarray([1.0], dtype=np.float32),
                    search_policy=np.asarray([1.0], dtype=np.float32),
                    q_values=np.asarray([0.0], dtype=np.float32),
                    metadata=SearchReplayMetadata(
                        side=side,
                        trigger_categories=(),
                        legal_actions=({"value": "a"},),
                        hierarchical_path=(),
                        raw_action=0,
                        search_action=0,
                        search_depth=2,
                        simulation_budget=32,
                        root_q_gap=0.0,
                        action_flipped=False,
                        root_value_target=target,
                        source_game=f"{side}-{outcome}",
                        complete_game=True,
                    ),
                )
            )
    sampled = GameOutcomeBalancedValueSampler(
        examples,
        seed=11,
        stratum_weights={("ussr", "loss"): 3.0},
    ).sample(600)
    counts: dict[tuple[str, str], int] = {}
    for example in sampled:
        outcome = "win" if example.metadata.root_value_target > 0 else "loss"
        key = (example.metadata.side, outcome)
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        ("us", "loss"): 100,
        ("us", "win"): 100,
        ("ussr", "loss"): 300,
        ("ussr", "win"): 100,
    }


def test_calibrated_value_loss_is_reported_and_updates_value_head() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_head = torch.nn.Linear(1, 1)
            self.value = torch.nn.Linear(1, 1)
            self._x = None

        def forward(self, input_dict, state, seq_lens):
            self._x = input_dict["obs"]["x"]
            return self.policy_head(self._x), state

        def value_function(self):
            return self.value(self._x).squeeze(-1)

    example = SearchReplayExample(
        observation={"x": np.asarray([1.0], dtype=np.float32)},
        raw_policy=np.asarray([1.0], dtype=np.float32),
        visit_counts=np.asarray([1.0], dtype=np.float32),
        search_policy=np.asarray([1.0], dtype=np.float32),
        q_values=np.asarray([0.0], dtype=np.float32),
        metadata=SearchReplayMetadata(
            side="us",
            trigger_categories=(),
            legal_actions=({"value": "a"},),
            hierarchical_path=(),
            raw_action=0,
            search_action=0,
            search_depth=2,
            simulation_budget=32,
            root_q_gap=0.0,
            action_flipped=False,
            root_value_target=1.0,
            source_game="game-us-win",
            complete_game=True,
            game_winner="us",
        ),
    )
    policy = SimpleNamespace(model=TinyModel())
    optimizer = build_search_optimizer(policy, lr=0.01, parameter_scope="value_head")
    metrics = distill_search_batch(
        policy,
        optimizer,
        [example],
        value_examples=[example],
        lambda_search=0.0,
        lambda_value=1.0,
        value_loss_calibrated=True,
        parameter_scope="value_head",
    )
    assert metrics["search_distill/value_loss_calibrated"] == 1.0
