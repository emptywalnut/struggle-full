from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from struggle_ai.search_replay import SearchReplayExample


def limit_examples_per_source_root(
    examples: list[SearchReplayExample],
    max_per_root: int | None,
) -> list[SearchReplayExample]:
    """Bound replay concentration while retaining hierarchy-level coverage."""

    if max_per_root is None or int(max_per_root) <= 0:
        return list(examples)
    cap = int(max_per_root)
    grouped: dict[tuple[str, int], list[SearchReplayExample]] = defaultdict(list)
    for example in examples:
        grouped[(example.metadata.source_game, example.metadata.source_step)].append(example)

    def quality(example: SearchReplayExample) -> tuple[float, float, float, int]:
        return (
            float(example.metadata.q_advantage_over_raw),
            float(example.metadata.teacher_visit_share),
            float(example.metadata.teacher_action_visits),
            int(example.metadata.action_flipped),
        )

    selected: list[SearchReplayExample] = []
    for group in grouped.values():
        if len(group) <= cap:
            selected.extend(group)
            continue
        by_level: dict[str, list[SearchReplayExample]] = defaultdict(list)
        for example in group:
            by_level[str(example.metadata.hierarchy_level or "root")].append(example)
        for bucket in by_level.values():
            bucket.sort(key=quality, reverse=True)
        representatives = sorted(
            (bucket[0] for bucket in by_level.values()),
            key=quality,
            reverse=True,
        )
        retained = representatives[:cap]
        retained_ids = {id(example) for example in retained}
        remaining = sorted(
            (example for example in group if id(example) not in retained_ids),
            key=quality,
            reverse=True,
        )
        retained.extend(remaining[: max(0, cap - len(retained))])
        selected.extend(retained)
    return selected


def actor_parameters(model: Any, *, scope: str = "actor") -> list[Any]:
    """Return actor/encoder parameters while leaving critic-only tensors frozen."""
    if scope not in {"actor", "head", "action_head"}:
        raise ValueError(f"unsupported search-distillation parameter scope: {scope}")
    head_prefixes = (
        "policy_query",
        "policy_query_us",
        "policy_query_ussr",
        "action_bias",
        "action_bias_us",
        "action_bias_ussr",
    )
    return [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("value.")
        and (
            scope == "actor"
            or name.startswith(head_prefixes)
            or (scope == "action_head" and name.startswith("action_encoder"))
        )
    ]


def search_training_parameters(model: Any, *, scope: str = "actor") -> list[Any]:
    if scope == "value_tower":
        return [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("value_state_encoder.")
        ]
    if scope == "value_tower_head":
        return [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and name.startswith(("value_state_encoder.", "value."))
        ]
    if scope == "value_adapter":
        return [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("value_adapter.")
        ]
    if scope == "value_head":
        return [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("value.")
        ]
    if scope == "actor_value":
        return [parameter for parameter in model.parameters() if parameter.requires_grad]
    if scope == "actor_value_heads":
        prefixes = (
            "action_encoder",
            "policy_query",
            "policy_query_us",
            "policy_query_ussr",
            "action_bias",
            "action_bias_us",
            "action_bias_ussr",
            "value",
        )
        return [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(prefixes)
        ]
    if scope == "policy_value_heads":
        prefixes = (
            "policy_query",
            "policy_query_us",
            "policy_query_ussr",
            "action_bias",
            "action_bias_us",
            "action_bias_ussr",
            "value",
        )
        return [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(prefixes)
        ]
    return actor_parameters(model, scope=scope)


class GameOutcomeBalancedValueSampler:
    """Sample complete roots by configured side/outcome weights and source game."""

    def __init__(
        self,
        examples: list[SearchReplayExample],
        *,
        seed: int = 1,
        stratum_weights: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.by_stratum_game: dict[
            tuple[str, str], dict[str, list[SearchReplayExample]]
        ] = defaultdict(lambda: defaultdict(list))
        for example in examples:
            target = example.metadata.root_value_target
            if target is None or not example.metadata.complete_game:
                continue
            outcome = "win" if target > 0.0 else "loss" if target < 0.0 else "draw"
            game = str(example.metadata.source_game or "unknown")
            self.by_stratum_game[(str(example.metadata.side), outcome)][game].append(example)
        self.strata = sorted(self.by_stratum_game)
        if not self.strata:
            raise ValueError("game-balanced value replay is empty")
        configured_weights = dict(stratum_weights or {})
        self.stratum_weights = {
            stratum: float(configured_weights.get(stratum, 1.0)) for stratum in self.strata
        }
        if any(weight <= 0.0 for weight in self.stratum_weights.values()):
            raise ValueError("value replay stratum weights must be positive")

    def sample(self, size: int) -> list[SearchReplayExample]:
        selected: list[SearchReplayExample] = []
        sample_size = max(1, int(size))
        total_weight = sum(self.stratum_weights.values())
        exact_counts = {
            stratum: sample_size * self.stratum_weights[stratum] / total_weight
            for stratum in self.strata
        }
        counts = {stratum: int(exact_counts[stratum]) for stratum in self.strata}
        remainder_order = list(self.strata)
        self.rng.shuffle(remainder_order)
        remainder_order.sort(
            key=lambda stratum: exact_counts[stratum] - counts[stratum], reverse=True
        )
        for stratum in remainder_order[: sample_size - sum(counts.values())]:
            counts[stratum] += 1
        for stratum in self.strata:
            by_game = self.by_stratum_game[stratum]
            games = tuple(sorted(by_game))
            for _ in range(counts[stratum]):
                game = self.rng.choice(games)
                selected.append(self.rng.choice(by_game[game]))
        self.rng.shuffle(selected)
        return selected


class SearchReplaySampler:
    """Side- and trigger-aware sampler for the dedicated joseki buffer."""

    def __init__(self, examples: list[SearchReplayExample], *, seed: int = 1) -> None:
        if not examples:
            raise ValueError("search replay is empty")
        self.examples = list(examples)
        self.rng = random.Random(seed)
        self.by_side: dict[str, list[SearchReplayExample]] = defaultdict(list)
        self.trigger_frequency: dict[str, int] = defaultdict(int)
        for example in self.examples:
            self.by_side[example.metadata.side].append(example)
            for trigger in example.metadata.trigger_categories:
                self.trigger_frequency[trigger] += 1

    def priority(self, example: SearchReplayExample) -> float:
        rarity = max(
            (len(self.examples) / max(1, self.trigger_frequency[trigger]) for trigger in example.metadata.trigger_categories),
            default=1.0,
        )
        q_advantage = max(0.0, float(example.metadata.q_advantage_over_raw))
        q_weight = 1.0 + min(3.0, q_advantage / 0.05)
        levels = {str(entry.get("level") or "") for entry in example.metadata.hierarchical_path}
        hierarchy_weight = 1.5 if (
            example.metadata.hierarchy_level in {
                "influence_allocation",
                "coup_target",
                "realignment_target",
            }
            or levels & {
            "influence_allocation",
            "coup_target",
            "realignment_target",
            }
        ) else 1.0
        strategic_weight = 1.25 if set(example.metadata.trigger_categories) & {
            "forced_response",
            "opponent_event",
            "scoring_card",
            "scoring_pressure",
        } else 1.0
        return (
            (2.0 if example.metadata.action_flipped else 1.0)
            * min(4.0, rarity**0.5)
            * q_weight
            * hierarchy_weight
            * strategic_weight
        )

    @staticmethod
    def era(example: SearchReplayExample) -> str:
        """Return the public-information war era for replay stratification."""

        global_features = np.asarray(example.observation.get("global", ()), dtype=np.float32)
        turn = int(round(float(global_features[2]) * 10.0)) if global_features.size > 2 else 0
        if turn <= 3:
            return "early"
        if turn <= 7:
            return "mid"
        return "late"

    @staticmethod
    def diversity_buckets(example: SearchReplayExample) -> dict[str, tuple[str, ...]]:
        """Return replay strata using only information-state data."""

        metadata = example.metadata
        action: dict[str, Any] = {}
        if 0 <= int(metadata.search_action) < len(metadata.legal_actions):
            action = dict(metadata.legal_actions[int(metadata.search_action)])
        global_features = np.asarray(example.observation.get("global", ()), dtype=np.float32)
        turn = int(round(float(global_features[2]) * 10.0)) if global_features.size > 2 else 0
        card = str(action.get("card") or "")
        if not card:
            prompt = str(action.get("prompt") or "").lower()
            if "card" in prompt or "headline" in prompt:
                card = str(action.get("value") or action.get("label") or "")
        if not card:
            for path_entry in metadata.hierarchical_path:
                if str(path_entry.get("level") or "") == "card":
                    card = str(path_entry.get("action") or "")
                    if card:
                        break
        region = str(action.get("region") or "")
        if not region:
            selector = str(action.get("selector") or "").lstrip(".")
            region_names = (
                "europe", "asia", "seasia", "mideast", "africa", "camerica", "samerica"
            )
            region = next((name for name in region_names if name in selector), "")
        if not region and str(action.get("decision") or "") == "country_click":
            region = f"country:{action.get('value') or action.get('label') or 'unknown'}"
        return {
            "game": (str(metadata.source_game or "unknown"),),
            "turn": (f"{metadata.side}:T{turn}",),
            "card": (card,) if card else (),
            "region": (region,) if region else (),
            "trigger": tuple(str(trigger) for trigger in metadata.trigger_categories),
        }

    def sample(
        self,
        batch_size: int,
        *,
        side: str | None = None,
        min_value_examples: int = 0,
        min_examples_per_era: int = 0,
        min_examples_per_side_era: int = 0,
        diversity_caps: dict[str, int] | None = None,
    ) -> list[SearchReplayExample]:
        population = self.by_side.get(side, []) if side else self.examples
        if not population:
            return []
        size = min(max(1, int(batch_size)), len(population))
        required_value = min(max(0, int(min_value_examples)), size)
        selected: list[SearchReplayExample] = []
        if required_value:
            value_by_side: dict[str, list[SearchReplayExample]] = defaultdict(list)
            for example in population:
                if example.metadata.root_value_target is not None:
                    value_by_side[str(example.metadata.side)].append(example)
            ordered_sides = [candidate for candidate in ("us", "ussr") if value_by_side[candidate]]
            while len(selected) < required_value and ordered_sides:
                made_progress = False
                for candidate_side in ordered_sides:
                    candidates = [
                        example
                        for example in value_by_side[candidate_side]
                        if id(example) not in {id(item) for item in selected}
                    ]
                    if not candidates:
                        continue
                    weights = [self.priority(example) for example in candidates]
                    selected.append(self.rng.choices(candidates, weights=weights, k=1)[0])
                    made_progress = True
                    if len(selected) >= required_value:
                        break
                if not made_progress:
                    break
            if len(selected) < required_value:
                candidates = [
                    example
                    for example in population
                    if example.metadata.root_value_target is not None
                    and id(example) not in {id(item) for item in selected}
                ]
                while candidates and len(selected) < required_value:
                    weights = [self.priority(example) for example in candidates]
                    chosen_index = self.rng.choices(
                        range(len(candidates)), weights=weights, k=1
                    )[0]
                    chosen = candidates.pop(chosen_index)
                    selected.append(chosen)

        def select_minimum(
            strata: list[tuple[str, str | None]],
            minimum: int,
        ) -> None:
            minimum = max(0, int(minimum))
            if not minimum:
                return
            if minimum * len(strata) > size:
                raise ValueError(
                    f"requested replay strata require {minimum * len(strata)} examples "
                    f"but batch size is {size}"
                )
            for era, stratum_side in strata:
                retained = sum(
                    self.era(example) == era
                    and (stratum_side is None or example.metadata.side == stratum_side)
                    for example in selected
                )
                needed = max(0, minimum - retained)
                if not needed:
                    continue
                selected_ids = {id(example) for example in selected}
                candidates = [
                    example
                    for example in population
                    if id(example) not in selected_ids
                    and self.era(example) == era
                    and (stratum_side is None or example.metadata.side == stratum_side)
                ]
                if len(candidates) < needed:
                    label = era if stratum_side is None else f"{stratum_side}:{era}"
                    raise ValueError(
                        f"replay stratum {label} has only {len(candidates) + retained} "
                        f"examples; {minimum} required"
                    )
                for _ in range(needed):
                    weights = [self.priority(example) for example in candidates]
                    chosen_index = self.rng.choices(
                        range(len(candidates)), weights=weights, k=1
                    )[0]
                    chosen = candidates.pop(chosen_index)
                    selected.append(chosen)

        eras = ["early", "mid", "late"]
        requested_sides = [side] if side else ["us", "ussr"]
        # Side-era minima run first so an era floor cannot accidentally be
        # satisfied entirely by the majority side.
        select_minimum(
            [(era, requested_side) for requested_side in requested_sides for era in eras],
            min_examples_per_side_era,
        )
        select_minimum([(era, None) for era in eras], min_examples_per_era)

        # Weighted sampling without replacement keeps rare tactical states from
        # disappearing while avoiding duplicate tensors in a small batch.
        caps = {
            str(key): int(value)
            for key, value in dict(diversity_caps or {}).items()
            if int(value) > 0
        }
        bucket_counts: dict[tuple[str, str], int] = defaultdict(int)

        def record_buckets(example: SearchReplayExample) -> None:
            for dimension, values in self.diversity_buckets(example).items():
                for value in values:
                    bucket_counts[(dimension, value)] += 1

        def allowed(example: SearchReplayExample) -> bool:
            for dimension, values in self.diversity_buckets(example).items():
                cap = caps.get(dimension)
                if cap is None:
                    continue
                if any(bucket_counts[(dimension, value)] >= cap for value in values):
                    return False
            return True

        for example in selected:
            record_buckets(example)
        selected_ids = {id(example) for example in selected}
        candidates = [example for example in population if id(example) not in selected_ids]
        candidate_weights = [self.priority(example) for example in candidates]
        for _ in range(size - len(selected)):
            eligible = [index for index, example in enumerate(candidates) if allowed(example)]
            # Caps are diversity guards, not a reason to return a short batch.
            choices = eligible or list(range(len(candidates)))
            weights = [candidate_weights[index] for index in choices]
            index = self.rng.choices(choices, weights=weights, k=1)[0]
            chosen = candidates.pop(index)
            selected.append(chosen)
            record_buckets(chosen)
            candidate_weights.pop(index)
        return selected


def build_search_optimizer(
    policy: Any,
    *,
    lr: float,
    weight_decay: float = 0.0,
    parameter_scope: str = "actor",
):
    import torch

    parameters = search_training_parameters(policy.model, scope=parameter_scope)
    if not parameters:
        raise RuntimeError("search distillation found no trainable actor parameters")
    return torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))


def distill_search_batch(
    policy: Any,
    optimizer: Any,
    examples: list[SearchReplayExample],
    *,
    anchor_examples: list[SearchReplayExample] | None = None,
    value_examples: list[SearchReplayExample] | None = None,
    lambda_search: float,
    lambda_pairwise: float = 0.0,
    lambda_anchor: float = 0.0,
    max_grad_norm: float = 1.0,
    parameter_scope: str = "actor",
    q_weight_scale: float = 0.05,
    max_q_weight: float = 4.0,
    hierarchy_weight: float = 1.25,
    balance_sides: bool = False,
    lambda_value: float = 0.0,
    value_loss_calibrated: bool = False,
    lambda_search_q: float = 0.0,
    value_anchor_model: Any | None = None,
    lambda_value_anchor: float = 0.0,
    confidence_weight_floor: float = 1.0,
    confidence_q_scale: float = 0.05,
    us_policy_weight: float = 1.0,
    ussr_policy_weight: float = 1.0,
    search_policy_mix: float = 1.0,
    pairwise_min_q_gap: float = 0.002,
    pairwise_q_scale: float = 0.02,
    pairwise_max_pairs: int = 8,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    if not examples:
        return {}
    if float(lambda_value_anchor) > 0.0 and value_anchor_model is None:
        raise ValueError("lambda_value_anchor requires a frozen value_anchor_model")
    model = policy.model
    device = next(model.parameters()).device
    observation_keys = tuple(sorted(examples[0].observation))
    for example in examples:
        if tuple(sorted(example.observation)) != observation_keys:
            raise ValueError("search distillation batch has inconsistent observation keys")
    obs = {
        key: torch.as_tensor(
            np.stack([example.observation[key] for example in examples]),
            dtype=torch.float32,
            device=device,
        )
        for key in observation_keys
    }
    if not 0.0 <= float(search_policy_mix) <= 1.0:
        raise ValueError("search_policy_mix must be in [0, 1]")
    search_targets = torch.as_tensor(
        np.stack([example.search_policy for example in examples]),
        dtype=torch.float32,
        device=device,
    )
    raw_policy_targets = torch.as_tensor(
        np.stack([example.raw_policy for example in examples]),
        dtype=torch.float32,
        device=device,
    )
    targets = (
        float(search_policy_mix) * search_targets
        + (1.0 - float(search_policy_mix)) * raw_policy_targets
    )
    def example_weight(example: SearchReplayExample) -> float:
        q_advantage = max(0.0, float(example.metadata.q_advantage_over_raw))
        q_weight = 1.0 + min(
            max(0.0, float(max_q_weight)),
            q_advantage / max(1e-6, float(q_weight_scale)),
        )
        levels = {str(entry.get("level") or "") for entry in example.metadata.hierarchical_path}
        hierarchical = float(hierarchy_weight) if (
            example.metadata.hierarchy_level in {
                "influence_allocation",
                "coup_target",
                "realignment_target",
            }
            or levels & {
            "influence_allocation",
            "coup_target",
            "realignment_target",
            }
        ) else 1.0
        confidence = float(confidence_weight_floor) + (1.0 - float(confidence_weight_floor)) * min(
            1.0,
            abs(float(example.metadata.root_q_gap)) / max(1e-6, float(confidence_q_scale)),
        )
        return (
            (2.0 if example.metadata.action_flipped else 1.0)
            * q_weight
            * hierarchical
            * confidence
        )

    sample_weights = torch.as_tensor(
        [example_weight(example) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    if balance_sides:
        side_indices: dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(examples):
            side_indices[str(example.metadata.side)].append(index)
        if len(side_indices) > 1:
            balanced_weights = sample_weights.clone()
            target_weight = sample_weights.sum() / float(len(side_indices))
            for indices in side_indices.values():
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
                side_weight = sample_weights.index_select(0, index_tensor).sum().clamp_min(1e-6)
                balanced_weights[index_tensor] *= target_weight / side_weight
            sample_weights = balanced_weights
    side_multipliers = torch.as_tensor(
        [
            float(us_policy_weight) if example.metadata.side == "us" else float(ussr_policy_weight)
            for example in examples
        ],
        dtype=torch.float32,
        device=device,
    )
    sample_weights *= side_multipliers
    model.train()
    logits, _state = model({"obs": obs}, [], None)
    policy_value_predictions = None
    if (
        float(lambda_value) > 0.0
        or float(lambda_search_q) > 0.0
        or (float(lambda_value_anchor) > 0.0 and value_examples is None)
    ):
        value_function = getattr(model, "value_function", None)
        if not callable(value_function):
            raise RuntimeError("search value training requires model.value_function()")
        policy_value_predictions = value_function().reshape(-1)
    log_probs = functional.log_softmax(logits, dim=-1)
    per_example = -(targets * log_probs).sum(dim=-1)
    policy_loss = (per_example * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
    pairwise_terms: list[Any] = []
    pairwise_weights: list[float] = []
    if float(lambda_pairwise) > 0.0:
        for row, example in enumerate(examples):
            mask = example.observation.get("action_mask")
            legal_count = (
                int(np.asarray(mask).sum())
                if mask is not None
                else len(example.metadata.legal_actions)
            )
            visits = np.asarray(example.visit_counts[:legal_count], dtype=np.float64)
            q_values = np.asarray(example.q_values[:legal_count], dtype=np.float64)
            valid = [index for index in range(legal_count) if visits[index] > 0.0]
            candidates = sorted(
                (
                    (float(q_values[better] - q_values[worse]), better, worse)
                    for better in valid
                    for worse in valid
                    if q_values[better] >= q_values[worse] + float(pairwise_min_q_gap)
                ),
                reverse=True,
            )[: max(0, int(pairwise_max_pairs))]
            for q_gap, better, worse in candidates:
                pairwise_terms.append(
                    functional.softplus(-(logits[row, better] - logits[row, worse]))
                )
                pairwise_weights.append(
                    min(4.0, q_gap / max(1e-6, float(pairwise_q_scale)))
                )
    pairwise_loss = torch.zeros((), dtype=torch.float32, device=device)
    if pairwise_terms:
        pair_tensor = torch.stack(pairwise_terms)
        pair_weights = torch.as_tensor(pairwise_weights, dtype=torch.float32, device=device)
        pairwise_loss = (pair_tensor * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
    probabilities = torch.softmax(logits, dim=-1)
    policy_entropy_per_example = -(probabilities * log_probs).sum(dim=-1)
    effective_anchor_examples = anchor_examples if anchor_examples is not None else examples
    for example in effective_anchor_examples:
        if tuple(sorted(example.observation)) != observation_keys:
            raise ValueError("search distillation anchor batch has inconsistent observation keys")
    if anchor_examples is None:
        anchor_log_probs = log_probs
    else:
        anchor_obs = {
            key: torch.as_tensor(
                np.stack([example.observation[key] for example in effective_anchor_examples]),
                dtype=torch.float32,
                device=device,
            )
            for key in observation_keys
        }
        anchor_logits, _anchor_state = model({"obs": anchor_obs}, [], None)
        anchor_log_probs = functional.log_softmax(anchor_logits, dim=-1)
    raw_targets = torch.as_tensor(
        np.stack([example.raw_policy for example in effective_anchor_examples]),
        dtype=torch.float32,
        device=device,
    )
    anchor_per_example = -(raw_targets * anchor_log_probs).sum(dim=-1)
    anchor_loss = anchor_per_example.mean()
    effective_value_examples = value_examples if value_examples is not None else examples
    for example in effective_value_examples:
        if tuple(sorted(example.observation)) != observation_keys:
            raise ValueError("search distillation value batch has inconsistent observation keys")
    if value_examples is None:
        value_predictions = policy_value_predictions
    elif float(lambda_value) > 0.0 or float(lambda_value_anchor) > 0.0:
        value_obs = {
            key: torch.as_tensor(
                np.stack([example.observation[key] for example in effective_value_examples]),
                dtype=torch.float32,
                device=device,
            )
            for key in observation_keys
        }
        model({"obs": value_obs}, [], None)
        value_predictions = model.value_function().reshape(-1)
    else:
        value_predictions = None
    value_loss = torch.zeros((), dtype=torch.float32, device=device)
    value_mask = torch.as_tensor(
        [example.metadata.root_value_target is not None for example in effective_value_examples],
        dtype=torch.bool,
        device=device,
    )
    if float(lambda_value) > 0.0:
        if not bool(value_mask.any()):
            raise ValueError("lambda_value requires complete-game root_value_target labels")
        value_targets = torch.as_tensor(
            [
                0.0 if example.metadata.root_value_target is None else example.metadata.root_value_target
                for example in effective_value_examples
            ],
            dtype=torch.float32,
            device=device,
        )
        value_loss_predictions = (
            torch.tanh(value_predictions[value_mask])
            if value_loss_calibrated
            else value_predictions[value_mask]
        )
        value_loss = functional.mse_loss(value_loss_predictions, value_targets[value_mask])
    search_q_loss = torch.zeros((), dtype=torch.float32, device=device)
    if float(lambda_search_q) > 0.0:
        q_values = torch.as_tensor(
            np.stack([example.q_values for example in examples]),
            dtype=torch.float32,
            device=device,
        )
        search_q_targets = (search_targets * q_values).sum(dim=-1)
        search_q_loss = functional.mse_loss(policy_value_predictions, search_q_targets)
    value_anchor_loss = torch.zeros((), dtype=torch.float32, device=device)
    if float(lambda_value_anchor) > 0.0:
        value_anchor_model.eval()
        with torch.no_grad():
            value_anchor_obs = {
                key: torch.as_tensor(
                    np.stack([example.observation[key] for example in effective_value_examples]),
                    dtype=torch.float32,
                    device=device,
                )
                for key in observation_keys
            }
            value_anchor_model({"obs": value_anchor_obs}, [], None)
            anchor_value_predictions = value_anchor_model.value_function().reshape(-1)
        value_anchor_loss = functional.mse_loss(value_predictions, anchor_value_predictions)
    loss = (
        float(lambda_search) * policy_loss
        + float(lambda_pairwise) * pairwise_loss
        + float(lambda_anchor) * anchor_loss
        + float(lambda_value) * value_loss
        + float(lambda_search_q) * search_q_loss
        + float(lambda_value_anchor) * value_anchor_loss
    )
    # Clear excluded-parameter gradients too. Head-only scopes deliberately
    # leave the shared encoder out of the optimizer, but autograd still reaches
    # it while producing policy/value head gradients.
    model.zero_grad(set_to_none=True)
    loss.backward()
    parameters = search_training_parameters(model, scope=parameter_scope)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
    optimizer.step()

    prediction = torch.argmax(logits.detach(), dim=-1)
    teacher = torch.argmax(search_targets, dim=-1)
    raw_action = torch.as_tensor([example.metadata.raw_action for example in examples], device=device)
    metrics = {
        "search_distill/loss": float(loss.detach().cpu()),
        "search_distill/policy_cross_entropy": float(policy_loss.detach().cpu()),
        "search_distill/pairwise_ranking_loss": float(pairwise_loss.detach().cpu()),
        "search_distill/pairwise_pairs": float(len(pairwise_terms)),
        "search_distill/policy_entropy": float(policy_entropy_per_example.detach().mean().cpu()),
        "search_distill/anchor_cross_entropy": float(anchor_loss.detach().cpu()),
        "search_distill/value_mse": float(value_loss.detach().cpu()),
        "search_distill/value_loss_calibrated": float(value_loss_calibrated),
        "search_distill/search_q_mse": float(search_q_loss.detach().cpu()),
        "search_distill/value_anchor_mse": float(value_anchor_loss.detach().cpu()),
        "search_distill/value_examples": float(value_mask.sum().detach().cpu()),
        "search_distill/separate_value_batch": float(value_examples is not None),
        "search_distill/grad_norm": float(grad_norm.detach().cpu()),
        "search_distill/teacher_accuracy": float((prediction == teacher).float().mean().cpu()),
        "search_distill/raw_teacher_agreement": float((raw_action == teacher).float().mean().cpu()),
        "search_distill/batch_examples": float(len(examples)),
        "search_distill/anchor_batch_examples": float(len(effective_anchor_examples)),
        "search_distill/flipped_fraction": float(
            sum(example.metadata.action_flipped for example in examples) / len(examples)
        ),
        "search_distill/mean_q_advantage": float(
            np.mean([max(0.0, example.metadata.q_advantage_over_raw) for example in examples])
        ),
        "search_distill/mean_sample_weight": float(sample_weights.detach().mean().cpu()),
        "search_distill/side_balanced": float(balance_sides),
        "search_distill/us_policy_weight": float(us_policy_weight),
        "search_distill/ussr_policy_weight": float(ussr_policy_weight),
        "search_distill/search_policy_mix": float(search_policy_mix),
    }
    total_sample_weight = sample_weights.detach().sum().clamp_min(1e-6)
    for side in ("us", "ussr"):
        indices = [index for index, example in enumerate(examples) if example.metadata.side == side]
        if indices:
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
            side_weight = sample_weights.detach().index_select(0, index_tensor).sum()
            metrics[f"search_distill/{side}_sample_weight_fraction"] = float(
                (side_weight / total_sample_weight).cpu()
            )
            metrics[f"search_distill/{side}_policy_entropy"] = float(
                policy_entropy_per_example.detach().index_select(0, index_tensor).mean().cpu()
            )
    if bool(value_mask.any()) and value_predictions is not None:
        selected_predictions = value_predictions.detach()[value_mask]
        selected_targets = torch.as_tensor(
            [
                float(example.metadata.root_value_target)
                for example in effective_value_examples
                if example.metadata.root_value_target is not None
            ],
            dtype=torch.float32,
            device=device,
        )
        metrics["search_distill/value_prediction_mean"] = float(selected_predictions.mean().cpu())
        metrics["search_distill/value_target_mean"] = float(selected_targets.mean().cpu())
        metrics["search_distill/value_sign_accuracy"] = float(
            (torch.sign(selected_predictions) == torch.sign(selected_targets)).float().mean().cpu()
        )
    return metrics


def run_search_distillation_updates(
    algo: Any,
    sampler: SearchReplaySampler,
    value_sampler: SearchReplaySampler | None,
    optimizers: dict[str, Any],
    policy_sides: dict[str, str | None],
    *,
    updates: int,
    batch_size: int,
    lambda_search: float,
    max_grad_norm: float,
    parameter_scope: str = "actor",
    balance_sides: bool = False,
    lambda_value: float = 0.0,
    lambda_search_q: float = 0.0,
    confidence_weight_floor: float = 1.0,
    search_policy_mix: float = 1.0,
    min_value_examples_per_batch: int = 0,
    diversity_caps: dict[str, int] | None = None,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    policy_batches: dict[str, int] = defaultdict(int)
    completed = 0
    for _ in range(max(0, int(updates))):
        for policy_id, side in policy_sides.items():
            examples = sampler.sample(
                batch_size,
                side=side,
                min_value_examples=(
                    0 if value_sampler is not None else min_value_examples_per_batch
                ),
                diversity_caps=diversity_caps,
            )
            if not examples:
                continue
            value_examples = None
            if value_sampler is not None and min_value_examples_per_batch > 0:
                if side in {"us", "ussr"}:
                    value_examples = value_sampler.sample(
                        min_value_examples_per_batch,
                        side=side,
                    )
                else:
                    us_count = min_value_examples_per_batch // 2
                    value_examples = value_sampler.sample(us_count, side="us")
                    value_examples.extend(
                        value_sampler.sample(
                            min_value_examples_per_batch - len(value_examples),
                            side="ussr",
                        )
                    )
                if len(value_examples) < min_value_examples_per_batch:
                    raise RuntimeError(
                        "search value replay cannot fill the requested complete-root batch "
                        f"for policy {policy_id}: requested={min_value_examples_per_batch}, "
                        f"available={len(value_examples)}"
                    )
            metrics = distill_search_batch(
                algo.get_policy(policy_id),
                optimizers[policy_id],
                examples,
                value_examples=value_examples,
                lambda_search=lambda_search,
                max_grad_norm=max_grad_norm,
                parameter_scope=parameter_scope,
                balance_sides=balance_sides,
                lambda_value=lambda_value,
                lambda_search_q=lambda_search_q,
                confidence_weight_floor=confidence_weight_floor,
                search_policy_mix=search_policy_mix,
            )
            for key, value in metrics.items():
                totals[f"{policy_id}/{key}"] += float(value)
            policy_batches[policy_id] += 1
            completed += 1
    if completed <= 0:
        return {}
    averaged = {
        key: value / max(1, policy_batches[key.split("/", 1)[0]])
        for key, value in totals.items()
    }
    averaged.update(
        {
            "search_distill/update_batches": float(completed),
            **{
                f"{policy_id}/search_distill/update_batches": float(count)
                for policy_id, count in policy_batches.items()
            },
        }
    )
    return averaged


def save_search_optimizer_state(checkpoint: Path | str, optimizers: dict[str, Any]) -> Path:
    import torch

    path = Path(checkpoint) / "search_distill_optimizers.pt"
    torch.save({policy_id: optimizer.state_dict() for policy_id, optimizer in optimizers.items()}, path)
    return path


def restore_search_optimizer_state(checkpoint: Path | str, optimizers: dict[str, Any]) -> bool:
    import torch

    path = Path(checkpoint) / "search_distill_optimizers.pt"
    if not path.exists():
        return False
    states = torch.load(path, map_location="cpu", weights_only=True)
    restored = False
    for policy_id, optimizer in optimizers.items():
        if policy_id not in states:
            continue
        try:
            optimizer.load_state_dict(states[policy_id])
        except ValueError:
            # A prior checkpoint may have trained a narrower parameter scope.
            # Its moments are not compatible with a broader campaign optimizer.
            continue
        restored = True
    return restored
