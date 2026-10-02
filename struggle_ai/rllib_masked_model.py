from __future__ import annotations

import copy
import math

import numpy as np

from struggle_ai.features import HEURISTIC_PRIOR_FEATURE


def register_masked_model() -> None:
    try:
        from ray.rllib.models import ModelCatalog
    except ImportError as exc:
        raise RuntimeError("RLlib is not installed") from exc

    ModelCatalog.register_custom_model("twilight_masked_model", TorchMaskedActions)


try:
    import torch
    import torch.nn as nn
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.utils.torch_utils import FLOAT_MIN
except ImportError:
    torch = None
    nn = None
    TorchModelV2 = None
    FLOAT_MIN = -3.4e38


if TorchModelV2 is not None:

    def mlp(input_size: int, hidden: int, output_size: int, layers: int = 2) -> nn.Sequential:
        modules: list[nn.Module] = []
        last = input_size
        for _ in range(max(0, layers - 1)):
            modules.extend([nn.Linear(last, hidden), nn.ReLU()])
            last = hidden
        modules.append(nn.Linear(last, output_size))
        modules.append(nn.ReLU())
        return nn.Sequential(*modules)


    class TorchMaskedActions(TorchModelV2, nn.Module):
        """Structured board/card/action encoder with legal-action masking.

        This keeps the RLlib action space as padded discrete candidates, but the
        logits are produced by scoring each legal action against an encoded game
        state instead of using one flat output head.
        """

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
            nn.Module.__init__(self)
            self._value_out = None

            original = getattr(obs_space, "original_space", obs_space)
            cfg = model_config.get("custom_model_config", {})
            hidden = int(cfg.get("hidden", 256))
            graph_layers = int(cfg.get("graph_layers", 2))
            self.graph_neighbor_hops = max(1, int(cfg.get("graph_neighbor_hops", 2)))
            self.heuristic_prior_scale = float(cfg.get("heuristic_prior_scale", 2.0))
            self.setup_heuristic_prior_scale = float(cfg.get("setup_heuristic_prior_scale", 0.0))
            self.policy_temperature = max(0.05, float(cfg.get("policy_temperature", 1.0)))
            self.model_arch = str(cfg.get("model_arch", "feedforward"))
            self.use_transformer_history = self.model_arch == "transformer_history"
            # Keep the legacy shared-head layout available so old RLlib
            # checkpoints can still construct and restore their original model.
            self.side_specific_heads = bool(cfg.get("side_specific_heads", False))
            self.value_adapter_hidden = max(0, int(cfg.get("value_adapter_hidden", 0)))
            self.value_state_tower = bool(cfg.get("value_state_tower", False))

            global_size = int(np.prod(original["global"].shape))
            event_size = int(np.prod(original["events"].shape))
            country_size = int(original["countries"].shape[-1])
            region_size = int(original["regions"].shape[-1])
            card_size = int(original["cards"].shape[-1])
            action_size = int(original["action_features"].shape[-1])
            original_spaces = getattr(original, "spaces", {})
            self.has_absolute_hand_masks = "us_hand" in original_spaces and "ussr_hand" in original_spaces
            if self.has_absolute_hand_masks:
                card_size += 2

            self.global_encoder = mlp(global_size + event_size, hidden, hidden)
            self.country_in = mlp(country_size, hidden, hidden)
            self.country_graph = nn.ModuleList(
                nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.LayerNorm(hidden))
                for _ in range(graph_layers)
            )
            self.region_encoder = mlp(region_size, hidden, hidden)
            self.card_encoder = mlp(card_size, hidden, hidden)
            self.action_encoder = mlp(action_size, hidden, hidden)
            if self.use_transformer_history:
                self.history_side_encoder = nn.Linear(1, hidden)
                self.history_turn_ar_encoder = nn.Linear(2, hidden)
                self.history_vp_defcon_encoder = nn.Linear(2, hidden)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=int(cfg.get("history_attention_heads", 4)),
                    dim_feedforward=hidden * 4,
                    dropout=float(cfg.get("history_dropout", 0.1)),
                    batch_first=True,
                    activation="gelu",
                )
                self.history_transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=int(cfg.get("history_layers", 2)),
                )
                self.card_history_encoder = mlp(int(original["card_history"].shape[-1]), hidden, hidden)
                card_encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=int(cfg.get("history_attention_heads", 4)),
                    dim_feedforward=hidden * 4,
                    dropout=float(cfg.get("history_dropout", 0.1)),
                    batch_first=True,
                    activation="gelu",
                )
                self.card_history_transformer = nn.TransformerEncoder(
                    card_encoder_layer,
                    num_layers=int(cfg.get("card_history_layers", cfg.get("history_layers", 2))),
                )
            self.state_encoder = nn.Sequential(
                nn.Linear(hidden * (6 if self.use_transformer_history else 4), hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            if self.value_state_tower:
                self.value_state_encoder = copy.deepcopy(self.state_encoder)
            if self.side_specific_heads:
                self.policy_query_us = nn.Linear(hidden, hidden)
                self.policy_query_ussr = nn.Linear(hidden, hidden)
                self.action_bias_us = nn.Linear(hidden, 1)
                self.action_bias_ussr = nn.Linear(hidden, 1)
                self.value = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 2),
                )
            else:
                self.policy_query = nn.Linear(hidden, hidden)
                self.action_bias = nn.Linear(hidden, 1)
                self.value = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
            if self.value_adapter_hidden:
                value_channels = 2 if self.side_specific_heads else 1
                self.value_adapter = nn.Sequential(
                    nn.Linear(hidden, self.value_adapter_hidden),
                    nn.ReLU(),
                    nn.Linear(self.value_adapter_hidden, value_channels),
                )
                nn.init.zeros_(self.value_adapter[-1].weight)
                nn.init.zeros_(self.value_adapter[-1].bias)

        def forward(self, input_dict, state, seq_lens):
            obs = input_dict["obs"]
            if self.value_state_tower:
                state_features = self.encode_state_features(obs)
                state_embedding = self.state_encoder(state_features)
                value_state_embedding = self.value_state_encoder(state_features)
            else:
                state_embedding = self.encode_state(obs)
                value_state_embedding = state_embedding
            action_embedding = self.action_encoder(obs["action_features"].float())
            side_is_us = obs["global"].float()[:, 0] > 0.0
            if self.side_specific_heads:
                query = self.select_side_output(
                    self.policy_query_us(state_embedding),
                    self.policy_query_ussr(state_embedding),
                    side_is_us,
                ).unsqueeze(1)
                action_bias = self.select_side_output(
                    self.action_bias_us(action_embedding),
                    self.action_bias_ussr(action_embedding),
                    side_is_us,
                ).squeeze(-1)
            else:
                query = self.policy_query(state_embedding).unsqueeze(1)
                action_bias = self.action_bias(action_embedding).squeeze(-1)
            logits = (action_embedding * query).sum(dim=-1) / math.sqrt(action_embedding.shape[-1])
            logits = logits + action_bias
            heuristic_prior = obs["action_features"].float()[..., HEURISTIC_PRIOR_FEATURE]
            logits = logits + self.heuristic_prior_scale * heuristic_prior
            if self.setup_heuristic_prior_scale:
                global_features = obs["global"].float()
                is_setup = (
                    (global_features[:, 2] <= 0.101)
                    & (global_features[:, 3] <= 0.001)
                    & (global_features[:, 23] > 0.5)
                ).float().unsqueeze(1)
                logits = logits + self.setup_heuristic_prior_scale * is_setup * heuristic_prior
            logits = logits / self.policy_temperature
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            mask = obs["action_mask"].float()
            inf_mask = torch.clamp(torch.log(mask), min=FLOAT_MIN)
            value_output = self.value(value_state_embedding)
            if self.value_adapter_hidden:
                value_output = value_output + self.value_adapter(value_state_embedding)
            if self.side_specific_heads:
                value_output = torch.where(side_is_us, value_output[:, 0], value_output[:, 1])
            else:
                value_output = value_output.squeeze(1)
            self._value_out = torch.nan_to_num(value_output, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            return logits + inf_mask, state

        @staticmethod
        def select_side_output(us_output, ussr_output, side_is_us):
            shape = [side_is_us.shape[0]] + [1] * (us_output.ndim - 1)
            return torch.where(side_is_us.reshape(shape), us_output, ussr_output)

        def encode_state(self, obs):
            return self.state_encoder(self.encode_state_features(obs))

        def encode_state_features(self, obs):
            global_embedding = self.global_encoder(
                torch.cat(
                    [
                        obs["global"].float().flatten(start_dim=1),
                        obs["events"].float().flatten(start_dim=1),
                    ],
                    dim=1,
                )
            )
            country_embedding = self.encode_countries(
                obs["countries"].float(),
                obs["country_adjacency"].float(),
            )
            region_embedding = self.region_encoder(obs["regions"].float()).mean(dim=1)
            card_features = obs["cards"].float()
            if self.has_absolute_hand_masks:
                hand_features = torch.stack(
                    [
                        obs["us_hand"].float(),
                        obs["ussr_hand"].float(),
                    ],
                    dim=-1,
                )
                card_features = torch.cat([card_features, hand_features], dim=-1)
            card_embedding = self.card_encoder(card_features).mean(dim=1)
            embeddings = [global_embedding, country_embedding, region_embedding, card_embedding]
            if self.use_transformer_history:
                embeddings.append(self.encode_history(obs))
                embeddings.append(self.encode_card_history(obs))
            return torch.cat(embeddings, dim=1)

        def encode_history(self, obs):
            mask = obs["history_mask"].float()
            start = self.common_left_padding(mask)
            action_embedding = self.action_encoder(obs["history_actions"].float()[:, start:])
            side_embedding = self.history_side_encoder(obs["history_sides"].float()[:, start:].unsqueeze(-1))
            turn_embedding = self.history_turn_ar_encoder(obs["history_turn_ar"].float()[:, start:])
            vp_defcon_embedding = self.history_vp_defcon_encoder(obs["history_vp_defcon"].float()[:, start:])
            h = action_embedding + side_embedding + turn_embedding + vp_defcon_embedding
            mask = mask[:, start:]
            key_padding_mask = mask <= 0.5
            if key_padding_mask.ndim == 2:
                all_empty = key_padding_mask.all(dim=1)
                if all_empty.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_empty, 0] = False
            encoded = self.history_transformer(h, src_key_padding_mask=key_padding_mask)
            weights = mask.unsqueeze(-1)
            return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        def encode_card_history(self, obs):
            mask = obs["card_history_mask"].float()
            start = self.common_left_padding(mask)
            h = self.card_history_encoder(obs["card_history"].float()[:, start:])
            mask = mask[:, start:]
            key_padding_mask = mask <= 0.5
            if key_padding_mask.ndim == 2:
                all_empty = key_padding_mask.all(dim=1)
                if all_empty.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_empty, 0] = False
            encoded = self.card_history_transformer(h, src_key_padding_mask=key_padding_mask)
            weights = mask.unsqueeze(-1)
            return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        @staticmethod
        def common_left_padding(mask):
            """Drop only sequence positions masked for every batch item."""
            active_columns = (mask > 0.5).any(dim=0)
            if not bool(active_columns.any().item()):
                return max(0, mask.shape[1] - 1)
            return int(torch.nonzero(active_columns, as_tuple=False)[0, 0].item())

        def encode_countries(self, countries, adjacency):
            h = self.country_in(countries)
            eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
            adj = adjacency + eye
            adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
            msg_adj = self.neighborhood_adjacency(adj)
            for layer in self.country_graph:
                msg = torch.bmm(msg_adj, h)
                h = h + layer(torch.cat([h, msg], dim=-1))
            return h.mean(dim=1)

        def neighborhood_adjacency(self, adj):
            if self.graph_neighbor_hops <= 1:
                return adj
            neighborhood = adj
            frontier = adj
            for _ in range(1, self.graph_neighbor_hops):
                frontier = torch.bmm(frontier, adj)
                neighborhood = neighborhood + frontier
            return neighborhood / neighborhood.sum(dim=-1, keepdim=True).clamp_min(1.0)

        def value_function(self):
            return self._value_out
else:

    class TorchMaskedActions:  # type: ignore[no-redef]
        pass
