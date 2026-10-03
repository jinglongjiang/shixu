"""One spatial encoder, one replaceable temporal encoder, one scalar value head."""

import torch
from torch import nn


def legacy_relations(robot, humans):
    """Historical convention, retained only for checkpoint-compatible comparisons."""
    relative = humans[..., :2] - robot[..., None, :2]
    velocity = humans[..., 3:5] - robot[..., None, 2:4]
    distance = torch.sqrt((relative ** 2).sum(-1) + 1e-6)
    speed = torch.sqrt((velocity ** 2).sum(-1) + 1e-6)
    closing = -(relative * velocity).sum(-1) / (distance + 1e-6)
    inverse = torch.where(closing > 0, closing / (distance + 1e-6), torch.zeros_like(closing))
    return torch.stack((relative[..., 0], relative[..., 1], distance,
                        velocity[..., 0], velocity[..., 1], speed, closing, inverse), -1)


class SpatialEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        unit = width // 4
        self.robot_encoder = nn.Sequential(nn.Linear(13, unit), nn.ReLU(), nn.LayerNorm(unit))
        self.human_encoder = nn.Sequential(nn.Linear(21, unit), nn.ReLU(), nn.LayerNorm(unit))
        self.human_attention = nn.MultiheadAttention(unit, 4, batch_first=True)
        self.dropout_fusion = nn.Dropout(0.1)
        self.fusion = nn.Sequential(nn.Linear(2 * unit, width), nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, tokens):
        batch, time = tokens.shape[:2]
        robot = tokens[:, :, 0]
        humans = tokens[:, :, 3:8]
        order = torch.argsort(humans[..., 2], dim=-1)
        humans = torch.gather(humans, 2, order[..., None].expand(-1, -1, -1, 13))
        relation = legacy_relations(robot, humans)
        encoded = self.human_encoder(torch.cat((humans, relation), dim=-1)).reshape(batch * time, 5, -1)
        attended, _ = self.human_attention(encoded, encoded, encoded)
        pooled = attended.max(dim=1).values.reshape(batch, time, -1)
        joined = torch.cat((self.robot_encoder(robot), pooled), dim=-1)
        return self.fusion(self.dropout_fusion(joined))


class TemporalEncoder(nn.Module):
    def __init__(self, kind, width, layers):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.backend = nn.GRU(width, width, layers, batch_first=True)
            self.norm = nn.LayerNorm(width)
        elif kind == "mamba":
            from mamba_ssm.modules.mamba_simple import Mamba
            self.backend = nn.ModuleList([nn.Sequential(Mamba(d_model=width, d_state=64, d_conv=4, expand=2),
                                                        nn.LayerNorm(width)) for _ in range(layers)])
        else:
            raise ValueError("Supported baseline substrates: gru, mamba")

    def forward(self, features):
        if self.kind == "gru":
            output, _ = self.backend(features)
            return self.norm(output)
        for block in self.backend:
            features = features + block(features)
        return features


class ValueModel(nn.Module):
    def __init__(self, backbone="gru", width=256, layers=4):
        super().__init__()
        if width % 16:
            raise ValueError("Width must be divisible by 16")
        self.spatial_encoder = SpatialEncoder(width)
        self.temporal_encoder = TemporalEncoder(backbone, width, layers)
        self.value_head = nn.Linear(width, 1)

    def forward(self, tokens):
        if tokens.ndim != 4 or tokens.shape[-2:] != (8, 13):
            raise ValueError("Expected a scene history [B,T,8,13]")
        spatial = self.spatial_encoder(tokens)
        temporal = self.temporal_encoder(spatial)
        return self.value_head(temporal[:, -1]).squeeze(-1)


class OrderedValueModel(nn.Module):
    """Same parameters and aligned inputs; only temporal/pooling order differs."""

    def __init__(self, order="scene", width=128, layers=2, feature_contract="observed"):
        super().__init__()
        if order not in ("scene", "actor") or width % 4:
            raise ValueError("Order must be scene/actor and width divisible by four")
        self.order = order
        if feature_contract not in ("observed", "legacy"):
            raise ValueError("Feature contract must be observed or legacy")
        self.feature_contract = feature_contract
        self.robot_encoder = nn.Sequential(nn.Linear(9 if feature_contract == "observed" else 13, width), nn.ReLU())
        self.human_encoder = nn.Sequential(nn.Linear(10 if feature_contract == "observed" else 21, width), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width), nn.ReLU())
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.temporal_encoder = TemporalEncoder("gru", width, layers)
        self.value_head = nn.Linear(width, 1)

    def _pool(self, features, visible):
        shape = features.shape
        values = features.reshape(-1, shape[-2], shape[-1])
        mask = visible.reshape(-1, visible.shape[-1]).clone()
        empty = ~mask.any(-1)
        mask[empty, 0] = True
        attended, _ = self.attention(values, values, values, key_padding_mask=~mask, need_weights=False)
        pooled = attended.masked_fill(~mask[..., None], -torch.inf).max(1).values
        return pooled.masked_fill(empty[:, None], 0).reshape(*shape[:-2], shape[-1])

    def _actor_last(self, features, visible):
        # Complete tracks use the fused sequence kernel; gaps preserve the previous state.
        stable = visible.all(1) | ~visible.any(1)
        if bool(stable.all()):
            return self.temporal_encoder(features)[:, -1]
        gru = self.temporal_encoder.backend
        hidden = features.new_zeros(gru.num_layers, len(features), gru.hidden_size)
        for tick in range(features.shape[1]):
            _, proposed = gru(features[:, tick:tick + 1], hidden)
            hidden = torch.where(visible[:, tick][None, :, None], proposed, hidden)
        return self.temporal_encoder.norm(hidden[-1])

    def forward(self, tokens):
        if tokens.ndim != 4 or tokens.shape[-2:] != (8, 13):
            raise ValueError("Expected aligned history [B,T,8,13]")
        robot, humans = tokens[:, :, 0], tokens[:, :, 3:8]
        visible = humans[..., 12] > 0
        if self.feature_contract == "observed":
            robot_input = robot[..., :9]
            human_input = torch.cat((humans[..., :9], visible[..., None].to(humans.dtype)), -1)
        else:
            robot_input = robot
            human_input = torch.cat((humans, legacy_relations(robot, humans)), -1)
        robot_features = self.robot_encoder(robot_input)[..., None, :].expand(-1, -1, 5, -1)
        human_features = self.human_encoder(human_input)
        features = self.fusion(torch.cat((robot_features, human_features), -1))
        if self.order == "scene":
            last = self.temporal_encoder(self._pool(features, visible))[:, -1]
        else:
            batch, time, people, width = features.shape
            sequences = features.permute(0, 2, 1, 3).reshape(batch * people, time, width)
            masks = visible.permute(0, 2, 1).reshape(batch * people, time)
            last = self._actor_last(sequences, masks).reshape(batch, people, width)
            last = self._pool(last, visible[:, -1])
        return self.value_head(last).squeeze(-1)


def load_weights(model, path, device):
    checkpoint = torch.load(str(path), map_location=device)
    raw = checkpoint.get("policy_state", checkpoint.get("model", checkpoint))
    ignored = ("q_head.", "future_pred_head.")
    state = {}
    for key, value in raw.items():
        key = key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key
        if not key.startswith(ignored):
            state[key] = value
    model.load_state_dict(state, strict=True)
    return checkpoint
