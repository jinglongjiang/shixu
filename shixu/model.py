"""One spatial encoder, one replaceable temporal encoder, one scalar value head."""

import torch
from torch import nn


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
        # Preserve the original relational feature convention when evaluating legacy weights.
        relative = humans[..., :2] - robot[..., None, :2]
        velocity = humans[..., 3:5] - robot[..., None, 2:4]
        distance = torch.sqrt((relative ** 2).sum(-1) + 1e-6)
        speed = torch.sqrt((velocity ** 2).sum(-1) + 1e-6)
        closing = -(relative * velocity).sum(-1) / (distance + 1e-6)
        inverse = torch.where(closing > 0, closing / (distance + 1e-6), torch.zeros_like(closing))
        relation = torch.stack((relative[..., 0], relative[..., 1], distance,
                                velocity[..., 0], velocity[..., 1], speed, closing, inverse), dim=-1)
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
