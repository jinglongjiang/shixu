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

    def _attend(self, features, visible):
        shape = features.shape
        values = features.reshape(-1, shape[-2], shape[-1])
        if values.shape[0] == 0:
            return features
        mask = visible.reshape(-1, visible.shape[-1]).clone()
        empty = ~mask.any(-1)
        mask[empty, 0] = True
        attended, _ = self.attention(values, values, values, key_padding_mask=~mask, need_weights=False)
        return attended.masked_fill(empty[:, None, None], 0).reshape(shape)

    @staticmethod
    def _max_pool(features, visible):
        pooled = features.masked_fill(~visible[..., None], -torch.inf).max(-2).values
        return pooled.masked_fill(~visible.any(-1)[..., None], 0)

    def _pool(self, features, visible):
        return self._max_pool(self._attend(features, visible), visible)

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

    def _features(self, tokens):
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
        return features, visible

    def forward(self, tokens):
        features, visible = self._features(tokens)
        if self.order == "scene":
            last = self.temporal_encoder(self._pool(features, visible))[:, -1]
        else:
            batch, time, people, width = features.shape
            sequences = features.permute(0, 2, 1, 3).reshape(batch * people, time, width)
            masks = visible.permute(0, 2, 1).reshape(batch * people, time)
            last = self._actor_last(sequences, masks).reshape(batch, people, width)
            last = self._pool(last, visible[:, -1])
        return self.value_head(last).squeeze(-1)


class MemoryValueModel(OrderedValueModel):
    """Same observed stem and planner; explicit shared-prefix temporal queries."""

    def __init__(self, substrate="kda", readout="evidence", width=128, layers=2):
        from .temporal import ActorMemory
        super().__init__("actor", width, layers, "observed")
        if readout not in ("full", "read", "static", "gate", "evidence", "revision"):
            raise ValueError("Unknown memory readout")
        self.substrate, self.readout = substrate, readout
        self.temporal_encoder = ActorMemory(substrate, width, layers, evidence_update=readout == "revision")
        self.evidence_gate = nn.Linear(2 * width + 4, width) if readout in ("gate", "evidence") else None
        self.memory_scale = nn.Parameter(torch.zeros(width)) if readout == "static" else None

    def encode_history(self, prefix):
        from .features import motion_evidence
        features, visible = self._features(prefix)
        batch, length, people, width = features.shape
        evidence = motion_evidence(prefix)
        sequences = features.permute(0, 2, 1, 3).reshape(batch * people, length, width)
        masks = visible.permute(0, 2, 1).reshape(batch * people, length)
        changes = evidence.permute(0, 2, 1, 3).reshape(batch * people, length, 4)
        states = self.temporal_encoder.encode(sequences, masks, changes)
        last_change = evidence[:, -1] if length else prefix.new_zeros(batch, people, 4)
        return states, visible.any(1), last_change

    def read_history(self, memory, queries):
        states, seen, evidence = memory
        features, visible = self._features(queries)
        batch, count, people, width = features.shape
        if seen.shape != (batch, people):
            raise ValueError("Prefix and query batches differ")
        repeated = []
        for state in states:
            if self.substrate == "gru":
                repeated.append(state.reshape(state.shape[0], batch, people, width)[:, :, None].expand(
                    -1, -1, count, -1, -1).reshape(state.shape[0], batch * count * people, width))
            else:
                repeated.append(state.reshape(batch, people, *state.shape[1:])[:, None].expand(
                    -1, count, -1, -1, -1, -1).reshape(batch * count * people, *state.shape[1:]))
        flat = features.reshape(batch * count * people, width)
        m = self.temporal_encoder.read(tuple(repeated), flat, visible.reshape(-1), advance=self.readout == "full")
        m = m.reshape(batch, count, people, width)
        usable = visible if self.readout == "full" else visible & seen[:, None]
        m = m * usable[..., None]
        if self.evidence_gate is not None:
            changes = evidence if self.readout == "evidence" else torch.zeros_like(evidence)
            changes = changes[:, None].expand(-1, count, -1, -1)
            gate = self.evidence_gate(torch.cat((features, m, changes), -1)).sigmoid()
            m = m * gate
        if self.memory_scale is not None:
            m = m * self.memory_scale.sigmoid()
        return self.value_head(self._pool(features + m, visible)).squeeze(-1)

    def score_candidates(self, prefix, queries):
        return self.read_history(self.encode_history(prefix), queries)

    def forward(self, tokens):
        if tokens.shape[1] < 1:
            raise ValueError("A value query needs at least one frame")
        return self.score_candidates(tokens[:, :-1], tokens[:, -1:]).squeeze(1)


class OcclusionValueModel(OrderedValueModel):
    """Measured actor history -> shared temporal memory -> retained-track value.

    Predicted locations are queries, never measurement writes. All actors share
    parameters; the number of visible/retained actors is not fixed to five.
    """

    def __init__(self, substrate="kda", width=128, layers=2, read_clock="candidate", interaction_order="read",
                 local_address=False, query_mode="read", write_mode="observed"):
        from .temporal import ActorMemory
        super().__init__("actor", width, layers, "observed")
        self.substrate = substrate
        if read_clock not in ("candidate", "observation"):
            raise ValueError("Memory reads use either the candidate or observation clock")
        self.read_clock = read_clock
        if interaction_order not in ("read", "write", "residual"):
            raise ValueError("Actor interaction order must be read, write or residual")
        if local_address and (substrate != "kda" or interaction_order != "residual" or read_clock != "candidate"):
            raise ValueError("Local addresses require candidate-read residual KDA")
        self.interaction_order = interaction_order
        self.local_address = local_address
        if query_mode not in ("read", "branch") or write_mode not in ("observed", "retained"):
            raise ValueError("Unknown successor-query or history-write contract")
        if query_mode == "branch" and (substrate != "kda" or read_clock != "candidate" or local_address):
            raise ValueError("Private successor branches require ordinary candidate-clock KDA")
        self.query_mode, self.write_mode = query_mode, write_mode
        self.human_encoder = nn.Sequential(nn.Linear(11, width), nn.ReLU())
        self.temporal_encoder = (ActorMemory(substrate, width, layers)
                                 if substrate != "current" else None)

    def _features(self, tokens):
        if tokens.ndim != 4 or tokens.shape[-1] != 13 or tokens.shape[-2] < 2:
            raise ValueError("Expected variable actor history [B,T,1+N,13], N>=1")
        humans = tokens[:, :, 1:]
        active = humans[..., 12] > 0
        measured = (humans[..., 10] > 0) & active
        # Observation validity controls writes, not the hypothetical query stem.
        actor_input = torch.cat((humans[..., :10], active[..., None].to(humans.dtype)), -1)
        robot = self.robot_encoder(tokens[:, :, 0, :9])[..., None, :].expand(
            -1, -1, humans.shape[2], -1)
        features = self.fusion(torch.cat((robot, self.human_encoder(actor_input)), -1))
        return features, active, measured

    def encode_history(self, prefix):
        features, active, measured = self._features(prefix)
        batch, length, people, width = features.shape
        if self.temporal_encoder is None:
            return None, measured.any(1), None
        addresses = features if self.local_address else None
        if self.interaction_order in ("write", "residual"):
            # Only arrived measurements supply neighbour context to real writes.
            context = self._attend(features, measured)
            features = features + context if self.interaction_order == "residual" else context
        sequence = features.permute(0, 2, 1, 3).reshape(batch * people, length, width)
        writes = active if self.write_mode == "retained" else measured
        masks = writes.permute(0, 2, 1).reshape(batch * people, length)
        if addresses is not None:
            addresses = addresses.permute(0, 2, 1, 3).reshape(batch * people, length, width)
        states = self.temporal_encoder.encode(sequence, masks, sequence.new_zeros(batch * people, length, 4),
                                              addresses=addresses)
        latest = features[:, -1] if length else features.new_zeros(batch, people, width)
        return states, measured.any(1), latest

    def read_history(self, memory, queries):
        features, active, _ = self._features(queries)
        own = features
        if self.interaction_order == "write":
            features = self._attend(features, active)
        elif self.interaction_order == "residual":
            features = features + self._attend(features, active)
        states, seen, latest = memory
        batch, count, people, width = features.shape
        if seen.shape != (batch, people):
            raise ValueError("Track slots must agree between history and candidates")
        if states is not None and self.read_clock == "observation":
            recalled = self.temporal_encoder.read(states, latest.reshape(batch * people, width), seen.reshape(-1))
            features = features + recalled.reshape(batch, 1, people, width) * (active & seen[:, None])[..., None]
        elif states is not None:
            shared = []
            for state in states:
                if self.substrate == "gru":
                    shared.append(state.reshape(state.shape[0], batch, 1, people, width))
                else:
                    shared.append(state.reshape(batch, 1, people, *state.shape[1:]))
            query = own if self.local_address else features
            branching = self.query_mode == "branch"
            recalled = self.temporal_encoder.read(tuple(shared), query, active, advance=branching)
            usable = active if branching else active & seen[:, None]
            features = features + recalled * usable[..., None]
        pooled = self._max_pool(features, active) if self.interaction_order in ("write", "residual") else self._pool(features, active)
        return self.value_head(pooled).squeeze(-1)

    def score_candidates(self, prefix, queries):
        return self.read_history(self.encode_history(prefix), queries)

    def forward(self, tokens):
        return self.score_candidates(tokens[:, :-1], tokens[:, -1:]).squeeze(1)


def build_model(config):
    section = config["model"]
    width, layers = int(section["width"]), int(section["layers"])
    if section.get("architecture") == "forecast_native":
        from .forecast import ForecastSuccessorValueModel
        return ForecastSuccessorValueModel(section.get("backbone", "kda"), width, layers,
                                           config.getfloat("env", "time_step", fallback=.25),
                                           section.getfloat("prediction_weight", fallback=.1))
    if section.get("architecture") == "forecast":
        from .forecast import ForecastValueModel
        return ForecastValueModel(section.get("backbone", "kda"), width, layers,
                                  config.getfloat("env", "time_step", fallback=.25),
                                  section.getfloat("prediction_weight", fallback=.1))
    if section.get("architecture") == "motion":
        from .motion import MotionValueModel
        return MotionValueModel(section.get("backbone", "kda"), width, layers,
                                section.get("clock", "elapsed"), section.getboolean("use_history", fallback=True),
                                config.getfloat("env", "time_step", fallback=.25),
                                section.get("motion_query", "physical"))
    if section.get("architecture") == "occlusion":
        return OcclusionValueModel(section.get("backbone", "kda"), width, layers,
                                   section.get("read_clock", "candidate"), section.get("interaction_order", "read"),
                                   section.getboolean("local_address", fallback=False),
                                   section.get("query_mode", "read"), section.get("write_mode", "observed"))
    if section.get("architecture") == "memory":
        return MemoryValueModel(section.get("backbone", "kda"), section.get("readout", "evidence"), width, layers)
    if section.get("order"):
        if section.get("backbone", "gru") != "gru":
            raise ValueError("The original processing-order model uses GRU")
        return OrderedValueModel(section["order"], width, layers, section.get("feature_contract", "observed"))
    return ValueModel(section.get("backbone", "gru"), width, layers)


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
