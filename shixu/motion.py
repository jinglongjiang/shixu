"""Physical actor memory, followed by current candidate-conditioned value fusion.

The no-short-convolution KDA mixer follows the MIT-licensed FLA layer at
9f38d24980c46d46bd38614e743cdacd21906578; see vendor/FLA_LICENSE.
Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li.
Elapsed observation intervals are the task-specific extension, not a new gate.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import OrderedValueModel
from .temporal import delta_scan


def observation_intervals(measured):
    """Legal intervals in control ticks; padding and never-seen slots have no past."""
    length = measured.shape[1]
    if not length:
        return measured.new_zeros(measured.shape, dtype=torch.float32), measured.new_ones(
            measured.shape[0], measured.shape[2], dtype=torch.float32)
    ticks = torch.arange(length, device=measured.device)[None, :, None]
    latest = torch.where(measured, ticks, -1).cummax(1).values
    previous = torch.cat((torch.full_like(latest[:, :1], -1), latest[:, :-1]), 1)
    gaps = torch.where(previous >= 0, ticks - previous, 1).float()
    query_gap = torch.where(latest[:, -1] >= 0, length - latest[:, -1], 1).float()
    return gaps, query_gap


class MotionKDACell(nn.Module):
    """Official-shaped no-convolution mixer, with private successor evaluation."""

    def __init__(self, width, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("KDA width must be divisible by its heads")
        self.heads, self.head_width = heads, width // heads
        self.input_norm = nn.LayerNorm(width)
        self.q, self.k, self.v = [nn.Linear(width, width, bias=False) for _ in range(3)]
        self.forget = nn.Sequential(nn.Linear(width, self.head_width, bias=False),
                                    nn.Linear(self.head_width, width, bias=False))
        self.strength = nn.Linear(width, heads, bias=False)
        self.log_rate = nn.Parameter(torch.empty(heads).uniform_(1, 16).log())
        dt = torch.empty(heads, self.head_width).uniform_(math.log(.001), math.log(.1)).exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.read_gate = nn.Sequential(nn.Linear(width, self.head_width, bias=False),
                                       nn.Linear(self.head_width, width))
        self.read_scale = nn.Parameter(torch.ones(self.head_width))
        self.output = nn.Linear(width, width, bias=False)

    def _heads(self, values):
        return values.reshape(*values.shape[:-1], self.heads, self.head_width)

    def parameters_for(self, features, intervals):
        x = self.input_norm(features)
        q, k = [F.normalize(self._heads(F.silu(layer(x))), dim=-1) for layer in (self.q, self.k)]
        v = self._heads(F.silu(self.v(x)))
        decay = -self.log_rate.exp()[:, None] * F.softplus(self._heads(self.forget(x)) + self.dt_bias)
        decay = decay * intervals[..., None, None]
        beta = self.strength(x).sigmoid()[..., None]
        return x, q, k, v, decay, beta

    def project(self, values, normalized_input):
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-5)
        values = values * self.read_scale * self._heads(self.read_gate(normalized_input)).sigmoid()
        return self.output(values.flatten(-2))

    def encode(self, features, measured, intervals, initial_state=None):
        x, q, k, v, decay, beta = self.parameters_for(features, intervals)
        values, state = delta_scan(q, k, v, decay, beta.expand_as(k), beta.expand_as(v), measured,
                                   initial_state=initial_state)
        return self.project(values, x), state

    def successor(self, state, features, intervals):
        x, q, k, v, decay, beta = self.parameters_for(features, intervals)
        discount = decay.exp()
        error = beta * (v - torch.einsum("...hk,...hkv->...hv", k * discount, state))
        values = (torch.einsum("...hk,...hkv->...hv", q * discount, state)
                  + (q * k).sum(-1, keepdim=True) * error) * self.head_width ** -.5
        return self.project(values, x)

    def read(self, state, features):
        x = self.input_norm(features)
        q = F.normalize(self._heads(F.silu(self.q(x))), dim=-1)
        values = torch.einsum("...hk,...hkv->...hv", q, state) * self.head_width ** -.5
        return self.project(values, x)


class MotionMemory(nn.Module):
    def __init__(self, kind, width, layers):
        super().__init__()
        self.kind, self.width, self.layers = kind, width, layers
        if kind == "kda":
            self.cells = nn.ModuleList([MotionKDACell(width) for _ in range(layers)])
        elif kind == "gru":
            self.gru = nn.GRU(width, width, layers, batch_first=True)
            self.norm = nn.LayerNorm(width)
        else:
            raise ValueError("Physical actor memory supports KDA or GRU")

    def empty(self, actors, reference):
        if self.kind == "gru":
            return reference.new_zeros(self.layers, actors, self.width)
        return tuple(reference.new_zeros(actors, cell.heads, cell.head_width, cell.head_width) for cell in self.cells)

    def encode(self, features, measured, intervals):
        if self.kind == "gru":
            if not features.shape[1]:
                return self.empty(len(features), features)
            order = torch.argsort((~measured).to(torch.int8), dim=1, stable=True)
            packed = nn.utils.rnn.pack_padded_sequence(
                features.gather(1, order[..., None].expand_as(features)), measured.sum(1).cpu().clamp_min(1),
                batch_first=True, enforce_sorted=False)
            _, state = self.gru(packed)
            return state * measured.any(1)[None, :, None]
        states = []
        for cell in self.cells:
            output, state = cell.encode(features, measured, intervals)
            features = features + output
            states.append(state)
        return tuple(states)

    def successor(self, states, query, intervals):
        if self.kind == "gru":
            _, state = self.gru(query[:, None], states)
            return self.norm(state[-1])
        for cell, state in zip(self.cells, states):
            query = query + cell.successor(state, query, intervals)
        return query

    def query(self, states, features, advance):
        if self.kind == "gru":
            if not advance:
                raise ValueError("A GRU control uses a private query step, not a matrix read")
            return self.successor(states, features, features.new_ones(features.shape[:-1]))
        for cell, state in zip(self.cells, states):
            recalled = (cell.successor(state, features, features.new_ones(features.shape[:-1]))
                        if advance else cell.read(state, features))
            features = features + recalled
        return features


class MotionValueModel(OrderedValueModel):
    """One physical actor stream; current geometry and goals enter only after it."""

    def __init__(self, kind="kda", width=128, layers=2, clock="elapsed", use_history=True, time_step=.25,
                 motion_query="physical"):
        super().__init__("actor", width, layers, "observed")
        self.width = width
        if clock not in ("elapsed", "observation", "imputed") or time_step <= 0:
            raise ValueError("Invalid motion-memory clock")
        self.clock, self.use_history, self.time_step = clock, use_history, time_step
        if motion_query not in ("physical", "branch", "read") or (kind == "gru" and motion_query == "read"):
            raise ValueError("Unknown physical-memory query contract")
        self.motion_query = motion_query
        self.human_encoder = nn.Sequential(nn.Linear(8, width), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(2 * width + 11, width), nn.ReLU())
        self.temporal_encoder = MotionMemory(kind, width, layers)

    def physical_features(self, tokens, intervals):
        humans = tokens[:, :, 1:]
        world_position = humans[..., :2] + tokens[:, :, 0, None, :2]
        physical = torch.cat((world_position, humans[..., 3:7],
                              (intervals * self.time_step)[..., None], humans[..., 12:13]), -1)
        return self.human_encoder(physical)

    def encode_history(self, prefix):
        active = prefix[:, :, 1:, 12] > 0
        measured = (prefix[:, :, 1:, 10] > 0) & active
        gaps, query_gap = observation_intervals(measured)
        features = self.physical_features(prefix, gaps)
        batch, length, people, width = features.shape
        flatten = lambda x: x.permute(0, 2, 1, *range(3, x.ndim)).reshape(batch * people, length, *x.shape[3:])
        if not self.use_history:
            states = self.temporal_encoder.empty(batch * people, features)
        else:
            writes = active if self.clock == "imputed" else measured
            durations = gaps if self.clock == "elapsed" else torch.ones_like(gaps)
            states = self.temporal_encoder.encode(flatten(features), flatten(writes), flatten(durations))
        return states, query_gap

    def read_history(self, memory, queries):
        states, gaps = memory
        batch, count, rows, _ = queries.shape
        people, width = rows - 1, self.width
        # CV human successors are common to every candidate robot action.
        physical = self.physical_features(queries[:, :1], gaps[:, None]).reshape(batch * people, width)
        durations = gaps if self.clock == "elapsed" else torch.ones_like(gaps)
        humans = queries[:, :, 1:]
        active = humans[..., 12] > 0
        geometry = torch.cat((humans[..., :10], active[..., None].to(humans.dtype)), -1)
        robots = self.robot_encoder(queries[:, :, 0, :9])[:, :, None].expand(-1, -1, people, -1)
        if self.motion_query == "physical":
            actors = self.temporal_encoder.successor(states, physical, durations.reshape(-1))
            actors = actors.reshape(batch, 1, people, width).expand(-1, count, -1, -1)
            features = self.fusion(torch.cat((robots, actors, geometry), -1))
        else:
            current = physical.reshape(batch, 1, people, width).expand(-1, count, -1, -1)
            features = self.fusion(torch.cat((robots, current, geometry), -1))
            if self.temporal_encoder.kind == "gru":
                shared = states.reshape(states.shape[0], batch, people, width)[:, :, None].expand(
                    -1, -1, count, -1, -1).reshape(states.shape[0], batch * count * people, width)
                features = self.temporal_encoder.query(shared, features.reshape(-1, width), True)
                features = features.reshape(batch, count, people, width)
            else:
                shared = tuple(state.reshape(batch, 1, people, *state.shape[1:]) for state in states)
                features = self.temporal_encoder.query(shared, features, self.motion_query == "branch")
        return self.value_head(self._pool(features, active)).squeeze(-1)

    def score_candidates(self, prefix, queries):
        return self.read_history(self.encode_history(prefix), queries)

    def forward(self, tokens):
        return self.score_candidates(tokens[:, :-1], tokens[:, -1:]).squeeze(1)
