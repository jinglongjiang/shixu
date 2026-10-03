"""Small actor memories with explicit observation-write and candidate-read APIs.

Delta recurrences follow the MIT-licensed FLA reference implementations:
https://github.com/fla-org/flash-linear-attention/tree/9f38d24980c46d46bd38614e743cdacd21906578/fla/ops
Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li.
See vendor/FLA_LICENSE. These compact blocks omit the language-model convolutions.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def delta_scan(q, k, v, log_decay, erase, write, visible, initial_state=None, chunk_size=8):
    """Exact short-chunk solve of the KDA/GDN2 recurrence; no inverse-decay overflow."""
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, log_decay, erase, write = [x.transpose(1, 2).to(dtype)
                                      for x in (q, k, v, log_decay, erase, write)]
    valid = visible[:, None, :, None].to(dtype)
    q, k, log_decay = q * valid, k * valid, log_decay * valid
    batch, heads, length, key_width = k.shape
    state = k.new_zeros(batch, heads, key_width, v.shape[-1]) if initial_state is None else initial_state.to(dtype)
    outputs = []
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        query, key, value, b, w = [x[:, :, start:stop] for x in (q, k, v, erase, write)]
        decay = log_decay[:, :, start:stop].cumsum(-2)
        size = stop - start
        causal = torch.ones(size, size, dtype=torch.bool, device=k.device).tril()
        difference = decay.unsqueeze(-2) - decay.unsqueeze(-3)
        # Future-to-past ratios can overflow even though their entries are discarded.
        ratios = torch.where(causal[..., None], difference, torch.zeros_like(difference)).exp()
        ratios = ratios * causal[..., None]
        transition = torch.einsum("bhik,bhjk,bhijk->bhij", b * key, key, ratios).tril(-1)
        rhs = w * value - ((b * key) * decay.exp()) @ state
        system = transition + torch.eye(size, dtype=dtype, device=k.device)
        updates = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        reads = torch.einsum("bhik,bhjk,bhijk->bhij", query, key, ratios)
        output = (query * decay.exp()) @ state + reads @ updates
        outputs.append(output * (key_width ** -0.5))
        tail = key * (decay[:, :, -1:] - decay).exp()
        state = state * decay[:, :, -1, :, None].exp() + tail.transpose(-1, -2) @ updates
    output = torch.cat(outputs, -2) if outputs else v[:, :, :0]
    return output.transpose(1, 2), state


class DeltaCell(nn.Module):
    def __init__(self, width, kind, heads=4):
        super().__init__()
        if kind not in ("kda", "gdn2") or width % heads:
            raise ValueError("Delta cells require kda/gdn2 and a width divisible by heads")
        self.kind, self.heads, self.head_width = kind, heads, width // heads
        self.input_norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.decay = nn.Linear(width, width, bias=False)
        self.log_rate = nn.Parameter(torch.zeros(heads))
        self.dt_bias = nn.Parameter(torch.full((heads, self.head_width), math.log(math.expm1(.02))))
        if kind == "kda":
            self.strength = nn.Linear(width, heads)
        else:
            self.erase = nn.Linear(width + 4, width)
            self.write = nn.Linear(width + 4, width)
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, width, bias=False)

    def _heads(self, values):
        return values.reshape(*values.shape[:-1], self.heads, self.head_width)

    def _parameters_for(self, features, evidence, addresses=None):
        x = self.input_norm(features)
        address = x if addresses is None else self.input_norm(addresses)
        q, k = [F.normalize(self._heads(layer(address)), dim=-1) for layer in (self.q, self.k)]
        v = self._heads(self.v(x))
        decay = -self.log_rate.exp()[:, None] * F.softplus(self._heads(self.decay(x)) + self.dt_bias)
        if self.kind == "kda":
            beta = self.strength(x).sigmoid()[..., None]
            b = beta.expand_as(k)
            w = beta.expand_as(v)
        else:
            gate_input = torch.cat((x, evidence), -1)
            b, w = [self._heads(layer(gate_input)).sigmoid() for layer in (self.erase, self.write)]
        return q, k, v, decay, b, w

    def encode(self, features, visible, evidence, initial_state=None, addresses=None):
        values, state = delta_scan(*self._parameters_for(features, evidence, addresses), visible,
                                   initial_state=initial_state)
        return self.output(self.output_norm(values.flatten(-2))), state

    def read(self, state, query):
        q = F.normalize(self._heads(self.q(self.input_norm(query))), dim=-1)
        values = (q[..., None] * state).sum(-2) * (self.head_width ** -0.5)
        return self.output(self.output_norm(values.flatten(-2)))


class ActorMemory(nn.Module):
    """One shared temporal operator; a separate state belongs to each actor."""

    def __init__(self, kind, width, layers, evidence_update=False):
        super().__init__()
        if evidence_update and kind != "gdn2":
            raise ValueError("Evidence-controlled write is a GDN2 diagnostic")
        self.kind, self.width, self.layers = kind, width, layers
        self.evidence_update = evidence_update
        if kind == "gru":
            self.gru = nn.GRU(width, width, layers, batch_first=True)
            self.norm = nn.LayerNorm(width)
            self.query = nn.Linear(width, width, bias=False)
            self.output = nn.Linear(width, width, bias=False)
        elif kind in ("kda", "gdn2"):
            self.cells = nn.ModuleList([DeltaCell(width, kind) for _ in range(layers)])
        else:
            raise ValueError("Actor memory supports gru/kda/gdn2")

    def encode(self, features, visible, evidence, addresses=None):
        if self.kind == "gru":
            if addresses is not None:
                raise ValueError("Separate addressing requires a delta memory")
            hidden = features.new_zeros(self.layers, len(features), self.width)
            if features.shape[1] and bool((visible.all(1) | ~visible.any(1)).all()):
                _, hidden = self.gru(features)
                hidden = hidden * visible.any(1)[None, :, None]
            else:
                for tick in range(features.shape[1]):
                    _, proposed = self.gru(features[:, tick:tick + 1], hidden)
                    hidden = torch.where(visible[:, tick][None, :, None], proposed, hidden)
            return (hidden,)
        states = []
        supplied = evidence if self.evidence_update else torch.zeros_like(evidence)
        for cell in self.cells:
            output, state = cell.encode(features, visible, supplied, addresses=addresses)
            features = features + output
            if addresses is not None:
                addresses = addresses + output
            states.append(state)
        return tuple(states)

    def read(self, states, query, visible, advance=False):
        if self.kind == "gru":
            if advance:
                _, hidden = self.gru(query[:, None], states[0])
                return self.norm(hidden[-1])
            return self.output(self.norm(states[0][-1]) * self.query(query).sigmoid())
        current = query
        for cell, state in zip(self.cells, states):
            if advance:
                output, _ = cell.encode(query[:, None], visible[:, None],
                                        query.new_zeros(len(query), 1, 4), initial_state=state)
                query = query + output[:, 0]
            else:
                query = query + cell.read(state, query)
        return query - current
