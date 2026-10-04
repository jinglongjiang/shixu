"""Legal motion forecasts are the only temporal input to candidate value."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import stack_histories, window
from .model import OcclusionValueModel, OrderedValueModel
from .motion import MotionMemory, observation_intervals
from .replay import Replay


class ActorMotionEstimator(nn.Module):
    """Physical positions from lawful actor history and current neighbours."""

    def __init__(self, kind="kda", width=128, layers=2, time_step=.25):
        super().__init__()
        self._init_estimator(kind, width, layers, time_step)

    def _init_estimator(self, kind, width, layers, time_step):
        if kind not in ("cv", "current", "gru", "kda") or width % 4 or time_step <= 0:
            raise ValueError("Invalid forecast substrate, width or control step")
        self.kind, self.width, self.time_step = kind, width, time_step
        self.register_buffer("times", torch.arange(10).float() * time_step)
        self.motion_encoder = None
        self.neighbor_encoder = None
        self.decoder = None
        self.memory = None
        if kind != "cv":
            self.motion_encoder = nn.Sequential(nn.Linear(8, width), nn.ReLU())
            self.neighbor_encoder = nn.Sequential(nn.Linear(6, width), nn.ReLU())
            self.decoder = nn.Linear(2 * width + 2, 18)
            nn.init.zeros_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
        if kind in ("gru", "kda"):
            self.memory = MotionMemory(kind, width, layers)
        self.last_losses = {}

    @staticmethod
    def _physical(tokens):
        humans = tokens[:, :, 1:]
        world = humans[..., :2] + tokens[:, :, 0, None, :2]
        return humans, world

    def forecast_positions(self, history):
        if history.ndim != 4 or history.shape[-1] != 13 or history.shape[1] < 1:
            raise ValueError("Forecasts require real track observations [B,T,1+N,13]")
        humans, world = self._physical(history)
        active = humans[..., 12] > 0
        measured = (humans[..., 10] > 0) & active
        anchor = world[:, -1]
        velocity = humans[:, -1, :, 3:5]
        displacement = velocity[:, :, None] * self.times[None, None, :, None]
        if self.kind != "cv":
            features = torch.cat((world - anchor[:, None], humans[..., 3:5], humans[..., 6:7],
                                  humans[..., 9:11], active[..., None].to(humans.dtype)), -1)
            features = self.motion_encoder(features)
            latest = features[:, -1]
            if self.memory is not None:
                batch, length, people, width = features.shape
                sequences = features.permute(0, 2, 1, 3).reshape(batch * people, length, width)
                writes = measured.permute(0, 2, 1).reshape(batch * people, length)
                gaps, _ = observation_intervals(measured)
                intervals = gaps.permute(0, 2, 1).reshape(batch * people, length)
                state = self.memory.encode(sequences, writes, intervals)
                if self.kind == "gru":
                    recalled = self.memory.norm(state[-1]).reshape(batch, people, width)
                    latest = latest + recalled
                else:
                    latest = self.memory.query(state, latest.reshape(batch * people, width), False)
                    latest = latest.reshape(batch, people, width)
            # Every learned estimator sees the same currently available crowd.
            relative = anchor[:, None] - anchor[:, :, None]
            relative_velocity = velocity[:, None] - velocity[:, :, None]
            radius = humans[:, -1, :, 6][:, None, :, None].expand(*relative.shape[:-1], 1)
            neighbours = self.neighbor_encoder(torch.cat((relative, relative_velocity,
                                                          relative.norm(dim=-1, keepdim=True), radius), -1))
            pairs = active[:, -1, :, None] & active[:, -1, None, :]
            pairs &= ~torch.eye(anchor.shape[1], device=anchor.device, dtype=torch.bool)[None]
            social = OrderedValueModel._max_pool(neighbours, pairs)
            residual = self.decoder(torch.cat((latest, social, anchor), -1)).reshape(*latest.shape[:2], 9, 2)
            residual = residual * self.times[None, None, 1:, None]
            displacement = displacement + torch.cat((torch.zeros_like(residual[:, :, :1]), residual), 2)
        return anchor[:, :, None] + displacement

    def motion_loss(self, history, predicted, future, valid):
        anchor = self._physical(history)[1][:, -1]
        displacement = predicted[:, :, 1:] - anchor[:, :, None]
        error = ((displacement - future) / self.times[None, None, 1:, None]).square().sum(-1)
        return (error * valid).sum() / (2 * valid.sum().clamp_min(1))


class ForecastValueModel(ActorMotionEstimator):
    """Prototype A: explicit forecast geometry replaces temporal value features.

    Observed/successor value uses offsets0/.25/.5/1/2s, shifted one native step
    for successors. Retained for exact loading of the completed A checkpoints.
    """

    full_observed_window = True

    def __init__(self, kind="kda", width=128, layers=2, time_step=.25, prediction_weight=.1):
        # Preserve A's initialization order and flat checkpoint parameter names.
        nn.Module.__init__(self)
        self.prediction_weight = prediction_weight
        self.prediction_steps = tuple(range(1, 10))
        self.robot_encoder = nn.Sequential(nn.Linear(9, width), nn.ReLU())
        self.human_encoder = nn.Sequential(nn.Linear(19, width), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width), nn.ReLU())
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.value_head = nn.Linear(width, 1)
        self._init_estimator(kind, width, layers, time_step)
        self.last_losses = {}

    def _value(self, history, queries, positions, query_offset):
        if query_offset not in (0, 1):
            raise ValueError("Only the observed frame or one native successor is supported")
        humans = history[:, -1, 1:]
        active = humans[..., 12] > 0
        indices = torch.as_tensor([0, 1, 2, 4, 8], device=positions.device) + query_offset
        future = positions.index_select(2, indices)
        elapsed = self.times[indices] - query_offset * self.time_step
        robot = queries[:, :, 0, :9]
        robot_future = robot[:, :, None, :2] + elapsed[None, None, :, None] * robot[:, :, None, 2:4]
        relative = future[:, None] - robot_future[:, :, None]
        clearance = relative.norm(dim=-1) - humans[:, None, :, None, 6] - robot[:, :, None, None, 4]
        velocity = humans[:, None, :, 3:5] - robot[:, :, None, 2:4]
        shape = relative.shape[:3]
        metadata = humans[:, None, :, [6, 9]].expand(*shape, 2)
        actor = torch.cat((relative.flatten(-2), clearance, velocity, metadata), -1)
        features = self.fusion(torch.cat((self.robot_encoder(robot)[:, :, None].expand(*shape, self.width),
                                         self.human_encoder(actor)), -1))
        values = features.reshape(-1, shape[2], self.width)
        mask = active[:, None].expand(*shape).reshape(-1, shape[2]).clone()
        empty = ~mask.any(-1)
        mask[empty, 0] = True
        attended, _ = self.attention(values, values, values, key_padding_mask=~mask, need_weights=False)
        pooled = OrderedValueModel._max_pool(attended, mask).masked_fill(empty[:, None], 0)
        return self.value_head(pooled).reshape(shape[0], shape[1])

    def score_candidates(self, history, queries):
        return self._value(history, queries, self.forecast_positions(history), 1)

    def value_and_forecast(self, history):
        positions = self.forecast_positions(history)
        return self._value(history, history[:, -1:], positions, 0).squeeze(1), positions

    def forward(self, history):
        return self.value_and_forecast(history)[0]

    def supervised_loss(self, histories, returns, future, valid):
        values, predicted = self.value_and_forecast(histories)
        value_loss = F.mse_loss(values, returns)
        if self.kind == "cv":
            motion_loss = value_loss.new_zeros(())
        else:
            motion_loss = self.motion_loss(histories, predicted, future, valid)
        self.last_losses = {"value_loss": float(value_loss.detach()), "forecast_loss": float(motion_loss.detach()),
                            "forecast_targets": int(valid.sum())}
        return value_loss + self.prediction_weight * motion_loss


class ForecastSuccessorValueModel(nn.Module):
    """Prototype B: physical estimates change native successor queries only.

    The inherited critic still learns scalar MC returns on real observations.
    No value gradient enters the independently supervised physical estimator.
    """

    full_observed_window = True

    def __init__(self, kind="kda", width=128, layers=2, time_step=.25, prediction_weight=.1):
        super().__init__()
        self.kind = kind
        self.prediction_weight = prediction_weight
        self.prediction_steps = tuple(range(1, 10))
        self.critic = OcclusionValueModel("gru", width, layers, interaction_order="write")
        self.predictor = ActorMotionEstimator(kind, width, layers, time_step)
        self.last_losses = {}

    @property
    def times(self):
        return self.predictor.times

    _physical = staticmethod(ActorMotionEstimator._physical)

    def forecast_positions(self, history):
        return self.predictor.forecast_positions(history)

    def corrected_queries(self, history, queries, positions):
        humans, world = self._physical(history)
        cv = world[:, -1] + humans[:, -1, :, 3:5] * self.times[1]
        residual = positions[:, :, 1] - cv
        active = queries[:, :, 1:, 12] > 0
        result = queries.clone()
        actor = result[:, :, 1:]
        actor[..., :2] += residual[:, None] * active[..., None]
        changed = active & residual[:, None].ne(0).any(-1)
        relative = actor[..., :2]
        distance = (relative.square().sum(-1) + 1e-6).sqrt()
        velocity = actor[..., 3:5] - result[:, :, 0, None, 2:4]
        inverse = (-(relative * velocity).sum(-1) / distance.square()).clamp(0, 10)
        actor[..., 2] = torch.where(changed, distance, actor[..., 2])
        actor[..., 7] = torch.where(changed, inverse, actor[..., 7])
        actor[..., 8] = torch.where(changed, (distance < 2).to(actor.dtype), actor[..., 8])
        return result

    def _value(self, history, queries, positions, query_offset):
        if query_offset != 1:
            raise ValueError("Forecast bridge changes only native successor queries")
        if self.kind != "cv":
            queries = self.corrected_queries(history, queries, positions)
        return self.critic.score_candidates(history[:, 1:], queries)

    def score_candidates(self, history, queries):
        if self.kind == "cv":
            return self.critic.score_candidates(history[:, 1:], queries)
        return self._value(history, queries, self.forecast_positions(history), 1)

    def forward(self, history):
        return self.critic(history)

    def gradient_groups(self):
        # Auxiliary gradients must not rescale the inherited critic's updates.
        return (self.critic.parameters(), self.predictor.parameters())

    def supervised_loss(self, histories, returns, future, valid):
        value_loss = F.mse_loss(self(histories), returns)
        motion_loss = (value_loss.new_zeros(()) if self.kind == "cv" else
                       self.predictor.motion_loss(histories, self.forecast_positions(histories), future, valid))
        self.last_losses = {"value_loss": float(value_loss.detach()), "forecast_loss": float(motion_loss.detach()),
                            "forecast_targets": int(valid.sum())}
        return value_loss + self.prediction_weight * motion_loss


class ForecastReplay(Replay):
    """Later legal measurements supervise prediction, never become input features."""

    def sample_with_forecasts(self, batch_size, device, rng):
        if not self.samples:
            raise ValueError("Cannot sample an empty replay")
        selected = [self.samples[i] for i in rng.integers(len(self.samples), size=batch_size)]
        histories = stack_histories([window(frames[max(0, tick + 1 - self.length):tick + 1],
                                            self.length, self.left_pad) for frames, tick, _ in selected])
        people = histories.shape[-2] - 1
        targets = np.zeros((batch_size, people, 9, 2), dtype=np.float32)
        valid = np.zeros((batch_size, people, 9), dtype=bool)
        for batch, (frames, tick, _) in enumerate(selected):
            current = frames[tick]
            count = current.shape[0] - 1
            anchor = current[1:, :2] + current[0, :2]
            for horizon in range(1, 10):
                if tick + horizon >= len(frames):
                    continue
                future = frames[tick + horizon]
                available = (current[1:, 12] > 0) & (future[1:, 12] > 0) & (future[1:, 10] > 0)
                targets[batch, :count, horizon - 1] = future[1:, :2] + future[0, :2] - anchor
                valid[batch, :count, horizon - 1] = available
        return (torch.as_tensor(histories, device=device),
                torch.as_tensor([row[2] for row in selected], dtype=torch.float32, device=device),
                torch.as_tensor(targets, device=device), torch.as_tensor(valid, device=device))
