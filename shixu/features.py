"""Checkpoint-compatible local camrl observation encoding."""

import numpy as np
import torch


def encode_state(state):
    robot = state.self_state.to_array().astype(np.float32)
    humans = np.zeros((5, 5), dtype=np.float32)
    for index, human in enumerate(state.human_states[:5]):
        humans[index] = human.to_array()
    return encode_packed(np.concatenate((robot, humans.ravel()))[None])[0]


def encode_aligned(state):
    if not hasattr(state, "track_ids"):
        raise ValueError("Actor-aligned input requires observed track association")
    if len(state.track_ids) != len(state.human_states) or len(set(state.track_ids)) != len(state.track_ids):
        raise ValueError("Track keys must be unique and match observations")
    robot = state.self_state
    robots = np.array([[robot.px, robot.py, robot.vx, robot.vy, robot.radius,
                        robot.gx, robot.gy, robot.v_pref, robot.theta]])
    humans = np.array([[h.px, h.py, h.vx, h.vy, h.radius] for h in state.human_states]).reshape(-1, 5)
    return aligned_tokens(robots, humans, state.track_ids)[0]


def aligned_tokens(robots, humans, keys):
    """Batch candidate robots against a shared observed/predicted human state."""
    tokens = robot_tokens(np.asarray(robots, dtype=np.float32))
    for key, human in zip(keys, humans):
        if key < 0 or not isinstance(key, (int, np.integer)):
            raise ValueError("Association keys must be nonnegative episode-local integers")
        if key >= 5:
            continue
        relative = human[:2] - robots[:, :2]
        velocity = human[2:4] - robots[:, 2:4]
        distance = np.sqrt(np.sum(relative * relative, axis=1) + 1e-6)
        closing = -np.sum(relative * velocity, axis=1) / distance
        inverse = np.zeros_like(distance)
        moving = closing > 1e-6
        inverse[moving] = 1 / np.maximum(distance[moving] / closing[moving], 0.1)
        tokens[:, 3 + key, :9] = np.column_stack((relative, distance,
                                                np.full(len(robots), human[2]), np.full(len(robots), human[3]),
                                                np.full(len(robots), np.hypot(human[2], human[3])),
                                                np.full(len(robots), human[4]), inverse, distance < 2))
        tokens[:, 3 + key, 12] = 1
    return tokens


def encode_packed(states):
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 34:
        raise ValueError("Expected packed legacy state [B,34]")
    tokens = robot_tokens(states[:, :9])
    for batch, packed in enumerate(states):
        humans = packed[9:].reshape(5, 5)
        humans = humans[np.any(humans != 0, axis=1)]
        relative = humans[:, :2] - packed[:2]
        velocity = humans[:, 2:4] - packed[2:4]
        distances = np.sqrt(np.sum(relative ** 2, axis=1) + 1e-6)
        closing = -np.sum(relative * velocity, axis=1) / (distances + 1e-6)
        ttc = np.where(closing > 0.1, distances / (closing + 1e-6), distances * 10)
        for row, index in enumerate(np.argsort(ttc), start=3):
            human, rel, dist = humans[index], relative[index], distances[index]
            closing_speed = -float(np.dot(rel, velocity[index])) / dist if dist > 1e-6 else 0
            inverse = 1 / max(dist / closing_speed, 0.1) if closing_speed > 1e-6 else 0
            if dist <= 1e-6:
                inverse = 10
            tokens[batch, row, :9] = [rel[0], rel[1], dist, human[2], human[3],
                                     np.hypot(human[2], human[3]), human[4], inverse, float(dist < 2)]
    return tokens


def robot_tokens(states):
    tokens = np.zeros((len(states), 8, 13), dtype=np.float32)
    px, py, vx, vy = [states[:, index] for index in range(4)]
    # These historical metadata indices are intentionally preserved for weight parity.
    pref, gx, gy = np.clip(states[:, 5], 0, 3), states[:, 6], states[:, 7]
    dx, dy = gx - px, gy - py
    distance, angle = np.hypot(dx, dy), np.arctan2(dy, dx)
    speed, heading = np.hypot(vx, vy), np.arctan2(vy, vx)
    tokens[:, 0, :9] = states[:, :9]
    tokens[:, 0, 9:13] = np.stack((distance, speed, pref, np.ones(len(states))), axis=-1)
    tokens[:, 1, :6] = np.stack((dx, dy, distance, angle, np.cos(angle), np.sin(angle)), axis=-1)
    moving = speed > 1e-6
    tokens[:, 2, :4] = np.stack((np.where(moving, speed, 0), np.where(moving, heading, 0),
                                np.where(moving, np.cos(heading), 0), np.where(moving, np.sin(heading), 0)), axis=-1)
    return tokens


def window(frames, length):
    if len(frames) == 0 or length <= 0:
        raise ValueError("A positive history length and at least one observed frame are required")
    selected = list(frames)[-length:]
    return np.asarray([selected[0]] * (length - len(selected)) + selected, dtype=np.float32)


def motion_evidence(tokens):
    """Causal velocity innovation at each real prefix frame, never at a query."""
    humans = tokens[:, :, 3:8]
    velocity, visible = humans[..., 3:5], humans[..., 12] > 0
    if tokens.shape[1] == 0:
        return tokens.new_zeros(tokens.shape[0], 0, 5, 4)
    valid = visible.to(velocity.dtype)
    sums = torch.cat((torch.zeros_like(velocity[:, :1]), (velocity * valid[..., None]).cumsum(1)), 1)
    counts = torch.cat((torch.zeros_like(valid[:, :1]), valid.cumsum(1)), 1)
    ticks = torch.arange(tokens.shape[1], device=tokens.device)
    starts = (ticks - 3).clamp_min(0)
    count = counts[:, ticks] - counts[:, starts]
    old = (sums[:, ticks] - sums[:, starts]) / count.clamp_min(1)[..., None]
    previous = torch.cat((torch.zeros_like(visible[:, :1]), visible[:, :-1]), 1)
    allowed = visible & previous & (count > 0)
    change = velocity - old
    speed = velocity.norm(dim=-1) - old.norm(dim=-1)
    row = torch.cat((change, speed[..., None], allowed[..., None].to(velocity.dtype)), -1)
    return row * allowed[..., None]
