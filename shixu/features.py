"""Checkpoint-compatible local camrl observation encoding."""

import numpy as np


def encode_state(state):
    robot = state.self_state.to_array().astype(np.float32)
    humans = np.zeros((5, 5), dtype=np.float32)
    for index, human in enumerate(state.human_states[:5]):
        humans[index] = human.to_array()
    return encode_packed(np.concatenate((robot, humans.ravel()))[None])[0]


def encode_packed(states):
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 34:
        raise ValueError("Expected packed legacy state [B,34]")
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


def window(frames, length):
    if len(frames) == 0 or length <= 0:
        raise ValueError("A positive history length and at least one observed frame are required")
    selected = list(frames)[-length:]
    return np.asarray([selected[0]] * (length - len(selected)) + selected, dtype=np.float32)
