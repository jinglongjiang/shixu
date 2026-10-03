"""Episode-level legal observations and Monte Carlo value targets."""

import numpy as np
import torch

from .features import window


def returns(rewards, gamma):
    targets = np.zeros(len(rewards), dtype=np.float32)
    total = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        total = rewards[index] + gamma * total
        targets[index] = total
    return targets


class Replay:
    def __init__(self, capacity=200000, length=24, gamma=0.99):
        if capacity <= 0 or length <= 0:
            raise ValueError("Replay capacity and history length must be positive")
        self.samples = []
        self.capacity, self.pointer = capacity, 0
        self.length, self.gamma = length, gamma

    def add(self, episode):
        targets = returns(episode["rewards"], self.gamma)
        frames = np.asarray(episode["tokens"], dtype=np.float32).copy()
        if len(frames) != len(targets):
            raise ValueError("Observation and reward sequence lengths differ")
        for index, target in enumerate(targets):
            sample = (frames, index, float(target))
            if len(self.samples) < self.capacity:
                self.samples.append(sample)
            else:
                self.samples[self.pointer] = sample
            self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, batch_size, device, rng):
        if not self.samples:
            raise ValueError("Cannot sample an empty replay")
        indices = rng.integers(len(self.samples), size=batch_size)
        selected = [self.samples[index] for index in indices]
        histories = torch.as_tensor(np.stack([window(row[0][max(0, row[1] + 1 - self.length):row[1] + 1], self.length)
                                             for row in selected]), device=device)
        labels = torch.as_tensor([row[2] for row in selected], device=device, dtype=torch.float32)
        return histories, labels
