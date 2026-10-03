"""ORCA Monte Carlo value pretraining followed by online MC value refinement."""

from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .replay import Replay
from .runner import orca_teacher, run_episode


def update(model, replay, optimizer, batch_size, device, rng):
    histories, labels = replay.sample(batch_size, device, rng)
    model.train()
    loss = F.mse_loss(model(histories), labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def train(env, policy, cfg, output, seed, il_episodes, rl_episodes):
    if il_episodes <= 0 or rl_episodes < 0:
        raise ValueError("IL episodes must be positive and RL episodes nonnegative")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    replay = Replay(cfg.getint("buffer", "capacity"), policy.length, policy.gamma)
    teacher = orca_teacher(cfg)
    accepted = 0
    limit = cfg.getint("imitation_learning", "max_il_prefill")
    for attempt in range(limit):
        episode = run_episode(env, policy, seed + attempt, teacher=teacher, phase="train")
        if episode["terminal"] == "reach_goal":
            replay.add(episode)
            accepted += 1
        if accepted == il_episodes:
            break
    if accepted < il_episodes:
        raise RuntimeError("ORCA did not supply the requested successful IL episodes")
    batch = cfg.getint("train", "il_batch_size")
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=cfg.getfloat("train", "il_learning_rate"), weight_decay=0.01)
    batches = max(1, (len(replay.samples) + batch - 1) // batch)
    total = cfg.getint("train", "il_epochs") * batches
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    for _ in range(total):
        update(policy.model, replay, optimizer, batch, policy.device, rng)
        schedule.step()
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=cfg.getfloat("train", "learning_rate"), weight_decay=0.01)
    for episode_number in range(rl_episodes):
        fraction = min(1.0, episode_number / cfg.getint("sarl", "epsilon_decay_episodes"))
        epsilon = cfg.getfloat("sarl", "epsilon_start") * (1 - fraction) + cfg.getfloat("sarl", "epsilon_end") * fraction
        episode = run_episode(env, policy, seed + limit + episode_number, epsilon=epsilon, phase="train")
        replay.add(episode)
        for _ in range(cfg.getint("train", "updates_per_ep")):
            update(policy.model, replay, optimizer, cfg.getint("train", "batch_size"), policy.device, rng)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.model.state_dict(), "seed": seed, "il_episodes": il_episodes,
                "rl_episodes": rl_episodes, "config": {section: dict(cfg[section]) for section in cfg.sections()},
                "framework": "clean MC-value baseline; not an exact legacy training replay"}, destination)
