"""Frozen PaS recent-grid interventions; not a reproduction or new method.

Run as a standalone script, without shixu/vendor on PYTHONPATH. The parent is
an existing locally adapted official-code fork with a locally trained model.
"""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PARENT = Path("/home/abc/temp/PaS_CrowdNav")
CHECKPOINT = Path("/home/abc/workspace/pas_formal_20260922_weights/pas_1207/checkpoints/last.pt")
OUT = ROOT / "outputs/temporal_problem_audit/pas"
RULES = {
    "provenance": "Official-code-derived LOCAL ADAPTED fork and locally trained15M-step seed1207. "
                  "Not original paper checkpoint, not full baseline reproduction.",
    "cases": [[n, g, c] for n in (5, 10) for g in ("circle_crossing", "square_crossing") for c in (91000, 91001)],
    "root": "First t>=2s with any currently invisible human at true centre distance<=3m. "
            "Offline event enrichment, no action/outcome selection, no replacement if absent.",
    "arms": "full4: original4-grid input; short2: repeat penultimate grid3times then current; "
            "current_grid: tile current grid4times; oracle_latent: frozen GT-VAE encodes current "
            "true occupancy instead of sensor-VAE. All share the original recurrent state and action head.",
    "interpretation": "current_grid is NOT a separately trained current-only policy: earlier GRU memory "
                      "is held fixed. Oracle uses the parent's matching-trained latent space but may "
                      "still be distribution-shifted. Shortened inputs are interventions, not trained controls.",
    "execution": "Only root action differs. Native control step, clipping, reward, dynamics and terminal "
                 "conditions; full4 legal observations and the original full4 recurrent update thereafter. "
                 "Use identical saved Torch RNG at root and common subsequent latent sampling stream.",
    "kill": "An action difference alone is not value. No gain closes this frozen intervention only. "
            "Any positive signal still needs trained simple controls and independent confirmation. No KDA training.",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    if path.exists():
        raise RuntimeError("Refusing to overwrite " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False,
                              default=lambda value: value.item() if isinstance(value, np.generic) else value))


def new_world(config, people, geometry, case):
    from crowd_sim.envs.crowd_sim_dict import CrowdSimDict
    cfg = copy.deepcopy(config)
    # Config stores sections on the class: copy each one before mutation.
    for key in ("env", "sim", "humans", "robot", "sensor", "pas", "noise", "reward", "action_space", "orca"):
        setattr(cfg, key, copy.deepcopy(getattr(config, key)))
    cfg.sim.human_num = people
    cfg.sim.test_sim = geometry
    env = CrowdSimDict()
    env.configure(cfg)
    env.phase, env.nenv, env.thisSeed = "test", 1, case
    return env, env.reset()


def tensors(observation):
    # Truth labels are deliberately excluded from the normal Policy.act call.
    vector = torch.as_tensor(observation["vector"], dtype=torch.float32)[None, None]
    grid = torch.as_tensor(observation["grid"], dtype=torch.float32)[None]
    return {"vector": vector, "grid": grid}


def clone_hidden(hidden):
    return {k: v.clone() for k, v in hidden.items()}


@torch.inference_mode()
def query(actor, observation, hidden, masks, rng, mode):
    torch.set_rng_state(rng.clone())
    inputs = tensors(observation)
    if mode == "current_grid":
        inputs["grid"] = inputs["grid"][:, -1:].expand(-1, 4, -1, -1).clone()
    elif mode == "short2":
        inputs["grid"] = torch.cat((inputs["grid"][:, -2:-1].expand(-1, 3, -1, -1), inputs["grid"][:, -1:]), 1)
    before = actor.base.Sensor_VAE.encode
    if mode == "oracle_latent":
        labels = torch.as_tensor(observation["label_grid"][[0]], dtype=torch.float32)[None]
        actor.base.Sensor_VAE.encode = lambda grid: actor.base.Label_VAE.encode(labels)
    try:
        value, action, _, after, decoded = actor.act(inputs, clone_hidden(hidden), masks, deterministic=True)
    finally:
        actor.base.Sensor_VAE.encode = before
    return action[0].numpy().copy(), clone_hidden(after), torch.get_rng_state().clone(), decoded.numpy().copy()


def event(env):
    visible = set(env.visible_ids[-1])
    hidden = [i for i, h in enumerate(env.humans) if i not in visible
              and np.linalg.norm(np.asarray(h.get_position()) - env.robot.get_position()) <= 3.]
    return hidden


def world_state(env):
    names = ("px", "py", "vx", "vy", "radius", "gx", "gy", "v_pref", "theta")
    return np.array([[getattr(a, k) for k in names] for a in [env.robot] + env.humans], dtype=np.float64)


def collect(actor, config, people, geometry, case):
    env, observation = new_world(config, people, geometry, case)
    torch.manual_seed(case)
    hidden = {k: torch.zeros(1, 1, 128) for k in ("pas", "policy")}
    masks = torch.ones(1, 4)
    actions, rewards, executed = [], [], []
    root, ticks, near_frames, ever_seen = None, 0, 0, set()
    while True:
        close = event(env)
        near_frames += bool(close)
        ever_seen.update(env.visible_ids[-1])
        if root is None and ticks >= 8 and close:
            root = dict(tick=ticks, observation=copy.deepcopy(observation), hidden=clone_hidden(hidden),
                        rng=torch.get_rng_state().clone(), world=world_state(env), target=close,
                        never_seen=[i for i in close if i not in ever_seen])
        action, hidden, after_rng, _ = query(actor, observation, hidden, masks, torch.get_rng_state(), "full4")
        if root is not None and ticks == root["tick"]:
            root.update(after_hidden=clone_hidden(hidden), after_rng=after_rng, native_action=action.copy())
        observation, reward, done, info = env.step(action)
        actions.append(action.tolist())
        rewards.append(float(reward))
        executed.append([env.robot.vx, env.robot.vy])
        ticks += 1
        if done:
            break
    record = dict(people=people, geometry=geometry, case=case, terminal=type(info["info"]).__name__,
                  actions=actions, rewards=rewards, executed=executed, near_frames=near_frames,
                  root_tick=None if root is None else root["tick"])
    return record, root


def branch(actor, config, record, root, action):
    env, observation = new_world(config, record["people"], record["geometry"], record["case"])
    for command in record["actions"][:root["tick"]]:
        observation, _, done, _ = env.step(np.asarray(command))
        if done:
            raise ValueError("Prefix terminated before root")
    if not np.array_equal(world_state(env), root["world"]):
        raise ValueError("World prefix parity failed")
    for key in ("vector", "grid", "label_grid"):
        np.testing.assert_array_equal(observation[key], root["observation"][key])
    hidden = clone_hidden(root["after_hidden"])
    torch.set_rng_state(root["after_rng"].clone())
    masks, total, commands, executed, clearance = torch.ones(1, 4), 0., [], [], 1e6
    while True:
        r0 = np.asarray(env.robot.get_position())
        h0 = np.asarray([h.get_position() for h in env.humans])
        radii = np.asarray([h.radius + env.robot.radius for h in env.humans])
        observation, reward, done, info = env.step(action)
        r1 = np.asarray(env.robot.get_position())
        h1 = np.asarray([h.get_position() for h in env.humans])
        relative, delta = h0 - r0, h1 - h0 - (r1 - r0)
        alpha = np.clip(-np.sum(relative * delta, 1) / np.maximum(np.sum(delta ** 2, 1), 1e-30), 0, 1)
        clearance = min(clearance, float(np.min(np.linalg.norm(relative + alpha[:, None] * delta, axis=1) - radii)))
        total += .99 ** len(commands) * float(reward)
        commands.append(action.tolist())
        executed.append([env.robot.vx, env.robot.vy])
        if done:
            break
        action, hidden, _, _ = query(actor, observation, hidden, masks, torch.get_rng_state(), "full4")
    return dict(terminal=type(info["info"]).__name__, return_=total, seconds=len(commands) * .25,
                minimum_clearance=clearance, commands=commands, executed=executed)


def main():
    sys.path.insert(0, str(PARENT))
    from crowd_nav.configs.config import Config
    from rl.model import Policy

    torch.set_num_threads(1)
    protocol = dict(rules=RULES, checkpoint_sha256=digest(CHECKPOINT), script_sha256=digest(Path(__file__)),
                    parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PARENT, text=True).strip(),
                    parent_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PARENT)),
                    config_sha256=digest(PARENT / "crowd_nav/configs/config.py"), torch=torch.__version__)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        frozen = json.loads(protocol_path.read_text())
        if {k: v for k, v in frozen.items() if k != "script_sha256"} != {
                k: v for k, v in protocol.items() if k != "script_sha256"}:
            raise ValueError("Frozen PaS protocol changed")
        save(OUT / "serialization_correction.json", dict(script_sha256=protocol["script_sha256"],
            original_sha256=frozen["script_sha256"], correction="Serialize NumPy scalars; no scientific changes."))
    else:
        save(protocol_path, protocol)
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    vae_path = OUT / "extracted_existing_label_vae.pt"
    if not vae_path.exists():
        torch.save({k[len("base.Label_VAE."):]: v for k, v in state_dict.items() if k.startswith("base.Label_VAE.")}, vae_path)
    config = Config()
    env, _ = new_world(config, 5, "circle_crossing", 91000)
    args = SimpleNamespace(num_steps=30, num_processes=12, num_mini_batch=2, rnn_input_size=64,
                           rnn_hidden_size=128, rnn_output_size=128, rnn_embedding_size=64, vae_weights=str(vae_path))
    actor = Policy(env.action_space, config=config, base="pas_rnn", base_kwargs=args).eval()
    actor.load_state_dict(state_dict, strict=True)
    masks = torch.ones(1, 4)
    for index, (people, geometry, case) in enumerate(RULES["cases"]):
        if (OUT / f"case_{index:02d}.json").exists():
            continue
        started = time.perf_counter()
        record, root = collect(actor, config, people, geometry, case)
        choices = {}
        if root is not None:
            for mode in ("full4", "short2", "current_grid", "oracle_latent"):
                action, _, _, _ = query(actor, root["observation"], root["hidden"], masks, root["rng"], mode)
                actual = branch(actor, config, record, root, action)
                choices[mode] = dict(action=action.tolist(), actual=actual)
            baseline = choices["full4"]["actual"]
            np.testing.assert_array_equal(baseline["commands"], record["actions"][root["tick"]:])
            if baseline["terminal"] != record["terminal"]:
                raise ValueError("Native continuation terminal changed")
            for mode, choice in choices.items():
                choice.update(delta_return=choice["actual"]["return_"] - baseline["return_"],
                              action_change=float(np.linalg.norm(np.asarray(choice["action"]) - choices["full4"]["action"])))
            root_path = OUT / f"root_{index:02d}.pt"
            if root_path.exists():
                saved = torch.load(root_path, weights_only=False)
                np.testing.assert_array_equal(root["world"], saved["world"])
                np.testing.assert_array_equal(root["native_action"], saved["native_action"])
            else:
                torch.save(root, root_path)
        save(OUT / f"case_{index:02d}.json", dict(record=record, choices=choices,
            parity_passed=root is not None, elapsed_seconds=time.perf_counter() - started))
        print("PAS_PROBE", index, people, geometry, case, record["terminal"], record["root_tick"],
              {m: (c["actual"]["terminal"], round(c["delta_return"], 6)) for m, c in choices.items()},
              "seconds", round(time.perf_counter() - started, 2), flush=True)


def coverage():
    """Annotate archived commands, without selecting new intervention roots."""
    sys.path.insert(0, str(PARENT))
    from crowd_nav.configs.config import Config
    rows = []
    for path in sorted(OUT.glob("case_*.json")):
        record = json.loads(path.read_text())["record"]
        env, _ = new_world(Config(), record["people"], record["geometry"], record["case"])
        seen, counts, first, root_flags = set(), dict(never_seen=0, outside_last4=0), {}, {}
        for tick, command in enumerate(record["actions"]):
            seen.update(env.visible_ids[-1])
            recent = {i for ids in env.visible_ids[-4:] for i in ids}
            close = [i for i, h in enumerate(env.humans)
                     if np.linalg.norm(np.asarray(h.get_position()) - env.robot.get_position()) <= 3.]
            for name, known in (("never_seen", seen), ("outside_last4", recent)):
                hit = bool(set(close) - known)
                counts[name] += hit
                if hit and tick >= 8 and name not in first:
                    first[name] = tick
                if tick == record["root_tick"]:
                    root_flags[name] = hit
            _, _, done, info = env.step(np.asarray(command))
        if type(info["info"]).__name__ != record["terminal"]:
            raise ValueError("Coverage replay terminal changed")
        rows.append(dict(people=record["people"], geometry=record["geometry"], case=record["case"],
                         frames=len(record["actions"]), counts=counts, first_eligible=first, root_flags=root_flags))
    save(OUT / "coverage.json", dict(scope="Post-test coverage annotation, not new outcome-driven root selection.",
                                     rows=rows, script_sha256=digest(Path(__file__))))


if __name__ == "__main__":
    coverage() if "--coverage" in sys.argv else main()
