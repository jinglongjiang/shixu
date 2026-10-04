"""Legal-history probes for hidden human dynamics; no navigation training."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.occlusion import configuration, source_hash
from shixu.observations import OccludedTracks
from shixu.runner import environment, orca_teacher


PROTOCOL = Path(__file__).with_name("latent_information_protocol.json")


def configuration_for_audit():
    parent = json.loads(Path(__file__).with_name("occlusion_budget_protocol.json").read_text())
    return configuration(parent, "motion_gru")


def legal_actors(state):
    rows = np.zeros((state.track_count, 8), dtype=np.float64)
    for key, human, measured, age in zip(state.track_ids, state.human_states, state.observed, state.ages):
        rows[key] = [human.px, human.py, human.vx, human.vy, human.radius, age, measured, 1]
    return rows


def snapshot(env, observer):
    indices = {human: index for index, human in enumerate(env.humans)}
    return {"robot": env.robot.get_full_state().to_array(),
            "humans": [human.get_full_state().to_array() for human in env.humans],
            "preferred": [None if human.policy._last_pref_vel is None else human.policy._last_pref_vel.copy()
                          for human in env.humans],
            "time": env.global_time,
            "tracks": [(indices[human], key, measurement.to_array(), stamp)
                       for human, (key, measurement, stamp) in observer.tracks.items()]}


def collect_case(task):
    name, geometry, case = task
    cfg = configuration_for_audit()
    teacher = orca_teacher(cfg)
    env = environment(cfg, teacher, geometry, 5)
    env.reset(options={"test_case": case})
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    frames, commands, rewards = [], [], []
    while True:
        state = observer.observe(env)
        saved = snapshot(env, observer)
        frames.append({"legal": legal_actors(state), "robot": saved["robot"], "snapshot": saved,
                       "visible": observer.counts["visible"], "retained_hidden": observer.counts["retained_hidden"]})
        action = teacher.predict(state)
        _, reward, done, truncated, info = env.step(action)
        commands.append([action.vx, action.vy])
        rewards.append(reward)
        if done or truncated:
            break
    return {"split": name, "geometry": geometry, "case": case, "frames": frames,
            "actions": commands, "rewards": rewards, "terminal": info["event"]}


def own_history(frames, tick, key, length=24):
    result = np.zeros((length, 8))
    selected = frames[max(0, tick - length + 1):tick + 1]
    for index, frame in enumerate(selected, start=length - len(selected)):
        if key < len(frame["legal"]):
            result[index] = frame["legal"][key]
    return result


def context(frame, key):
    actors = frame["legal"]
    own = actors[key]
    others = [row for index, row in enumerate(actors) if index != key and row[7] > 0]
    others.sort(key=lambda row: np.linalg.norm(row[:2] - own[:2]))
    padded = np.zeros((4, 8))
    if others:
        padded[:len(others)] = others
    return np.concatenate((own, frame["robot"], padded.ravel()))


def history_statistics(history, birth):
    measured = history[(history[:, 6] > 0) & (history[:, 7] > 0)]
    if not len(measured):
        return np.concatenate((birth, np.zeros(16)))
    velocity = measured[:, 2:4]
    return np.concatenate((birth, measured[0, :4], velocity.mean(0), velocity.std(0),
                           velocity[-1] - velocity[0], measured[-1, :2] - measured[0, :2],
                           np.asarray([len(measured), np.linalg.norm(velocity, axis=1).min(),
                                       np.linalg.norm(velocity, axis=1).max(), float(len(measured) > 1)])))


def build_samples(episodes, protocol):
    result = []
    dt = .25
    for episode in episodes:
        frames = episode["frames"]
        births = {}
        for tick, frame in enumerate(frames):
            for key, actor in enumerate(frame["legal"]):
                if actor[6] > 0 and actor[7] > 0 and key not in births:
                    births[key] = actor[:2].copy()
            if tick % protocol["sample_stride"] or tick + 8 >= len(frames):
                continue
            indices = {key: index for index, key, _, _ in frame["snapshot"]["tracks"]}
            for key, actor in enumerate(frame["legal"]):
                if actor[7] == 0:
                    continue
                index = indices[key]
                truth = frame["snapshot"]["humans"][index]
                pref = frame["snapshot"]["preferred"][index]
                if pref is None:
                    continue
                history = own_history(frames, tick, key)
                future = [frames[tick + round(seconds / dt)]["snapshot"]["humans"][index][:2] - actor[:2]
                          for seconds in protocol["future_seconds"]]
                result.append({"split": episode["split"], "geometry": episode["geometry"], "case": episode["case"],
                               "tick": tick, "key": key, "context": context(frame, key), "history": history,
                               "statistics": history_statistics(history, births[key]), "birth": births[key],
                               "target": np.concatenate((truth[5:7] - actor[:2], pref, *future)),
                               "goal_distance": float(np.linalg.norm(truth[5:7] - truth[:2])),
                               "hidden": bool(not actor[6]),
                               "near_robot": bool(np.linalg.norm(actor[:2] - frame["robot"][:2]) < 2),
                               "motion_change": bool(len(history[(history[:, 6] > 0) & (history[:, 7] > 0)]) >= 2
                                   and np.linalg.norm(history_statistics(history, births[key])[10:12]) >= .2)})
    return result


def design(rows, mode):
    current = np.stack([row["context"] for row in rows])
    if mode == "statistics":
        return np.concatenate((current, np.stack([row["statistics"] for row in rows])), axis=1)
    length = 1 if mode == "current" else int(mode[len("history"):])
    past = np.stack([row["history"][:-1].copy() for row in rows])
    past[:, :24 - length] = 0
    return np.concatenate((current, past.reshape(len(rows), -1)), axis=1)


def radial_features(x, centers, scale):
    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(x, centers, metric="sqeuclidean") / max(1, x.shape[1])
    return np.concatenate((x, np.exp(-distances / (2 * scale ** 2))), axis=1)


def fit_probe(train, validation, mode, protocol, landmark_indices, design_fn=design):
    from sklearn.linear_model import Ridge

    x, xv = design_fn(train, mode), design_fn(validation, mode)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1
    x, xv = (x - mean) / std, (xv - mean) / std
    y, yv = np.stack([row["target"] for row in train]), np.stack([row["target"] for row in validation])
    ym, ys = y.mean(0), y.std(0).clip(min=1e-6)
    centers = x[landmark_indices]
    best = None
    for scale in protocol["length_scales"]:
        features = radial_features(x, centers, scale)
        vf = radial_features(xv, centers, scale)
        for alpha in protocol["ridge_alphas"]:
            model = Ridge(alpha=alpha, solver="cholesky").fit(features, (y - ym) / ys)
            prediction = model.predict(vf) * ys + ym
            error = float(np.mean(((prediction - yv) / ys) ** 2))
            if best is None or error < best["validation_error"]:
                best = {"mode": mode, "mean": mean, "std": std, "centers": centers, "scale": scale,
                        "coef": model.coef_, "intercept": model.intercept_, "ymean": ym, "ystd": ys,
                        "alpha": alpha, "validation_error": error, "coefficients": int(model.coef_.size)}
    return best


def predict(model, rows, design_fn=design):
    x = (design_fn(rows, model["mode"]) - model["mean"]) / model["std"]
    features = radial_features(x, model["centers"], model["scale"])
    return (features @ model["coef"].T + model["intercept"]) * model["ystd"] + model["ymean"]


def errors(prediction, rows):
    target = np.stack([row["target"] for row in rows])
    cosine = (prediction[:, :2] * target[:, :2]).sum(1) / np.maximum(
        np.linalg.norm(prediction[:, :2], axis=1) * np.linalg.norm(target[:, :2], axis=1), 1e-12)
    return {"goal_angle_degrees": np.degrees(np.arccos(np.clip(cosine, -1, 1))),
            "goal_position_error_m": np.linalg.norm(prediction[:, :2] - target[:, :2], axis=1),
            "preferred_velocity_error_mps": np.linalg.norm(prediction[:, 2:4] - target[:, 2:4], axis=1),
            "future_1s_error_m": np.linalg.norm(prediction[:, 4:6] - target[:, 4:6], axis=1),
            "future_2s_error_m": np.linalg.norm(prediction[:, 6:8] - target[:, 6:8], axis=1)}


def summarize_errors(metrics, rows):
    groups = {"all": np.ones(len(rows), dtype=bool), "hidden": np.asarray([row["hidden"] for row in rows]),
              "visible": np.asarray([not row["hidden"] for row in rows]),
              "near_robot": np.asarray([row["near_robot"] for row in rows]),
              "motion_change": np.asarray([row["motion_change"] for row in rows])}
    result = {}
    for group, chosen in groups.items():
        result[group] = {"samples": int(chosen.sum())}
        for name, values in metrics.items():
            selected = chosen.copy()
            if name == "goal_angle_degrees":
                selected &= np.asarray([row["goal_distance"] >= .25 for row in rows])
            result[group][name] = float(values[selected].mean()) if selected.any() else None
    return result


def paired_errors(current, other, rows, repetitions):
    result = {}
    for name in current:
        case_means = []
        for case in sorted({row["case"] for row in rows}):
            chosen = np.asarray([row["case"] == case and (name != "goal_angle_degrees" or row["goal_distance"] >= .25)
                                 for row in rows])
            if chosen.any():
                case_means.append(float((current[name][chosen] - other[name][chosen]).mean()))
        values = np.asarray(case_means)
        if not len(values):
            result[name] = {"cases": 0, "current_minus_other": None, "paired_case_bootstrap_95": None}
            continue
        rng = np.random.default_rng(719)
        means = values[rng.integers(len(values), size=(repetitions, len(values)))].mean(1)
        result[name] = {"cases": len(values), "current_minus_other": float(values.mean()),
                        "paired_case_bootstrap_95": np.quantile(means, [.025, .975]).tolist()}
    return result


def probes(data, output):
    protocol = data["protocol"]
    rows = build_samples(data["episodes"], protocol)
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    indices = np.random.default_rng(713).choice(len(train), min(protocol["landmarks"], len(train)), replace=False)
    modes = ["current"] + [f"history{length}" for length in protocol["history_lengths"] if length > 1] + ["statistics"]
    models = {}
    for mode in modes:
        models[mode] = fit_probe(train, validation, mode, protocol, indices)
        print("FITTED", mode, "alpha", models[mode]["alpha"], "scale", models[mode]["scale"], flush=True)
    result = {"protocol": protocol, "source_sha256": data["source_sha256"],
              "models": [{key: model[key] for key in ("mode", "alpha", "scale", "validation_error", "coefficients")}
                         for model in models.values()], "tests": {}}
    for split in ("test", "ood"):
        selected = [row for row in rows if row["split"] == split]
        metrics = {mode: errors(predict(model, selected), selected) for mode, model in models.items()}
        cv = np.zeros((len(selected), 8))
        velocity = np.stack([row["context"][2:4] for row in selected])
        cv[:, :2] = -2 * np.stack([row["context"][:2] for row in selected])
        cv[:, 2:4], cv[:, 4:6], cv[:, 6:8] = velocity, velocity, 2 * velocity
        metrics["analytic_current"] = errors(cv, selected)
        birth = cv.copy()
        birth[:, :2] = -np.stack([row["birth"] for row in selected]) - np.stack([row["context"][:2] for row in selected])
        metrics["circle_birth_prior"] = errors(birth, selected)
        result["tests"][split] = {"cases": len({row["case"] for row in selected}), "samples": len(selected),
                                   "errors": {mode: summarize_errors(value, selected) for mode, value in metrics.items()},
                                   "paired": {mode: paired_errors(metrics["current"], value, selected, protocol["bootstrap_repetitions"])
                                              for mode, value in metrics.items() if mode != "current"}}
    result["scope"] = "Offline predictive probes, not policy training. Case-disjoint fits/validation/test; square is OOD. " \
                      "Privileged goals, prior preferred velocities and future states are labels only. Window probes use " \
                      "the same fixed input width, nonlinear landmark count and regression family; statistics is a cheaper " \
                      "compressed-history control including legal first-seen location. Goal direction metrics exclude actors " \
                      "within .25m of goal. Confidence intervals are exploratory, case-paired and unadjusted for multiple outcomes."
    torch.save({"models": models, "protocol": protocol}, output / "probes.pt")
    (output / "probe_results.json").write_text(json.dumps(result, indent=2))
    return result


def check_vendor():
    from crowd_sim.envs import crowd_sim
    if Path(crowd_sim.__file__).resolve() != Path(__file__).resolve().parents[1] / "vendor/crowd_sim/envs/crowd_sim.py":
        raise RuntimeError("Use PYTHONPATH=vendor:. for the frozen simulator")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "probe"))
    parser.add_argument("--output", type=Path, default=Path("outputs/latent_information"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    check_vendor()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "collect":
        protocol = json.loads(PROTOCOL.read_text())
        tasks = [(split["name"], split["geometry"], case) for split in protocol["splits"]
                 for case in range(split["start"], split["start"] + split["count"])]
        episodes = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for episode in executor.map(collect_case, tasks):
                episodes.append(episode)
                print("COLLECTED", len(episodes), len(tasks), episode["split"], episode["case"], episode["terminal"], flush=True)
        torch.save({"protocol": protocol, "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                    "source_sha256": source_hash(), "episodes": episodes}, args.output / "episodes.pt")
    else:
        data = torch.load(args.output / "episodes.pt", map_location="cpu", weights_only=False)
        probes(data, args.output)


if __name__ == "__main__":
    main()
