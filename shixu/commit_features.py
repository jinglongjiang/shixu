"""Legal features from the unmodified parent's selected successor evaluation."""

import numpy as np


SCALARS = ("goal_distance", "past_2s_progress", "window_ready", "grid_vx", "grid_vy",
           "command_vx", "command_vy", "selected_score", "top_score_gap", "selected_clearance",
           "safe_fraction", "action_switch_fraction", "robot_speed", "mean_command_change",
           "visible_fraction", "mean_track_age", "current_clearance", "goal_alignment")


def extract(policy, state, distances, grids, recent_commands):
    commands = policy.candidate_actions()
    captured = []
    critic = getattr(policy.model, "critic", policy.model)
    handle = critic.value_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().cpu().numpy().copy()))
    try:
        scores = policy.score(state, commands)
    finally:
        handle.remove()
    index = int(np.argmax(scores))
    _, _, clearances = policy.aligned_candidates(state, commands)
    robot = state.self_state
    goal = np.array([robot.gx-robot.px, robot.gy-robot.py])
    distance = float(np.linalg.norm(goal))
    past = list(distances)[-8:]+[distance]
    changes = list(grids)[-8:]+[index]
    command = np.asarray(commands[index])
    command_history = np.asarray(list(recent_commands)[-8:]+[command])
    finite = np.sort(scores[scores > -1e8])
    gap = float(finite[-1]-finite[-2]) if len(finite) > 1 else 0.
    observed = np.asarray(state.observed, bool)
    current_clearance = min((np.hypot(robot.px-h.px, robot.py-h.py)-robot.radius-h.radius
                             for h in state.human_states),default=5.)
    scale = distance*np.linalg.norm(command)
    scalar = np.asarray([distance,past[0]-past[-1] if len(past)==9 else 0.,float(len(past)==9),
        *policy.action_space[index],*command,float(scores[index]),gap,
        float(np.clip(clearances[index],-1.,5.)),float(np.mean(clearances >= .2)),
        float(np.mean(np.diff(changes)!=0)) if len(changes)>1 else 0.,
        float(np.hypot(robot.vx,robot.vy)),
        float(np.mean(np.linalg.norm(np.diff(command_history,axis=0),axis=1))) if len(command_history)>1 else 0.,
        float(np.mean(observed)) if len(observed) else 0.,
        float(np.mean(state.ages)) if state.ages else 0.,float(np.clip(current_clearance,-1.,5.)),
        float(np.dot(command,goal)/scale) if scale>0 else 0.],np.float64)
    if len(captured) != 1 or len(scalar) != len(SCALARS):
        raise ValueError("Unexpected parent representation")
    latent = captured[0].reshape(-1,captured[0].shape[-1])[index].astype(np.float64)
    feature = np.concatenate((scalar,latent))
    if not np.isfinite(feature).all():
        raise ValueError("Nonfinite legal gate input")
    return dict(features=feature,index=index,command=commands[index],score=float(scores[index]),
                clearance=float(clearances[index]),all_unsafe=not bool((clearances>=.2).any()))
