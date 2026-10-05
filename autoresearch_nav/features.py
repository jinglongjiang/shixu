"""Frozen, lawful proposal features; no future simulator state is accepted."""

import numpy as np


NAMES = ("goal_distance", "past_progress", "grid_speed", "command_speed", "goal_alignment",
         "selected_score", "score_gap", "quarter_clearance", "hold_cv_clearance", "current_clearance",
         "safe_fraction", "nearby_fraction", "min_ttc", "visible_fraction", "mean_age", "max_age",
         "grid_switch_fraction", "command_change", "robot_speed", "command_robot_alignment",
         "mean_recent_goal_progress", "heading_change", "relative_speed", "hold_goal_progress")


def build(policy, state, commands, scores, clearances, index):
    robot = state.self_state
    goal = np.array([robot.gx-robot.px, robot.gy-robot.py])
    distance = float(np.linalg.norm(goal))
    command = np.asarray(commands[index])
    grid = np.asarray(policy.action_space[index])
    velocity = np.array([robot.vx, robot.vy])
    humans = np.array([h.to_array() for h in state.human_states], np.float64).reshape(-1, 5)
    offset = humans[:, :2]-[robot.px, robot.py]
    radii = robot.radius+humans[:, 4]
    gaps = np.linalg.norm(offset, axis=1)-radii
    relative = humans[:, 2:4]-command
    vv = np.sum(relative*relative, axis=1)
    ttc = np.clip(-np.sum(offset*relative, axis=1)/np.maximum(vv, 1e-12), 0, 4)
    closest = np.linalg.norm(offset+ttc[:, None]*relative, axis=1)-radii
    ttc = np.where((closest < .2) & (np.sum(offset*relative, axis=1) < 0), ttc, 4.)
    position, moving, hold_min = np.array([robot.px, robot.py]), command.copy(), 5.
    # Hypothetical CV positions are derived only from lawful current measurements.
    for step in range(8):
        if step:
            moving = .3*moving+.7*grid
        position = position+.25*moving
        predicted = humans[:, :2]+(step+1)*.25*humans[:, 2:4]
        hold_min = min(hold_min, float(np.min(np.linalg.norm(predicted-position, axis=1)-radii)) if len(humans) else 5.)
    history = np.asarray(list(policy.recent_commands)+[command])
    grids = list(policy.recent_grids)+[index]
    meaningful_scores = np.sort(scores[scores > -1e8])
    def alignment(a, b):
        norm = np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a, b)/norm) if norm > 1e-12 else 0.
    changes = np.diff(history, axis=0)
    value = [distance, policy.distances[0]-distance, np.linalg.norm(grid), np.linalg.norm(command),
        alignment(command, goal), scores[index], meaningful_scores[-1]-meaningful_scores[-2] if len(meaningful_scores)>1 else 0.,
        np.clip(clearances[index], -1, 5), np.clip(hold_min, -1, 5), np.clip(min(gaps, default=5.), -1, 5),
        np.mean(clearances >= .2), np.mean(gaps < 2.) if len(humans) else 0., min(ttc, default=4.),
        np.mean(state.observed) if len(humans) else 0., np.mean(state.ages) if len(humans) else 0., max(state.ages, default=0.),
        np.mean(np.diff(grids) != 0) if len(grids)>1 else 0., np.mean(np.linalg.norm(changes, axis=1)) if len(changes) else 0.,
        np.linalg.norm(velocity), alignment(command, velocity), (policy.distances[0]-distance)/2.,
        1-alignment(command, history[-2]) if len(history)>1 else 0., np.mean(np.linalg.norm(relative, axis=1)) if len(humans) else 0.,
        distance-np.linalg.norm(position-[robot.gx, robot.gy])]
    result = np.asarray(value, np.float64)
    if result.shape != (len(NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid proposal features")
    return result
