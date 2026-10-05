"""Simplify the advantage regression to progress and encounter geometry."""

import numpy as np

SPEC = dict(name="linear-physical", parent="linear-advantage", family="linear", hidden=[],
            hypothesis="Remove critic and observational nuisance features; estimate advantage from hold progress, clearance and motion continuity.")
INDICES = (0, 2, 4, 8, 10, 12, 16, 17, 21, 23)


def fit(rows, budget):
    x = np.array([r["features"] for r in rows], np.float64)[:, INDICES]
    y = np.array([r["y"] for r in rows], np.float64)
    counts = {}
    for r in rows:
        key = r["seed"], r["case"]
        counts[key] = counts.get(key, 0)+1
    w = np.array([1/counts[r["seed"], r["case"]] for r in rows])
    mean = np.average(x, axis=0, weights=w)
    scale = np.maximum(np.sqrt(np.average((x-mean)**2, axis=0, weights=w)), 1e-6)
    design = np.column_stack(((x-mean)/scale, np.ones(len(x))))
    penalty = budget["ridge"]*np.eye(design.shape[1])
    penalty[-1, -1] = 0.
    coefficients = np.linalg.solve(design.T@(w[:, None]*design)+penalty, design.T@(w*y))
    return dict(mean=mean, scale=scale, coefficients=coefficients)


def accept(features, fitted):
    normalized = (features[list(INDICES)]-fitted["mean"])/fitted["scale"]
    return bool(normalized@fitted["coefficients"][:-1]+fitted["coefficients"][-1] > 0.)


def parameters(fitted):
    return len(fitted["coefficients"])
