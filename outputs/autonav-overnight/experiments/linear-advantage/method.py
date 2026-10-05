"""Episode-balanced linear estimate of the original commit advantage."""

import numpy as np

SPEC = dict(name="linear-advantage", parent="rule-forward", family="linear", hidden=[],
            hypothesis="Learn delta-Q directly from the fixed lawful features; commit only if predicted advantage is positive.")


def fit(rows, budget):
    x = np.array([r["features"] for r in rows], np.float64)
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
    normalized = (features-fitted["mean"])/fitted["scale"]
    return bool(normalized@fitted["coefficients"][:-1]+fitted["coefficients"][-1] > 0.)


def parameters(fitted):
    return len(fitted["coefficients"])
