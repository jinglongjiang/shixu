"""Cost-sensitive logistic initiation, weighted by the original return difference."""

import numpy as np
import torch

SPEC = dict(name="logistic-utility", parent="linear-advantage", family="logistic", hidden=[],
            hypothesis="Fit the advantage sign with |delta-Q| weights rather than value MSE; zero advantage has zero training weight.")


def fit(rows, budget):
    torch.manual_seed(budget["seed"])
    x = np.array([r["features"] for r in rows], np.float64)
    y = np.array([r["y"] for r in rows], np.float64)
    counts = {}
    for r in rows:
        key = r["seed"], r["case"]
        counts[key] = counts.get(key, 0)+1
    w = np.array([1/counts[r["seed"], r["case"]] for r in rows])
    mean = np.average(x, axis=0, weights=w)
    scale = np.maximum(np.sqrt(np.average((x-mean)**2, axis=0, weights=w)), 1e-6)
    inputs = torch.tensor((x-mean)/scale, dtype=torch.float32)
    targets = torch.tensor(y > 0., dtype=torch.float32)
    weights = torch.tensor(w*np.abs(y), dtype=torch.float32)
    weights = weights/weights.sum()
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=budget["lr"], weight_decay=budget["weight_decay"])
    for _ in range(budget["epochs"]):
        loss = (weights*torch.nn.functional.binary_cross_entropy_with_logits(model(inputs).flatten(), targets, reduction="none")).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    layers = [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy()) for m in model if isinstance(m, torch.nn.Linear)]
    return dict(mean=mean, scale=scale, layers=layers)


def accept(features, fitted):
    x = (features-fitted["mean"])/fitted["scale"]
    for i, (weight, bias) in enumerate(fitted["layers"]):
        x = weight@x+bias
        if i+1 < len(fitted["layers"]):
            x = np.maximum(x, 0.)
    return bool(float(x[0]) > 0.)


def parameters(fitted):
    return sum(w.size+b.size for w,b in fitted["layers"])
