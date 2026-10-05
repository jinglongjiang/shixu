"""Direct expected original-return advantage with the same tiny deployment network."""

import numpy as np
import torch

SPEC = dict(name="mlp-advantage", parent="mlp-utility", family="mlp", hidden=[16],
            hypothesis="Replace sign-classification loss by episode-balanced delta-Q regression; keep architecture, inputs and compute budget fixed.")


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
    target_scale = max(float(np.sqrt(np.average(y*y, weights=w))), 1e-6)
    targets = torch.tensor(y/target_scale, dtype=torch.float32)
    weights = torch.tensor(w, dtype=torch.float32)
    weights = weights/weights.sum()
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=budget["lr"], weight_decay=budget["weight_decay"])
    for _ in range(budget["epochs"]):
        loss = (weights*(model(inputs).flatten()-targets).square()).sum()
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
