"""Baseline experiment. This is the sole editable research file."""

SPEC = dict(name="rule-accept-all", parent=None, family="rule", hidden=[],
            hypothesis="Baseline V0.1: accept every original low-progress proposal.")


def fit(rows, budget):
    return {}


def accept(features, fitted):
    return True


def parameters(fitted):
    return 0
