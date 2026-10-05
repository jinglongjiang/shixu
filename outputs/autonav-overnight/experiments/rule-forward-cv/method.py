"""Veto initiation if lawful CV predicts a sub-margin encounter during the hold."""

SPEC = dict(name="rule-forward-cv", parent="rule-forward", family="rule", hidden=[],
            hypothesis="Combine progress and existing-margin CV feasibility, without fitting thresholds.")


def fit(rows, budget):
    return {}


def accept(features, fitted):
    return bool(features[23] > 0. and features[8] >= .2)


def parameters(fitted):
    return 0
