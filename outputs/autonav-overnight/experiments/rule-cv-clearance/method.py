"""Veto initiation if lawful CV predicts a sub-margin encounter during the hold."""

SPEC = dict(name="rule-cv-clearance", parent="rule-accept-all", family="rule", hidden=[],
            hypothesis="Use the unchanged 0.2m margin only as an initiation veto on the two-second CV hold.")


def fit(rows, budget):
    return {}


def accept(features, fitted):
    return bool(features[8] >= .2)


def parameters(fitted):
    return 0
