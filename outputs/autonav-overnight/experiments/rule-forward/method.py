"""Accept only a proposed hold that makes forward progress under lawful CV."""

SPEC = dict(name="rule-forward", parent="rule-accept-all", family="rule", hidden=[],
            hypothesis="Reject holds whose two-second smoothed CV endpoint does not reduce own-goal distance.")


def fit(rows, budget):
    return {}


def accept(features, fitted):
    return bool(features[23] > 0.)


def parameters(fitted):
    return 0
