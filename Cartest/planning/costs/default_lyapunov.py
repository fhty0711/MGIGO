"""Default Lyapunov objective used by the Cartest demo."""

from __future__ import annotations

from copy import deepcopy

from Cartest.planning.cost import make_objective


DEFAULT_PARAMS = {
    "omega_s": 1.0,
    "omega_d": 4.0,
    "alpha": 0.0,
    "acc_weight": 0.0,
    "jerk_weight": 0.0,
}

DEFAULT_CONSTRAINTS = {
    "enabled": ("obs", "lane", "speed", "acc", "jerk"),
    "specs": {
        "obs":   {"priority": 1, "mode": "hard", "aggregate": "max", "transform": "hard"},
        "lane":  {"priority": 2, "mode": "soft", "aggregate": "q95", "transform": "soft"},
        "speed": {"priority": 3, "mode": "soft", "aggregate": "max", "transform": "soft"},
        "acc":   {"priority": 4, "mode": "soft", "aggregate": "max", "transform": "soft"},
        "jerk":  {"priority": 5, "mode": "soft", "aggregate": "max", "transform": "soft"},
    },
    "constran": {"k_inner": 1.0, "obj_transform": "standard"},
}


def make_cost(gen, **params):
    """Build the default Lyapunov objective for a trajectory generator."""
    config = dict(DEFAULT_PARAMS)
    config.update(params)
    return make_objective(gen, **config)


def make_constraints_config(**_params):
    """Build the constraint/Constran config paired with this cost."""
    return deepcopy(DEFAULT_CONSTRAINTS)


make_cost.constraint_config_factory = make_constraints_config
