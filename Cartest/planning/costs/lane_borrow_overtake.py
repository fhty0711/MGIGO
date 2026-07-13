"""Lane-borrow overtake objective.

Reuses the default Lyapunov objective and replaces the lateral reference
with a smooth borrow-lane profile: start_lane -> borrow_lane -> return_lane.
"""

from __future__ import annotations

import jax.numpy as jnp

from Cartest.planning.cost import make_objective_with_lateral_reference
from Cartest.planning.costs.default_lyapunov import (
    make_constraints_config as make_default_constraints_config,
)


DEFAULT_PARAMS = {
    "omega_s": 1.0,
    "omega_d": 4.0,
    "alpha": 0.0,
    "start_lane_d": 0.0,
    "borrow_lane_d": 3.7,
    "return_lane_d": 0.0,
    "blocker_s": 80.0,
    "approach_distance": 22.0,
    "return_distance": 25.0,
    "transition_width": 8.0,
    "acc_weight": 10.0,
    "jerk_weight": 1.0,
}


def make_lateral_reference(*, start_lane_d, borrow_lane_d, return_lane_d,
                           blocker_s, approach_distance, return_distance,
                           transition_width, **_params):
    """Build a smooth lane-borrow reference d_ref(s)."""
    borrow_start_s = blocker_s - approach_distance
    return_start_s = blocker_s + return_distance

    def lateral_reference(s, _ctx):
        width = jnp.maximum(float(transition_width), 1e-3)
        borrow_gate = jnp.asarray(1.0) / (1.0 + jnp.exp(-(s - borrow_start_s) / width))
        return_gate = jnp.asarray(1.0) / (1.0 + jnp.exp(-(s - return_start_s) / width))
        return (
            start_lane_d
            + (borrow_lane_d - start_lane_d) * borrow_gate
            + (return_lane_d - borrow_lane_d) * return_gate
        )

    return lateral_reference


def make_cost(gen, **params):
    """Build the lane-borrow overtake Lyapunov objective."""
    config = dict(DEFAULT_PARAMS)
    config.update(params)
    lateral_reference = make_lateral_reference(**config)
    return make_objective_with_lateral_reference(
        gen,
        lateral_reference,
        omega_s=config["omega_s"],
        omega_d=config["omega_d"],
        alpha=config["alpha"],
        acc_weight=config["acc_weight"],
        jerk_weight=config["jerk_weight"],
    )


def make_constraints_config(**_params):
    """Build the constraint/Constran config paired with this cost."""
    return make_default_constraints_config()


make_cost.constraint_config_factory = make_constraints_config
