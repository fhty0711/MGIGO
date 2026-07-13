"""Curved cruise scenario on a variable-curvature (clothoid) reference.

Road layout (R = 40 m): straight - clothoid - circular arc - clothoid - straight,
total turn ~150 deg (< 180 deg so the centerline does not self-intersect).
The ego cruises through the curve, exercising the curvature-coupled Frenet
kinematics on a kappa(s) that varies along s (unlike the constant-kappa
CircularReference).
"""

from __future__ import annotations

import jax.numpy as jnp

from Cartest.core.reference_path import PiecewiseCurvatureReference

# R=40m variable-curvature centerline: straight - clothoid - arc - clothoid - straight
# total turn = (clothoid 25 + arc 80) / R = 105/40 = 2.625 rad ~ 150 deg
CURVED_SEGMENTS = [
    (0.0, 40.0, 0.0, 0.0),                  # straight
    (40.0, 65.0, 0.0, 1.0 / 40.0),          # clothoid kappa: 0 -> 1/40
    (65.0, 145.0, 1.0 / 40.0, 1.0 / 40.0),  # circular arc R=40, 80 m
    (145.0, 170.0, 1.0 / 40.0, 0.0),        # clothoid kappa: 1/40 -> 0
    (170.0, 250.0, 0.0, 0.0),               # straight
]

COST_NAME = "default_lyapunov"
COST_PARAMS = {"omega_s": 1.0, "omega_d": 4.0, "alpha": 0.0}

SCENARIO = {
    "ref_path":      PiecewiseCurvatureReference(CURVED_SEGMENTS),
    "road":          {"lane_hw": 4.0},
    "obstacles":     [],
    "safety":        {"obs_safe_dist": 0.1, "a_brake": 8.0,
                      "v_min": 2.0, "v_max": 35.0,
                      "acc_max": 5.0, "jerk_max": 2.0},
    # v_target kept low: on the R=40 arc, lateral accel a_n = kappa * v^2
    # = 0.025 * 10^2 = 2.5 m/s^2, comfortably below acc_max = 5.0.
    "behavior":      {"v_target": 10.0},
    "cost":          {"name": COST_NAME, "params": COST_PARAMS},
    "ego":           {"s": 0.0, "s_dot": 10.0, "s_ddot": 0.0,
                      "d":  0.0, "d_dot":  0.0, "d_ddot": 0.0,
                      "psi": 0.0},
}
