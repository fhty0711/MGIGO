"""State execution: extract commanded action from plan, simulate through vehicle model.

The B-spline trajectory is the planner's *desired* path.  The vehicle model
simulates actual dynamics — acceleration limits, integration, etc.
"""

from __future__ import annotations

import jax.numpy as jnp

from Cartest.vehicle_model import FrenetVehicleModel


def extract_command(gen, s, d, s_dot, d_dot, s_ddot, d_ddot):
    """Extract commanded acceleration from the plan at t=0.

    s_ddot[0] = s_ddot0 (clamped) — the acceleration the plan starts with.
    This is what the vehicle should try to execute *now*.
    """
    return float(s_ddot[0]), float(d_ddot[0])


def execute_step(gen, s, d, s_dot, d_dot, s_ddot, d_ddot,
                 vehicle_model: FrenetVehicleModel):
    """Execute one step: command from plan → vehicle model → next state.

    Args:
        gen:   FrenetBSplineTrajectory (provides dt)
        s..d_ddot: [T] evaluated trajectory
        vehicle_model: FrenetVehicleModel instance

    Returns:
        dict: next Frenet state {s0, s_dot0, s_ddot0, d0, d_dot0, d_ddot0}
    """
    # Commanded acceleration at t=0
    s_ddot_cmd, d_ddot_cmd = extract_command(gen, s, d, s_dot, d_dot, s_ddot, d_ddot)

    # Simulate vehicle response
    s_new, d_new, s_dot_new, d_dot_new = vehicle_model.step(
        float(s[0]), float(d[0]),
        float(s_dot[0]), float(d_dot[0]),
        s_ddot_cmd, d_ddot_cmd,
    )

    # Acceleration for next step: take plan's acceleration at t=dt
    # (rather than the clipped command, to let the plan propagate forward)
    return {
        's0':      s_new,
        's_dot0':  s_dot_new,
        's_ddot0': float(s_ddot[1]),
        'd0':      d_new,
        'd_dot0':  d_dot_new,
        'd_ddot0': float(d_ddot[1]),
    }
