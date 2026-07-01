"""Execution: bridge from plan to vehicle model.

1. Extract commanded acceleration from the plan at t=0.
2. Pass to vehicle model for forward simulation.
3. Return the model's full next state — execute trusts the model.
"""

from __future__ import annotations


def execute_step(gen, s, d, s_dot, d_dot, s_ddot, d_ddot,
                 vehicle_model):
    """Plan → vehicle model → next Frenet state.

    Args:
        gen:   FrenetBSplineTrajectory
        s..d_ddot: [T] evaluated trajectory
        vehicle_model: object with step(s0,d0,s_dot0,d_dot0,cmd_s,cmd_d)

    Returns:
        dict: {s0, s_dot0, s_ddot0, d0, d_dot0, d_ddot0}
    """
    # 1. Plan's intended acceleration (t=dt, after clamped boundary releases)
    s_ddot_cmd = float(s_ddot[1])
    d_ddot_cmd = float(d_ddot[1])

    # 2. Forward simulation through vehicle model
    s_new, d_new, s_dot_new, d_dot_new, ax, ay = vehicle_model.step(
        float(s[0]), float(d[0]),
        float(s_dot[0]), float(d_dot[0]),
        s_ddot_cmd, d_ddot_cmd,
    )

    # 3. Return model's output as next state
    return {
        's0':      s_new,
        's_dot0':  s_dot_new,
        's_ddot0': float(ax),
        'd0':      d_new,
        'd_dot0':  d_dot_new,
        'd_ddot0': float(ay),
    }
