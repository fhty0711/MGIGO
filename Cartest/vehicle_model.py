"""Vehicle model for simulation — decouples execution from planning.

The B-spline trajectory is the planner's *desired* path.  The vehicle
model simulates how the actual vehicle responds, with acceleration
limits and integration in Frenet coordinates.
"""

from __future__ import annotations

import jax.numpy as jnp


class FrenetVehicleModel:
    """Point-mass vehicle in Frenet (s, d) with acceleration limits.

    State:  (s, d, s_dot, d_dot) — position + velocity
    Input:  (s_ddot_cmd, d_ddot_cmd) — commanded acceleration from plan
    """

    def __init__(self, acc_max: float = 5.0, dt: float = 0.1):
        self.acc_max = acc_max
        self.dt = dt

    def step(self, s0, d0, s_dot0, d_dot0,
             s_ddot_cmd, d_ddot_cmd):
        """Euler-integrate commanded acceleration for one time step.

        Args:
            s0, d0:           current position [scalar]
            s_dot0, d_dot0:   current velocity [scalar]
            s_ddot_cmd:       desired longitudinal acc from plan [scalar]
            d_ddot_cmd:       desired lateral acc from plan [scalar]

        Returns:
            (s, d, s_dot, d_dot) after dt
        """
        # Clip acceleration to vehicle limits
        ax = jnp.clip(s_ddot_cmd, -self.acc_max, self.acc_max)
        ay = jnp.clip(d_ddot_cmd, -self.acc_max, self.acc_max)

        s_new     = s0     + s_dot0 * self.dt
        s_dot_new = s_dot0 + ax     * self.dt
        d_new     = d0     + d_dot0 * self.dt
        d_dot_new = d_dot0 + ay     * self.dt

        return s_new, d_new, s_dot_new, d_dot_new
