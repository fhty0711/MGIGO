"""Vehicle models for forward simulation.

Receives commanded acceleration (s̈_cmd, d̈_cmd) from the plan,
simulates realistic vehicle response, returns the full next state.

Interface:
  step(s0, d0, s_dot0, d_dot0, s_ddot_cmd, d_ddot_cmd)
      → (s, d, s_dot, d_dot, s_ddot, d_ddot)
"""

from __future__ import annotations

import jax.numpy as jnp


class FrenetVehicleModel:
    """Point-mass in Frenet (s, d) with friction-circle limits.

    μ = 0.85 → a_max ≈ 8.3 m/s²  (dry asphalt).
    """

    def __init__(self, mu: float = 0.85, dt: float = 0.1):
        self.a_max = mu * 9.81
        self.dt = dt

    def step(self, s0, d0, s_dot0, d_dot0,
             s_ddot_cmd, d_ddot_cmd):
        # ── Friction circle: clip combined, preserve direction ──
        a_cmd = jnp.sqrt(s_ddot_cmd ** 2 + d_ddot_cmd ** 2)
        scale = jnp.minimum(1.0, self.a_max / (a_cmd + 1e-6))
        ax = s_ddot_cmd * scale
        ay = d_ddot_cmd * scale

        # ── Euler integration ──
        s_new     = s0     + s_dot0 * self.dt
        s_dot_new = s_dot0 + ax     * self.dt
        d_new     = d0     + d_dot0 * self.dt
        d_dot_new = d_dot0 + ay     * self.dt

        return s_new, d_new, s_dot_new, d_dot_new, ax, ay
