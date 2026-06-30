"""Cost function builders for Frenet B-spline MPC.

Frenet kinematics are linear in control points, so the objective
landscape is well-conditioned for black-box IGO optimization.
"""

from __future__ import annotations

import jax.numpy as jnp


def _eval_traj(theta, ctx, gen):
    """Unpack theta → evaluate Frenet trajectory.

    theta = [ctrl_s_free(9) | ctrl_d_free(9)]  from M=2 IGO blocks.
    """
    n = gen.n_free
    ctrl_s_free = theta[:n]
    ctrl_d_free = theta[n:2 * n]

    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
        ctrl_s_free, ctrl_d_free,
        ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
        ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"],
    )
    return s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot


def make_objective(gen):
    """Build objective function: lateral tracking + speed tracking.

    No explicit smoothness penalty — the B-spline parameterization
    provides C⁴ continuity natively.
    """

    def obj_fn(theta, ctx):
        _, d, s_dot, d_dot, _, _, _, _ = _eval_traj(theta, ctx, gen)

        # Speed tracking: v = sqrt(ḃ² + ḋ²) → target
        v = jnp.sqrt(s_dot ** 2 + d_dot ** 2)
        speed_cost = jnp.sum((v - ctx["v_ref"]) ** 2)

        # Lateral tracking: d → 0 (lane center)
        lat_cost = jnp.sum(d ** 2)

        return speed_cost + lat_cost

    return obj_fn
