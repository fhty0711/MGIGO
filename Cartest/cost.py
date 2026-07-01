"""Cost function for Frenet B-spline MPC.

Objective: lateral tracking (d → 0) + speed tracking (v → v_target).

Both quantities from frenet_traj — cost and constraints share the
same vehicle-state pipeline, no more mixing raw Frenet derivatives.
"""

from __future__ import annotations

import jax.numpy as jnp


def _eval_all(theta, ctx, gen):
    """Unpack theta → Frenet trajectory → vehicle states.

    Returns (d, v) — the two quantities the objective cares about.
    """
    n = gen.n_free
    ctrl_s_free = theta[:n]
    ctrl_d_free = theta[n:2 * n]

    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
        ctrl_s_free, ctrl_d_free,
        ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
        ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"],
    )

    # Vehicle states [T, 9]: x, y, v, ψ, a_long, a_lat, j_long, j_lat, steer
    st = gen.to_vehicle_states(s, d, s_dot, d_dot,
                               s_ddot, d_ddot, s_dddot, d_dddot)
    v = st[:, 2]  # total speed (with (1-d·κ_r) correction)
    return d, v


def make_objective(gen):
    """Build objective: d → 0 (lane centre) + v → v_target.

    Smoothness is natively provided by the B-spline C⁴ continuity.
    """

    def obj_fn(theta, ctx):
        d, v = _eval_all(theta, ctx, gen)

        # Speed tracking: total speed → target
        speed_cost = jnp.sum((v - ctx["v_ref"]) ** 2)

        # Lateral tracking: Frenet d → 0 (lane centre)
        lat_cost = jnp.sum(d ** 2)

        return speed_cost + lat_cost

    return obj_fn
