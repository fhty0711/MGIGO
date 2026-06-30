"""Warm-start strategies for Frenet B-spline MPC.

Frenet parameterization makes warm-start trivial:
  - s-channel:  constant-speed forward extension  → ḃ ≈ v_target, s̈≈0, s⃛≈0
  - d-channel:  hold current lateral offset       → d̈≈0, d⃛≈0

Both initial and receding-horizon variants.
"""

from __future__ import annotations

import jax.numpy as jnp


def tangent_warmstart(gen, s0: float, v_target: float, d0: float = 0.0):
    """Initial warm-start: constant-speed extension using Greville abscissae.

    Uses the CORRECT Greville abscissae for free control points (P3..P11),
    NOT arange()*dt_knot — the clamped start makes the first few Greville
    abscissae non-uniform.

    With this formula: clamped P0,P1,P2 + Greville-based P3..P11 gives
    EXACT constant speed v_target and EXACT zero acceleration/jerk —
    the theoretically perfect warm-start for a straight road.
    """
    # Free control points at Greville abscissae × target speed
    ctrl_s = s0 + v_target * gen.greville[3:]
    ctrl_d = jnp.full((gen.n_free,), d0, dtype=jnp.float32)
    return ctrl_s, ctrl_d


def shift_warmstart(ctrl_s_old, ctrl_d_old, v_target: float, greville_free, dt_knot: float):
    """MPC receding-horizon warm-start: shift + extend previous solution.

    Shifts free control points left by one index, extends the last point
    forward at target speed. Preserves solver-optimized trajectory shape.

    Note: regenerating via tangent_warmstart is cleaner when no obstacle
    avoidance is needed. Use shift when preserving prior solution shape matters.
    """
    n = len(ctrl_s_old)
    ctrl_s = jnp.zeros_like(ctrl_s_old)
    ctrl_d = jnp.zeros_like(ctrl_d_old)

    ctrl_s = ctrl_s.at[:-1].set(ctrl_s_old[1:])
    ctrl_d = ctrl_d.at[:-1].set(ctrl_d_old[1:])

    # Extend last point: use the last Greville spacing (≈ dt_knot in uniform region)
    ctrl_s = ctrl_s.at[-1].set(ctrl_s_old[-1] + v_target * dt_knot)
    ctrl_d = ctrl_d.at[-1].set(ctrl_d_old[-1])

    return ctrl_s, ctrl_d
