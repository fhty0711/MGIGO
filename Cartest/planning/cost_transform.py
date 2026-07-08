"""Constructive Lyapunov cost with error-space linear transformation.

Supports two levels of coupling:

  1. Scalar coupling (T ∈ GL(2)):
       [z̃]   [1   α] [e_s]         det = 1 − αβ > 0
       [w̃] = [β   1] [e_d]
     Same α,β applied at every order (position, velocity, acceleration).

  2. Cross‑order coupling (C ∈ ℝ³ˣ³ blocks):
       [𝐳_A]   [ I     C_{B→A} ] [𝐞_A]      𝐞_A = [e_s, ė_s, ë_s]ᵀ
       [𝐳_B] = [C_{A→B}   I    ] [𝐞_B]      𝐞_B = [e_d, ė_d, ë_d]ᵀ

     C_{B→A}[i,j] couples lateral order‑j error into longitudinal order‑i.
     Stability requires det(I − C_{B→A}·C_{A→B}) ≠ 0.

All variants reduce to the standard decoupled cost when coupling matrices
are zero.
"""

from __future__ import annotations

import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════
# Context builder
# ═══════════════════════════════════════════════════════════════════════

def build_context(gen, state, v_ref, lane_hw, obs_pos, obs_rad):
    return {
        'v_ref':   jnp.full(gen.T, v_ref) if isinstance(v_ref, (int, float)) else v_ref,
        'lane_hw': lane_hw,
        'obs_pos': obs_pos,
        'obs_rad': obs_rad,
        **state.to_ctx(),
    }


# ═══════════════════════════════════════════════════════════════════════
# 1. Scalar coupling (degenerate — identical α,β at all orders)
# ═══════════════════════════════════════════════════════════════════════

def make_objective_transform(gen,
                             omega_z: float = 1.0,
                             omega_w: float = 4.0,
                             alpha_t: float = 0.0,
                             beta_t:  float = 0.0):
    """Scalar-coupled Lyapunov cost.

    Equivalent to C_{B→A} = α_t·I, C_{A→B} = β_t·I in the cross‑order
    formulation.
    """
    if 1.0 - alpha_t * beta_t <= 0:
        raise ValueError(f"T must be invertible: 1 − α·β ≤ 0")

    t_arr = jnp.arange(gen.T) * gen.dt

    def obj_fn(theta, ctx):
        s, d, d_dot, d_ddot, d_dddot, s_dot, s_ddot, s_dddot = _eval_all(
            gen, theta[:gen.n_free], theta[gen.n_free:2*gen.n_free], ctx)

        # Reference
        v_tgt = ctx["v_ref"][0]
        lam   = omega_z
        exp_term = jnp.exp(-lam * t_arr)
        dv = ctx["s_dot0"] - v_tgt
        s_ref      = ctx["s0"] + v_tgt * t_arr + dv / lam * (1.0 - exp_term)
        s_ref_dot  = v_tgt + dv * exp_term
        s_ref_ddot = -dv * lam * exp_term

        es = s - s_ref;          ed = d
        es_dot = s_dot - s_ref_dot;  ed_dot = d_dot
        es_ddot = s_ddot - s_ref_ddot; ed_ddot = d_ddot

        z_tilde      = es      + alpha_t * ed
        w_tilde      = beta_t  * es + ed
        z_tilde_dot  = es_dot  + alpha_t * ed_dot
        w_tilde_dot  = beta_t  * es_dot + ed_dot
        z_tilde_ddot = es_ddot + alpha_t * ed_ddot
        w_tilde_ddot = beta_t  * es_ddot + ed_ddot

        return _lyapunov_cost(z_tilde, w_tilde,
                              z_tilde_dot, w_tilde_dot,
                              z_tilde_ddot, w_tilde_ddot,
                              omega_z, omega_w)

    return obj_fn


# ═══════════════════════════════════════════════════════════════════════
# 2. Cross‑order coupling (general C_{B→A}, C_{A→B})
# ═══════════════════════════════════════════════════════════════════════

def make_objective_cross_order(gen,
                               omega_z: float = 1.0,
                               omega_w: float = 4.0,
                               C_ba: tuple = None,   # 3×3: lateral → longitudinal
                               C_ab: tuple = None):  # 3×3: longitudinal → lateral
    """Cross‑order coupled Lyapunov cost.

    Args:
        gen:     FrenetBSplineTrajectory
        omega_z: Lyapunov gain for 𝐳_A (longitudinal‑dominant) channel
        omega_w: Lyapunov gain for 𝐳_B (lateral‑dominant) channel
        C_ba:    3×3 matrix (9 elements row‑major).
                 C_ba[i,j] couples lateral order‑j into longitudinal order‑i.
                 Order indices: 0=position, 1=velocity, 2=acceleration.
                 None → zeros.
        C_ab:    3×3 matrix coupling longitudinal → lateral.  None → zeros.

    Returns:
        obj_fn(theta, ctx) → scalar cost
    """
    # Parse coupling matrices
    if C_ba is None:
        C_ba_flat = jnp.zeros(9)
    else:
        C_ba_flat = jnp.array(C_ba, dtype=jnp.float32).reshape(-1)

    if C_ab is None:
        C_ab_flat = jnp.zeros(9)
    else:
        C_ab_flat = jnp.array(C_ab, dtype=jnp.float32).reshape(-1)

    # Stability check: det(I − C_{B→A}·C_{A→B}) > 0
    C_ba_mat = C_ba_flat.reshape(3, 3)
    C_ab_mat = C_ab_flat.reshape(3, 3)
    det = jnp.linalg.det(jnp.eye(3) - C_ba_mat @ C_ab_mat)
    if det <= 0:
        raise ValueError(f"det(I − C_ba·C_ab) = {det:.4f} ≤ 0 — coupling not invertible")

    t_arr = jnp.arange(gen.T) * gen.dt

    def obj_fn(theta, ctx):
        s, d, d_dot, d_ddot, d_dddot, s_dot, s_ddot, s_dddot = _eval_all(
            gen, theta[:gen.n_free], theta[gen.n_free:2*gen.n_free], ctx)

        # Reference
        v_tgt = ctx["v_ref"][0]
        lam   = omega_z
        exp_term = jnp.exp(-lam * t_arr)
        dv = ctx["s_dot0"] - v_tgt
        s_ref      = ctx["s0"] + v_tgt * t_arr + dv / lam * (1.0 - exp_term)
        s_ref_dot  = v_tgt + dv * exp_term
        s_ref_ddot = -dv * lam * exp_term

        # Raw error vectors (3 components each)
        eA_0 = s - s_ref;          eB_0 = d
        eA_1 = s_dot - s_ref_dot;  eB_1 = d_dot
        eA_2 = s_ddot - s_ref_ddot; eB_2 = d_ddot

        # Cross‑order coupling: 𝐳_A = 𝐞_A + C_{B→A} @ 𝐞_B
        #   zA_i = eA_i + Σ_j C_ba[i,j] * eB_j
        zA_0 = eA_0 + C_ba_flat[0]*eB_0 + C_ba_flat[1]*eB_1 + C_ba_flat[2]*eB_2
        zA_1 = eA_1 + C_ba_flat[3]*eB_0 + C_ba_flat[4]*eB_1 + C_ba_flat[5]*eB_2
        zA_2 = eA_2 + C_ba_flat[6]*eB_0 + C_ba_flat[7]*eB_1 + C_ba_flat[8]*eB_2

        zB_0 = eB_0 + C_ab_flat[0]*eA_0 + C_ab_flat[1]*eA_1 + C_ab_flat[2]*eA_2
        zB_1 = eB_1 + C_ab_flat[3]*eA_0 + C_ab_flat[4]*eA_1 + C_ab_flat[5]*eA_2
        zB_2 = eB_2 + C_ab_flat[6]*eA_0 + C_ab_flat[7]*eA_1 + C_ab_flat[8]*eA_2

        return _lyapunov_cost(zA_0, zB_0, zA_1, zB_1, zA_2, zB_2,
                              omega_z, omega_w)

    return obj_fn


# ═══════════════════════════════════════════════════════════════════════
# Shared: decoupled Lyapunov hierarchy on transformed errors
# ═══════════════════════════════════════════════════════════════════════

def _lyapunov_cost(z0, w0, z1, w1, z2, w2, omega_z, omega_w):
    """Compute the 3‑level constructive Lyapunov cost on (z, w)."""
    term0 = z0**2 + w0**2

    v1_z = z1 + omega_z * z0
    v1_w = w1 + omega_w * w0
    term1 = v1_z**2 + v1_w**2

    v2_z = z2 + 2.0 * omega_z * z1 + omega_z**2 * z0
    v2_w = w2 + 2.0 * omega_w * w1 + omega_w**2 * w0
    term2 = v2_z**2 + v2_w**2

    return jnp.sum(term0) + jnp.sum(term1) + jnp.sum(term2)


# ═══════════════════════════════════════════════════════════════════════
# Aggressiveness templates
# ═══════════════════════════════════════════════════════════════════════

def template_coupling(name: str):
    """Return (C_ba, C_ab, ω_z, ω_w, ACC_MAX, JERK_MAX) for a named template.

    Templates:
      'conservative'  — no coupling, tight constraints
      'standard'      — no coupling, default constraints (current baseline)
      'active'        — mild cross‑order coupling, moderate constraints
      'aggressive'    — stronger coupling, relaxed constraints
      'emergency'     — full cross‑order coupling, very relaxed constraints
    """
    # fmt: off
    templates = {
        'conservative': (
            # C_ba (row-major): 0, C_ab: 0
            None, None,
            1.0, 4.0,   # ω_z, ω_w
            3.0, 1.5,   # ACC_MAX, JERK_MAX
        ),
        'standard': (
            None, None,
            1.0, 4.0,
            5.0, 2.0,
        ),
        'active': (
            # C_ba: lateral pos → longitudinal pos (0.1), lat vel → long vel (0.05)
            (0.1, 0.0, 0.0,   # pos row:  lat-pos→long-pos
             0.0, 0.05, 0.0,  # vel row:  lat-vel→long-vel
             0.0, 0.0, 0.0),  # acc row:  none
            # C_ab: longitudinal pos → lateral vel (−0.05) — longitudinal prep for lateral
            (0.0,  0.0,   0.0,
             -0.05, 0.0,  0.0,
             0.0,  0.0,   0.0),
            1.5, 6.0,   # ω_z, ω_w
            7.0, 3.0,   # ACC_MAX, JERK_MAX
        ),
        'aggressive': (
            # C_ba: lat-pos→long-pos(0.15), lat-vel→long-pos(0.1), lat-vel→long-vel(0.1)
            (0.15, 0.10, 0.0,
             0.0,  0.10, 0.0,
             0.0,  0.0,  0.0),
            # C_ab: long-pos→lat-vel(−0.1), long-vel→lat-pos(0.05)
            (0.0,  0.0,  0.0,
             -0.10, 0.0, 0.0,
             0.05, 0.0,  0.0),
            2.0, 8.0,
            10.0, 5.0,
        ),
        'emergency': (
            # C_ba: full lower‑triangular — lateral feeds into all longitudinal orders
            (0.2,  0.15, 0.0,
             0.1,  0.15, 0.0,
             0.0,  0.1,  0.1),
            # C_ab: longitudinal pos feeds lateral vel + acc
            (0.0,  0.0,  0.0,
             -0.15, 0.0, 0.0,
             -0.1,  0.0, 0.0),
            2.0, 8.0,
            15.0, 8.0,
        ),
    }
    # fmt: on
    if name not in templates:
        raise ValueError(f"Unknown template: {name}. "
                         f"Available: {list(templates.keys())}")
    return templates[name]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _eval_all(gen, ctrl_s_free, ctrl_d_free, ctx):
    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
        ctrl_s_free, ctrl_d_free,
        ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
        ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"],
    )
    return s, d, d_dot, d_ddot, d_dddot, s_dot, s_ddot, s_dddot
