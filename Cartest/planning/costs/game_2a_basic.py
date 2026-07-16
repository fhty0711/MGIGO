"""Two-agent basic Lyapunov game objectives (B-spline + RNE).

Migrated from the old standalone ``Simple_game_2a.py`` demo.  Each agent
owns a (s, d) B-spline control-point pair; objectives are Lyapunov
tracking costs with no explicit collision term.
"""

from __future__ import annotations

import jax.numpy as jnp


def _agent_ctx(ctx, agent_idx):
    return {
        "s0": ctx[f"s0_a{agent_idx}"],
        "s_dot0": ctx[f"s_dot0_a{agent_idx}"],
        "s_ddot0": ctx.get(f"s_ddot0_a{agent_idx}", 0.0),
        "d0": ctx[f"d0_a{agent_idx}"],
        "d_dot0": ctx.get(f"d_dot0_a{agent_idx}", 0.0),
        "d_ddot0": ctx.get(f"d_ddot0_a{agent_idx}", 0.0),
    }


def _theta_for_agent(joint_x, agent_idx, n_free):
    base = agent_idx * 2 * n_free
    return joint_x[base:base + 2 * n_free]


def _eval_agent_plan(gen, joint_x, ctx, agent_idx):
    theta = _theta_for_agent(joint_x, agent_idx, gen.n_free)
    return gen.evaluate_plan(theta[:gen.n_free], theta[gen.n_free:], _agent_ctx(ctx, agent_idx))


def pair_distance_violation(x0, y0, x1, y1, safe_dist):
    """Soft-collision violation: positive when two Cartesian paths get too close."""
    dist = jnp.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + 1e-6)
    return jnp.maximum(0.0, safe_dist - dist)


def make_agent_specs(gen, scenario):
    """Build per-agent objective/constraint specs for the 2-agent basic game."""
    n_free = gen.n_free
    t_arr = jnp.arange(gen.T) * gen.dt
    omega_s = float(scenario["cost"]["params"].get("omega_s", 1.0))
    omega_d = float(scenario["cost"]["params"].get("omega_d", 4.0))
    target_d = float(scenario["behavior"].get("target_d", 0.0))

    def make_objective(agent_idx):
        v_target = float(scenario["agents"][agent_idx]["v_target"])

        def objective(joint_x, ctx):
            frenet, _vehicle, _cart = _eval_agent_plan(gen, joint_x, ctx, agent_idx)
            s, d, s_dot, d_dot, s_ddot, d_ddot, _s3, _d3 = frenet
            s0 = ctx[f"s0_a{agent_idx}"]
            v0 = ctx[f"s_dot0_a{agent_idx}"]
            dv = v0 - v_target
            exp_term = jnp.exp(-omega_s * t_arr)
            s_ref = s0 + v_target * t_arr + dv / omega_s * (1.0 - exp_term)
            s_dot_ref = v_target + dv * exp_term
            s_ddot_ref = -dv * omega_s * exp_term
            d_ref = jnp.full_like(d, target_d)
            es = s - s_ref
            ed = d - d_ref
            es_dot = s_dot - s_dot_ref
            ed_dot = d_dot
            es_ddot = s_ddot - s_ddot_ref
            ed_ddot = d_ddot
            return (
                jnp.sum(es ** 2 + ed ** 2)
                + jnp.sum((es_dot + omega_s * es) ** 2 + (ed_dot + omega_d * ed) ** 2)
                + jnp.sum((es_ddot + 2.0 * omega_s * es_dot + omega_s ** 2 * es) ** 2
                          + (ed_ddot + 2.0 * omega_d * ed_dot + omega_d ** 2 * ed) ** 2)
            )

        return objective

    return {idx: (make_objective(idx), []) for idx in range(len(scenario["agents"]))}
