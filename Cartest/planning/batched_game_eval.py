"""Batched three-agent B-spline game evaluation for Cartest.

This module is intentionally Cartest-specific.  It avoids the generic
black-box fitness interface so B-spline trajectories can be evaluated once
and then reused across B x M_inner game-cost combinations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import vmap


def agent_ctx(ctx, agent_idx):
    return {
        "s0": ctx[f"s0_a{agent_idx}"],
        "s_dot0": ctx[f"s_dot0_a{agent_idx}"],
        "s_ddot0": ctx.get(f"s_ddot0_a{agent_idx}", 0.0),
        "d0": ctx[f"d0_a{agent_idx}"],
        "d_dot0": ctx.get(f"d_dot0_a{agent_idx}", 0.0),
        "d_ddot0": ctx.get(f"d_ddot0_a{agent_idx}", 0.0),
    }


def theta_for_agent(joint_x, agent_idx, n_free):
    base = agent_idx * 2 * n_free
    return joint_x[base:base + 2 * n_free]


def evaluate_agent_plan_batch(gen, joint_batch, ctx, agent_idx):
    """Evaluate one agent's plan for a batch of joint vectors."""
    n_free = gen.n_free
    a_ctx = agent_ctx(ctx, agent_idx)

    def one(joint_x):
        theta = theta_for_agent(joint_x, agent_idx, n_free)
        frenet, vehicle, (x, y) = gen.evaluate_plan(theta[:n_free], theta[n_free:], a_ctx)
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = frenet
        return {
            "s": s,
            "d": d,
            "s_dot": s_dot,
            "d_dot": d_dot,
            "s_ddot": s_ddot,
            "d_ddot": d_ddot,
            "s_dddot": s_dddot,
            "d_dddot": d_dddot,
            "vehicle": vehicle,
            "x": x,
            "y": y,
        }

    return vmap(one)(joint_batch)


def evaluate_joint_plan_batch(gen, joint_batch, ctx, agent_count=3):
    """Evaluate all agent plans for joint vectors shaped [batch, joint_dim]."""
    return tuple(
        evaluate_agent_plan_batch(gen, joint_batch, ctx, agent_idx)
        for agent_idx in range(agent_count)
    )


def pair_distance_violation(x0, y0, x1, y1, safe_dist):
    dist = jnp.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + 1e-6)
    return jnp.maximum(0.0, safe_dist - dist)


def batched_agent_costs_from_plans(plans, scenario):
    """Return scalar objective costs shaped [batch, 3].

    This intentionally matches the objective portion of
    Cartest.planning.costs.three_agent_track.  Constran constraints are
    handled by separate batched violation functions in later tasks.
    """
    ego = plans[0]
    front = plans[1]
    rear = plans[2]
    ego_target_d = float(scenario["behavior"].get("ego_target_d", 3.5))

    ego_v_ref = float(scenario["agents"][0]["v_target"])
    front_v_ref = float(scenario["agents"][1]["v_target"])
    rear_v_ref = float(scenario["agents"][2]["v_target"])

    ego_cost = (
        3.0 * jnp.sum((ego["s_dot"] - ego_v_ref) ** 2, axis=-1)
        + 10.0 * jnp.sum((ego["d"] - ego_target_d) ** 2, axis=-1)
        + 5.0 * jnp.sum(ego["d_dot"] ** 2, axis=-1)
        + 0.5 * jnp.sum(ego["d_ddot"] ** 2, axis=-1)
        + jnp.sum(ego["s_dddot"] ** 2 + ego["d_dddot"] ** 2, axis=-1)
    )
    front_cost = (
        3.0 * jnp.sum((front["s_dot"] - front_v_ref) ** 2, axis=-1)
        + 2.0 * jnp.sum(front["s_ddot"] ** 2, axis=-1)
    )
    rear_cost = (
        3.0 * jnp.sum((rear["s_dot"] - rear_v_ref) ** 2, axis=-1)
        + 2.0 * jnp.sum(rear["s_ddot"] ** 2, axis=-1)
    )
    return jnp.stack([ego_cost, front_cost, rear_cost], axis=-1)
