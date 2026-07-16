"""Batched three-agent B-spline game evaluation for Cartest.

This module is intentionally Cartest-specific.  It avoids the generic
black-box fitness interface so B-spline trajectories can be evaluated once
and then reused across B x M_inner game-cost combinations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import vmap

from Constraintdealer.Constran import (
    T_alpha,
    sigma_k,
    OBJ_PRESETS,
    OBJ_TRANSFORM_STANDARD,
    TRANSFORM_SOFT,
    TRANSFORM_HARD,
)


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


# ───────────────────────────────────────────────────────────────────────
# Constraint violations + sigma-nested cost (mirrors Constran._assemble_nest)
# ───────────────────────────────────────────────────────────────────────

def _aggregate(values, mode):
    if mode == "max":
        return jnp.max(values, axis=-1)
    if mode == "q95":
        return jnp.quantile(values, 0.95, axis=-1)
    raise ValueError(f"unsupported aggregate {mode!r}")


def batched_constraint_violations_from_plans(plans, scenario):
    """Per-agent raw constraint violations g[batch, T] for the three-agent game.

    Matches ``Cartest.planning.costs.three_agent_track`` g_fn semantics:
    lane/speed/acc/jerk are identical for all agents; the collision term is
    role-differentiated (ego full-horizon both neighbours, front/rear
    short-horizon, rear additionally guards longitudinal clearance).
    """
    safe_gap = float(scenario["safety"].get("safe_gap", 3.0))
    vehicle_length = float(scenario["safety"].get("vehicle_length", 5.0))
    v_min = float(scenario["safety"].get("v_min", 2.0))
    v_max = float(scenario["safety"].get("v_max", 35.0))
    acc_max = float(scenario["safety"].get("acc_max", 5.0))
    jerk_max = float(scenario["safety"].get("jerk_max", 2.0))
    lane_min, lane_max = scenario["road"].get("lane_bounds_d", (-1.75, 5.25))

    out = []
    for aid in range(3):
        plan = plans[aid]
        vehicle = plan["vehicle"]
        lane = jnp.maximum(jnp.maximum(0.0, lane_min - plan["d"]),
                           jnp.maximum(0.0, plan["d"] - lane_max))
        v = vehicle[..., 2]
        speed = jnp.maximum(jnp.maximum(0.0, v_min - v), jnp.maximum(0.0, v - v_max))
        a_long, a_lat = vehicle[..., 4], vehicle[..., 5]
        a_mag = jnp.sqrt(a_long ** 2 + a_lat ** 2)
        acc = jnp.maximum(
            jnp.maximum(0.0, jnp.abs(a_long) - acc_max),
            jnp.maximum(jnp.maximum(0.0, jnp.abs(a_lat) - acc_max), jnp.maximum(0.0, a_mag - acc_max)),
        )
        j_long, j_lat = vehicle[..., 6], vehicle[..., 7]
        j_mag = jnp.sqrt(j_long ** 2 + j_lat ** 2)
        jerk = jnp.maximum(
            jnp.maximum(0.0, jnp.abs(j_long) - jerk_max),
            jnp.maximum(jnp.maximum(0.0, jnp.abs(j_lat) - jerk_max), jnp.maximum(0.0, j_mag - jerk_max)),
        )

        if aid == 0:
            col = jnp.maximum(
                pair_distance_violation(plan["x"], plan["y"], plans[1]["x"], plans[1]["y"], safe_gap),
                pair_distance_violation(plan["x"], plan["y"], plans[2]["x"], plans[2]["y"], safe_gap),
            )
        elif aid == 1:
            short = slice(0, 2)
            col = jnp.zeros_like(plan["x"])
            ego_short = pair_distance_violation(
                plan["x"][..., short], plan["y"][..., short],
                plans[0]["x"][..., short], plans[0]["y"][..., short], safe_gap)
            rear_short = pair_distance_violation(
                plan["x"][..., short], plan["y"][..., short],
                plans[2]["x"][..., short], plans[2]["y"][..., short], safe_gap)
            col = col.at[..., short].set(jnp.maximum(ego_short, rear_short))
        else:
            short = slice(0, 2)
            col = jnp.zeros_like(plan["x"])
            ego_short = pair_distance_violation(
                plan["x"][..., short], plan["y"][..., short],
                plans[0]["x"][..., short], plans[0]["y"][..., short], safe_gap)
            col = col.at[..., short].set(ego_short)
            clearance = vehicle_length + safe_gap
            clearance_violation = jnp.maximum(0.0, clearance - (plans[1]["s"] - plan["s"]))
            col = jnp.maximum(col, clearance_violation)

        out.append({"lane": lane, "speed": speed, "acc": acc, "jerk": jerk, "collision": col})
    return tuple(out)


# Constraint layer config for three_agent_track, ordered by priority
# (low -> high = innermost -> outermost), mirroring the Deterministic specs:
#   lane(soft,q95) speed(soft,max) acc(soft,max) jerk(soft,max) collision(hard,max)
# Each entry: (name, aggregate, transform_table, baseline, resolution).
# resolution = first knot of the transform table (the mode's "分辨率").
_THREE_AGENT_LAYERS = [
    ("lane", "q95", TRANSFORM_SOFT, 0.5, float(TRANSFORM_SOFT[0][0])),
    ("speed", "max", TRANSFORM_SOFT, 0.5, float(TRANSFORM_SOFT[0][0])),
    ("acc", "max", TRANSFORM_SOFT, 0.5, float(TRANSFORM_SOFT[0][0])),
    ("jerk", "max", TRANSFORM_SOFT, 0.5, float(TRANSFORM_SOFT[0][0])),
    ("collision", "max", TRANSFORM_HARD, 2.0, float(TRANSFORM_HARD[0][0])),
]


def _obj_table(obj_transform):
    if isinstance(obj_transform, tuple):
        return obj_transform
    return OBJ_PRESETS.get(obj_transform, OBJ_TRANSFORM_STANDARD)


def batched_nested_costs_from_plans(plans, scenario, k_inner=1.0, obj_transform="standard"):
    """Three-agent sigma-nested cost shaped [batch, 3].

    This is a batched replication of ``Constran._assemble_nest``: the raw
    objective is the innermost seed, and each constraint layer wraps it as
    ``inner = sqrt(2) * sigma_1(inner) + Phi`` with
    ``Phi = max(0, T_alpha(g)) + baseline * 1[max(0,g) > resolution]``.
    Constraints are applied in priority order (innermost -> outermost) and
    the objective uses ``sigma_k`` with ``k_inner``.
    """
    M = jnp.sqrt(2.0)
    obj_knots_g, obj_knots_T = _obj_table(obj_transform)
    n_total = len(_THREE_AGENT_LAYERS) + 1  # n constraints + objective's own sigma wrap

    costs = batched_agent_costs_from_plans(plans, scenario)          # [batch, 3]
    violations = batched_constraint_violations_from_plans(plans, scenario)

    result = []
    for aid in range(3):
        inner = T_alpha(costs[:, aid], obj_knots_g, obj_knots_T)     # objective transform
        inner = inner / (M ** n_total)                                # pre-scale by sqrt(2)**n_total
        inner = sigma_k(inner, k=k_inner)                             # k only for objective

        for _name, agg, table, baseline, resolution in _THREE_AGENT_LAYERS:
            g_raw = _aggregate(violations[aid][_name], agg)          # [batch]
            t_val = jnp.maximum(0.0, T_alpha(g_raw, table[0], table[1]))
            Phi = t_val
            violated = jnp.maximum(0.0, g_raw) > resolution
            Phi = jnp.where(violated, Phi + baseline, Phi)
            inner = M * sigma_k(inner, k=1.0) + Phi                   # constraint layer

        inner = M * sigma_k(inner, k=1.0)                             # final sigma wrap
        result.append(inner)
    return jnp.stack(result, axis=-1)


# ───────────────────────────────────────────────────────────────────────
# Fixed-sample expected cost (f_hat) for one agent
# ───────────────────────────────────────────────────────────────────────

def evaluate_agent_control_batch(gen, ctrl_s_batch, ctrl_d_batch, ctx, agent_idx):
    """Evaluate controls shaped [batch, n_free] for one agent only."""
    a_ctx = agent_ctx(ctx, agent_idx)

    def one(ctrl_s, ctrl_d):
        frenet, vehicle, (x, y) = gen.evaluate_plan(ctrl_s, ctrl_d, a_ctx)
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

    return vmap(one)(ctrl_s_batch, ctrl_d_batch)


def _plans_for_agent_source(gen, source, ctx, agent_idx):
    """Evaluate one agent's plan from a block sample tensor [N_blocks, batch, D]."""
    s_block = agent_idx * 2
    d_block = s_block + 1
    return evaluate_agent_control_batch(gen, source[s_block], source[d_block], ctx, agent_idx)


def _broadcast_plan(plan, B, M_inner, use_candidate):
    """Tile a [N, ...] plan into [B * M_inner, ...] for pairwise pairing.

    Candidate plans (N=B) are repeated across the M_inner axis; background
    plans (N=M_inner) are repeated across the B axis.  Row ``b * M_inner + m``
    then carries candidate ``b`` (for the acting agent) and background ``m``
    (for the opponents), matching the joint construction in MPC_G_MS.
    """
    if use_candidate:
        return {key: jnp.repeat(value[:, None, ...], M_inner, axis=1)
                       .reshape((B * M_inner,) + value.shape[1:])
                for key, value in plan.items()}
    return {key: jnp.repeat(value[None, :, ...], B, axis=0)
                   .reshape((B * M_inner,) + value.shape[1:])
            for key, value in plan.items()}


def batched_expected_cost_for_agent(gen, samples_b, samples_m, ctx, scenario, agent_idx,
                                    k_inner=1.0, obj_transform="standard"):
    """Compute f_hat[B] for one agent from fixed block samples.

    samples_b: [N_blocks, B, D]       candidate (own-action) samples
    samples_m: [N_blocks, M_inner, D] background (opponent) samples

    Trajectories are evaluated only ``B + 2 * M_inner`` times (the acting
    agent's B candidates plus each opponent's M_inner backgrounds) and then
    broadcast over the B x M_inner pairing - instead of evaluating all
    ``B * M_inner`` joint trajectories.  The pairwise game cost is computed
    by broadcasting the cached ego/front/rear plans.
    """
    B = samples_b.shape[1]
    M_inner = samples_m.shape[1]

    # Only the plans actually needed: the agent's own candidates (from
    # samples_b) and the other agents' backgrounds (from samples_m).
    plans = [None, None, None]
    plans[agent_idx] = _plans_for_agent_source(gen, samples_b, ctx, agent_idx)
    for aid in range(3):
        if aid != agent_idx:
            plans[aid] = _plans_for_agent_source(gen, samples_m, ctx, aid)

    broadcast = tuple(
        _broadcast_plan(plans[aid], B, M_inner, aid == agent_idx) for aid in range(3)
    )
    costs = batched_nested_costs_from_plans(broadcast, scenario, k_inner=k_inner,
                                            obj_transform=obj_transform)
    return costs[:, agent_idx].reshape((B, M_inner)).mean(axis=1)
