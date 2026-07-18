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
)
from Cartest.planning.costs.three_agent_track import three_agent_batched_layers
from Cartest.planning.costs.three_agent_track_components import (
    _plan_s_dot,
    acc_limit_violation_per_t,
    bridged_jerk_cost,
    collision_clearances,
    collision_prefix,
    collision_violation_per_t,
    forward_motion_violation_per_t,
    jerk_limit_violation_per_t,
    lane_boundary_violation_per_t,
    lane_objective,
    pair_footprint_violation,
    progress_objective,
    role_soft_objective,
    role_target_d,
    rss_cvar_risk,
    speed_limit_violation_per_t,
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


def expected_cost_for_agent_controls(
        gen, own_controls, frozen_joint_controls, ctx, scenario, agent_idx,
        k_inner=0.1, obj_transform="standard"):
    """Expected nested cost for unilateral controls against frozen opponents.

    ``own_controls`` is ``[B, 2, D]`` and ``frozen_joint_controls`` is
    ``[M, 6, D]``.  The acting agent's two blocks in every frozen joint sample
    are replaced before trajectory evaluation, so those source entries may be
    masked with NaNs to make accidental use visible.
    """
    own_controls = jnp.asarray(own_controls)
    frozen_joint_controls = jnp.asarray(frozen_joint_controls)
    candidate_count = own_controls.shape[0]
    opponent_count = frozen_joint_controls.shape[0]
    first_block = 2 * agent_idx

    joint = jnp.broadcast_to(
        frozen_joint_controls[None],
        (candidate_count,) + frozen_joint_controls.shape,
    )
    replacement = jnp.broadcast_to(
        own_controls[:, None],
        (candidate_count, opponent_count) + own_controls.shape[1:],
    )
    joint = joint.at[:, :, first_block:first_block + 2].set(replacement)
    plans = evaluate_joint_plan_batch(
        gen,
        joint.reshape((candidate_count * opponent_count, -1)),
        ctx,
        agent_count=3,
    )
    costs = batched_nested_costs_from_plans(
        plans,
        scenario,
        gen.dt,
        k_inner=k_inner,
        obj_transform=obj_transform,
        ctx=ctx,
    )[:, agent_idx]
    return costs.reshape((candidate_count, opponent_count)).mean(axis=1)


def batched_agent_costs_from_plans(plans, scenario, dt, ctx=None):
    """Return scalar objective costs shaped [batch, 3] using shared components.

    Mirrors the objective portion of
    ``Cartest.planning.costs.three_agent_track`` via the shared
    ``role_soft_objective`` so the scalar and batched paths cannot drift.
    """
    ctx = {} if ctx is None else ctx
    return jnp.stack([
        role_soft_objective(plans, scenario, ctx, aid, dt)
        for aid in range(3)
    ], axis=-1)


# ───────────────────────────────────────────────────────────────────────
# Constraint violations + sigma-nested cost (mirrors Constran._assemble_nest)
# ───────────────────────────────────────────────────────────────────────

def _aggregate(values, mode):
    if mode == "max":
        return jnp.max(values, axis=-1)
    if mode == "q95":
        return jnp.quantile(values, 0.95, axis=-1)
    raise ValueError(f"unsupported aggregate {mode!r}")


def _violations_for_agent(plans, scenario, aid):
    """Raw feasibility-layer violations g[batch, T] for one agent.

    Reduced layer set matching ``three_agent_track.THREE_AGENT_CONSTRAINT_DEFS``:
    ``speed``, ``kinematics = max(acc, jerk)``, ``forward``, and
    ``safety_envelope = max(collision, lane_boundary)``.  Per-timestep values
    come from the shared component module so scalar and batched paths use
    identical formulas.
    """
    own = plans[aid]
    speed = speed_limit_violation_per_t(own, scenario)
    acc = acc_limit_violation_per_t(own, scenario)
    jerk = jerk_limit_violation_per_t(own, scenario)
    forward = forward_motion_violation_per_t(own, scenario)
    collision = collision_violation_per_t(plans, scenario, aid)
    lane_boundary = lane_boundary_violation_per_t(own, scenario)
    return {
        "speed": speed,
        "kinematics": jnp.maximum(acc, jerk),
        "forward": forward,
        "safety_envelope": jnp.maximum(collision, lane_boundary),
    }


def batched_constraint_violations_from_plans(plans, scenario):
    """Per-agent raw constraint violations for all three agents."""
    return tuple(_violations_for_agent(plans, scenario, aid) for aid in range(3))


# Static JIT metadata compiled from the same definitions used by scalar specs.
_THREE_AGENT_LAYERS = three_agent_batched_layers()


def _obj_table(obj_transform):
    if isinstance(obj_transform, tuple):
        return obj_transform
    return OBJ_PRESETS.get(obj_transform, OBJ_TRANSFORM_STANDARD)


def _objective_for_agent(plans, scenario, aid, dt, ctx=None):
    """Raw objective [batch] for one agent via the shared role objective.

    Takes the full plan tuple so RSS interaction risk can be evaluated against
    the other agents.  Used by ``batched_nested_costs_from_plans`` (the
    full-plan validation path) so it matches the scalar specs exactly.
    """
    ctx = {} if ctx is None else ctx
    return role_soft_objective(plans, scenario, ctx, aid, dt)


def _nest_one_agent(obj, violations, k_inner=1.0, obj_transform="standard"):
    """Sigma-nested cost [batch] for one agent (replicates Constran._assemble_nest)."""
    M = jnp.sqrt(2.0)
    obj_knots_g, obj_knots_T = _obj_table(obj_transform)
    n_total = len(_THREE_AGENT_LAYERS) + 1  # n constraints + objective's own sigma wrap

    inner = T_alpha(obj, obj_knots_g, obj_knots_T)                  # objective transform
    inner = inner / (M ** n_total)                                   # pre-scale by sqrt(2)**n_total
    inner = sigma_k(inner, k=k_inner)                                # k only for objective

    for _name, agg, table, baseline, resolution in _THREE_AGENT_LAYERS:
        g_raw = _aggregate(violations[_name], agg)                  # [batch]
        t_val = jnp.maximum(0.0, T_alpha(g_raw, table[0], table[1]))
        Phi = t_val
        violated = jnp.maximum(0.0, g_raw) > resolution
        Phi = jnp.where(violated, Phi + baseline, Phi)
        inner = M * sigma_k(inner, k=1.0) + Phi                      # constraint layer

    inner = M * sigma_k(inner, k=1.0)                                # final sigma wrap
    return inner


def _nest_one_agent_from_aggregates(obj, aggregate_values, k_inner=1.0,
                                    obj_transform="standard"):
    """Sigma-nested cost from already-aggregated constraint values.

    ``obj`` and non-collision aggregates are typically shaped ``[B, 1]``;
    collision is shaped ``[B, M_inner]``.  JAX broadcasting then evaluates the
    same nested scalar cost for every candidate/background pair without first
    materializing full broadcast trajectory dictionaries.
    """
    M = jnp.sqrt(2.0)
    obj_knots_g, obj_knots_T = _obj_table(obj_transform)
    n_total = len(_THREE_AGENT_LAYERS) + 1

    inner = T_alpha(obj, obj_knots_g, obj_knots_T)
    inner = inner / (M ** n_total)
    inner = sigma_k(inner, k=k_inner)

    for name, _agg, table, baseline, resolution in _THREE_AGENT_LAYERS:
        g_raw = aggregate_values[name]
        t_val = jnp.maximum(0.0, T_alpha(g_raw, table[0], table[1]))
        Phi = t_val
        violated = jnp.maximum(0.0, g_raw) > resolution
        Phi = jnp.where(violated, Phi + baseline, Phi)
        inner = M * sigma_k(inner, k=1.0) + Phi

    return M * sigma_k(inner, k=1.0)


def batched_nested_costs_from_plans(plans, scenario, dt, k_inner=1.0,
                                    obj_transform="standard", ctx=None):
    """Three-agent sigma-nested cost shaped [batch, 3].

    Batched replication of ``Constran._assemble_nest``: the raw objective is
    the innermost seed, and each constraint layer wraps it as
    ``inner = sqrt(2) * sigma_1(inner) + Phi`` with
    ``Phi = max(0, T_alpha(g)) + baseline * 1[max(0,g) > resolution]``.
    Constraints are applied in priority order (innermost -> outermost) and
    the objective uses ``sigma_k`` with ``k_inner``.
    """
    ctx = {} if ctx is None else ctx
    return jnp.stack([
        _nest_one_agent(
            _objective_for_agent(plans, scenario, aid, dt, ctx),
            _violations_for_agent(plans, scenario, aid),
            k_inner, obj_transform)
        for aid in range(3)
    ], axis=-1)


def _own_constraint_aggregates(plan, scenario):
    """Aggregate non-interaction (candidate-only) constraints for one batch.

    Returns the reduced-layer aggregates with collision deferred to the
    pairwise path: ``speed``, ``kinematics = max(acc, jerk)``, ``forward``,
    and ``lane_boundary`` (the caller folds ``lane_boundary`` into
    ``safety_envelope`` with the pairwise collision term).
    """
    speed = speed_limit_violation_per_t(plan, scenario)
    acc = acc_limit_violation_per_t(plan, scenario)
    jerk = jerk_limit_violation_per_t(plan, scenario)
    forward = forward_motion_violation_per_t(plan, scenario)
    lane_boundary = lane_boundary_violation_per_t(plan, scenario)
    return {
        "speed": _aggregate(speed, "max"),
        "kinematics": _aggregate(jnp.maximum(acc, jerk), "max"),
        "forward": _aggregate(forward, "max"),
        "lane_boundary": _aggregate(lane_boundary, "max"),
    }


def _pair_footprint_violation_bm(candidate, background, longitudinal_clearance,
                                 lateral_clearance):
    """Pairwise footprint violation for candidate [B,T] and background [M,T]."""
    return pair_footprint_violation(
        candidate["s"][:, None, :], candidate["d"][:, None, :],
        background["s"][None, :, :], background["d"][None, :, :],
        longitudinal_clearance, lateral_clearance)


def _collision_aggregate_bm(candidate, backgrounds, scenario, aid):
    """Collision aggregate shaped [B, M_inner] for one acting agent."""
    longitudinal_clearance, lateral_clearance = collision_clearances(scenario)
    short = collision_prefix(scenario)
    own = candidate[aid]

    if aid == 0:
        front = backgrounds[1]
        rear = backgrounds[2]
        ego_front = _pair_footprint_violation_bm(
            own, front, longitudinal_clearance, lateral_clearance)
        ego_rear = _pair_footprint_violation_bm(
            own, rear, longitudinal_clearance, lateral_clearance)
        return jnp.max(jnp.maximum(ego_front, ego_rear), axis=-1)

    if aid == 1:
        ego = backgrounds[0]
        rear = backgrounds[2]
        front_ego = _pair_footprint_violation_bm(
            own, ego, longitudinal_clearance, lateral_clearance)[..., short]
        front_rear = _pair_footprint_violation_bm(
            own, rear, longitudinal_clearance, lateral_clearance)[..., short]
        return jnp.max(jnp.maximum(front_ego, front_rear), axis=-1)

    ego = backgrounds[0]
    front = backgrounds[1]
    rear_ego_short = _pair_footprint_violation_bm(
        own, ego, longitudinal_clearance, lateral_clearance)[..., short]
    rear_ego = jnp.max(rear_ego_short, axis=-1)
    rear_front = jnp.max(_pair_footprint_violation_bm(
        own, front, longitudinal_clearance, lateral_clearance), axis=-1)
    return jnp.maximum(rear_ego, rear_front)


def _rss_pairwise_bm(own, neighbor_backgrounds, scenario, dt):
    """RSS CVaR risk shaped [B, M] for a candidate batch vs background neighbours.

    Reshapes the acting candidate to ``[B, 1, T]`` and each background neighbour
    to ``[1, M, T]`` so the shared ``rss_cvar_risk`` broadcasts over the
    ``B x M`` candidate/background grid.  Element ``[b, m]`` is exactly the
    scalar RSS risk of the joint (candidate_b, background_m) plan.
    """
    own_b = {
        "s": own["s"][:, None, :],
        "d": own["d"][:, None, :],
        "s_dot": _plan_s_dot(own)[:, None, :],
    }
    nbrs_b = tuple(
        {
            "s": nb["s"][None, :, :],
            "d": nb["d"][None, :, :],
            "s_dot": _plan_s_dot(nb)[None, :, :],
        }
        for nb in neighbor_backgrounds
    )
    return rss_cvar_risk(own_b, nbrs_b, scenario, dt)


def _objective_pairwise_bm(candidate, background, scenario, aid, dt, ctx=None):
    """Objective shaped [B, M] for the fast expected-cost path.

    Candidate-only terms (progress / lane / comfort) broadcast as ``[B, 1]``
    and the RSS interaction risk is computed pairwise against the background
    neighbour batches as ``[B, M]``.  This matches the scalar cost evaluated on
    the joint (candidate_b, background_m) plan, which is what
    ``test_batched_f_hat_matches_black_box_scalar_loop`` verifies.
    """
    ctx = {} if ctx is None else ctx
    own = candidate[aid]
    v_target = float(scenario["agents"][aid]["v_target"])
    target_d = role_target_d(scenario, aid)
    progress = progress_objective(own, v_target)
    lane = lane_objective(own, target_d)
    comfort = bridged_jerk_cost(own, ctx, aid, dt)
    neighbor_bgs = tuple(background[j] for j in range(3) if j != aid)
    rss = _rss_pairwise_bm(own, neighbor_bgs, scenario, dt)
    return (progress + lane + comfort)[:, None] + rss


def _fast_expected_cost_for_agent(candidate, background, scenario, aid,
                                  dt, ctx=None, k_inner=1.0, obj_transform="standard"):
    """Expected cost [B] for one acting agent without full-plan broadcasting."""
    B = candidate[aid]["s"].shape[0]
    M_inner = background[aid]["s"].shape[0]
    own = candidate[aid]
    obj = _objective_pairwise_bm(candidate, background, scenario, aid, dt, ctx)  # [B, M]
    own_aggs = {k: v[:, None] for k, v in _own_constraint_aggregates(own, scenario).items()}
    collision = _collision_aggregate_bm(candidate, background, scenario, aid)  # [B, M]
    own_aggs["safety_envelope"] = jnp.maximum(collision, own_aggs.pop("lane_boundary"))
    pair_cost = _nest_one_agent_from_aggregates(
        obj, own_aggs, k_inner=k_inner, obj_transform=obj_transform)
    return pair_cost.reshape((B, M_inner)).mean(axis=1)


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


def batched_expected_costs_for_all_agents(gen, samples_b, samples_m, ctx, scenario,
                                          k_inner=1.0, obj_transform="standard"):
    """Compute f_hat [M_agent, B] for all agents with a shared plan cache.

    Candidate plans (from samples_b) and background plans (from samples_m)
    are evaluated once per agent - 6 trajectory-eval calls total - and reused
    across all three agents' expected-cost evaluations.  For each agent only
    its own nested cost is computed (not all three), so this is the
    solver-facing path avoids the old full-plan broadcast used during
    development and evaluates only aggregate values needed by each agent's
    nested cost.
    """
    candidate = [_plans_for_agent_source(gen, samples_b, ctx, aid) for aid in range(3)]
    background = [_plans_for_agent_source(gen, samples_m, ctx, aid) for aid in range(3)]
    return jnp.stack([
        _fast_expected_cost_for_agent(candidate, background, scenario, aid,
                                      dt=gen.dt, ctx=ctx, k_inner=k_inner,
                                      obj_transform=obj_transform)
        for aid in range(3)
    ])
