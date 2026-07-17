"""Shared three-agent cost components (B-spline + RNE).

This module owns the normalized cost formulas migrated from ``mpc_source``:
hard feasibility diagnostics (boundary / collision / speed / acceleration /
jerk / forward motion) and the soft role objective (progress, lane target,
bridged-jerk comfort, RSS/CVaR interaction risk).

Both the scalar Constran path (``Cartest.planning.costs.three_agent_track``)
and the batched path (``Cartest.planning.costs.three_agent_track_batched``) import these
formulas so the two paths cannot drift.  Per-timestep helpers are
shape-agnostic over leading axes (they work on ``[T]`` scalar plans, on
``[batch, T]`` batched plans, and on ``[B, M, T]`` pairwise plans); the
``*_violation`` aggregates reduce over time with ``max`` so they can be used
directly as scalar Constran ``g_fn`` outputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Physical normalization tolerances (diagnostic scales; Constran's T_alpha
# transform table handles the solver-facing scaling, so violations below are
# returned in raw physical units to match the existing nesting behaviour).
# ---------------------------------------------------------------------------
LANE_BOUNDARY_TOL = 0.05
COLLISION_TOL = 0.05
SPEED_TOL = 0.5
ACC_TOL = 0.5
JERK_TOL = 0.25
FORWARD_TOL = 0.5

# Comfort (bridged jerk) scales.
JERK_LONG_SCALE = 4.0
JERK_LAT_SCALE = 4.0

# Soft objective scales.
SPEED_RUN_SCALE = 2.0
LANE_KEEP_SCALE = 1.0

# RSS / CVaR interaction-risk parameters (ported from mpc_source).
RSS_DEFICIT_SCALE = 2.0
RSS_CVAR_FRACTION = 0.10
RSS_TIME_HALF_LIFE = 3.0
RSS_CVAR_NORMALIZER = 45.0
RSS_RISK_POWER = 2.0
RSS_LATERAL_MARGIN = 0.3
RSS_LATERAL_SOFTNESS = 0.15
RSS_RHO = 1.0
RSS_A_MAX = 2.0
RSS_B_REAR = 4.0
RSS_B_EGO = 4.0
RSS_B_FRONT = 4.0
RSS_B_EGO_FRONT = 4.0
RSS_MIN_GAP = 2.0
RSS_GAP_MAX = 60.0


# ---------------------------------------------------------------------------
# Shared geometry / footprint helpers
# ---------------------------------------------------------------------------

def collision_prefix(scenario):
    """Short-horizon slice that includes the state executed by the MPC."""
    execute_index = int(scenario.get("game", {}).get("execute_index", 1))
    return slice(0, max(1, execute_index + 1))


def collision_lateral_clearance(scenario):
    """Lateral center clearance for the three-agent footprint model."""
    vehicle_width = float(scenario["safety"].get("vehicle_width", 2.0))
    safe_gap = float(scenario["safety"].get("safe_gap", 3.0))
    lane_width = float(scenario["road"].get("lane_width", vehicle_width + safe_gap))
    return min(vehicle_width + safe_gap, lane_width)


def lane_footprint_bounds(scenario):
    """Road bounds tightened by half vehicle width so the body stays on-road."""
    lane_min, lane_max = scenario["road"].get("lane_bounds_d", (-1.75, 5.25))
    half_width = 0.5 * float(scenario["safety"].get("vehicle_width", 2.0))
    return lane_min + half_width, lane_max - half_width


def upper_lane_center(scenario):
    """Preferred lateral centerline for the upper-lane agents."""
    behavior = scenario.get("behavior", {})
    if "upper_lane_d" in behavior:
        return float(behavior["upper_lane_d"])
    lane_centers = scenario["road"].get("lane_centers_d", (0.0, 3.5))
    return float(lane_centers[-1])


def pair_footprint_violation(s0, d0, s1, d1, longitudinal_clearance,
                             lateral_clearance):
    """Positive when two Frenet vehicle envelopes overlap.

    Rectangular Frenet envelope; returns the smaller penetration depth so it
    is positive only when both longitudinal and lateral envelopes overlap.
    Shape-agnostic over leading axes (last axis = time).
    """
    longitudinal = jnp.maximum(0.0, longitudinal_clearance - jnp.abs(s1 - s0))
    lateral = jnp.maximum(0.0, lateral_clearance - jnp.abs(d1 - d0))
    return jnp.minimum(longitudinal, lateral)


def _plan_s_dot(plan):
    # Conditional lookup: dict.get(k, default) evaluates default eagerly, so a
    # plan dict without "vehicle" would raise KeyError on the default branch.
    if "s_dot" in plan:
        return plan["s_dot"]
    return plan["vehicle"][..., 2]


# ---------------------------------------------------------------------------
# Per-timestep violation cores (shape-agnostic, last axis = time)
# ---------------------------------------------------------------------------

def lane_boundary_violation_per_t(plan, scenario):
    """Body-tightened road boundary violation per timestep ``[..., T]``."""
    lane_min, lane_max = lane_footprint_bounds(scenario)
    d = plan["d"]
    return jnp.maximum(jnp.maximum(0.0, lane_min - d),
                       jnp.maximum(0.0, d - lane_max))


def speed_limit_violation_per_t(plan, scenario):
    """Lower/upper speed violation for vehicle speed and Frenet ``s_dot``."""
    v_min = float(scenario["safety"].get("v_min", 2.0))
    v_max = float(scenario["safety"].get("v_max", 35.0))
    vehicle = plan["vehicle"]
    v = vehicle[..., 2]
    s_dot = _plan_s_dot(plan)
    return jnp.maximum(
        jnp.maximum(jnp.maximum(0.0, v_min - v), jnp.maximum(0.0, v - v_max)),
        jnp.maximum(jnp.maximum(0.0, v_min - s_dot), jnp.maximum(0.0, s_dot - v_max)),
    )


def acc_limit_violation_per_t(plan, scenario):
    """Acceleration-limit violation (long / lat / magnitude convention)."""
    acc_max = float(scenario["safety"].get("acc_max", 5.0))
    vehicle = plan["vehicle"]
    a_long, a_lat = vehicle[..., 4], vehicle[..., 5]
    a_mag = jnp.sqrt(a_long ** 2 + a_lat ** 2)
    return jnp.maximum(
        jnp.maximum(0.0, jnp.abs(a_long) - acc_max),
        jnp.maximum(jnp.maximum(0.0, jnp.abs(a_lat) - acc_max),
                    jnp.maximum(0.0, a_mag - acc_max)),
    )


def jerk_limit_violation_per_t(plan, scenario):
    """Jerk-limit violation (long / lat / magnitude convention)."""
    jerk_max = float(scenario["safety"].get("jerk_max", 2.0))
    vehicle = plan["vehicle"]
    j_long, j_lat = vehicle[..., 6], vehicle[..., 7]
    j_mag = jnp.sqrt(j_long ** 2 + j_lat ** 2)
    return jnp.maximum(
        jnp.maximum(0.0, jnp.abs(j_long) - jerk_max),
        jnp.maximum(jnp.maximum(0.0, jnp.abs(j_lat) - jerk_max),
                    jnp.maximum(0.0, j_mag - jerk_max)),
    )


def forward_motion_violation_per_t(plan, scenario):
    """Forward-motion violation: positive when ``s_dot`` goes negative."""
    s_dot = _plan_s_dot(plan)
    return jnp.maximum(0.0, -s_dot)


def collision_violation_per_t(plans, scenario, agent_idx):
    """Role-dependent footprint collision violation per timestep ``[..., T]``.

    Ego checks full horizon against both neighbours; front checks a
    short-horizon prefix against ego/rear; rear checks a short-horizon prefix
    against ego plus the full horizon against front.
    """
    vehicle_length = float(scenario["safety"].get("vehicle_length", 5.0))
    safe_gap = float(scenario["safety"].get("safe_gap", 3.0))
    longitudinal_clearance = vehicle_length + safe_gap
    lateral_clearance = collision_lateral_clearance(scenario)
    short = collision_prefix(scenario)

    own = plans[agent_idx]
    si, di = own["s"], own["d"]

    if agent_idx == 0:
        front, rear = plans[1], plans[2]
        return jnp.maximum(
            pair_footprint_violation(si, di, front["s"], front["d"],
                                     longitudinal_clearance, lateral_clearance),
            pair_footprint_violation(si, di, rear["s"], rear["d"],
                                     longitudinal_clearance, lateral_clearance),
        )
    if agent_idx == 1:
        ego, rear = plans[0], plans[2]
        out = jnp.zeros_like(si)
        ego_vals = pair_footprint_violation(
            si[..., short], di[..., short], ego["s"][..., short], ego["d"][..., short],
            longitudinal_clearance, lateral_clearance)
        rear_vals = pair_footprint_violation(
            si[..., short], di[..., short], rear["s"][..., short], rear["d"][..., short],
            longitudinal_clearance, lateral_clearance)
        return out.at[..., short].set(jnp.maximum(ego_vals, rear_vals))

    ego, front = plans[0], plans[1]
    out = jnp.zeros_like(si)
    ego_vals = pair_footprint_violation(
        si[..., short], di[..., short], ego["s"][..., short], ego["d"][..., short],
        longitudinal_clearance, lateral_clearance)
    out = out.at[..., short].set(ego_vals)
    front_vals = pair_footprint_violation(si, di, front["s"], front["d"],
                                          longitudinal_clearance, lateral_clearance)
    return jnp.maximum(out, front_vals)


# ---------------------------------------------------------------------------
# Scalar aggregated violations (max over time -> scalar).  These double as
# scalar Constran g_fn outputs: Constran's _wrap_aggregate applies jnp.max,
# which is the identity on an already-aggregated scalar.
# ---------------------------------------------------------------------------

def lane_boundary_violation(plan, scenario):
    return jnp.max(lane_boundary_violation_per_t(plan, scenario))


def speed_limit_violation(plan, scenario):
    return jnp.max(speed_limit_violation_per_t(plan, scenario))


def acc_limit_violation(plan, scenario):
    return jnp.max(acc_limit_violation_per_t(plan, scenario))


def jerk_limit_violation(plan, scenario):
    return jnp.max(jerk_limit_violation_per_t(plan, scenario))


def kinematics_violation(plan, scenario):
    """``max(acc, jerk)`` feasibility layer (single Constran layer)."""
    return jnp.maximum(acc_limit_violation(plan, scenario),
                       jerk_limit_violation(plan, scenario))


def forward_motion_violation(plan, scenario):
    return jnp.max(forward_motion_violation_per_t(plan, scenario))


def collision_violation(plans, scenario, agent_idx):
    return jnp.max(collision_violation_per_t(plans, scenario, agent_idx))


def safety_envelope_violation(plans, scenario, agent_idx):
    """``max(collision, lane_boundary)`` outer feasibility layer."""
    return jnp.maximum(collision_violation(plans, scenario, agent_idx),
                       lane_boundary_violation(plans[agent_idx], scenario))


def constraint_layer_violations(plans, scenario, agent_idx):
    """Reduced feasibility-layer violations keyed by solver layer name."""
    own = plans[agent_idx]
    speed = jnp.max(speed_limit_violation_per_t(own, scenario))
    acc = jnp.max(acc_limit_violation_per_t(own, scenario))
    jerk = jnp.max(jerk_limit_violation_per_t(own, scenario))
    forward = jnp.max(forward_motion_violation_per_t(own, scenario))
    collision = jnp.max(collision_violation_per_t(plans, scenario, agent_idx))
    lane_boundary = jnp.max(lane_boundary_violation_per_t(own, scenario))
    return {
        "speed": speed,
        "kinematics": jnp.maximum(acc, jerk),
        "forward": forward,
        "safety_envelope": jnp.maximum(collision, lane_boundary),
    }


# ---------------------------------------------------------------------------
# Soft objective components
# ---------------------------------------------------------------------------

def progress_objective(plan, v_target):
    """Normalized longitudinal speed-tracking term (mean over horizon)."""
    s_dot = _plan_s_dot(plan)
    return jnp.mean(((s_dot - v_target) / SPEED_RUN_SCALE) ** 2, axis=-1)


def lane_objective(plan, target_d):
    """Normalized target-lane / lane-keeping term (mean over horizon)."""
    d = plan["d"]
    return jnp.mean(((d - target_d) / LANE_KEEP_SCALE) ** 2, axis=-1)


def bridged_jerk_cost(plan, ctx, agent_idx, dt):
    """Bridged jerk comfort: bridge the horizon with the previous executed accel.

    Uses ``ctx["a_long_prev_a{agent_idx}"]`` / ``ctx["a_lat_prev_a{agent_idx}"]``
    when present (filled by the closed-loop runner in a later change) and
    falls back to zero otherwise.
    """
    vehicle = plan["vehicle"]
    a_long = vehicle[..., 4]
    a_lat = vehicle[..., 5]
    if isinstance(ctx, dict):
        prev_long = ctx.get(f"a_long_prev_a{agent_idx}", 0.0)
        prev_lat = ctx.get(f"a_lat_prev_a{agent_idx}", 0.0)
    else:
        prev_long = prev_lat = 0.0
    leading_shape = a_long.shape[:-1] + (1,)
    a_long_full = jnp.concatenate(
        [jnp.full(leading_shape, prev_long, dtype=a_long.dtype), a_long], axis=-1)
    a_lat_full = jnp.concatenate(
        [jnp.full(leading_shape, prev_lat, dtype=a_lat.dtype), a_lat], axis=-1)
    j_long = jnp.diff(a_long_full, axis=-1) / dt
    j_lat = jnp.diff(a_lat_full, axis=-1) / dt
    return jnp.mean((j_long / JERK_LONG_SCALE) ** 2
                    + (j_lat / JERK_LAT_SCALE) ** 2, axis=-1)


def _cvar_topk(weighted, fraction):
    """Top-k mean over the last axis (CVaR tail risk)."""
    n = weighted.shape[-1]
    k = max(1, int(round(fraction * n)))
    k = min(k, n)
    topk_vals = jax.lax.top_k(weighted, k)[0]
    return jnp.mean(topk_vals, axis=-1)


def rss_cvar_risk(plan, neighbor_plans, scenario, dt):
    """RSS/CVaR interaction-risk shaping.

    Uses Frenet longitudinal gap, a lateral-overlap gate, exponential time
    weighting, and a top-k CVaR over the horizon.  Shape-agnostic: returns a
    scalar for ``[T]`` plans, ``[batch]`` for ``[batch, T]`` plans, and
    ``[B, M]`` for broadcasted pairwise plans.
    """
    vehicle_length = float(scenario["safety"].get("vehicle_length", 5.0))
    vehicle_width = float(scenario["safety"].get("vehicle_width", 2.0))
    half_len = vehicle_length / 2.0
    half_wid = vehicle_width / 2.0
    ego_s = plan["s"]
    ego_d = plan["d"]
    ego_v = _plan_s_dot(plan)
    horizon = ego_s.shape[-1]
    t_arr = jnp.arange(horizon) * dt
    time_weight = jnp.exp(-jnp.log(2.0) * t_arr / RSS_TIME_HALF_LIFE)

    r_time = jnp.zeros_like(ego_s)
    length_buffer = 2.0 * half_len
    for nbr in neighbor_plans:
        obs_s = nbr["s"]
        obs_d = nbr["d"]
        obs_v = _plan_s_dot(nbr)
        lat_gap = jnp.abs(ego_d - obs_d)
        lat_overlap = jax.nn.sigmoid(
            (2.0 * half_wid + RSS_LATERAL_MARGIN - lat_gap) / RSS_LATERAL_SOFTNESS)
        rel = obs_s - ego_s
        gap_front = rel - length_buffer
        gap_rear = -rel - length_buffer

        v_ego_react = ego_v + RSS_A_MAX * RSS_RHO
        req_front = (ego_v * RSS_RHO + 0.5 * RSS_A_MAX * RSS_RHO ** 2
                     + v_ego_react ** 2 / (2.0 * RSS_B_EGO_FRONT)
                     - obs_v ** 2 / (2.0 * RSS_B_FRONT))
        req_front = jnp.maximum(RSS_MIN_GAP, req_front)
        v_obs_react = obs_v + RSS_A_MAX * RSS_RHO
        req_rear = (obs_v * RSS_RHO + 0.5 * RSS_A_MAX * RSS_RHO ** 2
                    + v_obs_react ** 2 / (2.0 * RSS_B_REAR)
                    - ego_v ** 2 / (2.0 * RSS_B_EGO))
        req_rear = jnp.maximum(RSS_MIN_GAP, req_rear)

        front_gate = jax.nn.sigmoid(rel / 2.0)
        rear_gate = jax.nn.sigmoid(-rel / 2.0)
        deficit_front = jnp.maximum(0.0, req_front - gap_front) / RSS_DEFICIT_SCALE
        deficit_rear = jnp.maximum(0.0, req_rear - gap_rear) / RSS_DEFICIT_SCALE
        risk_front = front_gate * lat_overlap * (
            jax.nn.softplus(deficit_front) / jnp.log(2.0)) ** RSS_RISK_POWER
        risk_rear = rear_gate * lat_overlap * (
            jax.nn.softplus(deficit_rear) / jnp.log(2.0)) ** RSS_RISK_POWER
        r_time = r_time + risk_front + risk_rear

    weighted = r_time * time_weight
    return _cvar_topk(weighted, RSS_CVAR_FRACTION) / RSS_CVAR_NORMALIZER


def role_target_d(scenario, agent_idx):
    """Lane target for a role: ego merges to ``ego_target_d``; others keep upper lane."""
    if agent_idx == 0:
        return float(scenario.get("behavior", {}).get("ego_target_d", 3.5))
    return upper_lane_center(scenario)


def role_soft_objective(plans, scenario, ctx, agent_idx, dt):
    """Role-aware soft objective: progress + lane target + comfort + RSS risk.

    All agents share the same component formulas; only the lane target and the
    RSS neighbour set are role-dependent (ego merges to ``ego_target_d``;
    front/rear keep ``upper_lane_d``).  Shape-agnostic over leading axes.
    """
    own = plans[agent_idx]
    neighbors = tuple(plans[j] for j in range(len(plans)) if j != agent_idx)
    v_target = float(scenario["agents"][agent_idx]["v_target"])
    target_d = role_target_d(scenario, agent_idx)
    progress = progress_objective(own, v_target)
    lane = lane_objective(own, target_d)
    comfort = bridged_jerk_cost(own, ctx, agent_idx, dt)
    rss = rss_cvar_risk(own, neighbors, scenario, dt)
    return progress + lane + comfort + rss


# ---------------------------------------------------------------------------
# Selected-plan diagnostics (no optimization effect)
# ---------------------------------------------------------------------------

def selected_plan_component_report(plans, scenario, ctx, dt):
    """Return per-agent named cost components for logging/debugging.

    Does not affect optimization.  Returns a Python dict keyed by agent index;
    each value holds both raw feasibility diagnostics (``speed``, ``acc``,
    ``jerk``, ``forward``, ``collision``, ``lane_boundary``), the solver-layer
    aggregates (``kinematics = max(acc, jerk)``, ``safety_envelope =
    max(collision, lane_boundary)``), and the objective tradeoff terms
    (``progress``, ``lane_preference``, ``comfort``, ``rss``).  All values are
    max-over-horizon scalars (or mean-over-horizon for objective terms).
    """
    reports = {}
    for agent_idx in range(len(plans)):
        own = plans[agent_idx]
        neighbors = tuple(plans[j] for j in range(len(plans)) if j != agent_idx)
        v_target = float(scenario["agents"][agent_idx]["v_target"])
        target_d = role_target_d(scenario, agent_idx)
        speed = speed_limit_violation(own, scenario)
        acc = acc_limit_violation(own, scenario)
        jerk = jerk_limit_violation(own, scenario)
        forward = forward_motion_violation(own, scenario)
        collision = collision_violation(plans, scenario, agent_idx)
        lane_boundary = lane_boundary_violation(own, scenario)
        reports[agent_idx] = {
            "lane_boundary": lane_boundary,
            "collision": collision,
            "speed": speed,
            "acc": acc,
            "jerk": jerk,
            "kinematics": jnp.maximum(acc, jerk),
            "forward": forward,
            "safety_envelope": jnp.maximum(collision, lane_boundary),
            "progress": progress_objective(own, v_target),
            "lane_preference": lane_objective(own, target_d),
            "comfort": bridged_jerk_cost(own, ctx, agent_idx, dt),
            "rss": rss_cvar_risk(own, neighbors, scenario, dt),
        }
    return reports
