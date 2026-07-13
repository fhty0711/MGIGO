"""Constraint builders for Frenet B-spline MPC.

All kinematics constraints (acc, jerk, speed) use to_vehicle_states()
- the correct Frenet->vehicle transformation with curvature coupling.

Lane and obstacle use Frenet / Cartesian directly (no curvature coupling needed).

Per-sample penalty: max(max(0, |long|-LIM), max(0, |lat|-LIM))
Only the *worse* component is penalised, not both.
"""

from __future__ import annotations

import jax.numpy as jnp
from Constraintdealer.Constran import Deterministic


# ═══════════════════════════════════════════════════════════════════════
# Constraint parameters
# ═══════════════════════════════════════════════════════════════════════

# Fallback physical limits. Scenarios normally provide these under "safety".
V_MIN, V_MAX = 2.0, 35.0
ACC_MAX = 5.0          # m/s²
JERK_MAX = 2.0         # m/s³  (tight: comfort limit)
A_BRAKE = 8.0


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _eval_frenet(theta, ctx, gen):
    """Unpack theta -> evaluate Frenet trajectory."""
    n = gen.n_free
    return gen.evaluate(
        theta[:n], theta[n:2 * n],
        ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
        ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"],
    )


def _eval_vehicle_states(theta, ctx, gen):
    """Unpack theta -> evaluate -> vehicle states [T, 9]."""
    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = _eval_frenet(theta, ctx, gen)
    return gen.to_vehicle_states(s, d, s_dot, d_dot,
                                 s_ddot, d_ddot, s_dddot, d_dddot)


# ═══════════════════════════════════════════════════════════════════════
# Constraint factory
# ═══════════════════════════════════════════════════════════════════════

def _constraint_spec(name, g_fn, config):
    spec = dict(config["specs"][name])
    return Deterministic(
        g_fn,
        mode=spec.get("mode", "soft"),
        priority=spec["priority"],
        aggregate=spec.get("aggregate", ""),
        transform=spec.get("transform", ""),
        baseline=spec.get("baseline"),
    )


def make_constraints(gen, road, safety, config):
    """Build constraint list for Frenet B-spline MPC.

    Args:
        gen:     FrenetBSplineTrajectory
        road:    {"lane_hw": float, "lane_bounds_d": (d_min, d_max)  (optional)}
        safety:  {"obs_safe_dist": float, "a_brake": float, ...}
        config:  {"enabled": (...), "specs": {...}, "constran": {...},
                  "safety_overrides": {...}  (optional)}
    """
    enabled = tuple(config["enabled"])

    safety = dict(safety)
    safety.update(config.get("safety_overrides", {}))

    lane_hw = road["lane_hw"]
    lane_bounds_d = road.get("lane_bounds_d", (-lane_hw, lane_hw))
    lane_bounds_d = jnp.asarray(lane_bounds_d, dtype=jnp.float32)
    obs_safe_dist = safety["obs_safe_dist"]
    a_brake = safety.get("a_brake", A_BRAKE)
    v_min = safety.get("v_min", V_MIN)
    v_max = safety.get("v_max", V_MAX)
    acc_max = safety.get("acc_max", ACC_MAX)
    jerk_max = safety.get("jerk_max", JERK_MAX)

    def obs_g(theta, ctx):
        """RSS: longitudinal + lateral safe distance per obstacle."""
        n_obs = ctx["obs_pos"].shape[1]
        if n_obs == 0:
            return jnp.zeros(gen.T)

        st = _eval_vehicle_states(theta, ctx, gen)
        x, y, v = st[:, 0], st[:, 1], st[:, 2]
        rho = obs_safe_dist

        d_rss = v * rho + v ** 2 / (2.0 * a_brake)                    # [T]

        dx = x[:, None] - ctx["obs_pos"][:, :, 0]                     # [T, N]
        dy = y[:, None] - ctx["obs_pos"][:, :, 1]                     # [T, N]
        r  = ctx["obs_rad"]                                           # [T, N]

        pen_x = jnp.maximum(0., d_rss[:, None] + r - jnp.abs(dx))
        pen_y = jnp.maximum(0., r - jnp.abs(dy))

        return jnp.maximum(pen_x, pen_y).max(axis=-1)

    def lane_g(theta, ctx):
        """Lane boundary.  Supports asymmetric bounds via lane_bounds_d."""
        _, d, _, _, _, _, _, _ = _eval_frenet(theta, ctx, gen)
        return jnp.maximum(
            jnp.maximum(0., lane_bounds_d[0] - d),
            jnp.maximum(0., d - lane_bounds_d[1]),
        )

    def speed_g(theta, ctx):
        """Speed: max(v_min-v, v-v_max, 0) per sample."""
        st = _eval_vehicle_states(theta, ctx, gen)
        v = st[:, 2]
        return jnp.maximum(
            jnp.maximum(0., v_min - v),
            jnp.maximum(0., v - v_max),
        )

    def acc_g(theta, ctx):
        """Acc: max(|long|-LIM, |lat|-LIM, |total|-LIM, 0) per sample."""
        st = _eval_vehicle_states(theta, ctx, gen)
        a_long, a_lat = st[:, 4], st[:, 5]
        am = jnp.sqrt(a_long ** 2 + a_lat ** 2)
        return jnp.maximum(
            jnp.maximum(0., jnp.abs(a_long) - acc_max),
            jnp.maximum(
                jnp.maximum(0., jnp.abs(a_lat) - acc_max),
                jnp.maximum(0., am            - acc_max),
            ),
        )

    def jerk_g(theta, ctx):
        """Jerk: max(|long|-LIM, |lat|-LIM, |total|-LIM, 0) per sample."""
        st = _eval_vehicle_states(theta, ctx, gen)
        j_long, j_lat = st[:, 6], st[:, 7]
        jm = jnp.sqrt(j_long ** 2 + j_lat ** 2)
        return jnp.maximum(
            jnp.maximum(0., jnp.abs(j_long) - jerk_max),
            jnp.maximum(
                jnp.maximum(0., jnp.abs(j_lat) - jerk_max),
                jnp.maximum(0., jm            - jerk_max),
            ),
        )

    g_fns = {
        "obs": obs_g,
        "lane": lane_g,
        "speed": speed_g,
        "acc": acc_g,
        "jerk": jerk_g,
    }

    return [_constraint_spec(name, g_fns[name], config) for name in enabled]


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_g_values(st, d, x_cart, y_cart, obs_pos, obs_rad,
                     lane_hw, safety):
    """Compute per-constraint g-values for reporting.

    Args:
        obs_pos: [T, N, 2] or [N, 2], obs_rad: [T, N] or [N]
        lane_hw: scalar (symmetric fallback)
        safety:  scenario safety dict (may contain lane_bounds_d)
    """
    v = st[:, 2]
    a_long, a_lat = st[:, 4], st[:, 5]
    j_long, j_lat = st[:, 6], st[:, 7]
    am = jnp.sqrt(a_long ** 2 + a_lat ** 2)
    jm = jnp.sqrt(j_long ** 2 + j_lat ** 2)

    bounds = jnp.asarray(safety.get("lane_bounds_d", (-lane_hw, lane_hw)),
                         dtype=jnp.float32)
    g_lane = jnp.quantile(
        jnp.maximum(jnp.maximum(0., bounds[0] - d),
                    jnp.maximum(0., d - bounds[1])),
        0.9,
    )

    obs_safe_dist = safety["obs_safe_dist"]
    a_brake = safety.get("a_brake", A_BRAKE)
    v_min = safety.get("v_min", V_MIN)
    v_max = safety.get("v_max", V_MAX)
    acc_max = safety.get("acc_max", ACC_MAX)
    jerk_max = safety.get("jerk_max", JERK_MAX)

    n_obs = obs_pos.shape[1] if obs_pos.ndim == 3 else obs_pos.shape[0]
    if n_obs == 0:
        g_obs = 0.0
    else:
        rho = obs_safe_dist
        d_rss = v * rho + v ** 2 / (2.0 * a_brake)
        if obs_pos.ndim == 3:
            dx = x_cart[:, None] - obs_pos[:, :, 0]
            dy = y_cart[:, None] - obs_pos[:, :, 1]
            r  = obs_rad
        else:
            dx = x_cart[:, None] - obs_pos[None, :, 0]
            dy = y_cart[:, None] - obs_pos[None, :, 1]
            r  = obs_rad[None, :]
        pen_x = jnp.maximum(0., d_rss[:, None] + r - jnp.abs(dx))
        pen_y = jnp.maximum(0., r - jnp.abs(dy))
        g_obs = float(jnp.max(jnp.maximum(pen_x, pen_y)))

    g_jerk = float(jnp.max(
        jnp.maximum(
            jnp.maximum(0., jnp.abs(j_long) - jerk_max),
            jnp.maximum(jnp.maximum(0., jnp.abs(j_lat) - jerk_max),
                        jnp.maximum(0., jm - jerk_max)),
        )))

    g_acc = float(jnp.max(
        jnp.maximum(
            jnp.maximum(0., jnp.abs(a_long) - acc_max),
            jnp.maximum(jnp.maximum(0., jnp.abs(a_lat) - acc_max),
                        jnp.maximum(0., am - acc_max)),
        )))

    g_spd = jnp.quantile(
        jnp.maximum(jnp.maximum(0., v_min - v), jnp.maximum(0., v - v_max)), 0.9)

    return {
        'lane': float(g_lane), 'obs': float(g_obs),
        'jerk': float(g_jerk), 'acc': float(g_acc), 'spd': float(g_spd),
    }


def compute_summary(st, d, x_cart, y_cart, obs_pos, obs_rad):
    """Compute summary metrics: min_obs_dist, max |a_long|, |a_lat|, |jerk|."""
    a_long, a_lat = st[:, 4], st[:, 5]
    j_long, j_lat = st[:, 6], st[:, 7]
    jm = jnp.sqrt(j_long ** 2 + j_lat ** 2)

    n_obs = obs_pos.shape[1] if obs_pos.ndim == 3 else obs_pos.shape[0]
    if n_obs == 0:
        min_obs = 1e9
    else:
        if obs_pos.ndim == 3:
            dist = jnp.sqrt((x_cart[:, None] - obs_pos[:, :, 0]) ** 2 +
                            (y_cart[:, None] - obs_pos[:, :, 1]) ** 2) - obs_rad
        else:
            dist = jnp.sqrt((x_cart[:, None] - obs_pos[None, :, 0]) ** 2 +
                            (y_cart[:, None] - obs_pos[None, :, 1]) ** 2) - obs_rad[None, :]
        min_obs = float(jnp.min(dist))
    return {
        'min_obs': min_obs,
        'max_a_long': float(jnp.max(jnp.abs(a_long))),
        'max_a_lat':  float(jnp.max(jnp.abs(a_lat))),
        'max_jerk':   float(jnp.max(jm)),
        'v':          float(st[0, 2]),
    }
