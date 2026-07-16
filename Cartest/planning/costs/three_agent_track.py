"""Three-agent ego/front/rear game objectives (B-spline + RNE).

Inspired by ``MultipleTest/Trackgame.py``: the ego agent wants to merge
toward the upper lane and pays for full-horizon collisions with both
neighbours; the front agent is short-horizon risk-averse; the rear agent
also guards longitudinal clearance against the front vehicle.
"""

from __future__ import annotations

import jax.numpy as jnp

from Constraintdealer.Constran import Deterministic

from Cartest.planning.costs.game_2a_basic import (
    _eval_agent_plan,
    pair_distance_violation,
)


THREE_AGENT_CONSTRAINT_DEFS = (
    ("lane", "soft", 1, "q95", "soft"),
    ("speed", "soft", 2, "max", "soft"),
    ("acc", "soft", 3, "max", "soft"),
    ("jerk", "soft", 4, "max", "soft"),
    ("collision", "hard", 5, "max", "hard"),
)


def _make_constraint_from_definition(definition, g_fn):
    _name, mode, priority, aggregate, transform = definition
    return Deterministic(g_fn, mode=mode, priority=priority,
                         aggregate=aggregate, transform=transform)


def three_agent_batched_layers():
    """Compile static nesting metadata through the real ConstraintSpec rules."""
    placeholder = lambda x, ctx: x
    layers = []
    for definition in THREE_AGENT_CONSTRAINT_DEFS:
        name = definition[0]
        spec = _make_constraint_from_definition(definition, placeholder)
        table = spec.get_transform_table()
        resolution = float(table[0][0]) if table is not None else 0.0
        layers.append((name, spec.aggregate, table, spec.baseline, resolution))
    return tuple(layers)


def _collision_prefix(scenario):
    """Short-horizon slice that includes the state executed by the MPC."""
    execute_index = int(scenario.get("game", {}).get("execute_index", 1))
    return slice(0, max(1, execute_index + 1))


def eval_joint_plans(gen, joint_x, ctx, agent_count=3):
    """Evaluate all agent B-spline plans once for a joint decision vector."""
    return tuple(_eval_agent_plan(gen, joint_x, ctx, idx) for idx in range(agent_count))


def _prepared_plans(gen, joint_x, ctx, agent_count):
    if isinstance(ctx, tuple) and len(ctx) == 2:
        return ctx[1]
    return eval_joint_plans(gen, joint_x, ctx, agent_count)


def _agent_plan(gen, joint_x, ctx, agent_idx, agent_count):
    return _prepared_plans(gen, joint_x, ctx, agent_count)[agent_idx]


def _attach_joint_plan_prepare(objective, gen, agent_count):
    def prepare(joint_x, ctx):
        return eval_joint_plans(gen, joint_x, ctx, agent_count)

    objective._constran_prepare = prepare
    return objective


def make_agent_specs(gen, scenario):
    """Build per-agent objective/constraint specs for the 3-agent track game."""
    agent_count = len(scenario["agents"])
    safe_gap = float(scenario["safety"].get("safe_gap", 3.0))
    vehicle_length = float(scenario["safety"].get("vehicle_length", 5.0))
    v_min = float(scenario["safety"].get("v_min", 2.0))
    v_max = float(scenario["safety"].get("v_max", 35.0))
    acc_max = float(scenario["safety"].get("acc_max", 5.0))
    jerk_max = float(scenario["safety"].get("jerk_max", 2.0))
    lane_min, lane_max = scenario["road"].get("lane_bounds_d", (-1.75, 5.25))
    ego_target_d = float(scenario["behavior"].get("ego_target_d", 3.5))

    def ego_objective(joint_x, ctx):
        fr_e, _st_e, _cart_e = _agent_plan(gen, joint_x, ctx, 0, agent_count)
        d, s_dot, d_dot, d_ddot = fr_e[1], fr_e[2], fr_e[3], fr_e[5]
        s_dddot, d_dddot = fr_e[6], fr_e[7]
        v_ref = float(scenario["agents"][0]["v_target"])
        return (
            3.0 * jnp.sum((s_dot - v_ref) ** 2)
            + 10.0 * jnp.sum((d - ego_target_d) ** 2)
            + 5.0 * jnp.sum(d_dot ** 2)
            + 0.5 * jnp.sum(d_ddot ** 2)
            + jnp.sum(s_dddot ** 2 + d_dddot ** 2)
        )

    def front_objective(joint_x, ctx):
        fr_f, _st_f, _cart_f = _agent_plan(gen, joint_x, ctx, 1, agent_count)
        v_ref = float(scenario["agents"][1]["v_target"])
        return (
            3.0 * jnp.sum((fr_f[2] - v_ref) ** 2)
            + 2.0 * jnp.sum(fr_f[4] ** 2)
        )

    def rear_objective(joint_x, ctx):
        fr_r, _st_r, _cart_r = _agent_plan(gen, joint_x, ctx, 2, agent_count)
        v_ref = float(scenario["agents"][2]["v_target"])
        return (
            3.0 * jnp.sum((fr_r[2] - v_ref) ** 2)
            + 2.0 * jnp.sum(fr_r[4] ** 2)
        )

    def make_lane_g(agent_idx):
        def lane_g(joint_x, ctx):
            frenet, _vehicle, _cart = _agent_plan(gen, joint_x, ctx, agent_idx, agent_count)
            d = frenet[1]
            return jnp.maximum(jnp.maximum(0.0, lane_min - d),
                               jnp.maximum(0.0, d - lane_max))
        return lane_g

    def make_speed_g(agent_idx):
        def speed_g(joint_x, ctx):
            _frenet, vehicle, _cart = _agent_plan(gen, joint_x, ctx, agent_idx, agent_count)
            v = vehicle[:, 2]
            return jnp.maximum(jnp.maximum(0.0, v_min - v), jnp.maximum(0.0, v - v_max))
        return speed_g

    def make_acc_g(agent_idx):
        def acc_g(joint_x, ctx):
            _frenet, vehicle, _cart = _agent_plan(gen, joint_x, ctx, agent_idx, agent_count)
            a_long, a_lat = vehicle[:, 4], vehicle[:, 5]
            a_mag = jnp.sqrt(a_long ** 2 + a_lat ** 2)
            return jnp.maximum(
                jnp.maximum(0.0, jnp.abs(a_long) - acc_max),
                jnp.maximum(jnp.maximum(0.0, jnp.abs(a_lat) - acc_max),
                            jnp.maximum(0.0, a_mag - acc_max)),
            )
        return acc_g

    def make_jerk_g(agent_idx):
        def jerk_g(joint_x, ctx):
            _frenet, vehicle, _cart = _agent_plan(gen, joint_x, ctx, agent_idx, agent_count)
            j_long, j_lat = vehicle[:, 6], vehicle[:, 7]
            j_mag = jnp.sqrt(j_long ** 2 + j_lat ** 2)
            return jnp.maximum(
                jnp.maximum(0.0, jnp.abs(j_long) - jerk_max),
                jnp.maximum(jnp.maximum(0.0, jnp.abs(j_lat) - jerk_max),
                            jnp.maximum(0.0, j_mag - jerk_max)),
            )
        return jerk_g

    def make_collision_g(agent_idx):
        def collision_g(joint_x, ctx):
            plans = _prepared_plans(gen, joint_x, ctx, agent_count)
            short = _collision_prefix(scenario)
            fr_i, _st_i, (xi, yi) = plans[agent_idx]
            if agent_idx == 0:
                _fr_f, _st_f, (xf, yf) = plans[1]
                _fr_r, _st_r, (xr, yr) = plans[2]
                return jnp.maximum(pair_distance_violation(xi, yi, xf, yf, safe_gap),
                                   pair_distance_violation(xi, yi, xr, yr, safe_gap))
            if agent_idx == 1:
                _fr_e, _st_e, (xe, ye) = plans[0]
                _fr_r, _st_r, (xr, yr) = plans[2]
                out = jnp.zeros(gen.T)
                values = pair_distance_violation(xi[short], yi[short],
                                                 xe[short], ye[short], safe_gap)
                rear_values = pair_distance_violation(xi[short], yi[short],
                                                      xr[short], yr[short], safe_gap)
                return out.at[short].set(jnp.maximum(values, rear_values))

            _fr_e, _st_e, (xe, ye) = plans[0]
            fr_f, _st_f, _cart_f = plans[1]
            out = jnp.zeros(gen.T)
            ego_values = pair_distance_violation(xi[short], yi[short],
                                                 xe[short], ye[short], safe_gap)
            dx_front_rear = fr_f[0] - fr_i[0]
            clearance = vehicle_length + safe_gap
            clearance_violation = jnp.maximum(0.0, clearance - dx_front_rear)
            return jnp.maximum(out.at[short].set(ego_values), clearance_violation)
        return collision_g

    def make_constraints(agent_idx):
        violation_fns = {
            "lane": make_lane_g(agent_idx),
            "speed": make_speed_g(agent_idx),
            "acc": make_acc_g(agent_idx),
            "jerk": make_jerk_g(agent_idx),
            "collision": make_collision_g(agent_idx),
        }
        return [
            _make_constraint_from_definition(definition, violation_fns[definition[0]])
            for definition in THREE_AGENT_CONSTRAINT_DEFS
        ]

    return {
        0: (_attach_joint_plan_prepare(ego_objective, gen, agent_count), make_constraints(0)),
        1: (_attach_joint_plan_prepare(front_objective, gen, agent_count), make_constraints(1)),
        2: (_attach_joint_plan_prepare(rear_objective, gen, agent_count), make_constraints(2)),
    }
