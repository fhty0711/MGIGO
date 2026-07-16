"""Two-agent Constran collision-game objectives (B-spline + RNE).

Migrated from the old standalone ``Simple_game_2b_constran.py`` demo.
Agents share the basic Lyapunov tracking objective; collision avoidance
and lane bounds are handled by Constran constraints.
"""

from __future__ import annotations

import jax.numpy as jnp

from Constraintdealer.Constran import Deterministic

from Cartest.planning.costs.game_2a_basic import (
    _eval_agent_plan,
    pair_distance_violation,
    make_agent_specs as make_basic_specs,
)


def make_agent_specs(gen, scenario):
    """Build per-agent objective/constraint specs for the 2-agent Constran game."""
    safe_dist = float(scenario["safety"].get("safe_dist", 3.0))
    lane_hw = float(scenario["road"].get("lane_hw", 4.0))
    v_min = float(scenario["safety"].get("v_min", 2.0))
    v_max = float(scenario["safety"].get("v_max", 35.0))
    acc_max = float(scenario["safety"].get("acc_max", 5.0))
    jerk_max = float(scenario["safety"].get("jerk_max", 2.0))
    basic_specs = make_basic_specs(gen, scenario)

    def make_collision_g(agent_idx):
        def collision_g(joint_x, ctx):
            _f0, _v0, (x0, y0) = _eval_agent_plan(gen, joint_x, ctx, 0)
            _f1, _v1, (x1, y1) = _eval_agent_plan(gen, joint_x, ctx, 1)
            return pair_distance_violation(x0, y0, x1, y1, safe_dist)
        return collision_g

    def make_lane_g(agent_idx):
        def lane_g(joint_x, ctx):
            frenet, _vehicle, _cart = _eval_agent_plan(gen, joint_x, ctx, agent_idx)
            return jnp.maximum(0.0, jnp.abs(frenet[1]) - lane_hw)
        return lane_g

    def make_speed_g(agent_idx):
        def speed_g(joint_x, ctx):
            _frenet, vehicle, _cart = _eval_agent_plan(gen, joint_x, ctx, agent_idx)
            v = vehicle[:, 2]
            return jnp.maximum(jnp.maximum(0.0, v_min - v), jnp.maximum(0.0, v - v_max))
        return speed_g

    def make_acc_g(agent_idx):
        def acc_g(joint_x, ctx):
            _frenet, vehicle, _cart = _eval_agent_plan(gen, joint_x, ctx, agent_idx)
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
            _frenet, vehicle, _cart = _eval_agent_plan(gen, joint_x, ctx, agent_idx)
            j_long, j_lat = vehicle[:, 6], vehicle[:, 7]
            j_mag = jnp.sqrt(j_long ** 2 + j_lat ** 2)
            return jnp.maximum(
                jnp.maximum(0.0, jnp.abs(j_long) - jerk_max),
                jnp.maximum(jnp.maximum(0.0, jnp.abs(j_lat) - jerk_max),
                            jnp.maximum(0.0, j_mag - jerk_max)),
            )
        return jerk_g

    specs = {}
    for aid in range(len(scenario["agents"])):
        constraints = [
            Deterministic(make_lane_g(aid), mode="soft", priority=1,
                          aggregate="q95", transform="soft"),
            Deterministic(make_speed_g(aid), mode="soft", priority=2,
                          aggregate="max", transform="soft"),
            Deterministic(make_acc_g(aid), mode="soft", priority=3,
                          aggregate="max", transform="soft"),
            Deterministic(make_jerk_g(aid), mode="soft", priority=4,
                          aggregate="max", transform="soft"),
            Deterministic(make_collision_g(aid), mode="hard", priority=5,
                          aggregate="max", transform="hard"),
        ]
        specs[aid] = (basic_specs[aid][0], constraints)
    return specs
