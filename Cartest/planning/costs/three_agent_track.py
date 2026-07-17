"""Three-agent ego/front/rear game objectives (B-spline + RNE).

The role objectives and feasibility-layer violations live in
``Cartest.planning.costs.three_agent_track_components``; this module wires
them into the scalar Constran ``g_fn`` interface used by the game solver.
The three agents share component formulas and physical scales; only the lane
target and RSS neighbour set differ by role (ego merges toward
``ego_target_d``; front/rear keep ``upper_lane_d``).
"""

from __future__ import annotations

import jax.numpy as jnp

from Constraintdealer.Constran import Deterministic

from Cartest.planning.costs.game_2a_basic import _eval_agent_plan
from Cartest.planning.costs.three_agent_track_components import (
    collision_lateral_clearance,
    collision_prefix,
    forward_motion_violation,
    kinematics_violation,
    lane_footprint_bounds,
    pair_footprint_violation,
    role_soft_objective,
    safety_envelope_violation,
    speed_limit_violation,
    upper_lane_center,
)

# Backward-compatible aliases for callers that import the underscore name.
_collision_prefix = collision_prefix


THREE_AGENT_CONSTRAINT_DEFS = (
    ("speed", "hard", 3, "max", "hard"),
    ("kinematics", "hard", 4, "max", "hard"),
    ("forward", "hard", 5, "max", "hard"),
    ("safety_envelope", "hard", 6, "max", "hard"),
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


def longitudinal_tracking_reference(s0, s_dot0, v_target, omega_s, T, dt):
    """Reference s trajectory and derivatives (retained for the batched path)."""
    t_arr = jnp.arange(T) * dt
    dv = s_dot0 - v_target
    exp_term = jnp.exp(-omega_s * t_arr)
    s_ref = s0 + v_target * t_arr + dv / omega_s * (1.0 - exp_term)
    s_dot_ref = v_target + dv * exp_term
    s_ddot_ref = -dv * omega_s * exp_term
    return s_ref, s_dot_ref, s_ddot_ref


def _track_dt(gen, scenario):
    return float(getattr(gen, "dt", scenario.get("game", {}).get("dt", 0.15)))


def _raw_ctx(ctx):
    if isinstance(ctx, tuple) and len(ctx) == 2:
        return ctx[0]
    return ctx


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


def _plan_dict(plan_tuple):
    """Convert a (frenet, vehicle, cart) tuple into the component-module dict shape."""
    frenet, vehicle, _cart = plan_tuple
    return {
        "s": frenet[0], "d": frenet[1], "s_dot": frenet[2], "d_dot": frenet[3],
        "s_ddot": frenet[4], "d_ddot": frenet[5], "s_dddot": frenet[6],
        "d_dddot": frenet[7], "vehicle": vehicle,
    }


def make_agent_specs(gen, scenario):
    """Build per-agent objective/constraint specs for the 3-agent track game."""
    agent_count = len(scenario["agents"])

    def _plans_dicts(joint_x, ctx):
        plans = _prepared_plans(gen, joint_x, ctx, agent_count)
        return tuple(_plan_dict(plan) for plan in plans)

    def _objective(agent_idx):
        def objective(joint_x, ctx):
            plan_dicts = _plans_dicts(joint_x, ctx)
            return role_soft_objective(
                plan_dicts, scenario, _raw_ctx(ctx),
                agent_idx=agent_idx, dt=_track_dt(gen, scenario))
        return objective

    ego_objective = _objective(0)
    front_objective = _objective(1)
    rear_objective = _objective(2)

    def make_speed_g(agent_idx):
        def speed_g(joint_x, ctx):
            plan_dicts = _plans_dicts(joint_x, ctx)
            return speed_limit_violation(plan_dicts[agent_idx], scenario)
        return speed_g

    def make_kinematics_g(agent_idx):
        def kinematics_g(joint_x, ctx):
            plan_dicts = _plans_dicts(joint_x, ctx)
            return kinematics_violation(plan_dicts[agent_idx], scenario)
        return kinematics_g

    def make_forward_g(agent_idx):
        def forward_g(joint_x, ctx):
            plan_dicts = _plans_dicts(joint_x, ctx)
            return forward_motion_violation(plan_dicts[agent_idx], scenario)
        return forward_g

    def make_safety_envelope_g(agent_idx):
        def safety_envelope_g(joint_x, ctx):
            plan_dicts = _plans_dicts(joint_x, ctx)
            return safety_envelope_violation(plan_dicts, scenario, agent_idx)
        return safety_envelope_g

    def make_constraints(agent_idx):
        violation_fns = {
            "speed": make_speed_g(agent_idx),
            "kinematics": make_kinematics_g(agent_idx),
            "forward": make_forward_g(agent_idx),
            "safety_envelope": make_safety_envelope_g(agent_idx),
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
