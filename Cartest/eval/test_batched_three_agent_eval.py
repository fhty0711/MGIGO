"""Tests for three-agent batched B-spline game evaluation."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
import sys
from unittest.mock import patch

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.execution.execute import FrenetState
from Cartest.planning.scenarios import get_scenario
from Cartest.planning.solver_modes import build_multi_agent_context, build_multi_agent_warmstart
from Cartest.planning.costs.three_agent_track import make_agent_specs


BASIS = ROOT / "Cartest" / "basis" / "bspline_basis.npz"


def _states(scenario):
    return [
        FrenetState(
            s=agent["s"], s_dot=agent["s_dot"], s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"], d_dot=agent.get("d_dot", 0.0), d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]


def _joint_from_mu(mu):
    parts = []
    for agent_idx in range(3):
        s_block = agent_idx * 2
        d_block = s_block + 1
        parts.append(mu[s_block, 0])
        parts.append(mu[d_block, 0])
    return jnp.concatenate(parts)


def test_t_alpha_uses_exact_small_table_lookup():
    from Constraintdealer.Constran import (
        CONSTRAINT_KNOTS_G,
        CONSTRAINT_KNOTS_T,
        T_alpha,
    )

    source = inspect.getsource(T_alpha)
    assert 'method="compare_all"' in source

    values = jnp.asarray([
        0.0,
        -1e-12,
        1e-12,
        -float(CONSTRAINT_KNOTS_G[0]),
        float(CONSTRAINT_KNOTS_G[0]),
        -float(CONSTRAINT_KNOTS_G[-1]),
        float(CONSTRAINT_KNOTS_G[-1]),
        -1e6,
        1e6,
    ])
    log_knots = jnp.log(CONSTRAINT_KNOTS_G)
    log_ax = jnp.log(jnp.maximum(jnp.abs(values), jnp.nextafter(0.0, 1.0)))
    indices = jnp.searchsorted(log_knots, log_ax, side="right", method="scan") - 1
    indices = jnp.clip(indices, 0, len(CONSTRAINT_KNOTS_G) - 2)
    x0, x1 = log_knots[indices], log_knots[indices + 1]
    y0 = jnp.asarray(CONSTRAINT_KNOTS_T)[indices]
    y1 = jnp.asarray(CONSTRAINT_KNOTS_T)[indices + 1]
    t = jnp.maximum((log_ax - x0) / (x1 - x0 + 1e-12), 0.0)
    expected = jnp.sign(values) * (y0 + t * (y1 - y0))

    assert jnp.allclose(T_alpha(values), expected, rtol=0.0, atol=0.0)


def test_batched_plan_eval_shapes_match_three_agents():
    from Cartest.planning.costs.three_agent_track_batched import evaluate_joint_plan_batch

    scenario = get_scenario("three_agent_track")
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    ctx = build_multi_agent_context(_states(scenario))
    mu, _ = build_multi_agent_warmstart(gen, scenario, _states(scenario), jax.random.PRNGKey(0))

    samples = jnp.stack([_joint_from_mu(mu), _joint_from_mu(mu) + 0.05], axis=0)
    plans = evaluate_joint_plan_batch(gen, samples, ctx, agent_count=3)

    assert len(plans) == 3
    assert plans[0]["s"].shape == (2, gen.T)
    assert plans[0]["x"].shape == (2, gen.T)
    assert plans[0]["vehicle"].shape == (2, gen.T, 9)


def test_batched_agent_cost_matches_scalar_cost_for_fixed_joint_samples():
    from Cartest.planning.costs.three_agent_track_batched import evaluate_joint_plan_batch, batched_agent_costs_from_plans

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, _ = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(1))
    joint = _joint_from_mu(mu)
    joint_batch = jnp.stack([joint, joint + 0.01], axis=0)

    specs = make_agent_specs(gen, scenario)
    scalar = jnp.stack([
        jnp.stack([specs[aid][0](sample, ctx) for aid in range(3)])
        for sample in joint_batch
    ], axis=0)

    plans = evaluate_joint_plan_batch(gen, joint_batch, ctx, agent_count=3)
    batched = batched_agent_costs_from_plans(plans, scenario, gen.dt)

    assert batched.shape == (2, 3)
    assert jnp.all(jnp.isfinite(batched))
    assert jnp.allclose(batched, scalar, rtol=2e-4, atol=2e-3)


def test_front_and_rear_objectives_penalize_upper_lane_center_offset():
    from Cartest.planning.costs.three_agent_track_batched import batched_agent_costs_from_plans

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    horizon = 6
    zeros = jnp.zeros(horizon)
    vehicle = jnp.zeros((horizon, 9))
    upper_lane_d = scenario["behavior"]["upper_lane_d"]

    def scalar_plan(d_value):
        d = jnp.full(horizon, d_value)
        frenet = (zeros, d, zeros, zeros, zeros, zeros, zeros, zeros)
        return frenet, vehicle, (zeros, d)

    centered_scalar = (
        scalar_plan(0.0), scalar_plan(upper_lane_d), scalar_plan(upper_lane_d))
    offset_scalar = (
        scalar_plan(0.0), scalar_plan(upper_lane_d - 0.4),
        scalar_plan(upper_lane_d - 0.4))

    class FakeGen:
        T = horizon

    specs = make_agent_specs(FakeGen(), scenario)
    ctx = {
        "s0_a1": 0.0,
        "s_dot0_a1": scenario["agents"][1]["s_dot"],
        "s0_a2": 0.0,
        "s_dot0_a2": scenario["agents"][2]["s_dot"],
    }
    with patch("Cartest.planning.costs.three_agent_track._prepared_plans",
               return_value=centered_scalar):
        centered_front = specs[1][0](jnp.zeros(1), ctx)
        centered_rear = specs[2][0](jnp.zeros(1), ctx)
    with patch("Cartest.planning.costs.three_agent_track._prepared_plans",
               return_value=offset_scalar):
        offset_front = specs[1][0](jnp.zeros(1), ctx)
        offset_rear = specs[2][0](jnp.zeros(1), ctx)

    assert offset_front > centered_front
    assert offset_rear > centered_rear

    def batched_plan(d_value):
        d = jnp.full((1, horizon), d_value)
        return {
            "s": jnp.zeros((1, horizon)),
            "d": d,
            "s_dot": jnp.zeros((1, horizon)),
            "d_dot": jnp.zeros((1, horizon)),
            "s_ddot": jnp.zeros((1, horizon)),
            "d_ddot": jnp.zeros((1, horizon)),
            "s_dddot": jnp.zeros((1, horizon)),
            "d_dddot": jnp.zeros((1, horizon)),
            "vehicle": vehicle[None],
            "x": jnp.zeros((1, horizon)),
            "y": d,
        }

    centered_batched = (
        batched_plan(0.0), batched_plan(upper_lane_d), batched_plan(upper_lane_d))
    offset_batched = (
        batched_plan(0.0), batched_plan(upper_lane_d - 0.4),
        batched_plan(upper_lane_d - 0.4))
    centered_cost = batched_agent_costs_from_plans(centered_batched, scenario, 0.1)[0]
    offset_cost = batched_agent_costs_from_plans(offset_batched, scenario, 0.1)[0]

    assert offset_cost[1] > centered_cost[1]
    assert offset_cost[2] > centered_cost[2]


def test_front_and_rear_objectives_penalize_reverse_longitudinal_motion():
    from Cartest.planning.costs.three_agent_track_batched import batched_agent_costs_from_plans

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    horizon = 6
    zeros = jnp.zeros(horizon)
    vehicle = jnp.zeros((horizon, 9))
    upper_lane_d = scenario["behavior"]["upper_lane_d"]

    def plan(s_start, s_dot_value):
        s = s_start + jnp.arange(horizon) * s_dot_value * 0.1
        d = jnp.full(horizon, upper_lane_d)
        frenet = (s, d, jnp.full(horizon, s_dot_value), zeros, zeros, zeros, zeros, zeros)
        return {
            "s": s[None],
            "d": d[None],
            "s_dot": jnp.full((1, horizon), s_dot_value),
            "d_dot": zeros[None],
            "s_ddot": zeros[None],
            "d_ddot": zeros[None],
            "s_dddot": zeros[None],
            "d_dddot": zeros[None],
            "vehicle": vehicle[None],
            "x": s[None],
            "y": d[None],
        }

    forward = (plan(35.0, 10.0), plan(45.0, 10.0), plan(15.0, 10.0))
    reverse = (plan(35.0, 10.0), plan(45.0, 10.0), plan(15.0, -2.0))
    forward_cost = batched_agent_costs_from_plans(forward, scenario, 0.1)[0]
    reverse_cost = batched_agent_costs_from_plans(reverse, scenario, 0.1)[0]

    assert reverse_cost[2] > forward_cost[2]


def test_speed_constraint_rejects_negative_frenet_progress_even_with_positive_vehicle_speed():
    from Cartest.planning.costs import three_agent_track_batched as batched_game_eval

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    horizon = 4
    upper_lane_d = scenario["behavior"]["upper_lane_d"]
    v_min = scenario["safety"]["v_min"]
    vehicle = jnp.zeros((1, horizon, 9)).at[..., 2].set(12.0)

    def plan(s_dot_value):
        s_dot = jnp.full((1, horizon), s_dot_value)
        return {
            "s": jnp.arange(horizon, dtype=jnp.float32)[None],
            "d": jnp.full((1, horizon), upper_lane_d),
            "s_dot": s_dot,
            "d_dot": jnp.zeros((1, horizon)),
            "s_ddot": jnp.zeros((1, horizon)),
            "d_ddot": jnp.zeros((1, horizon)),
            "s_dddot": jnp.zeros((1, horizon)),
            "d_dddot": jnp.zeros((1, horizon)),
            "vehicle": vehicle,
            "x": jnp.arange(horizon, dtype=jnp.float32)[None],
            "y": jnp.full((1, horizon), upper_lane_d),
        }

    plans = (plan(12.0), plan(12.0), plan(-1.0))
    speed = batched_game_eval._violations_for_agent(plans, scenario, 2)["speed"][0]

    assert jnp.max(speed) >= v_min + 1.0


def test_batched_pairwise_rss_accepts_minimal_s_dot_plan_dicts():
    from Cartest.planning.costs import three_agent_track_batched as batched_game_eval

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    horizon = 5
    own = {
        "s": jnp.linspace(10.0, 14.0, horizon)[None],
        "d": jnp.full((1, horizon), 3.5),
        "s_dot": jnp.full((1, horizon), 15.0),
    }
    neighbor = {
        "s": jnp.linspace(20.0, 24.0, horizon)[None],
        "d": jnp.full((1, horizon), 3.5),
        "s_dot": jnp.full((1, horizon), 12.0),
    }

    risk = batched_game_eval._rss_pairwise_bm(own, (neighbor,), scenario, dt=0.15)

    assert risk.shape == (1, 1)
    assert jnp.all(jnp.isfinite(risk))


def test_pairwise_objective_uses_bridged_jerk_context():
    from Cartest.planning.costs import three_agent_track_batched as batched_game_eval
    from Cartest.planning.costs.three_agent_track_components import role_soft_objective

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    horizon = 4
    dt = 0.1
    vehicle = jnp.zeros((1, horizon, 9), dtype=jnp.float32).at[..., 2].set(15.0)

    def plan(s0, d0, speed):
        s_dot = jnp.full((1, horizon), speed, dtype=jnp.float32)
        s = s0 + jnp.arange(horizon, dtype=jnp.float32)[None] * speed * dt
        d = jnp.full((1, horizon), d0, dtype=jnp.float32)
        return {
            "s": s,
            "d": d,
            "s_dot": s_dot,
            "d_dot": jnp.zeros((1, horizon), dtype=jnp.float32),
            "s_ddot": jnp.zeros((1, horizon), dtype=jnp.float32),
            "d_ddot": jnp.zeros((1, horizon), dtype=jnp.float32),
            "s_dddot": jnp.zeros((1, horizon), dtype=jnp.float32),
            "d_dddot": jnp.zeros((1, horizon), dtype=jnp.float32),
            "vehicle": vehicle,
        }

    candidate = (plan(0.0, 3.5, 15.0), plan(80.0, 3.5, 15.0), plan(-80.0, 3.5, 15.0))
    background = (plan(0.0, 3.5, 15.0), plan(80.0, 3.5, 15.0), plan(-80.0, 3.5, 15.0))
    ctx = {"a_long_prev_a0": 40.0, "a_lat_prev_a0": -20.0}

    def candidate_to_pairwise(p):
        return {k: v[:, None, ...] for k, v in p.items()}

    def background_to_pairwise(p):
        return {k: v[None, ...] for k, v in p.items()}

    pairwise_plans = (
        candidate_to_pairwise(candidate[0]),
        background_to_pairwise(background[1]),
        background_to_pairwise(background[2]),
    )
    expected = role_soft_objective(pairwise_plans, scenario, ctx, agent_idx=0, dt=dt)
    actual = batched_game_eval._objective_pairwise_bm(
        candidate, background, scenario, 0, dt, ctx=ctx)

    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_batched_nested_cost_matches_constran_scalar_specs():
    from Cartest.planning.costs.three_agent_track_batched import evaluate_joint_plan_batch, batched_nested_costs_from_plans
    from Constraintdealer.Constran import build_multi_agent

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, _ = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(2))
    joint = _joint_from_mu(mu)
    joint_batch = jnp.stack([joint, joint + 0.02], axis=0)

    specs = make_agent_specs(gen, scenario)
    scalar_fns = build_multi_agent(specs, k_inner=1.0, obj_transform="standard")
    scalar = jnp.stack([
        jnp.stack([scalar_fns[aid](aid, sample, ctx) for aid in range(3)])
        for sample in joint_batch
    ], axis=0)

    plans = evaluate_joint_plan_batch(gen, joint_batch, ctx, agent_count=3)
    batched = batched_nested_costs_from_plans(
        plans, scenario, gen.dt, k_inner=1.0, obj_transform="standard")

    assert batched.shape == (2, 3)
    assert jnp.all(jnp.isfinite(batched))
    assert jnp.allclose(batched, scalar, rtol=2e-4, atol=2e-3)


def _plan_dict(s, d, vehicle, zeros):
    """Build a scalar [T] plan dict in the component-module shape."""
    return {
        "s": s, "d": d, "s_dot": zeros, "d_dot": zeros, "s_ddot": zeros,
        "d_ddot": zeros, "s_dddot": zeros, "d_dddot": zeros, "vehicle": vehicle,
    }


def _batched_plan_dict(s, d, vehicle, zeros):
    """Build a batched [1, T] plan dict in the component-module shape."""
    return {k: v[None] for k, v in _plan_dict(s, d, vehicle, zeros).items()}


def _collision_fixture_plans():
    horizon = 5
    ego_x = jnp.array([100.0, 100.0, 100.0, 0.0, 100.0])
    front_x = jnp.array([10.0, 10.0, 10.0, 0.0, 10.0])
    rear_x = jnp.array([20.0, 20.0, 20.0, 0.0, 20.0])
    zeros = jnp.zeros(horizon)
    vehicle = jnp.zeros((horizon, 9))

    scalar = (
        _plan_dict(ego_x, zeros, vehicle, zeros),
        _plan_dict(front_x, zeros, vehicle, zeros),
        _plan_dict(rear_x, zeros, vehicle, zeros),
    )
    batched = (
        _batched_plan_dict(ego_x, zeros, vehicle, zeros),
        _batched_plan_dict(front_x, zeros, vehicle, zeros),
        _batched_plan_dict(rear_x, zeros, vehicle, zeros),
    )
    return scalar, batched


def test_three_agent_collision_uses_vehicle_footprint_not_point_distance():
    from Cartest.planning.costs import three_agent_track_components as components

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    horizon = 5
    zeros = jnp.zeros(horizon)
    vehicle = jnp.zeros((horizon, 9))
    ego_s = jnp.zeros(horizon)
    front_s = jnp.full(horizon, 4.5)
    ego_d = jnp.zeros(horizon)
    front_d = jnp.full(horizon, 1.5)
    far_s = jnp.full(horizon, 100.0)
    far_d = jnp.full(horizon, 3.5)

    # Both axis separations are inside the expanded physical-body envelope,
    # so the rectangular footprint model must fire.
    longitudinal, lateral = components.collision_clearances(scenario)
    assert front_s[0] - ego_s[0] < longitudinal
    assert front_d[0] - ego_d[0] < lateral

    scalar_plans = (
        _plan_dict(ego_s, ego_d, vehicle, zeros),
        _plan_dict(front_s, front_d, vehicle, zeros),
        _plan_dict(far_s, far_d, vehicle, zeros),
    )
    batched_plans = (
        _batched_plan_dict(ego_s, ego_d, vehicle, zeros),
        _batched_plan_dict(front_s, front_d, vehicle, zeros),
        _batched_plan_dict(far_s, far_d, vehicle, zeros),
    )

    scalar_ego = components.collision_violation_per_t(scalar_plans, scenario, 0)
    batched_ego = components.collision_violation_per_t(batched_plans, scenario, 0)[0]

    assert jnp.max(scalar_ego) > 0.0
    assert jnp.max(batched_ego) > 0.0


def test_three_agent_lane_bounds_use_vehicle_footprint_not_center_only():
    from Cartest.planning.costs import three_agent_track_components as components

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    horizon = 4
    zeros = jnp.zeros(horizon)
    vehicle = jnp.zeros((horizon, 9))
    ego_d = jnp.full(horizon, -1.2)
    safe_center = jnp.full(horizon, -0.2)

    lane_min, lane_max = components.lane_footprint_bounds(scenario)
    assert ego_d[0] < lane_min

    scalar_plans = (
        _plan_dict(zeros, ego_d, vehicle, zeros),
        _plan_dict(zeros, safe_center, vehicle, zeros),
        _plan_dict(zeros, safe_center, vehicle, zeros),
    )
    batched_plans = (
        _batched_plan_dict(zeros, ego_d, vehicle, zeros),
        _batched_plan_dict(zeros, safe_center, vehicle, zeros),
        _batched_plan_dict(zeros, safe_center, vehicle, zeros),
    )

    scalar_lane = components.lane_boundary_violation_per_t(scalar_plans[0], scenario)
    batched_lane = components.lane_boundary_violation_per_t(batched_plans[0], scenario)[0]

    assert jnp.max(scalar_lane) > 0.0
    assert jnp.max(batched_lane) > 0.0


def test_collision_prefix_includes_execute_index_for_scalar_and_batched_paths():
    from Cartest.planning.costs import three_agent_track_components as components

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"]["execute_index"] = 3
    scalar_plans, batched_plans = _collision_fixture_plans()

    scalar_front = components.collision_violation_per_t(scalar_plans, scenario, 1)
    scalar_rear = components.collision_violation_per_t(scalar_plans, scenario, 2)
    batched_front = components.collision_violation_per_t(batched_plans, scenario, 1)[0]
    batched_rear = components.collision_violation_per_t(batched_plans, scenario, 2)[0]

    for values in (scalar_front, scalar_rear, batched_front, batched_rear):
        assert values[3] > 0.0
        assert values[4] == 0.0


def test_scalar_and_batched_constraints_share_layer_metadata():
    from Cartest.planning.costs.three_agent_track import (
        THREE_AGENT_CONSTRAINT_DEFS,
        three_agent_batched_layers,
    )

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    scalar_specs = make_agent_specs(gen, scenario)[0][1]
    batched_layers = three_agent_batched_layers()

    assert [definition[0] for definition in THREE_AGENT_CONSTRAINT_DEFS] == [
        "speed", "kinematics", "forward", "safety_envelope",
    ]
    assert len(scalar_specs) == len(batched_layers)
    for definition, spec, layer in zip(
            THREE_AGENT_CONSTRAINT_DEFS, scalar_specs, batched_layers):
        name, mode, priority, aggregate, transform = definition
        layer_name, layer_aggregate, table, baseline, resolution = layer
        assert layer_name == name
        assert (spec.mode, spec.priority, spec.aggregate, spec.transform) == (
            mode, priority, aggregate, transform)
        assert layer_aggregate == spec.aggregate
        assert baseline == spec.baseline
        assert resolution == float(spec.get_transform_table()[0][0])
        assert jnp.array_equal(table[0], spec.get_transform_table()[0])
        assert jnp.array_equal(table[1], spec.get_transform_table()[1])


def test_three_agent_batched_constraint_aggregates_use_reduced_layers():
    from Cartest.planning.costs.three_agent_track_batched import (
        batched_constraint_violations_from_plans,
        evaluate_joint_plan_batch,
    )

    scenario = get_scenario("three_agent_track")
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, _ = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(2))
    joint_x = mu[:, 0].reshape(-1)
    plans = evaluate_joint_plan_batch(gen, joint_x[None], ctx, agent_count=3)
    violations = batched_constraint_violations_from_plans(plans, scenario)
    for agent_values in violations:
        assert tuple(agent_values.keys()) == ("speed", "kinematics", "forward", "safety_envelope")


def test_three_agent_batched_nested_cost_matches_scalar_specs():
    from Cartest.planning.costs.three_agent_track_batched import (
        batched_nested_costs_from_plans,
        evaluate_joint_plan_batch,
    )
    from Cartest.planning.costs.registry import make_agent_specs_from_scenario
    from Constraintdealer.Constran import build_multi_agent

    scenario = get_scenario("three_agent_track")
    small = dict(scenario)
    small["game"] = dict(scenario["game"], T=2, B=4, B0=2, M_inner=2)
    gen = FrenetBSplineTrajectory(BASIS, small["ref_path"])
    states = _states(small)
    ctx = build_multi_agent_context(states)
    mu, _ = build_multi_agent_warmstart(gen, small, states, jax.random.PRNGKey(3))
    joint_x = mu[:, 0].reshape(-1)
    plans = evaluate_joint_plan_batch(gen, joint_x[None], ctx, agent_count=3)
    batched_cost = batched_nested_costs_from_plans(plans, small, gen.dt)[0]
    scalar = build_multi_agent(make_agent_specs_from_scenario(gen, small), k_inner=1.0, obj_transform="standard")
    scalar_cost = jnp.array([scalar[aid](aid, joint_x, ctx) for aid in range(3)])
    assert jnp.all(jnp.isfinite(batched_cost))
    assert jnp.allclose(batched_cost, scalar_cost, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_t_alpha_uses_exact_small_table_lookup()
    test_batched_plan_eval_shapes_match_three_agents()
    test_batched_agent_cost_matches_scalar_cost_for_fixed_joint_samples()
    test_front_and_rear_objectives_penalize_upper_lane_center_offset()
    test_front_and_rear_objectives_penalize_reverse_longitudinal_motion()
    test_speed_constraint_rejects_negative_frenet_progress_even_with_positive_vehicle_speed()
    test_batched_pairwise_rss_accepts_minimal_s_dot_plan_dicts()
    test_pairwise_objective_uses_bridged_jerk_context()
    test_batched_nested_cost_matches_constran_scalar_specs()
    test_three_agent_collision_uses_vehicle_footprint_not_point_distance()
    test_three_agent_lane_bounds_use_vehicle_footprint_not_center_only()
    test_collision_prefix_includes_execute_index_for_scalar_and_batched_paths()
    test_scalar_and_batched_constraints_share_layer_metadata()
    test_three_agent_batched_constraint_aggregates_use_reduced_layers()
    test_three_agent_batched_nested_cost_matches_scalar_specs()
    print("batched three-agent eval tests ok")
