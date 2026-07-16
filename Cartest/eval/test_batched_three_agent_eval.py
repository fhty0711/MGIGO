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
    from Cartest.planning.batched_game_eval import evaluate_joint_plan_batch

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
    from Cartest.planning.batched_game_eval import evaluate_joint_plan_batch, batched_agent_costs_from_plans

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
    batched = batched_agent_costs_from_plans(plans, scenario)

    assert batched.shape == (2, 3)
    assert jnp.all(jnp.isfinite(batched))
    assert jnp.allclose(batched, scalar, rtol=2e-4, atol=2e-3)


def test_batched_nested_cost_matches_constran_scalar_specs():
    from Cartest.planning.batched_game_eval import evaluate_joint_plan_batch, batched_nested_costs_from_plans
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
    batched = batched_nested_costs_from_plans(plans, scenario, k_inner=1.0, obj_transform="standard")

    assert batched.shape == (2, 3)
    assert jnp.all(jnp.isfinite(batched))
    assert jnp.allclose(batched, scalar, rtol=2e-4, atol=2e-3)


def _collision_fixture_plans():
    horizon = 5
    ego_x = jnp.array([100.0, 100.0, 100.0, 0.0, 100.0])
    front_x = jnp.array([10.0, 10.0, 10.0, 0.0, 10.0])
    rear_x = jnp.array([20.0, 20.0, 20.0, 0.0, 20.0])
    zeros = jnp.zeros(horizon)
    vehicle = jnp.zeros((horizon, 9))

    def scalar_plan(s, x):
        frenet = (s, zeros, zeros, zeros, zeros, zeros, zeros, zeros)
        return frenet, vehicle, (x, zeros)

    scalar = (
        scalar_plan(jnp.zeros(horizon), ego_x),
        scalar_plan(jnp.full(horizon, 100.0), front_x),
        scalar_plan(jnp.zeros(horizon), rear_x),
    )

    def batched_plan(s, x):
        return {
            "s": s[None], "d": zeros[None], "vehicle": vehicle[None],
            "x": x[None], "y": zeros[None],
        }

    batched = (
        batched_plan(jnp.zeros(horizon), ego_x),
        batched_plan(jnp.full(horizon, 100.0), front_x),
        batched_plan(jnp.zeros(horizon), rear_x),
    )
    return scalar, batched


def test_collision_prefix_includes_execute_index_for_scalar_and_batched_paths():
    from Cartest.planning import batched_game_eval
    from Cartest.planning.costs import three_agent_track as scalar_cost

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"]["execute_index"] = 3
    scalar_plans, batched_plans = _collision_fixture_plans()

    class FakeGen:
        T = 5

    specs = scalar_cost.make_agent_specs(FakeGen(), scenario)
    with patch.object(scalar_cost, "_prepared_plans", return_value=scalar_plans):
        scalar_front = specs[1][1][-1].g_fn(jnp.zeros(1), {})
        scalar_rear = specs[2][1][-1].g_fn(jnp.zeros(1), {})

    batched_front = batched_game_eval._violations_for_agent(
        batched_plans, scenario, 1)["collision"][0]
    batched_rear = batched_game_eval._violations_for_agent(
        batched_plans, scenario, 2)["collision"][0]

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
        "lane", "speed", "acc", "jerk", "collision",
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


if __name__ == "__main__":
    test_t_alpha_uses_exact_small_table_lookup()
    test_batched_plan_eval_shapes_match_three_agents()
    test_batched_agent_cost_matches_scalar_cost_for_fixed_joint_samples()
    test_batched_nested_cost_matches_constran_scalar_specs()
    test_collision_prefix_includes_execute_index_for_scalar_and_batched_paths()
    test_scalar_and_batched_constraints_share_layer_metadata()
    print("batched three-agent eval tests ok")
