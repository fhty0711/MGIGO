"""Smoke tests for Cartest multi-agent solver mode helpers."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
from unittest.mock import patch

from gmm_igo.solver_builder import build_nash_solver
from Constraintdealer.Constran import build as build_constran_cost


def test_nash_solver_exposes_block_level_diagnostics():
    def objective_a0(joint_x, ctx):
        del ctx
        return (joint_x[0] - 1.0) ** 2 + 0.1 * joint_x[1] ** 2

    def objective_a1(joint_x, ctx):
        del ctx
        return (joint_x[2] + 1.0) ** 2 + 0.1 * joint_x[3] ** 2

    solver = build_nash_solver(
        {0: (objective_a0, []), 1: (objective_a1, [])},
        dims=(1, 1, 1, 1),
        solver="rne_blocks",
        block_to_agent=(0, 0, 1, 1),
        T=2,
        dt=0.05,
        K=2,
        B=4,
        B0=2,
        T_0=2,
        M_inner=1,
    )

    result = solver(jax.random.PRNGKey(0), context={})

    assert result.joint_x.shape == (4,)
    assert result.per_agent_cost.shape == (2,)
    assert result.diag["mu"].shape == (4, 2, 1)
    assert result.diag["S_or_L"].shape == (4, 2, 1, 1)
    assert result.diag["pi"].shape == (4, 2)
    assert result.diag["v"].shape == (4, 1)
    expected_v = jnp.log(result.diag["pi"][:, :-1] / (result.diag["pi"][:, -1:] + 1e-10))
    assert jnp.allclose(result.diag["v"], expected_v)
    assert tuple(result.diag["block_to_agent"]) == (0, 0, 1, 1)


if __name__ == "__main__":
    test_nash_solver_exposes_block_level_diagnostics()
    print("nash block diagnostics test ok")


from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.planning.costs.registry import make_agent_specs_from_scenario
from Cartest.planning.scenarios import get_scenario

BASIS = ROOT / "Cartest" / "basis" / "bspline_basis.npz"


def _joint_zero(gen, scenario):
    return jnp.zeros((len(scenario["game"]["block_layout"]) * gen.n_free,), dtype=jnp.float32)


def _ctx_from_agents(scenario):
    ctx = {}
    for idx, agent in enumerate(scenario["agents"]):
        ctx[f"s0_a{idx}"] = float(agent["s"])
        ctx[f"s_dot0_a{idx}"] = float(agent["s_dot"])
        ctx[f"s_ddot0_a{idx}"] = float(agent.get("s_ddot", 0.0))
        ctx[f"d0_a{idx}"] = float(agent["d"])
        ctx[f"d_dot0_a{idx}"] = float(agent.get("d_dot", 0.0))
        ctx[f"d_ddot0_a{idx}"] = float(agent.get("d_ddot", 0.0))
    return ctx


def test_game_agent_specs_return_finite_costs():
    for name in ("game_2a_basic", "game_2b_constran", "game_2b_asymmetric", "three_agent_track"):
        scenario = get_scenario(name)
        gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
        specs = make_agent_specs_from_scenario(gen, scenario)
        joint_x = _joint_zero(gen, scenario)
        ctx = _ctx_from_agents(scenario)

        assert sorted(specs) == list(range(len(scenario["agents"])))
        for aid, (objective, constraints) in specs.items():
            value = objective(joint_x, ctx)
            assert jnp.isfinite(value), f"{name} agent {aid} cost not finite"
            assert constraints == [] or isinstance(constraints, list)


def test_game_2b_constran_keeps_full_constraint_stack():
    scenario = get_scenario("game_2b_constran")
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    specs = make_agent_specs_from_scenario(gen, scenario)

    for _aid, (_objective, constraints) in specs.items():
        assert len(constraints) == 5
        assert [(c.mode, c.priority, c.aggregate) for c in constraints] == [
            ("soft", 1, "q95"),
            ("soft", 2, "max"),
            ("soft", 3, "max"),
            ("soft", 4, "max"),
            ("hard", 5, "max"),
        ]


def test_three_agent_track_uses_constran_constraints_and_valid_scene():
    scenario = get_scenario("three_agent_track")
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    specs = make_agent_specs_from_scenario(gen, scenario)

    lane_min, lane_max = scenario["road"]["lane_bounds_d"]
    for agent in scenario["agents"]:
        assert lane_min <= agent["d"] <= lane_max
        assert scenario["safety"]["v_min"] <= agent["s_dot"] <= scenario["safety"]["v_max"]

    for _aid, (_objective, constraints) in specs.items():
        assert len(constraints) == 4
        assert [(c.mode, c.priority, c.aggregate) for c in constraints] == [
            ("hard", 3, "max"),
            ("hard", 4, "max"),
            ("hard", 5, "max"),
            ("hard", 6, "max"),
        ]

    joint_x = _joint_zero(gen, scenario)
    ctx = _ctx_from_agents(scenario)
    for _aid, (_objective, constraints) in specs.items():
        safety_values = constraints[-1].g_fn(joint_x, ctx)
        assert jnp.all(jnp.isfinite(safety_values))


def test_three_agent_track_cost_prepares_joint_plans_once_per_agent_cost():
    scenario = get_scenario("three_agent_track")
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    joint_x = _joint_zero(gen, scenario)
    ctx = _ctx_from_agents(scenario)
    calls = []

    from Cartest.planning.costs import three_agent_track as cost_module

    real_eval = cost_module.eval_joint_plans


    def counted_eval(*args, **kwargs):
        calls.append(1)
        return real_eval(*args, **kwargs)

    with patch.object(cost_module, "eval_joint_plans", side_effect=counted_eval):
        specs = make_agent_specs_from_scenario(gen, scenario)
        objective, constraints = specs[0]
        cost_fn = build_constran_cost(objective, constraints, jit_cost=False)
        value = cost_fn(joint_x, ctx)

    assert jnp.isfinite(value)
    assert len(calls) == 1


def test_three_agent_track_uses_cartest_batched_rne_solver_mode():
    scenario = get_scenario("three_agent_track")
    assert scenario["game"]["solver"] == "cartest_batched_rne_blocks"


def test_cartest_batched_initial_pi_converts_to_natural_parameters():
    from Cartest.planning.solver_modes import _pi_to_v
    from Cartest.planning.solvers.batched_rne_solver import _v_to_pi

    pi = jnp.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    recovered = jax.vmap(_v_to_pi)(_pi_to_v(pi))
    assert jnp.allclose(recovered, pi)


def test_cartest_batched_initial_pi_rejects_invalid_probabilities():
    from Cartest.planning.solver_modes import _pi_to_v

    invalid = (
        jnp.array([[0.0, 0.5, 0.5]]),
        jnp.array([[-0.1, 0.6, 0.5]]),
        jnp.array([[jnp.inf, 0.4, 0.6]]),
        jnp.array([[0.2, 0.2, 0.2]]),
    )
    for pi in invalid:
        try:
            _pi_to_v(pi)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid initial_pi accepted: {pi}")


def test_cartest_batched_explicit_initial_pi_overrides_warm_start_v():
    from Cartest.planning.costs import three_agent_track_batched as batched_game_eval
    from Cartest.planning.solvers import batched_rne_solver
    from Cartest.planning.solver_modes import _build_cartest_batched_solver, _pi_to_v

    captured = {}
    blocks, components, dim = 6, 2, 1

    def fake_factory(*_args, **_kwargs):
        def fake_solver(_key, *, context, initial_mu, initial_L_inv, initial_v):
            del context
            captured["initial_v"] = initial_v
            return {
                "mu": initial_mu,
                "L_inv": initial_L_inv,
                "pi": jnp.full((blocks, components), 0.5),
                "v": initial_v,
                "metrics": {},
            }
        return fake_solver

    class FakeGen:
        n_free = dim

    scenario = {
        "agents": ({}, {}, {}),
        "game": {"block_to_agent": (0, 0, 1, 1, 2, 2)},
    }
    mu = jnp.zeros((blocks, components, dim))
    L_inv = jnp.ones((blocks, components, dim, dim))
    initial_pi = jnp.tile(jnp.array([[0.8, 0.2]]), (blocks, 1))
    warm_v = jnp.full((blocks, components - 1), -9.0)

    with patch.object(batched_rne_solver, "make_cartest_batched_rne_blocks_solver",
                      side_effect=fake_factory), \
         patch.object(batched_game_eval, "evaluate_joint_plan_batch", return_value=None), \
         patch.object(batched_game_eval, "batched_nested_costs_from_plans",
                      return_value=jnp.zeros((1, 3))):
        solver = _build_cartest_batched_solver(FakeGen(), scenario, (dim,) * blocks)
        solver(jax.random.PRNGKey(0), context={}, initial_mu=mu,
               initial_S_or_L=L_inv, initial_pi=initial_pi,
               warm_start={"v": warm_v})

    assert jnp.allclose(captured["initial_v"], _pi_to_v(initial_pi))


if __name__ == "__main__":
    test_nash_solver_exposes_block_level_diagnostics()
    test_game_agent_specs_return_finite_costs()
    test_game_2b_constran_keeps_full_constraint_stack()
    test_three_agent_track_uses_constran_constraints_and_valid_scene()
    test_three_agent_track_cost_prepares_joint_plans_once_per_agent_cost()
    test_three_agent_track_uses_cartest_batched_rne_solver_mode()
    test_cartest_batched_initial_pi_converts_to_natural_parameters()
    test_cartest_batched_initial_pi_rejects_invalid_probabilities()
    test_cartest_batched_explicit_initial_pi_overrides_warm_start_v()
    print("game solver modes tests ok")


from Cartest.execution.execute import FrenetState
from Cartest.planning.solver_modes import (
    build_cartest_nash_solver,
    build_multi_agent_context,
    build_multi_agent_warmstart,
    select_nash_plan,
)


def test_cartest_nash_solver_helpers_match_scenario_shapes():
    scenario = get_scenario("game_2a_basic")
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = [
        FrenetState(s=a["s"], s_dot=a["s_dot"], s_ddot=a.get("s_ddot", 0.0),
                    d=a["d"], d_dot=a.get("d_dot", 0.0), d_ddot=a.get("d_ddot", 0.0),
                    psi=a.get("psi", 0.0))
        for a in scenario["agents"]
    ]
    ctx = build_multi_agent_context(states)
    mu, L_inv = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(1))

    assert ctx["s0_a0"] == scenario["agents"][0]["s"]
    assert mu.shape == (4, scenario["game"]["K"], gen.n_free)
    assert L_inv.shape == (4, scenario["game"]["K"], gen.n_free, gen.n_free)

    small = dict(scenario)
    small["game"] = dict(scenario["game"], T=2, B=4, B0=2, M_inner=1)
    solver = build_cartest_nash_solver(gen, small)
    result = solver(jax.random.PRNGKey(2), context=ctx, initial_mu=mu, initial_S_or_L=L_inv)
    plans = select_nash_plan(result, small)

    assert len(plans) == 2
    assert plans[0]["ctrl_s"].shape == (gen.n_free,)
    assert plans[0]["ctrl_d"].shape == (gen.n_free,)
    assert plans[1]["ctrl_s"].shape == (gen.n_free,)
    assert plans[1]["ctrl_d"].shape == (gen.n_free,)


def _states(scenario):
    return [
        FrenetState(
            s=agent["s"], s_dot=agent["s_dot"], s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"], d_dot=agent.get("d_dot", 0.0), d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]


def test_three_agent_track_constraint_layers_are_reduced_and_ordered():
    from Cartest.planning.costs.three_agent_track import three_agent_batched_layers

    layers = three_agent_batched_layers()
    names = tuple(layer[0] for layer in layers)
    assert names == ("speed", "kinematics", "forward", "safety_envelope")


def test_three_agent_track_specs_return_finite_component_costs():
    scenario = get_scenario("three_agent_track")
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, _ = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(1))
    joint_x = mu[:, 0].reshape(-1)
    specs = make_agent_specs_from_scenario(gen, scenario)
    for aid, (obj, constraints) in specs.items():
        value = obj(joint_x, ctx)
        assert jnp.isfinite(value), f"objective agent {aid} not finite"
        prepared = getattr(obj, "_constran_prepare", lambda x, c: None)(joint_x, ctx)
        cctx = (ctx, prepared) if prepared is not None else ctx
        for constraint in constraints:
            g = constraint.g_fn(joint_x, cctx)
            assert jnp.all(jnp.isfinite(g)), f"constraint agent {aid} not finite"
