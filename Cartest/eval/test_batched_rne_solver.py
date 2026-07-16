"""Tests for Cartest batched RNE helper logic."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
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
from Cartest.planning.solver_modes import build_cartest_nash_solver, select_nash_plan
from Cartest.planning.costs.three_agent_track import make_agent_specs
from Constraintdealer.Constran import build_multi_agent


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


def _make_samples(gen, scenario):
    states = _states(scenario)
    mu, _ = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(4))
    base = mu[:, 0]
    B = 4
    M = 3
    samples_b = jnp.stack([base + 0.01 * i for i in range(B)], axis=1)
    samples_m = jnp.stack([base - 0.02 * i for i in range(M)], axis=1)
    return samples_b, samples_m


def test_batched_f_hat_matches_black_box_scalar_loop():
    from Cartest.planning.batched_game_eval import batched_expected_costs_for_all_agents

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    ctx = build_multi_agent_context(_states(scenario))
    samples_b, samples_m = _make_samples(gen, scenario)

    specs = make_agent_specs(gen, scenario)
    scalar_fns = build_multi_agent(specs, k_inner=1.0, obj_transform="standard")

    expected_by_agent = []
    for aid in range(3):
        block_mask = jnp.asarray(scenario["game"]["block_to_agent"]) == aid
        expected = []
        for b in range(samples_b.shape[1]):
            vals = []
            for m in range(samples_m.shape[1]):
                joint = jnp.where(block_mask[:, None], samples_b[:, b, :], samples_m[:, m, :]).reshape(-1)
                vals.append(scalar_fns[aid](aid, joint, ctx))
            expected.append(jnp.mean(jnp.stack(vals)))
        expected_by_agent.append(jnp.stack(expected))

    expected = jnp.stack(expected_by_agent)
    actual = batched_expected_costs_for_all_agents(gen, samples_b, samples_m, ctx, scenario)
    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected, rtol=2e-4, atol=2e-3)


def test_batched_f_hat_uses_shared_b_plus_m_trajectory_evaluations(monkeypatch=None):
    from Cartest.planning import batched_game_eval

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    ctx = build_multi_agent_context(_states(scenario))
    samples_b, samples_m = _make_samples(gen, scenario)

    counts = {"calls": 0}
    original = batched_game_eval.evaluate_agent_control_batch

    def counted(*args, **kwargs):
        counts["calls"] += 1
        return original(*args, **kwargs)

    batched_game_eval.evaluate_agent_control_batch = counted
    try:
        _ = batched_game_eval.batched_expected_costs_for_all_agents(gen, samples_b, samples_m, ctx, scenario)
    finally:
        batched_game_eval.evaluate_agent_control_batch = original

    # All three agents share 3 candidate batches + 3 background batches.
    assert counts["calls"] == 6


def test_cartest_batched_rne_solver_runs_small_problem():
    from Cartest.planning.batched_rne_solver import make_cartest_batched_rne_blocks_solver

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"] = dict(scenario["game"], T=2, B=4, B0=2, M_inner=3, K=2)
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, L_inv = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(5))
    mu = mu[:, :2]
    L_inv = L_inv[:, :2]

    solver = make_cartest_batched_rne_blocks_solver(gen, scenario)
    result = solver(jax.random.PRNGKey(6), context=ctx, initial_mu=mu, initial_L_inv=L_inv)

    assert result["mu"].shape == mu.shape
    assert result["L_inv"].shape == L_inv.shape
    assert result["pi"].shape == (6, 2)
    assert jnp.all(jnp.isfinite(result["mu"]))


def test_fast_sampling_avoids_per_sample_covariance_inverse():
    from Cartest.planning import batched_rne_solver

    source = inspect.getsource(batched_rne_solver._sample_all_blocks_fast)
    assert "jnp.linalg.inv" not in source
    assert "multivariate_normal" not in source


def test_tie_aware_elite_weights_match_rank_weights_without_ties():
    from Cartest.planning.batched_rne_solver import _tie_aware_elite_weights

    actual = _tie_aware_elite_weights(jnp.array([0.0, 1.0, 2.0, 3.0]), 2)
    assert jnp.allclose(actual, jnp.array([0.25, 0.25, 0.0, 0.0]))


def test_tie_aware_elite_weights_split_boundary_ties():
    from Cartest.planning.batched_rne_solver import _tie_aware_elite_weights

    actual = _tie_aware_elite_weights(jnp.array([0.0, 1.0, 1.0, 3.0]), 2)
    assert jnp.allclose(actual, jnp.array([0.25, 0.125, 0.125, 0.0]))


def test_tie_aware_elite_weights_split_all_equal_costs_and_preserve_mass():
    from Cartest.planning.batched_rne_solver import _tie_aware_elite_weights

    costs = jnp.array([[1.0, 1.0, 1.0, 1.0],
                       [0.0, 2.0, 2.0, 4.0]])
    actual = _tie_aware_elite_weights(costs, 2)
    expected = jnp.array([[0.125, 0.125, 0.125, 0.125],
                          [0.25, 0.125, 0.125, 0.0]])
    assert jnp.allclose(actual, expected)
    assert jnp.allclose(jnp.sum(actual, axis=-1), jnp.array([0.5, 0.5]))


def test_mixture_weights_do_not_reset_at_iteration_zero():
    from Cartest.planning.batched_rne_solver import _should_reset_mixture_weights

    assert not bool(_should_reset_mixture_weights(0, 3))
    assert not bool(_should_reset_mixture_weights(2, 3))
    assert bool(_should_reset_mixture_weights(3, 3))


def test_batched_rne_exposes_reusable_solver_factory():
    from Cartest.planning import batched_rne_solver

    assert hasattr(batched_rne_solver, "make_cartest_batched_rne_blocks_solver")
    assert not hasattr(batched_rne_solver, "cartest_batched_rne_blocks_solver")


def test_batched_postprocess_is_jitted_and_selects_joint_plan():
    from Cartest.planning import solver_modes

    assert hasattr(solver_modes, "_build_cartest_batched_postprocess")

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, _ = build_multi_agent_warmstart(
        gen, scenario, states, jax.random.PRNGKey(9))
    pi = jnp.tile(jnp.asarray([[0.1, 0.8, 0.1]]), (mu.shape[0], 1))

    postprocess = solver_modes._build_cartest_batched_postprocess(gen, scenario)
    selected, best, joint_x, costs = postprocess(mu, pi, ctx)

    assert hasattr(postprocess, "lower")
    assert selected.shape == (mu.shape[0], gen.n_free)
    assert jnp.array_equal(best, jnp.ones((mu.shape[0],), dtype=best.dtype))
    assert jnp.allclose(selected, mu[:, 1])
    assert joint_x.shape == (mu.shape[0] * gen.n_free,)
    assert costs.shape == (len(scenario["agents"]),)
    assert jnp.all(jnp.isfinite(costs))


def test_select_nash_plan_reuses_preselected_blocks_without_argmax():
    selected = jnp.arange(18, dtype=jnp.float32).reshape(6, 3)
    best = jnp.asarray([2, 1, 0, 2, 0, 1])
    result = SimpleNamespace(diag={
        "mu": jnp.zeros((6, 3, 3)),
        "pi": jnp.full((6, 3), 1.0 / 3.0),
        "selected_blocks": selected,
        "best_components": best,
    })
    scenario = {
        "agents": ({}, {}, {}),
        "game": {"block_to_agent": (0, 0, 1, 1, 2, 2)},
    }

    with patch.object(jnp, "argmax", side_effect=AssertionError("unexpected sync")):
        plans = select_nash_plan(result, scenario)

    assert jnp.array_equal(plans[1]["ctrl_s"], selected[2])
    assert jnp.array_equal(plans[1]["ctrl_d"], selected[3])
    assert plans[1]["best_components"] == (best[2], best[3])


def test_cartest_batched_solver_matches_generic_rne_blocks_small_problem():
    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"] = dict(scenario["game"], T=2, B=4, B0=2, M_inner=2, K=2)
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, L_inv = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(7))
    mu = mu[:, :2]
    L_inv = L_inv[:, :2]

    generic_scenario = copy.deepcopy(scenario)
    generic_scenario["game"] = dict(scenario["game"], solver="rne_blocks")
    batched_scenario = copy.deepcopy(scenario)
    batched_scenario["game"] = dict(scenario["game"], solver="cartest_batched_rne_blocks")

    generic = build_cartest_nash_solver(gen, generic_scenario)(
        jax.random.PRNGKey(8), context=ctx, initial_mu=mu, initial_S_or_L=L_inv)
    batched = build_cartest_nash_solver(gen, batched_scenario)(
        jax.random.PRNGKey(8), context=ctx, initial_mu=mu, initial_S_or_L=L_inv)

    assert jnp.allclose(batched.diag["mu"], generic.diag["mu"], rtol=2e-4, atol=2e-3)
    assert jnp.allclose(batched.diag["pi"], generic.diag["pi"], rtol=2e-4, atol=2e-3)
    assert jnp.allclose(batched.per_agent_cost, generic.per_agent_cost, rtol=2e-4, atol=2e-3)


if __name__ == "__main__":
    test_batched_f_hat_matches_black_box_scalar_loop()
    test_batched_f_hat_uses_shared_b_plus_m_trajectory_evaluations()
    test_cartest_batched_rne_solver_runs_small_problem()
    test_fast_sampling_avoids_per_sample_covariance_inverse()
    test_tie_aware_elite_weights_match_rank_weights_without_ties()
    test_tie_aware_elite_weights_split_boundary_ties()
    test_tie_aware_elite_weights_split_all_equal_costs_and_preserve_mass()
    test_mixture_weights_do_not_reset_at_iteration_zero()
    test_batched_rne_exposes_reusable_solver_factory()
    test_batched_postprocess_is_jitted_and_selects_joint_plan()
    test_select_nash_plan_reuses_preselected_blocks_without_argmax()
    test_cartest_batched_solver_matches_generic_rne_blocks_small_problem()
    print("batched rne helper tests ok")
