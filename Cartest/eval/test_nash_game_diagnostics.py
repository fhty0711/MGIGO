"""Tests for three-agent unilateral-deviation diagnostics."""

from __future__ import annotations

from pathlib import Path
import copy
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Cartest.eval.nash_game_diagnostics import (
    build_component_mean_diagnostic,
    make_empirical_best_response_solver,
    mode_residual,
    run_empirical_best_response,
    sample_frozen_opponent_blocks,
    select_nash_keyframes,
    summarize_best_response_restarts,
)
from Cartest.eval.three_agent_analysis import run_three_agent_analysis
from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.execution.execute import FrenetState
from Cartest.planning.scenarios import get_scenario
from Cartest.planning.solver_modes import (
    build_cartest_nash_solver,
    build_multi_agent_context,
    build_multi_agent_warmstart,
)
from Cartest.planning.costs.three_agent_track_batched import (
    batched_nested_costs_from_plans,
    evaluate_joint_plan_batch,
    expected_cost_for_agent_controls,
)


BASIS = ROOT / "Cartest" / "basis" / "bspline_basis.npz"


def test_mode_residual_clips_negative_roundoff():
    selected = jnp.array([1.0, 2.0, 3.0])
    best = jnp.array([1.0 + 1e-7, 1.5, 3.0])
    assert jnp.allclose(
        mode_residual(selected, best), jnp.array([0.0, 0.5, 0.0]))


def test_keyframes_are_unique_and_follow_rollout_events():
    states = np.zeros((9, 3, 6), dtype=float)
    states[:, 0, 1] = [0.0, 0.2, 0.8, 1.8, 2.8, 3.1, 3.4, 3.5, 3.5]
    rss = np.zeros((8, 3), dtype=float)
    rss[1] = [0.2, 0.4, 0.3]

    frames = select_nash_keyframes(
        states,
        rss,
        target_d=3.5,
        completion_tolerance=0.5,
        completion_hold=3,
    )

    assert frames["initial"] == 0
    assert frames["max_rss"] == 1
    # Reports are indexed by the MPC action that produced state_history[i].
    assert frames["lane_midpoint"] == 2
    assert frames["lane_complete"] == 4
    assert len(set(frames.values())) == len(frames.values())


def test_component_mean_diagnostic_returns_finite_nonnegative_residuals():
    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"] = dict(
        scenario["game"], T=2, B=4, B0=2, M_inner=2, T_0=2)
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = [
        FrenetState(
            s=agent["s"], s_dot=agent["s_dot"],
            s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"], d_dot=agent.get("d_dot", 0.0),
            d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]
    context = build_multi_agent_context(states)
    mu, L = build_multi_agent_warmstart(
        gen, scenario, states, jax.random.PRNGKey(1))
    result = build_cartest_nash_solver(gen, scenario)(
        jax.random.PRNGKey(2), context=context,
        initial_mu=mu, initial_S_or_L=L)
    diagnostic = build_component_mean_diagnostic(gen, scenario)

    best_cost, best_index = diagnostic(
        result.diag["mu"], result.diag["selected_blocks"], context)
    residual = mode_residual(result.per_agent_cost, best_cost)

    assert residual.shape == (3,)
    assert best_index.shape == (3,)
    assert jnp.all(jnp.isfinite(residual))
    assert jnp.all(residual >= 0.0)


def test_expected_cost_for_agent_controls_matches_explicit_joint_loop():
    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = [
        FrenetState(
            s=agent["s"], s_dot=agent["s_dot"],
            s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"], d_dot=agent.get("d_dot", 0.0),
            d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]
    context = build_multi_agent_context(states)
    mu, _ = build_multi_agent_warmstart(
        gen, scenario, states, jax.random.PRNGKey(3))
    selected = mu[:, 1]
    own = jnp.stack([
        selected[0:2],
        selected[0:2] + 0.02,
    ])
    frozen = jnp.stack([
        selected,
        selected.at[2].add(0.01),
        selected.at[4].add(-0.01),
    ])

    actual = expected_cost_for_agent_controls(
        gen, own, frozen, context, scenario, agent_idx=0)
    explicit = []
    for candidate in own:
        joint = frozen.at[:, 0:2].set(candidate)
        plans = evaluate_joint_plan_batch(
            gen, joint.reshape((joint.shape[0], -1)), context)
        explicit.append(batched_nested_costs_from_plans(
            plans, scenario, gen.dt, k_inner=0.1,
            obj_transform="standard", ctx=context)[:, 0].mean())

    assert jnp.allclose(actual, jnp.stack(explicit), rtol=1e-5, atol=1e-5)


def test_frozen_opponent_samples_have_expected_shape_and_mask_own_blocks():
    block_count, component_count, dim = 6, 3, 4
    mu = jnp.zeros((block_count, component_count, dim))
    precision_cholesky = jnp.broadcast_to(
        jnp.eye(dim), (block_count, component_count, dim, dim))
    pi = jnp.full((block_count, component_count), 1.0 / component_count)

    samples = sample_frozen_opponent_blocks(
        mu, precision_cholesky, pi, count=5,
        key=jax.random.PRNGKey(4), agent_idx=1)

    assert samples.shape == (5, 6, 4)
    assert jnp.all(jnp.isnan(samples[:, 2:4]))
    assert jnp.all(jnp.isfinite(samples[:, 0:2]))
    assert jnp.all(jnp.isfinite(samples[:, 4:6]))


def test_best_response_restart_summary_uses_lowest_cost_and_clips_residual():
    summary = summarize_best_response_restarts(
        selected_cost=4.0,
        restart_costs=jnp.array([3.5, 3.8, 4.0]),
    )
    assert summary["best_restart"] == 0
    assert np.isclose(summary["best_cost"], 3.5)
    assert np.isclose(summary["residual"], 0.5)

    roundoff = summarize_best_response_restarts(
        selected_cost=4.0,
        restart_costs=jnp.array([4.0 + 1e-7]),
    )
    assert roundoff["residual"] == 0.0


def test_empirical_best_response_replays_keys_and_returns_finite_plan():
    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"] = dict(
        scenario["game"], T=1, B=4, B0=2, M_inner=2, T_0=2)
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = [
        FrenetState(
            s=agent["s"], s_dot=agent["s_dot"],
            s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"], d_dot=agent.get("d_dot", 0.0),
            d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]
    context = build_multi_agent_context(states)
    mu, precision_cholesky = build_multi_agent_warmstart(
        gen, scenario, states, jax.random.PRNGKey(5))
    pi = jnp.full((6, 3), 1.0 / 3.0)
    selected = mu[:, 1]
    solver = make_empirical_best_response_solver(gen, scenario, agent_idx=0)
    kwargs = dict(
        gen=gen,
        scenario=scenario,
        agent_idx=0,
        point_equilibrium_cost=1.25,
        mu=mu,
        precision_cholesky=precision_cholesky,
        pi=pi,
        selected_blocks=selected,
        context=context,
        background_key=jax.random.PRNGKey(6),
        restart_keys=[jax.random.PRNGKey(7), jax.random.PRNGKey(8)],
        solver=solver,
    )

    first = run_empirical_best_response(**kwargs)
    replay = run_empirical_best_response(**kwargs)

    assert np.asarray(first["best_controls"]).shape == (2, gen.n_free)
    assert np.all(np.isfinite(first["best_controls"]))
    assert np.all(np.isfinite(first["best_response_xy"]))
    assert np.allclose(first["restart_costs"], replay["restart_costs"])
    assert np.isclose(
        first["equilibrium_expected_cost"],
        replay["equilibrium_expected_cost"])
    assert first["epsilon_br"] >= 0.0
    assert first["method"] == "empirical_distributional_best_response_2_restart"


def test_empirical_best_response_isolates_invalid_restart():
    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"] = dict(scenario["game"], M_inner=2)
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = [
        FrenetState(
            s=agent["s"], s_dot=agent["s_dot"],
            s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"], d_dot=agent.get("d_dot", 0.0),
            d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]
    context = build_multi_agent_context(states)
    mu, precision_cholesky = build_multi_agent_warmstart(
        gen, scenario, states, jax.random.PRNGKey(11))
    pi = jnp.full((6, 3), 1.0 / 3.0)
    selected = mu[:, 1]
    calls = {"count": 0}

    def flaky_solver(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(x=None)
        return SimpleNamespace(x=selected[0:2].reshape(-1))

    result = run_empirical_best_response(
        gen=gen,
        scenario=scenario,
        agent_idx=0,
        point_equilibrium_cost=1.0,
        mu=mu,
        precision_cholesky=precision_cholesky,
        pi=pi,
        selected_blocks=selected,
        context=context,
        background_key=jax.random.PRNGKey(12),
        restart_keys=[jax.random.PRNGKey(13), jax.random.PRNGKey(14)],
        solver=flaky_solver,
    )

    assert result["restart_status"] == ["invalid", "ok"]
    assert np.isclose(
        result["restart_costs"][0], result["equilibrium_expected_cost"])
    assert np.all(np.isfinite(result["best_controls"]))


def test_analysis_summary_separates_solver_and_diagnostic_timing(tmp_path):
    summary = run_three_agent_analysis(
        steps=2,
        T=1,
        seed=0,
        output_dir=tmp_path,
        render_video=False,
        run_best_response=False,
        save_keyframes=True,
        game_overrides={"B": 4, "B0": 2, "M_inner": 2, "T_0": 2},
    )

    assert summary["configuration"]["steps"] == 2
    assert len(summary["timing"]["solve_ms"]) == 2
    assert "diagnostic_ms" in summary["timing"]
    assert isinstance(summary["timing"]["solve_ms"], list)
    assert Path(summary["artifacts"]["json"]).exists()
    assert Path(summary["artifacts"]["npz"]).exists()
    assert Path(summary["artifacts"]["contact_sheet"]).exists()
    assert summary["artifacts"]["keyframes"]
