"""Tests for three-agent unilateral-deviation diagnostics."""

from __future__ import annotations

from pathlib import Path
import copy
import sys

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Cartest.eval.nash_game_diagnostics import (
    build_component_mean_diagnostic,
    mode_residual,
    select_nash_keyframes,
)
from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.execution.execute import FrenetState
from Cartest.planning.scenarios import get_scenario
from Cartest.planning.solver_modes import (
    build_cartest_nash_solver,
    build_multi_agent_context,
    build_multi_agent_warmstart,
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
