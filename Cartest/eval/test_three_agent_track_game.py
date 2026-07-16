"""Smoke test for the three-agent Trackgame-inspired Cartest scenario."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.execution.execute import FrenetState
from Cartest.planning.scenarios import get_scenario
from Cartest.planning.solver_modes import (
    build_cartest_nash_solver,
    build_multi_agent_context,
    build_multi_agent_warmstart,
    select_nash_plan,
)


def test_three_agent_track_one_small_rne_solve():
    scenario = get_scenario("three_agent_track")
    gen = FrenetBSplineTrajectory(
        ROOT / "Cartest" / "basis" / "bspline_basis.npz", scenario["ref_path"]
    )
    states = [
        FrenetState(
            s=a["s"], s_dot=a["s_dot"], s_ddot=a.get("s_ddot", 0.0),
            d=a["d"], d_dot=a.get("d_dot", 0.0), d_ddot=a.get("d_ddot", 0.0),
            psi=a.get("psi", 0.0),
        )
        for a in scenario["agents"]
    ]
    small = dict(scenario)
    small["game"] = dict(scenario["game"], T=2, B=4, B0=2, M_inner=1)

    solver = build_cartest_nash_solver(gen, small)
    ctx = build_multi_agent_context(states)
    mu, L_inv = build_multi_agent_warmstart(gen, small, states, jax.random.PRNGKey(0))
    result = solver(jax.random.PRNGKey(1), context=ctx,
                    initial_mu=mu, initial_S_or_L=L_inv)
    plans = select_nash_plan(result, small)

    assert len(plans) == 3
    assert result.per_agent_cost.shape == (3,)
    assert jnp.all(jnp.isfinite(result.per_agent_cost))


if __name__ == "__main__":
    test_three_agent_track_one_small_rne_solve()
    print("three-agent track game test ok")
