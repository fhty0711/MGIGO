"""Benchmark generic vs Cartest batched three-agent RNE on GPU."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import time

import jax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.execution.execute import FrenetState
from Cartest.planning.scenarios import get_scenario
from Cartest.planning.solver_modes import build_cartest_nash_solver, build_multi_agent_context, build_multi_agent_warmstart


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


def _block(result):
    leaves = jax.tree_util.tree_leaves(result)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def run_one(label, solver_name):
    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"] = dict(scenario["game"], solver=solver_name)
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, L_inv = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(10))
    solver = build_cartest_nash_solver(gen, scenario)

    t0 = time.perf_counter()
    r0 = solver(jax.random.PRNGKey(11), context=ctx, initial_mu=mu, initial_S_or_L=L_inv)
    _block(r0)
    compile_plus_first = time.perf_counter() - t0

    t1 = time.perf_counter()
    r1 = solver(jax.random.PRNGKey(12), context=ctx, initial_mu=mu, initial_S_or_L=L_inv)
    _block(r1)
    steady = time.perf_counter() - t1
    print(f"{label}: compile+first={compile_plus_first:.3f}s steady={steady:.3f}s backend={jax.default_backend()}")


def main():
    print("devices:", jax.devices())
    run_one("generic_rne_blocks", "rne_blocks")
    run_one("cartest_batched_rne_blocks", "cartest_batched_rne_blocks")


if __name__ == "__main__":
    main()
