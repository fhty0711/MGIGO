"""Tests for three-agent batched B-spline game evaluation."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

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


if __name__ == "__main__":
    test_batched_plan_eval_shapes_match_three_agents()
    test_batched_agent_cost_matches_scalar_cost_for_fixed_joint_samples()
    print("batched three-agent eval tests ok")
