"""Tests for Cartest batched RNE helper logic."""

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
    from Cartest.planning.batched_game_eval import batched_expected_cost_for_agent

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    ctx = build_multi_agent_context(_states(scenario))
    samples_b, samples_m = _make_samples(gen, scenario)

    specs = make_agent_specs(gen, scenario)
    scalar_fns = build_multi_agent(specs, k_inner=1.0, obj_transform="standard")

    for aid in range(3):
        block_mask = jnp.asarray(scenario["game"]["block_to_agent"]) == aid
        expected = []
        for b in range(samples_b.shape[1]):
            vals = []
            for m in range(samples_m.shape[1]):
                joint = jnp.where(block_mask[:, None], samples_b[:, b, :], samples_m[:, m, :]).reshape(-1)
                vals.append(scalar_fns[aid](aid, joint, ctx))
            expected.append(jnp.mean(jnp.stack(vals)))
        expected = jnp.stack(expected)

        actual = batched_expected_cost_for_agent(gen, samples_b, samples_m, ctx, scenario, aid)
        assert actual.shape == expected.shape
        assert jnp.allclose(actual, expected, rtol=2e-4, atol=2e-3)


def test_batched_f_hat_uses_b_plus_m_trajectory_evaluations(monkeypatch=None):
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
        _ = batched_game_eval.batched_expected_cost_for_agent(gen, samples_b, samples_m, ctx, scenario, 0)
    finally:
        batched_game_eval.evaluate_agent_control_batch = original

    # Agent 0 needs: ego B candidates + front M backgrounds + rear M backgrounds.
    assert counts["calls"] == 3


if __name__ == "__main__":
    test_batched_f_hat_matches_black_box_scalar_loop()
    test_batched_f_hat_uses_b_plus_m_trajectory_evaluations()
    print("batched rne helper tests ok")
