"""Unilateral-deviation diagnostics for the three-agent RNE game."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from Cartest.planning.costs.three_agent_track_batched import (
    batched_nested_costs_from_plans,
    evaluate_agent_control_batch,
    evaluate_joint_plan_batch,
    expected_cost_for_agent_controls,
)
from Cartest.planning.solvers.batched_rne_solver import _sample_all_blocks
from gmm_igo.solver_builder import build_solver


def mode_residual(selected_cost, best_cost):
    """Non-negative improvement available over restricted component means."""
    return jnp.maximum(
        jnp.asarray(selected_cost) - jnp.asarray(best_cost), 0.0)


def sample_frozen_opponent_blocks(
        mu, precision_cholesky, pi, count, key, agent_idx):
    """Sample joint blocks from the final GMM and mask the acting agent.

    Solver diagnostics store a Cholesky factor ``L`` of each precision matrix.
    The compatibility sampler accepts the precision itself, so reconstructing
    ``L @ L.T`` here preserves the same component and Gaussian sample stream.
    """
    precision = jnp.matmul(
        precision_cholesky,
        jnp.swapaxes(precision_cholesky, -1, -2),
    )
    samples = jnp.swapaxes(
        _sample_all_blocks(mu, precision, pi, count, key), 0, 1)
    first_block = 2 * agent_idx
    return samples.at[:, first_block:first_block + 2].set(jnp.nan)


def summarize_best_response_restarts(selected_cost, restart_costs):
    """Summarize finite-sample best-response restarts for one agent."""
    costs = np.asarray(restart_costs, dtype=float)
    best_restart = int(np.argmin(costs))
    best_cost = float(costs[best_restart])
    residual = float(max(float(selected_cost) - best_cost, 0.0))
    relative = residual / max(abs(float(selected_cost)), 1e-6)
    return {
        "selected_cost": float(selected_cost),
        "restart_costs": costs.tolist(),
        "best_restart": best_restart,
        "best_cost": best_cost,
        "residual": residual,
        "best_response_cost": best_cost,
        "epsilon_br": residual,
        "epsilon_br_relative": relative,
    }


def make_empirical_best_response_solver(gen, scenario, agent_idx):
    """Build a two-block solver whose opponent samples arrive via context.

    Keeping the samples in the context rather than in the Python closure lets
    keyframes with the same sample shape reuse one compiled objective signature.
    """
    def objective(x, ctx):
        own = x.reshape((1, 2, gen.n_free))
        cost = expected_cost_for_agent_controls(
            gen,
            own,
            ctx["br_frozen_joint_controls"],
            ctx,
            scenario,
            agent_idx,
        )[0]
        # M22 explores an unbounded Gaussian support.  Extremely remote
        # B-spline controls can overflow intermediate trajectory/cost terms;
        # keep the optimizer's ranking function total in that region.
        return jnp.nan_to_num(
            cost, nan=1e6, posinf=1e6, neginf=1e6)

    game = scenario["game"]
    return build_solver(
        objective,
        dims=(gen.n_free, gen.n_free),
        solver="m22",
        T=int(game.get("T", 300)),
        dt=float(game.get("dt", 0.15)),
        K=int(game.get("K", 3)),
        B=int(game.get("B", 60)),
        B0=int(game.get("B0", 25)),
        T_0=int(game.get("T_0", 300)),
    )


def _key_as_list(key):
    return np.asarray(key, dtype=np.uint32).tolist()


def run_empirical_best_response(
        gen, scenario, agent_idx, point_equilibrium_cost, mu,
        precision_cholesky, pi, selected_blocks, context, background_key,
        restart_keys, solver=None):
    """Run deterministic empirical best responses against one frozen sample set.

    The returned ``epsilon_br`` compares two expectations over the identical
    opponent samples.  ``point_equilibrium_cost`` is retained separately and
    is never mixed into that residual.
    """
    opponent_count = int(scenario["game"].get("M_inner", 30))
    frozen = sample_frozen_opponent_blocks(
        mu,
        precision_cholesky,
        pi,
        opponent_count,
        background_key,
        agent_idx,
    )
    br_context = dict(context)
    br_context["br_frozen_joint_controls"] = frozen
    first_block = 2 * agent_idx
    equilibrium_controls = jnp.asarray(
        selected_blocks[first_block:first_block + 2])
    equilibrium_expected = expected_cost_for_agent_controls(
        gen,
        equilibrium_controls[None],
        frozen,
        br_context,
        scenario,
        agent_idx,
    )[0]
    equilibrium_expected = float(jax.device_get(equilibrium_expected))

    if solver is None:
        solver = make_empirical_best_response_solver(gen, scenario, agent_idx)

    agent_mu = jnp.asarray(mu[first_block:first_block + 2])
    agent_l = jnp.asarray(
        precision_cholesky[first_block:first_block + 2])
    agent_pi = jnp.asarray(pi[first_block:first_block + 2])
    restart_costs = []
    restart_controls = []
    restart_runtime_ms = []
    restart_status = []

    for restart_index, restart_key in enumerate(restart_keys):
        if restart_index == 0:
            initial_mu = agent_mu
        else:
            perturbation = 0.2 * jax.random.normal(
                restart_key, agent_mu.shape, dtype=agent_mu.dtype)
            initial_mu = agent_mu + perturbation
        started = time.perf_counter_ns()
        result = solver(
            restart_key,
            context=br_context,
            initial_mu=initial_mu,
            initial_S_or_L=agent_l,
            initial_pi=agent_pi,
        )
        raw_controls = result.x
        valid = raw_controls is not None
        if valid:
            controls = jnp.asarray(raw_controls).reshape((2, gen.n_free))
            controls.block_until_ready()
            valid = bool(np.all(np.isfinite(
                np.asarray(jax.device_get(controls)))))
        if not valid:
            # A failed restart is conservatively equivalent to "no deviation".
            # This preserves the finite equilibrium baseline and prevents one
            # unstable multistart branch from fabricating an improvement or
            # discarding the other valid restarts.
            controls = equilibrium_controls
        runtime_ms = (time.perf_counter_ns() - started) / 1e6
        cost = expected_cost_for_agent_controls(
            gen,
            controls[None],
            frozen,
            br_context,
            scenario,
            agent_idx,
        )[0]
        cost_value = float(jax.device_get(cost))
        if not np.isfinite(cost_value):
            controls = equilibrium_controls
            cost_value = equilibrium_expected
            valid = False
        restart_costs.append(cost_value)
        restart_controls.append(np.asarray(jax.device_get(controls)))
        restart_runtime_ms.append(float(runtime_ms))
        restart_status.append("ok" if valid else "invalid")

    summary = summarize_best_response_restarts(
        equilibrium_expected, restart_costs)
    best_controls = restart_controls[summary["best_restart"]]
    plan = evaluate_agent_control_batch(
        gen,
        jnp.asarray(best_controls[0])[None],
        jnp.asarray(best_controls[1])[None],
        br_context,
        agent_idx,
    )
    best_response_xy = np.stack([
        np.asarray(jax.device_get(plan["x"][0])),
        np.asarray(jax.device_get(plan["y"][0])),
    ], axis=-1)
    summary.update({
        "method": (
            "empirical_distributional_best_response_"
            f"{len(restart_keys)}_restart"),
        "agent_idx": int(agent_idx),
        "point_equilibrium_cost": float(point_equilibrium_cost),
        "equilibrium_expected_cost": equilibrium_expected,
        "best_controls": best_controls.tolist(),
        "best_response_xy": best_response_xy.tolist(),
        "restart_runtime_ms": restart_runtime_ms,
        "restart_status": restart_status,
        "diagnostic_runtime_ms": float(sum(restart_runtime_ms)),
        "background_key": _key_as_list(background_key),
        "restart_keys": [_key_as_list(key) for key in restart_keys],
        "opponent_sample_count": opponent_count,
    })
    return summary


def build_component_mean_diagnostic(gen, scenario):
    """Compile restricted unilateral search over each agent's 3x3 means."""
    combinations = jnp.asarray(
        [(ks, kd) for ks in range(3) for kd in range(3)],
        dtype=jnp.int32,
    )
    combination_count = int(combinations.shape[0])

    @jax.jit
    def diagnostic(final_mu, selected_blocks, context):
        best_costs = []
        best_indices = []
        for aid in range(3):
            s_block, d_block = 2 * aid, 2 * aid + 1
            variants = jnp.broadcast_to(
                selected_blocks[None],
                (combination_count,) + selected_blocks.shape,
            )
            variants = variants.at[:, s_block].set(
                final_mu[s_block, combinations[:, 0]])
            variants = variants.at[:, d_block].set(
                final_mu[d_block, combinations[:, 1]])
            plans = evaluate_joint_plan_batch(
                gen,
                variants.reshape((combination_count, -1)),
                context,
                agent_count=3,
            )
            costs = batched_nested_costs_from_plans(
                plans,
                scenario,
                gen.dt,
                k_inner=0.1,
                obj_transform="standard",
                ctx=context,
            )[:, aid]
            best_index = jnp.argmin(costs)
            best_costs.append(costs[best_index])
            best_indices.append(best_index)
        return jnp.stack(best_costs), jnp.stack(best_indices)

    return diagnostic


def _state_event_to_report_index(state_index):
    return max(0, int(state_index) - 1)


def _first_true(values, fallback):
    indices = np.flatnonzero(values)
    return int(indices[0]) if indices.size else int(fallback)


def select_nash_keyframes(
        state_history,
        rss_history,
        *,
        target_d=3.5,
        completion_tolerance=0.5,
        completion_hold=5):
    """Choose unique report indices for four meaningful game events."""
    states = np.asarray(state_history)
    rss = np.asarray(rss_history)
    d = states[:, 0, 1]
    last_report = max(0, len(states) - 2)

    midpoint_state = _first_true(d >= 0.5 * target_d, len(states) - 1)
    inside = np.abs(d - target_d) <= completion_tolerance
    complete_state = len(states) - 1
    for index in range(max(0, len(inside) - completion_hold + 1)):
        if np.all(inside[index:index + completion_hold]):
            complete_state = index
            break

    candidates = [
        ("initial", 0),
        ("max_rss", int(np.argmax(np.sum(rss, axis=1))) if rss.size else 0),
        ("lane_midpoint", _state_event_to_report_index(midpoint_state)),
        ("lane_complete", _state_event_to_report_index(complete_state)),
    ]
    unique = {}
    used = set()
    for name, index in candidates:
        index = min(max(0, int(index)), last_report)
        if index not in used:
            unique[name] = index
            used.add(index)
    return unique
