"""Unilateral-deviation diagnostics for the three-agent RNE game."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from Cartest.planning.costs.three_agent_track_batched import (
    batched_nested_costs_from_plans,
    evaluate_joint_plan_batch,
)


def mode_residual(selected_cost, best_cost):
    """Non-negative improvement available over restricted component means."""
    return jnp.maximum(
        jnp.asarray(selected_cost) - jnp.asarray(best_cost), 0.0)


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
