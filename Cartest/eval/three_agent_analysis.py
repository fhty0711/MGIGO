"""Reproducible closed-loop analysis for the three-agent B-spline game.

The control rollout is completed before any stochastic best-response work.
Consequently diagnostic keys, compilation, and runtime cannot alter executed
controls or contaminate the reported MPC solve times.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.execution.execute import FrenetState, execute_perfect_tracking_at
from Cartest.eval.nash_game_diagnostics import (
    build_component_mean_diagnostic,
    make_empirical_best_response_solver,
    mode_residual,
    run_empirical_best_response,
    select_nash_keyframes,
)
from Cartest.planning.costs.three_agent_track_components import (
    selected_plan_component_report,
)
from Cartest.planning.scenarios import get_scenario
from Cartest.planning.solver_modes import (
    build_cartest_nash_solver,
    build_multi_agent_context,
    build_multi_agent_warmstart,
    select_nash_plan,
)


ROOT = Path(__file__).resolve().parents[2]
BASIS = ROOT / "Cartest" / "basis" / "bspline_basis.npz"
STATE_FIELDS = ("s", "d", "s_dot", "d_dot", "s_ddot", "d_ddot")
OBJECTIVE_TERMS = ("progress", "lane_preference", "comfort", "rss")
CONSTRAINT_TERMS = (
    "speed", "acc", "jerk", "forward", "collision", "lane_boundary",
    "kinematics", "safety_envelope",
)


def _block_until_ready(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _make_states(scenario):
    return [
        FrenetState(
            s=agent["s"],
            s_dot=agent["s_dot"],
            s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"],
            d_dot=agent.get("d_dot", 0.0),
            d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]


def _state_row(states):
    return [
        [float(getattr(state, field)) for field in STATE_FIELDS]
        for state in states
    ]


def _sign_changes(values, tolerance=5e-3):
    values = np.asarray(values)
    signs = np.sign(values[np.abs(values) > tolerance])
    return int(np.sum(signs[1:] != signs[:-1])) if len(signs) >= 2 else 0


def _first_sustained_lane_state(d_values, target=3.5, tolerance=0.5, hold=5):
    inside = np.abs(np.asarray(d_values) - target) <= tolerance
    for index in range(max(0, len(inside) - hold + 1)):
        if np.all(inside[index:index + hold]):
            return index
    return None


def _percentile_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _output_paths(output_dir, T, steps, seed):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = output_dir / (
        f"three_agent_track_T{T}_steps{steps}_seed{seed}_{stamp}")
    return {
        "json": stem.with_suffix(".json"),
        "npz": stem.with_suffix(".npz"),
        "video": stem.with_suffix(".mp4"),
        "keyframe_dir": Path(f"{stem}_keyframes"),
        "contact_sheet": Path(f"{stem}_keyframes_contact_sheet.png"),
    }


def _save_keyframe_artifacts(reports, keyframes, paths, road):
    from Cartest.visualization.game_renderer import (
        compute_game_limits,
        save_game_frame,
    )

    keyframe_dir = paths["keyframe_dir"]
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = {}
    for name, step in keyframes.items():
        output = keyframe_dir / f"{int(step):03d}_{name}.png"
        limits = compute_game_limits(reports[:step + 1], road=road)
        save_game_frame(reports[step], output, road=road, limits=limits)
        frame_paths[name] = output

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(frame_paths)
    columns = min(2, max(1, count))
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(14 * columns, 4.05 * rows),
        dpi=100, squeeze=False)
    fig.patch.set_facecolor("#0a1120")
    for ax in axes.flat:
        ax.axis("off")
        ax.set_facecolor("#0a1120")
    for ax, (name, image_path) in zip(axes.flat, frame_paths.items()):
        ax.imshow(plt.imread(image_path))
        ax.set_title(name, color="white", fontsize=12)
        ax.axis("off")
    fig.tight_layout(pad=0.35)
    fig.savefig(paths["contact_sheet"], facecolor=fig.get_facecolor())
    plt.close(fig)
    return frame_paths


def run_three_agent_analysis(
        steps=150, T=300, seed=0, output_dir=None, *, render_video=True,
        run_best_response=True, save_keyframes=True, game_overrides=None,
        fps=5):
    """Run the game, then compute diagnostics and save JSON/NPZ/video outputs."""
    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"] = dict(scenario["game"], T=int(T))
    if game_overrides:
        scenario["game"].update(game_overrides)
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    solver = build_cartest_nash_solver(gen, scenario)
    mode_diagnostic = build_component_mean_diagnostic(gen, scenario)
    states = _make_states(scenario)
    names = [agent["name"] for agent in scenario["agents"]]
    target_speeds = [float(agent["v_target"]) for agent in scenario["agents"]]
    execute_index = int(scenario["game"].get("execute_index", 1))
    execute_dt = execute_index * float(gen.dt)
    output_dir = output_dir or (
        ROOT / "Cartest" / "output" / "t300_150_nash_analysis")
    paths = _output_paths(output_dir, T, steps, seed)

    warm_context = build_multi_agent_context(states)
    warm_mu, warm_l = build_multi_agent_warmstart(
        gen, scenario, states, jax.random.PRNGKey(999))
    compile_started = time.perf_counter_ns()
    warm_result = solver(
        jax.random.PRNGKey(1000),
        context=warm_context,
        initial_mu=warm_mu,
        initial_S_or_L=warm_l,
    )
    _block_until_ready(warm_result)
    solver_compile_ms = (time.perf_counter_ns() - compile_started) / 1e6
    mode_compile_started = time.perf_counter_ns()
    warm_mode = mode_diagnostic(
        warm_result.diag["mu"],
        warm_result.diag["selected_blocks"],
        {key: jnp.asarray(value) for key, value in warm_context.items()},
    )
    _block_until_ready(warm_mode)
    mode_compile_ms = (time.perf_counter_ns() - mode_compile_started) / 1e6

    solve_ms = []
    mode_diagnostic_ms = []
    state_history = [_state_row(states)]
    history_xy = []
    reports = []
    snapshots = []
    nested_costs = []
    component_history = []
    rss_history = []
    pi_history = []
    mode_residual_history = []
    mode_best_index_history = []
    mu_update_norm = []
    solver_names = []
    iteration_counts = []
    plan_jerk_max = np.zeros((steps, 3, 2), dtype=float)
    replan_rms = np.full((steps, 3, len(STATE_FIELDS)), np.nan)
    replan_max = np.full((steps, 3, len(STATE_FIELDS)), np.nan)
    previous_plans = None
    control_key = jax.random.PRNGKey(seed)

    for step in range(steps):
        control_key, solve_key, warm_key = jax.random.split(control_key, 3)
        context = build_multi_agent_context(states)
        initial_mu, initial_l = build_multi_agent_warmstart(
            gen, scenario, states, warm_key)

        started = time.perf_counter_ns()
        result = solver(
            solve_key,
            context=context,
            initial_mu=initial_mu,
            initial_S_or_L=initial_l,
        )
        _block_until_ready(result)
        solve_ms.append((time.perf_counter_ns() - started) / 1e6)

        solver_names.append(result.solver_name)
        metrics = np.asarray(result.diag["metrics"]["mean_fitness"])
        iteration_counts.append(int(metrics.shape[0]))
        mu_update_norm.append(float(np.linalg.norm(
            np.asarray(result.diag["mu"]) - np.asarray(initial_mu))))

        context_jax = {
            key: jnp.asarray(value) for key, value in context.items()
        }
        diagnostic_started = time.perf_counter_ns()
        best_mode_cost, best_mode_index = mode_diagnostic(
            result.diag["mu"],
            result.diag["selected_blocks"],
            context_jax,
        )
        _block_until_ready((best_mode_cost, best_mode_index))
        mode_diagnostic_ms.append(
            (time.perf_counter_ns() - diagnostic_started) / 1e6)
        selected_cost = np.asarray(result.per_agent_cost, dtype=float)
        residual = np.asarray(mode_residual(
            selected_cost, best_mode_cost), dtype=float)
        nested_costs.append(selected_cost)
        mode_residual_history.append(residual)
        mode_best_index_history.append(np.asarray(best_mode_index, dtype=int))
        pi_history.append(np.asarray(result.diag["pi"], dtype=float))

        selected = select_nash_plan(result, scenario)
        plan_dicts = []
        predicted_xy = []
        current_xy = []
        next_states = []
        for plan in selected:
            aid = int(plan["agent_idx"])
            frenet, vehicle, (x, y) = gen.evaluate_plan(
                plan["ctrl_s"], plan["ctrl_d"], states[aid].to_ctx())
            frenet_np = [np.asarray(value) for value in frenet]
            vehicle_np = np.asarray(vehicle)
            x_np, y_np = np.asarray(x), np.asarray(y)
            plan_np = {
                field: frenet_np[index]
                for index, field in enumerate(STATE_FIELDS)
            }
            plan_np.update({
                "s_dddot": frenet_np[6],
                "d_dddot": frenet_np[7],
                "vehicle": vehicle_np,
                "x": x_np,
                "y": y_np,
            })
            plan_dicts.append(plan_np)
            predicted_xy.append(np.stack([x_np, y_np], axis=-1))
            index = max(1, min(execute_index, vehicle_np.shape[0] - 1))
            current_xy.append([vehicle_np[index, 0], vehicle_np[index, 1]])
            next_states.append(execute_perfect_tracking_at(
                *frenet[:6], vehicle[:, 3], index=index))
            plan_jerk_max[step, aid, 0] = float(
                np.max(np.abs(vehicle_np[:, 6])))
            plan_jerk_max[step, aid, 1] = float(
                np.max(np.abs(vehicle_np[:, 7])))

            if previous_plans is not None:
                previous = previous_plans[aid]
                overlap = min(
                    len(previous["s"]) - execute_index,
                    len(plan_np["s"]),
                )
                if overlap > 0:
                    for field_index, field in enumerate(STATE_FIELDS):
                        delta = (
                            previous[field][execute_index:execute_index + overlap]
                            - plan_np[field][:overlap]
                        )
                        replan_rms[step, aid, field_index] = float(
                            np.sqrt(np.mean(delta ** 2)))
                        replan_max[step, aid, field_index] = float(
                            np.max(np.abs(delta)))

        raw_components = selected_plan_component_report(
            tuple(plan_dicts), scenario, context, gen.dt)
        component_step = np.zeros(
            (3, len(OBJECTIVE_TERMS) + len(CONSTRAINT_TERMS)), dtype=float)
        for aid in range(3):
            for term_index, term in enumerate(
                    OBJECTIVE_TERMS + CONSTRAINT_TERMS):
                component_step[aid, term_index] = float(
                    raw_components[aid][term])
        component_history.append(component_step)
        rss_history.append(component_step[:, OBJECTIVE_TERMS.index("rss")])

        history_xy.append(np.asarray(current_xy, dtype=float))
        best_components = np.asarray(result.diag["best_components"], dtype=int)
        agent_status = []
        for aid, plan in enumerate(selected):
            agent_status.append({
                "name": names[aid],
                "speed": float(states[aid].s_dot),
                "target_speed": target_speeds[aid],
                "mode": [
                    int(best_components[2 * aid]),
                    int(best_components[2 * aid + 1]),
                ],
                "epsilon_mode": float(residual[aid]),
                "pi_s": np.asarray(plan["pi_s"], dtype=float).tolist(),
                "pi_d": np.asarray(plan["pi_d"], dtype=float).tolist(),
            })
        reports.append({
            "step": step,
            "history_xy": np.asarray(history_xy, dtype=float),
            "predicted_xy": predicted_xy,
            "agent_names": names,
            "pi": [np.asarray(plan["pi_s"]) for plan in selected],
            "solve_ms": solve_ms[-1],
            "agent_status": agent_status,
        })
        snapshots.append({
            "context": context_jax,
            "mu": jnp.asarray(result.diag["mu"]),
            "S_or_L": jnp.asarray(result.diag["S_or_L"]),
            "pi": jnp.asarray(result.diag["pi"]),
            "selected_blocks": jnp.asarray(result.diag["selected_blocks"]),
            "per_agent_cost": selected_cost,
        })
        states = next_states
        state_history.append(_state_row(states))
        previous_plans = plan_dicts

    state_array = np.asarray(state_history, dtype=float)
    solve_array = np.asarray(solve_ms, dtype=float)
    nested_array = np.asarray(nested_costs, dtype=float)
    component_array = np.asarray(component_history, dtype=float)
    rss_array = np.asarray(rss_history, dtype=float)
    mode_residual_array = np.asarray(mode_residual_history, dtype=float)
    mode_index_array = np.asarray(mode_best_index_history, dtype=int)
    pi_array = np.asarray(pi_history, dtype=float)
    keyframes = select_nash_keyframes(
        state_array,
        rss_array,
        target_d=float(scenario.get("behavior", {}).get("ego_target_d", 3.5)),
    )

    best_responses = {}
    best_response_ms = 0.0
    if run_best_response:
        diagnostic_root = jax.random.fold_in(
            jax.random.PRNGKey(seed), 0x4E415348)
        br_solvers = [
            make_empirical_best_response_solver(gen, scenario, aid)
            for aid in range(3)
        ]
        for frame_name, step in keyframes.items():
            snapshot = snapshots[step]
            frame_key = jax.random.fold_in(diagnostic_root, int(step))
            frame_results = {}
            ghost_paths = []
            for aid in range(3):
                background_key = jax.random.fold_in(frame_key, aid)
                restart_keys = [
                    jax.random.fold_in(frame_key, 100 + aid * 3 + restart)
                    for restart in range(3)
                ]
                br = run_empirical_best_response(
                    gen,
                    scenario,
                    aid,
                    float(snapshot["per_agent_cost"][aid]),
                    snapshot["mu"],
                    snapshot["S_or_L"],
                    snapshot["pi"],
                    snapshot["selected_blocks"],
                    snapshot["context"],
                    background_key,
                    restart_keys,
                    solver=br_solvers[aid],
                )
                frame_results[names[aid]] = br
                ghost_paths.append(np.asarray(br["best_response_xy"]))
                best_response_ms += br["diagnostic_runtime_ms"]
            best_responses[frame_name] = {
                "step": int(step),
                "agents": frame_results,
            }
            reports[step]["keyframe_name"] = frame_name
            reports[step]["best_response_xy"] = ghost_paths
            reports[step]["best_response_diag"] = frame_results

    keyframe_paths = {}
    if save_keyframes:
        keyframe_paths = _save_keyframe_artifacts(
            reports, keyframes, paths, scenario.get("road", {}))

    ego_d = state_array[:, 0, 1]
    lane_target = float(
        scenario.get("behavior", {}).get("ego_target_d", 3.5))
    lane_state = _first_sustained_lane_state(ego_d, target=lane_target)
    executed_jerk = np.diff(state_array[:, :, 4:6], axis=0) / execute_dt
    cost_totals = {}
    cost_shares = {}
    constraint_max = {}
    for aid, name in enumerate(names):
        totals = {
            term: float(np.sum(component_array[:, aid, term_index]))
            for term_index, term in enumerate(OBJECTIVE_TERMS)
        }
        total = sum(totals.values())
        cost_totals[name] = totals
        cost_shares[name] = {
            term: (100.0 * value / total if total else 0.0)
            for term, value in totals.items()
        }
        constraint_max[name] = {
            term: float(np.max(component_array[
                :, aid, len(OBJECTIVE_TERMS) + term_index]))
            for term_index, term in enumerate(CONSTRAINT_TERMS)
        }

    replan_summary = {}
    for aid, name in enumerate(names):
        replan_summary[name] = {}
        for field_index, field in enumerate(STATE_FIELDS):
            finite_rms = replan_rms[:, aid, field_index]
            finite_max = replan_max[:, aid, field_index]
            finite_rms = finite_rms[np.isfinite(finite_rms)]
            finite_max = finite_max[np.isfinite(finite_max)]
            replan_summary[name][field] = {
                "mean_rms": float(np.mean(finite_rms)) if len(finite_rms) else None,
                "p95_rms": float(np.percentile(finite_rms, 95)) if len(finite_rms) else None,
                "max_abs": float(np.max(finite_max)) if len(finite_max) else None,
            }

    summary = {
        "configuration": {
            "T": int(T),
            "steps": int(steps),
            "seed": int(seed),
            "execute_index": execute_index,
            "execute_dt": execute_dt,
            "solver": scenario["game"]["solver"],
            "B": int(scenario["game"]["B"]),
            "B0": int(scenario["game"]["B0"]),
            "M_inner": int(scenario["game"]["M_inner"]),
            "K": int(scenario["game"]["K"]),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "timing": {
            "solver_compile_and_first_ms": solver_compile_ms,
            "mode_compile_and_first_ms": mode_compile_ms,
            "solve_ms": solve_array.tolist(),
            "solve_summary_ms": _percentile_summary(solve_array),
            "diagnostic_ms": {
                "mode_per_step": mode_diagnostic_ms,
                "mode_total": float(np.sum(mode_diagnostic_ms)),
                "best_response_total": float(best_response_ms),
            },
        },
        "lane_change": {
            "completed": lane_state is not None,
            "first_sustained_state_index": lane_state,
            "first_sustained_mpc_step": (
                None if lane_state is None else max(0, lane_state - 1)),
            "initial_d": float(ego_d[0]),
            "minimum_d": float(np.min(ego_d)),
            "maximum_d": float(np.max(ego_d)),
            "final_d": float(ego_d[-1]),
            "delta_d_sign_changes": _sign_changes(np.diff(ego_d)),
            "negative_lateral_travel": float(
                -np.sum(np.diff(ego_d)[np.diff(ego_d) < 0.0])),
        },
        "smoothness": {
            "executed_jerk_rms": {
                names[aid]: {
                    "long": float(np.sqrt(np.mean(executed_jerk[:, aid, 0] ** 2))),
                    "lat": float(np.sqrt(np.mean(executed_jerk[:, aid, 1] ** 2))),
                }
                for aid in range(3)
            },
            "executed_jerk_max": {
                names[aid]: {
                    "long": float(np.max(np.abs(executed_jerk[:, aid, 0]))),
                    "lat": float(np.max(np.abs(executed_jerk[:, aid, 1]))),
                }
                for aid in range(3)
            },
            "planned_jerk_max": {
                names[aid]: {
                    "long": float(np.max(plan_jerk_max[:, aid, 0])),
                    "lat": float(np.max(plan_jerk_max[:, aid, 1])),
                }
                for aid in range(3)
            },
            "replan_jump": replan_summary,
        },
        "cost_totals": cost_totals,
        "cost_shares_percent": cost_shares,
        "constraint_max": constraint_max,
        "nash_diagnostics": {
            "method_mode": "restricted_component_mean",
            "method_best_response": (
                "empirical_distributional_best_response_3_restart"
                if run_best_response else None),
            "solver_names": sorted(set(solver_names)),
            "iteration_counts": sorted(set(iteration_counts)),
            "all_agents_mu_updated_each_step": bool(
                np.all(np.asarray(mu_update_norm) > 0.0)),
            "component_mean_unilateral_residual": {
                names[aid]: {
                    **_percentile_summary(mode_residual_array[:, aid]),
                    "fraction_le_1e-3": float(np.mean(
                        mode_residual_array[:, aid] <= 1e-3)),
                    "fraction_le_1e-2": float(np.mean(
                        mode_residual_array[:, aid] <= 1e-2)),
                }
                for aid in range(3)
            },
            "keyframes": {name: int(step) for name, step in keyframes.items()},
            "best_responses": best_responses,
            "qualification": (
                "Finite-sample unilateral-deviation evidence; not a formal "
                "proof of an exact Nash equilibrium."),
        },
        "artifacts": {
            "video": str(paths["video"]) if render_video else None,
            "json": str(paths["json"]),
            "npz": str(paths["npz"]),
            "keyframes": {
                name: str(path) for name, path in keyframe_paths.items()
            },
            "contact_sheet": (
                str(paths["contact_sheet"]) if save_keyframes else None),
        },
    }

    np.savez_compressed(
        paths["npz"],
        state_history=state_array,
        solve_times_ms=solve_array,
        nested_costs=nested_array,
        components=component_array,
        rss_history=rss_array,
        replan_rms=replan_rms,
        replan_max=replan_max,
        plan_jerk_max=plan_jerk_max,
        pi_history=pi_array,
        mu_update_norm=np.asarray(mu_update_norm),
        nash_mode_residual=mode_residual_array,
        nash_best_indices=mode_index_array,
    )
    paths["json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if render_video:
        from Cartest.visualization.game_renderer import save_game_video
        save_game_video(
            reports,
            paths["video"],
            road=scenario.get("road", {}),
            fps=fps,
        )
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the three-agent B-spline Nash MPC rollout")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-best-response", action="store_true")
    parser.add_argument("--no-keyframes", action="store_true")
    parser.add_argument(
        "--keyframes-only",
        action="store_true",
        help="save diagnostic PNGs/contact sheet without rendering the MP4",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    summary = run_three_agent_analysis(
        steps=args.steps,
        T=args.T,
        seed=args.seed,
        output_dir=args.output_dir,
        render_video=not args.no_video and not args.keyframes_only,
        run_best_response=not args.no_best_response,
        save_keyframes=not args.no_keyframes,
        fps=args.fps,
    )
    print(json.dumps({
        "artifacts": summary["artifacts"],
        "solve_summary_ms": summary["timing"]["solve_summary_ms"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
