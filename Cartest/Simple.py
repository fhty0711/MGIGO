"""Frenet B-spline trajectory MPC demo."""

from __future__ import annotations

import argparse, sys, time
from datetime import datetime
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
from jax import random

jax.config.update("jax_default_matmul_precision", "highest")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.planning.warmstart import build_initial_mu
from Cartest.planning.cost import build_context
from Cartest.execution.execute import FrenetState, execute_perfect_tracking, execute_perfect_tracking_at
from Cartest.planning.constraints import make_constraints, compute_g_values, compute_summary
from Cartest.eval.reporting import StepReport
from Cartest.visualization.plotting import setup_axes, render_frame, save_animation
from Cartest.eval.diagnostics import diagnose, print_diag
from Cartest.planning.scenarios import (
    SCENARIOS, get_scenario, make_initial_state,
    build_obstacle_predictions, first_frame_obstacles,
)
from Cartest.planning.costs import (
    make_objective_from_scenario,
    make_constraint_config_from_scenario,
)
from gmm_igo.solver_builder import build_solver
from Cartest.planning.scenarios import scenario_kind
from Cartest.planning.solver_modes import (
    build_cartest_nash_solver,
    build_multi_agent_context,
    build_multi_agent_warmstart,
    select_nash_plan,
)
from Cartest.planning.costs.three_agent_track_components import (
    selected_plan_component_report,
)
from Cartest.visualization.game_renderer import save_game_video


ROOT = Path(__file__).resolve().parent
BASIS = ROOT / "basis"
OUTPUT = ROOT / "output"


def make_output_video_path(scenario_name, now=None):
    """Return the standard timestamped MP4 output path for a scenario."""
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    return OUTPUT / f"{scenario_name}_{stamp}.mp4"


def default_steps_for_scenario(scenario_name):
    """Resolve the default closed-loop step count for a scenario."""
    scenario = get_scenario(scenario_name)
    if scenario_kind(scenario) == "multi_agent_game":
        return int(scenario.get("game", {}).get("default_steps", 30))
    return int(scenario.get("default_steps", 150))


def block_until_ready(result):
    """Synchronize JAX async work for accurate wall-clock timing."""
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


# ═══════════════════════════════════════════════════════════════════════
# MPC
# ═══════════════════════════════════════════════════════════════════════

def run_single_agent_mpc(steps=150, seed=0, plot=True, scenario_name="empty",
                         include_compile_time=False):
    scenario = get_scenario(scenario_name)
    ref_path = scenario["ref_path"]
    gen = FrenetBSplineTrajectory(BASIS / "bspline_basis.npz", ref_path)

    # Build solver ONCE
    # ── Scenario ──
    road = scenario["road"]
    safety = scenario["safety"]
    lane_hw = road["lane_hw"]
    lane_bounds_d = road.get("lane_bounds_d", (-lane_hw, lane_hw))
    safe_dist = safety["obs_safe_dist"]
    v_target = scenario["behavior"]["v_target"]
    obs_pos, obs_rad = build_obstacle_predictions(scenario, gen)
    obs_list = first_frame_obstacles(obs_pos, obs_rad)

    constraint_cfg = make_constraint_config_from_scenario(scenario)
    constran_cfg = constraint_cfg["constran"]

    solver = build_solver(
        make_objective_from_scenario(gen, scenario),
        dims=(gen.n_free, gen.n_free),
        constraints=make_constraints(gen, road, safety, constraint_cfg),
        solver='m22', T=300, dt=0.2, K=3, B=64, B0=30, T_0=300,
        k_inner=constran_cfg["k_inner"], obj_transform=constran_cfg["obj_transform"],
    )

    key = random.PRNGKey(seed)

    state = make_initial_state(scenario)

    if not include_compile_time:
        # Compile all JAX functions before the first timed step.
        ctx_warm = build_context(gen, state, v_target, lane_hw, obs_pos, obs_rad,
                                 lane_bounds_d=lane_bounds_d)
        mu_warm = build_initial_mu(gen, state.s, state.s_dot, state.d)
        block_until_ready(solver(random.PRNGKey(999), context=ctx_warm,
                                 initial_mu=mu_warm))

    if plot:
        fig, ax_t, ax_k = setup_axes()

    hx, hy, hv = [state.s], [state.d], [state.s_dot]
    frames, reports = [], []

    for step in range(steps):
        key, sk = random.split(key)

        obs_pos, obs_rad = build_obstacle_predictions(scenario, gen, mpc_time=step * gen.dt)
        ctx = build_context(gen, state, v_target, lane_hw, obs_pos, obs_rad,
                            lane_bounds_d=lane_bounds_d)
        mu_init = build_initial_mu(gen, state.s, state.s_dot, state.d)

        t0 = time.time()
        result = solver(sk, context=ctx, initial_mu=mu_init)
        ms = (time.time() - t0) * 1000

        ctrl_s, ctrl_d = result.x[:gen.n_free], result.x[gen.n_free:]

        # Evaluate plan -> Frenet + vehicle states + Cartesian (one call)
        frenet, st, (x_cart, y_cart) = gen.evaluate_plan(ctrl_s, ctrl_d, ctx)
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = frenet

        # Metrics
        gv = compute_g_values(st, d, x_cart, y_cart, obs_pos, obs_rad, lane_hw, safety)
        sm = compute_summary(st, d, x_cart, y_cart, obs_pos, obs_rad)

        # Execute: use plan's predicted next state (perfect tracking)
        state = execute_perfect_tracking(s, d, s_dot, d_dot, s_ddot, d_ddot, st[1, 3])
        hx.append(state.s); hy.append(state.d); hv.append(state.s_dot)

        # Record
        report = StepReport(
            step=step,
            hx=np.array(hx), hy=np.array(hy), hv=np.array(hv),
            px=np.array(x_cart), py=np.array(y_cart), sp=np.array(st[:, 2]),
            a_long=np.array(st[:, 4]), a_lat=np.array(st[:, 5]),
            jm=np.array(jnp.sqrt(st[:, 6]**2 + st[:, 7]**2)),
            solve_ms=ms, min_obs=sm['min_obs'],
            max_along=sm['max_a_long'], max_alat=sm['max_a_lat'],
            max_jerk=sm['max_jerk'], cost=result.cost, g_values=gv,
        )
        reports.append(report)
        frames.append(report.to_frame_dict())

        print(report.print_cost())
        print(report.print_line())
        if step % 5 == 0:
            print_diag(diagnose(gen, result.x, ctx, safe_dist, state))

    if plot:
        np.savez(ROOT / "frenet_demo.npz",
                 hx=np.array(hx), hy=np.array(hy), hv=np.array(hv),
                 frames=np.array(frames, dtype=object))
        save_animation(fig, reports,
                       lambda i: render_frame(ax_t, ax_k, reports[i],
                                              obs_list, safe_dist, gen.dt,
                                              scenario=scenario),
                      make_output_video_path(scenario_name))


def _make_agent_states(scenario):
    return [
        FrenetState(
            s=agent["s"], s_dot=agent["s_dot"], s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"], d_dot=agent.get("d_dot", 0.0), d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]


def run_multi_agent_game(steps=30, seed=0, plot=True, scenario_name="game_2a_basic",
                         include_compile_time=False):
    scenario = get_scenario(scenario_name)
    gen = FrenetBSplineTrajectory(BASIS / "bspline_basis.npz", scenario["ref_path"])
    solver = build_cartest_nash_solver(gen, scenario)
    execute_index = int(scenario.get("game", {}).get("execute_index", 1))
    key = random.PRNGKey(seed)
    states = _make_agent_states(scenario)

    if not include_compile_time:
        ctx_warm = build_multi_agent_context(states)
        mu_warm, L_warm = build_multi_agent_warmstart(
            gen, scenario, states, random.PRNGKey(999))
        block_until_ready(solver(
            random.PRNGKey(1000), context=ctx_warm,
            initial_mu=mu_warm, initial_S_or_L=L_warm))

    history_xy = []
    reports = []

    for step in range(steps):
        key, solve_key, warm_key = random.split(key, 3)
        ctx = build_multi_agent_context(states)
        mu_init, L_inv_init = build_multi_agent_warmstart(gen, scenario, states, warm_key)

        t0 = time.time()
        result = solver(solve_key, context=ctx, initial_mu=mu_init,
                        initial_S_or_L=L_inv_init)
        block_until_ready(result)
        ms = (time.time() - t0) * 1000
        plans = select_nash_plan(result, scenario)

        current_xy = []
        predicted_xy = []
        pi_values = []
        next_states = []
        plan_dicts = []
        for plan in plans:
            agent_idx = plan["agent_idx"]
            state = states[agent_idx]
            agent_ctx = state.to_ctx()
            frenet, vehicle, (x_cart, y_cart) = gen.evaluate_plan(
                plan["ctrl_s"], plan["ctrl_d"], agent_ctx)
            s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = frenet
            idx = max(1, min(execute_index, vehicle.shape[0] - 1))
            next_states.append(execute_perfect_tracking_at(
                s, d, s_dot, d_dot, s_ddot, d_ddot, vehicle[:, 3], index=idx))
            current_xy.append([float(vehicle[idx, 0]), float(vehicle[idx, 1])])
            predicted_xy.append(np.stack([np.array(x_cart), np.array(y_cart)], axis=-1))
            pi_values.append(np.array(plan["pi_s"]))
            plan_dicts.append({
                "s": s, "d": d, "s_dot": s_dot, "d_dot": d_dot,
                "s_ddot": s_ddot, "d_ddot": d_ddot, "s_dddot": s_dddot,
                "d_dddot": d_dddot, "vehicle": vehicle,
            })

        history_xy.append(np.array(current_xy))
        states = next_states
        report = {
            "step": step,
            "history_xy": np.array(history_xy),
            "predicted_xy": predicted_xy,
            "agent_names": [agent["name"] for agent in scenario["agents"]],
            "pi": pi_values,
            "solve_ms": ms,
        }
        # Selected-plan cost diagnostics (three-agent scenarios only; the
        # component module is role-specific to ego/front/rear).  Pure-Python
        # floats so the report stays serializer-friendly; does not affect the
        # solver or warmstart.
        if len(plan_dicts) == 3:
            raw_report = selected_plan_component_report(
                tuple(plan_dicts), scenario, ctx, gen.dt)
            report["component_report"] = {
                aid: {key: float(val) for key, val in comps.items()}
                for aid, comps in raw_report.items()
            }
        reports.append(report)

        state_line = " ".join(
            f"{scenario['agents'][i]['name']}:s={states[i].s:.1f},d={states[i].d:.2f},v={states[i].s_dot:.1f}"
            for i in range(len(states))
        )
        print(f"step {step:02d} | {state_line} | solve={ms:.1f}ms")

    if plot and reports:
        output = make_output_video_path(scenario_name)
        save_game_video(reports, output, road=scenario.get("road", {}))
        print(f"Saved game video: {output}")


def run(steps=150, seed=0, plot=True, scenario_name="empty",
        include_compile_time=False):
    scenario = get_scenario(scenario_name)
    kind = scenario_kind(scenario)
    if kind == "single_agent":
        return run_single_agent_mpc(steps=steps, seed=seed, plot=plot,
                                    scenario_name=scenario_name,
                                    include_compile_time=include_compile_time)
    if kind == "multi_agent_game":
        return run_multi_agent_game(steps=steps, seed=seed, plot=plot,
                                    scenario_name=scenario_name,
                                    include_compile_time=include_compile_time)
    raise ValueError(f"Unknown scenario type {kind!r} for scenario {scenario_name!r}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("scenario", nargs="?", choices=tuple(sorted(SCENARIOS)),
                   help="scenario name")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario", dest="scenario_flag",
                   choices=tuple(sorted(SCENARIOS)))
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--include-compile-time", action="store_true",
                   help="include first-call JAX compile time in step 0 solve timing")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    scenario_name = args.scenario_flag or args.scenario or "empty"
    steps = args.steps if args.steps is not None else default_steps_for_scenario(scenario_name)
    run(steps=steps, seed=args.seed, plot=not args.no_plot,
        scenario_name=scenario_name,
        include_compile_time=args.include_compile_time)
