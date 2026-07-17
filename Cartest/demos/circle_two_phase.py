"""Level 1a: Two-phase MPC on CircularReference.

Phase 1: loose constraints (exploration) -> warm-start for Phase 2
Phase 2: tight constraints (refinement) -> execution trajectory
"""

from __future__ import annotations

import sys, time
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.core.reference_path import CircularReference
from Cartest.planning.warmstart import build_initial_mu
from Cartest.planning.cost import make_objective, build_context
from Cartest.execution.execute import execute_perfect_tracking, FrenetState
from Cartest.planning.constraints import make_constraints
from Cartest.planning.scenarios import get_scenario, make_initial_state, build_obstacle_predictions
from Cartest.planning.costs.default_lyapunov import DEFAULT_CONSTRAINTS
from gmm_igo.solver_builder import build_solver

CARTEST_ROOT = Path(__file__).resolve().parents[1]
BASIS = CARTEST_ROOT / "basis"


def run_two_phase(R=100.0, steps=60, seed=42):
    ref_path = CircularReference(R, 0.0, 0.0)
    gen = FrenetBSplineTrajectory(BASIS / "bspline_basis.npz", ref_path)

    scenario = get_scenario("circle_track")
    road = scenario["road"]
    safety = scenario["safety"]
    lane_hw = road["lane_hw"]
    safe_dist = safety["obs_safe_dist"]
    v_target = scenario["behavior"]["v_target"]
    obs_pos, obs_rad = build_obstacle_predictions(scenario, gen)

    # ── Phase 1 solver: loose constraints ──
    solver_p1 = build_solver(
        make_objective(gen, omega_s=1.0, omega_d=4.0, alpha=0.0),
        dims=(gen.n_free, gen.n_free),
        constraints=make_constraints(gen, road, safety, DEFAULT_CONSTRAINTS),
        solver='m22', T=300, dt=0.25, K=3, B=128, B0=50, T_0=100,
        k_inner=1.0, obj_transform='standard',
    )

    # ── Phase 2 solver: tight constraints ──
    solver_p2 = build_solver(
        make_objective(gen, omega_s=1.0, omega_d=4.0, alpha=0.0),
        dims=(gen.n_free, gen.n_free),
        constraints=make_constraints(gen, road, safety, DEFAULT_CONSTRAINTS),
        solver='m22', T=300, dt=0.20, K=3, B=64, B0=30, T_0=300,
        k_inner=1.0, obj_transform='standard',
    )

    key = random.PRNGKey(seed)
    state = make_initial_state(scenario)
    state = FrenetState(s=state.s, s_dot=state.s_dot, s_ddot=state.s_ddot,
                        d=-3.0, d_dot=state.d_dot, d_ddot=state.d_ddot,
                        psi=state.psi)

    # ── Warmup ──
    ctx_warm = build_context(gen, state, v_target, lane_hw, obs_pos, obs_rad)
    mu_warm = build_initial_mu(gen, state.s, state.s_dot, state.d)
    _ = solver_p1(random.PRNGKey(999), context=ctx_warm, initial_mu=mu_warm)
    _ = solver_p2(random.PRNGKey(998), context=ctx_warm, initial_mu=mu_warm)

    d_hist, v_hist, a_hist = [], [], []
    t_p1_hist, t_p2_hist = [], []
    n_free = gen.n_free

    for step in range(steps):
        key, k1, k2 = random.split(key, 3)
        ctx = build_context(gen, state, v_target, lane_hw, obs_pos, obs_rad)
        mu_init = build_initial_mu(gen, state.s, state.s_dot, state.d)

        # ── Phase 1: exploration ──
        t0 = time.time()
        result_p1 = solver_p1(k1, context=ctx, initial_mu=mu_init)
        t_p1 = (time.time() - t0) * 1000

        # ── Extract z_ref from Phase 1's B-spline output ──
        ctrl_s_p1, ctrl_d_p1 = result_p1.x[:n_free], result_p1.x[n_free:]
        frenet_p1 = gen.evaluate(
                ctrl_s_p1, ctrl_d_p1,
                ctx["s0"], ctx["s_dot0"], ctx["s_ddot0"],
                ctx["d0"], ctx["d_dot0"], ctx["d_ddot0"])
        z_ref = {
            's_ref': frenet_p1[0], 's_dot_ref': frenet_p1[2],
            's_ddot_ref': frenet_p1[4],
            'd_ref': frenet_p1[1], 'd_dot_ref': frenet_p1[3],
            'd_ddot_ref': frenet_p1[5],
        }

        # ── Phase 2: track z_ref with tight constraints ──
        ctx_p2 = {**ctx, 'z_ref': z_ref}
        t0 = time.time()
        result_p2 = solver_p2(k2, context=ctx_p2, warm_start=result_p1)
        t_p2 = (time.time() - t0) * 1000

        ctrl_s, ctrl_d = result_p2.x[:n_free], result_p2.x[n_free:]
        frenet, st, _ = gen.evaluate_plan(ctrl_s, ctrl_d, ctx)
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = frenet
        state = execute_perfect_tracking(s, d, s_dot, d_dot, s_ddot, d_ddot, st[1, 3])

        d_hist.append(float(state.d))
        v_hist.append(float(state.s_dot))
        a_hist.append(float(st[0, 5]))
        t_p1_hist.append(t_p1)
        t_p2_hist.append(t_p2)

        if step % 10 == 0:
            print(f"step {step:3d} | d={state.d:.2f} v={state.s_dot:.1f} "
                  f"| P1={t_p1:.0f}ms P2={t_p2:.0f}ms "
                  f"| a_lat={st[0,5]:.1f}")

    d_arr = np.array(d_hist)
    v_arr = np.array(v_hist)
    settle = next((i for i in range(5, len(d_arr))
                   if all(abs(d_arr[j]) < 0.1 for j in range(i, min(i+5, len(d_arr))))), steps)

    print(f"\nR={R}: d={d_arr[0]:.0f}->{d_arr[-1]:.2f} settle={settle*0.1:.1f}s "
          f"v={v_arr[0]:.0f}->{v_arr[-1]:.1f} "
          f"P1_avg={np.mean(t_p1_hist):.0f}ms P2_avg={np.mean(t_p2_hist):.0f}ms "
          f"total={np.mean(t_p1_hist)+np.mean(t_p2_hist):.0f}ms")
    return d_arr, v_arr


if __name__ == "__main__":
    print("=== Two-phase MPC on CircularReference ===")
    print()
    print("--- R=100m (safe: a_lat=v²/100 ≤ 5.0 for v≤22) ---")
    run_two_phase(R=100.0)
    print()
    print("--- R=50m (borderline: a_lat=v²/50 ≤ 5.0 for v≤15.8) ---")
    run_two_phase(R=50.0)
