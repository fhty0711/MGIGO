"""Frenet B-spline trajectory MPC demo.

B-spline control points live in (s, d) Frenet space.
All kinematics are linear in control points — no arctan2/curvature chain.
Only obstacle checking maps to Cartesian via the reference path.
"""

from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Cartest.frenet_traj import FrenetBSplineTrajectory
from Cartest.reference_path import StraightReference
from Cartest.warmstart import tangent_warmstart, shift_warmstart
from Cartest.cost import make_objective
from Cartest.execute import execute_step
from Cartest.constraints import make_constraints, LANE_HW, OBS_SAFE_DIST, V_MIN, V_MAX, ACC_MAX, JERK_MAX
from Cartest.vehicle_model import FrenetVehicleModel
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from Constraintdealer.Constran import build


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

V_TARGET = 18.0
IGO_STEPS, IGO_DT = 500, 0.15
B_SAMPLES, B0_ELITE, T0_RESET = 64, 20, 250
M, K = 2, 3  # block 0: ctrl_s (9d), block 1: ctrl_d (9d)

WHEEL_BASE = 2.8

OBSTACLES = [{"x": 60.0, "y": 2.5, "r": 2.0}]

OUTPUT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════
# MPC
# ═══════════════════════════════════════════════════════════════════════

def run(steps=150, seed=0, plot=True):
    ref_path = StraightReference()
    gen = FrenetBSplineTrajectory(OUTPUT / "bspline_basis.npz", ref_path)
    D = gen.n_free * 2

    # Build cost function ONCE
    obj_fn = make_objective(gen)
    constraints = make_constraints(gen)
    cost_fn = build(obj_fn, constraints, k_inner=0.1, obj_transform='standard')

    vehicle = FrenetVehicleModel(acc_max=ACC_MAX, dt=gen.dt)

    key = random.PRNGKey(seed)

    # Initial Frenet state
    s0,  s_dot0,  s_ddot0  = 0.0, 12.0, 0.0
    d0,  d_dot0,  d_ddot0  = -3.0, 0.0, 0.0

    # Initial warm-start
    ctrl_s, ctrl_d = tangent_warmstart(gen, s0, s_dot0, d0=d0)
    mu_init = jnp.stack([
        jnp.stack([ctrl_s, ctrl_s, ctrl_s], axis=0),
        jnp.stack([ctrl_d, ctrl_d, ctrl_d], axis=0),
    ], axis=0).astype(jnp.float32)  # [2, 3, 9]
    D_max = gen.n_free  # 9
    dims = (D_max, D_max)
    L_inv = jnp.tile(jnp.eye(D_max, dtype=jnp.float32)[None, None, :, :], (M, K, 1, 1))
    v_init = jnp.zeros((M, K - 1), dtype=jnp.float32)

    obs_pos = jnp.array([[o["x"], o["y"]] for o in OBSTACLES], dtype=jnp.float32)
    obs_rad = jnp.array([o["r"] for o in OBSTACLES], dtype=jnp.float32)

    if plot:
        plt.ion(); fig, (ax_t, ax_k) = plt.subplots(1, 2, figsize=(16, 5))

    # History for plotting
    hx, hy, hv = [float(s0)], [float(d0)], [float(s_dot0)]
    frames = []

    for step in range(steps):
        key, sk = random.split(key)

        ctx = {
            "v_ref": jnp.full(gen.T, V_TARGET),
            "lane_hw": LANE_HW,
            "obs_pos": obs_pos, "obs_rad": obs_rad,
            "s0": s0, "s_dot0": s_dot0, "s_ddot0": s_ddot0,
            "d0": d0, "d_dot0": d_dot0, "d_ddot0": d_ddot0,
        }

        t0 = time.time()
        mu_k, L_k, pi_k = mmog_igo_optimizer_mpc(
            sk, IGO_STEPS, IGO_DT, M, K, B_SAMPLES, B0_ELITE,
            dims, T0_RESET, cost_fn, mu_init, L_inv, v_init, ctx)
        mu_k.block_until_ready()
        ms = (time.time() - t0) * 1000

        best = int(jnp.argmax(pi_k[0]))
        ctrl_s = mu_k[0, best, :D_max]
        ctrl_d = mu_k[1, best, :D_max]

        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
            ctrl_s, ctrl_d, s0, s_dot0, s_ddot0, d0, d_dot0, d_ddot0)

        v = jnp.sqrt(s_dot ** 2 + d_dot ** 2)
        jm = jnp.sqrt(s_dddot ** 2 + d_dddot ** 2)
        am = jnp.sqrt(s_ddot ** 2 + d_ddot ** 2)

        # Per-constraint g values (q90 for reporting)
        x_cart, y_cart = gen.to_cartesian(s, d)
        g_lane = float(jnp.quantile(jnp.maximum(0., jnp.abs(d) - LANE_HW), 0.9))
        g_obs  = float(jnp.quantile(jnp.maximum(0., OBS_SAFE_DIST + obs_rad[0] -
            jnp.sqrt((x_cart - obs_pos[0, 0]) ** 2 + (y_cart - obs_pos[0, 1]) ** 2)), 0.9))
        g_jerk = float(jnp.quantile(jnp.maximum(0., jm - JERK_MAX), 0.9))
        g_acc  = float(jnp.quantile(jnp.maximum(0., am - ACC_MAX), 0.9))
        g_spd  = float(jnp.quantile(jnp.maximum(0., jnp.maximum(V_MIN - v, v - V_MAX)), 0.9))
        total_cost = float(cost_fn(jnp.concatenate([ctrl_s, ctrl_d]), ctx))
        print(f"        cost={total_cost:.4f} g=[lane={g_lane:.2f} obs={g_obs:.2f} jerk={g_jerk:.1f} acc={g_acc:.1f} spd={g_spd:.1f}]")

        # Execute: plan's commanded acc → vehicle model → next state
        ns = execute_step(gen, s, d, s_dot, d_dot, s_ddot, d_ddot, vehicle)
        s0, s_dot0, s_ddot0 = ns['s0'], ns['s_dot0'], ns['s_ddot0']
        d0, d_dot0, d_ddot0 = ns['d0'], ns['d_dot0'], ns['d_ddot0']

        hx.append(float(s0)); hy.append(float(d0)); hv.append(float(v[0]))

        min_obs = float(jnp.min(jnp.sqrt(
            (x_cart[:, None] - obs_pos[None, :, 0]) ** 2 +
            (y_cart[:, None] - obs_pos[None, :, 1]) ** 2
        ) - obs_rad[None, :]))
        ma = float(jnp.max(jnp.abs(s_ddot)))
        ml = float(jnp.max(jnp.abs(d_ddot)))
        mj = float(jnp.max(jnp.abs(jm)))

        frames.append(dict(
            hx=np.array(hx), hy=np.array(hy), hv=np.array(hv),
            px=np.array(x_cart), py=np.array(y_cart), sp=np.array(v),
            al=np.array(s_ddot), alat=np.array(d_ddot), jl=np.array(jm),
            mo=ms, md=min_obs, ma=ma, ml=ml, mj=mj))

        # Warm-start: regenerate from current Frenet state.
        # Uses s_dot0 (current speed) for consistency with clamped boundary
        # → zero-jerk constant-speed initial guess. Optimizer finds acceleration.
        ctrl_s_ws, ctrl_d_ws = tangent_warmstart(gen, s0, float(s_dot0), d0=d0)
        mu_init = jnp.stack([
            jnp.stack([ctrl_s_ws, ctrl_s_ws, ctrl_s_ws], axis=0),
            jnp.stack([ctrl_d_ws, ctrl_d_ws, ctrl_d_ws], axis=0),
        ], axis=0).astype(jnp.float32)

        print(f"step {step:3d} | x={s0:6.1f} y={d0:5.1f} v={v[0]:5.1f} | "
              f"a_long={ma:5.1f} a_lat={ml:5.1f} jerk={mj:5.1f} | "
              f"obs={min_obs:4.1f}m | {ms:5.0f}ms")

    if plot:
        np.savez(OUTPUT / "frenet_demo.npz", hx=np.array(hx), hy=np.array(hy), hv=np.array(hv),
                 frames=np.array(frames, dtype=object))
        def render(i):
            f = frames[i]; ax_t.cla(); ax_k.cla()
            ax_t.plot(f["hx"], f["hy"], "g-", lw=2, label="exec")
            ax_t.plot(f["px"], f["py"], "b--", lw=2, label="plan")
            for o in OBSTACLES:
                ax_t.add_patch(Circle((o["x"], o["y"]), o["r"], fc="r", alpha=.3))
                ax_t.add_patch(Circle((o["x"], o["y"]), o["r"] + OBS_SAFE_DIST,
                                       fc="none", ec="r", alpha=.3, ls="--"))
            ax_t.set_aspect("equal"); ax_t.legend(loc="upper left"); ax_t.grid(alpha=.25)
            ax_t.set_xlim(f["hx"][0] - 10, f["px"].max() + 20); ax_t.set_ylim(-6, 6)
            ax_t.set_title(f"step {i}  v={f['hv'][-1]:.1f}  obs={f['md']:.1f}m  {f['mo']:.0f}ms")
            t_ = np.arange(len(f["sp"])) * gen.dt
            ax_k.plot(t_, f["al"], label="s̈ (a_long)"); ax_k.plot(t_, f["alat"], label="d̈ (a_lat)")
            ax_k.plot(t_, f["jl"], label="jerk"); ax_k.legend(); ax_k.grid(alpha=.25)
            ax_k.set_title(f"max |s̈|={f['ma']:.1f}  |d̈|={f['ml']:.1f}  jerk={f['mj']:.1f}")
        anim = FuncAnimation(fig, render, frames=len(frames), interval=80)
        anim.save(OUTPUT / "frenet_demo.gif", writer=PillowWriter(fps=12))
        plt.close(fig)
        print(f"saved {OUTPUT / 'frenet_demo.gif'}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(steps=args.steps, seed=args.seed, plot=not args.no_plot)
