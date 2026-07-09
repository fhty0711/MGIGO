"""Level 2b with Constran: collision as Deterministic(mode='hard'), auto σ-nesting.

Replaces hand-tuned collision weight with Constran.build().
Tests whether symmetric scenario breaks symmetry via mixed-strategy Nash.
"""

from __future__ import annotations

import sys, time
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
from jax import random, lax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.core.reference_path import StraightReference
from Cartest.execution.execute import execute_perfect_tracking, FrenetState
from Constraintdealer.Constran import Deterministic, build as constran_build
from gmm_igo.MPC_G_MS import mmog_igo_rne_blocks_solver

ROOT = Path(__file__).resolve().parent
BASIS = ROOT / "basis"


# ═══════════════════════════════════════════════════════════════════════
# Per-agent Lyapunov goal cost (no weights — bare objective)
# ═══════════════════════════════════════════════════════════════════════

def make_goal_cost(gen, agent_idx, n_free, v_target, omega_s=1.0, omega_d=4.0):
    """Bare Lyapunov tracking cost for one agent. Reads from joint_x."""
    t_arr = jnp.arange(gen.T) * gen.dt
    base = agent_idx * 2 * n_free

    def goal_cost(joint_x, ctx):
        theta = joint_x[base : base + 2*n_free]
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
            theta[:n_free], theta[n_free:2*n_free],
            ctx['s0'], ctx['s_dot0'], ctx['s_ddot0'],
            ctx['d0'], ctx['d_dot0'], ctx['d_ddot0'])

        # ── Reference (hardcoded exponential speed profile, d_ref=0) ──
        v_tgt = float(v_target)
        s0 = ctx['s0']; v0 = ctx['s_dot0']
        dv = v0 - v_tgt; lam = omega_s
        exp_term = jnp.exp(-lam * t_arr)
        s_ref = s0 + v_tgt * t_arr + dv / lam * (1.0 - exp_term)
        s_dot_ref = v_tgt + dv * exp_term
        s_ddot_ref = -dv * lam * exp_term
        d_ref = jnp.zeros_like(s)
        d_dot_ref = jnp.zeros_like(s)
        d_ddot_ref = jnp.zeros_like(s)

        es = s - s_ref; ed = d - d_ref
        es_dot = s_dot - s_dot_ref; ed_dot = d_dot - d_dot_ref
        es_ddot = s_ddot - s_ddot_ref; ed_ddot = d_ddot - d_ddot_ref

        K00, K01 = omega_s, 0.0
        K10, K11 = 0.0, omega_d
        K2_00 = K00**2; K2_01 = 0.0; K2_10 = 0.0; K2_11 = K11**2

        term0 = es**2 + ed**2
        v1_s = es_dot + K00*es; v1_d = ed_dot + K11*ed
        term1 = v1_s**2 + v1_d**2
        v2_s = es_ddot + 2.0*K00*es_dot + K2_00*es
        v2_d = ed_ddot + 2.0*K11*ed_dot + K2_11*ed
        term2 = v2_s**2 + v2_d**2

        return jnp.sum(term0) + jnp.sum(term1) + jnp.sum(term2)
    return goal_cost


# ═══════════════════════════════════════════════════════════════════════
# Per-agent constraint g_fns (work on joint_x, extract own theta)
# ═══════════════════════════════════════════════════════════════════════

V_MIN, V_MAX = 2.0, 35.0
_ACC_MAX = 5.0
_JERK_MAX = 2.0

def _eval_my_frenet(gen, agent_idx, n_free, joint_x, ctx):
    base = agent_idx * 2 * n_free
    theta = joint_x[base : base + 2*n_free]
    return gen.evaluate(theta[:n_free], theta[n_free:2*n_free],
                        ctx['s0'], ctx['s_dot0'], ctx['s_ddot0'],
                        ctx['d0'], ctx['d_dot0'], ctx['d_ddot0'])

def _eval_my_vehicle(gen, agent_idx, n_free, joint_x, ctx):
    s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = _eval_my_frenet(
        gen, agent_idx, n_free, joint_x, ctx)
    return gen.to_vehicle_states(s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot)

def make_lane_g(gen, agent_idx, n_free, lane_hw):
    def lane_g(joint_x, ctx):
        _, d, _, _, _, _, _, _ = _eval_my_frenet(gen, agent_idx, n_free, joint_x, ctx)
        return jnp.maximum(0., jnp.abs(d) - lane_hw)
    return lane_g

def make_speed_g(gen, agent_idx, n_free):
    def speed_g(joint_x, ctx):
        st = _eval_my_vehicle(gen, agent_idx, n_free, joint_x, ctx)
        v = st[:, 2]
        return jnp.maximum(jnp.maximum(0., V_MIN - v), jnp.maximum(0., v - V_MAX))
    return speed_g

def make_acc_g(gen, agent_idx, n_free):
    def acc_g(joint_x, ctx):
        st = _eval_my_vehicle(gen, agent_idx, n_free, joint_x, ctx)
        a_long, a_lat = st[:, 4], st[:, 5]
        am = jnp.sqrt(a_long**2 + a_lat**2)
        return jnp.maximum(
            jnp.maximum(0., jnp.abs(a_long) - _ACC_MAX),
            jnp.maximum(jnp.maximum(0., jnp.abs(a_lat) - _ACC_MAX),
                        jnp.maximum(0., am - _ACC_MAX)))
    return acc_g

def make_jerk_g(gen, agent_idx, n_free):
    def jerk_g(joint_x, ctx):
        st = _eval_my_vehicle(gen, agent_idx, n_free, joint_x, ctx)
        j_long, j_lat = st[:, 6], st[:, 7]
        jm = jnp.sqrt(j_long**2 + j_lat**2)
        return jnp.maximum(
            jnp.maximum(0., jnp.abs(j_long) - _JERK_MAX),
            jnp.maximum(jnp.maximum(0., jnp.abs(j_lat) - _JERK_MAX),
                        jnp.maximum(0., jm - _JERK_MAX)))
    return jerk_g


# ═══════════════════════════════════════════════════════════════════════
# Collision g_fn: geometric RSS on Cartesian trajectory samples
# ═══════════════════════════════════════════════════════════════════════

def make_collision_g(gen, agent_idx, n_free, safe_dist=3.0):
    """Collision violation between this agent and the other.

    Computes Cartesian distance at each B-spline sampling point.
    Positive g = violation (distance below safe threshold).
    """
    def collision_g(joint_x, ctx):
        # Decode both agents' trajectories
        carts = []
        for a in range(2):
            base = a * 2 * n_free
            theta = joint_x[base : base + 2*n_free]
            s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
                theta[:n_free], theta[n_free:2*n_free],
                ctx[f's0_a{a}'], ctx[f's_dot0_a{a}'], 0.0,
                ctx[f'd0_a{a}'], 0.0, 0.0)
            x, y = gen.to_cartesian(s, d)
            carts.append((x, y))

        dist = jnp.sqrt((carts[0][0] - carts[1][0])**2 + (carts[0][1] - carts[1][1])**2)
        return jnp.maximum(0.0, safe_dist - dist)  # [T] violation per time step

    return collision_g


# ═══════════════════════════════════════════════════════════════════════
# Build per-agent Constran cost
# ═══════════════════════════════════════════════════════════════════════

def build_agent_constran(gen, agent_idx, n_free, v_target, lane_hw=4.0):
    """Constran.build: goal_cost + lane/speed/acc/jerk/collision → single scalar.

    Collision as Deterministic(mode='hard', priority=5) — outer layer, hard baseline.
    Lane/speed/acc/jerk as soft inner layers.
    No hand-tuned weights — Constran σ-nesting handles everything.
    """
    goal_fn = make_goal_cost(gen, agent_idx, n_free, v_target)

    constraints = [
        Deterministic(make_lane_g(gen, agent_idx, n_free, lane_hw),
                      mode='soft', priority=1, aggregate='q95', transform='soft'),
        Deterministic(make_speed_g(gen, agent_idx, n_free),
                      mode='soft', priority=2, aggregate='max', transform='soft'),
        Deterministic(make_acc_g(gen, agent_idx, n_free),
                      mode='soft', priority=3, aggregate='max', transform='soft'),
        Deterministic(make_jerk_g(gen, agent_idx, n_free),
                      mode='soft', priority=4, aggregate='max', transform='soft'),
        Deterministic(make_collision_g(gen, agent_idx, n_free),
                      mode='hard', priority=5, aggregate='max', transform='hard'),
    ]

    return constran_build(goal_fn, constraints, k_inner=0.5)


# ═══════════════════════════════════════════════════════════════════════
# Game fitness (Constran-built per agent)
# ═══════════════════════════════════════════════════════════════════════

def make_game_fitness_constran(gen, v_targets):
    n_free = gen.n_free
    cost_a0 = build_agent_constran(gen, 0, n_free, v_targets[0])
    cost_a1 = build_agent_constran(gen, 1, n_free, v_targets[1])

    def fitness_fn_j(agent_idx, joint_x, ctx):
        return lax.cond(
            agent_idx == 0,
            lambda: cost_a0(joint_x, ctx),
            lambda: cost_a1(joint_x, ctx),
        )
    return fitness_fn_j


# ═══════════════════════════════════════════════════════════════════════
# Warm-start
# ═══════════════════════════════════════════════════════════════════════

def build_game_warmstart(gen, agent_s0, agent_s_dot0, agent_d0, K=3, key=None):
    n_free = gen.n_free; N_blocks = 4
    mu_init = jnp.zeros((N_blocks, K, n_free))
    if key is None: key = random.PRNGKey(0)
    speed_factors = jnp.array([1.15, 1.0, 0.5])
    for a in range(2):
        for k in range(K):
            v_k = agent_s_dot0[a] * speed_factors[k]
            ctrl_s = agent_s0[a] + v_k * gen.greville[2:gen.n_ctrl]
            ctrl_d = jnp.full(n_free, agent_d0[a])
            key, sk = random.split(key)
            noise = random.normal(sk, (n_free,)) * 0.1
            mu_init = mu_init.at[a*2, k].set(ctrl_s + noise)
            mu_init = mu_init.at[a*2+1, k].set(ctrl_d)
    L_inv = jnp.stack([jnp.eye(n_free)] * (N_blocks * K)).reshape(N_blocks, K, n_free, n_free)
    return mu_init, L_inv


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def run(scenario="symmetric", steps=30, seed=42):
    gen = FrenetBSplineTrajectory(BASIS / "bspline_basis.npz", StraightReference())
    n_free = gen.n_free

    if scenario == "symmetric":
        # Both at same s, same speed, same d → symmetric Nash
        v_targets = [18.0, 18.0]
        agent_s0 = [0.0, 0.0]
        agent_s_dot0 = [14.0, 14.0]
        agent_d0 = [-3.0, -3.0]
        label = "SYMMETRIC"
    else:
        # Asymmetric merge
        v_targets = [18.0, 10.0]
        agent_s0 = [0.0, 8.0]
        agent_s_dot0 = [16.0, 8.0]
        agent_d0 = [-3.0, 0.0]
        label = "ASYMMETRIC"

    states = [FrenetState(s=agent_s0[i], s_dot=agent_s_dot0[i], s_ddot=0.0,
                          d=agent_d0[i], d_dot=0.0, d_ddot=0.0, psi=0.0)
              for i in range(2)]

    fitness_fn = make_game_fitness_constran(gen, v_targets)

    N_blocks, M_agent, K = 4, 2, 3
    T_rne, dt_rne, T_0 = 200, 0.15, 100
    B, B0, M_inner = 64, 20, 30
    dims = (n_free,) * 4
    block_to_agent = (0, 0, 1, 1)

    key = random.PRNGKey(seed)
    mu_init, L_inv_init = build_game_warmstart(gen, agent_s0, agent_s_dot0, agent_d0, K, key=key)

    # Warmup
    ctx_warm = {'s0_a0': 0.0, 's_dot0_a0': float(agent_s_dot0[0]), 'd0_a0': float(agent_d0[0]),
                's0_a1': 0.0, 's_dot0_a1': float(agent_s_dot0[1]), 'd0_a1': float(agent_d0[1]),
                's0': 0.0, 's_dot0': float(agent_s_dot0[0]), 's_ddot0': 0.0,
                'd0': float(agent_d0[0]), 'd_dot0': 0.0, 'd_ddot0': 0.0}
    _ = mmog_igo_rne_blocks_solver(
        random.PRNGKey(999), T_rne, dt_rne, N_blocks, M_agent, K, B, B0,
        dims, T_0, fitness_fn, mu_init, L_inv_init, ctx_warm, M_inner, block_to_agent)

    d_hist, v_hist, pi_hist = [[], []], [[], []], [[], []]
    print(f"=== {label} (Constran auto-build, no hand-tuned weights) ===")

    for step in range(steps):
        key, sk = random.split(key)
        game_ctx = {
            's0_a0': float(states[0].s), 's_dot0_a0': float(states[0].s_dot),
            'd0_a0': float(states[0].d),
            's0_a1': float(states[1].s), 's_dot0_a1': float(states[1].s_dot),
            'd0_a1': float(states[1].d),
            's0': float(states[0].s), 's_dot0': float(states[0].s_dot), 's_ddot0': 0.0,
            'd0': float(states[0].d), 'd_dot0': 0.0, 'd_ddot0': 0.0,
        }
        t0 = time.time()
        final_mu, final_L, final_pi, metrics = mmog_igo_rne_blocks_solver(
            sk, T_rne, dt_rne, N_blocks, M_agent, K, B, B0,
            dims, T_0, fitness_fn, mu_init, L_inv_init, game_ctx, M_inner, block_to_agent)
        ms = (time.time() - t0) * 1000

        for a in range(2):
            best_k = int(jnp.argmax(final_pi[a]))
            ctrl_s = final_mu[a*2, best_k]; ctrl_d = final_mu[a*2+1, best_k]
            s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
                ctrl_s, ctrl_d,
                states[a].s, states[a].s_dot, states[a].s_ddot,
                states[a].d, states[a].d_dot, states[a].d_ddot)
            st = gen.to_vehicle_states(s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot)
            states[a] = execute_perfect_tracking(s, d, s_dot, d_dot, s_ddot, d_ddot, st[1, 3])
            d_hist[a].append(float(states[a].d)); v_hist[a].append(float(states[a].s_dot))
            pi_hist[a].append(np.array(final_pi[a]))

        key, wk = random.split(key)
        mu_init, L_inv_init = build_game_warmstart(
            gen, [states[0].s, states[1].s],
            [states[0].s_dot, states[1].s_dot],
            [states[0].d, states[1].d], K, key=wk)

        if step % 5 == 0:
            pi0, pi1 = np.array(final_pi[0]), np.array(final_pi[1])
            print(f"step {step:2d}: d0={states[0].d:+.2f} d1={states[1].d:+.2f} "
                  f"v0={states[0].s_dot:.1f} v1={states[1].s_dot:.1f} "
                  f"π0={pi0} π1={pi1} | {ms:.0f}ms")

    # Summary
    for a, name in enumerate(["Agent 0", "Agent 1"]):
        d = np.array(d_hist[a]); v = np.array(v_hist[a])
        settle = next((i for i in range(5, len(d))
                       if all(abs(d[j]) < 0.1 for j in range(i, min(i+5, len(d))))), steps)
        pi = np.array(pi_hist[a])
        print(f"{name}: d={d[0]:.0f}→{d[-1]:.2f} settle={settle*0.1:.1f}s "
              f"v={v[0]:.0f}→{v[-1]:.1f} π_mean={np.mean(pi, axis=0)}")


if __name__ == "__main__":
    run("symmetric")
    print()
    run("asymmetric")
