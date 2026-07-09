"""Level 2a: Minimal B-spline + MPC_G_MS integration.

2 agents on straight road, each converging to lane center.
No collision — pure verification that B-spline + RNE work together.
"""

from __future__ import annotations

import sys, time
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.core.reference_path import StraightReference
# (use gen.evaluate directly — simpler than importing cost internals)
from Cartest.execution.execute import execute_perfect_tracking, FrenetState
from gmm_igo.MPC_G_MS import mmog_igo_rne_blocks_solver

ROOT = Path(__file__).resolve().parent
BASIS = ROOT / "basis"

# ═══════════════════════════════════════════════════════════════════════
# B-spline + Lyapunov cost (same as cost.py, for per-agent evaluation)
# ═══════════════════════════════════════════════════════════════════════

def make_agent_cost(gen, omega_s=1.0, omega_d=4.0, alpha=0.0):
    """Per-agent Lyapunov cost, reads z_ref from ctx if present."""
    t_arr = jnp.arange(gen.T) * gen.dt

    def agent_cost(theta, ctx):
        n = gen.n_free
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
            theta[:n], theta[n:2*n],
            ctx['s0'], ctx['s_dot0'], ctx['s_ddot0'],
            ctx['d0'], ctx['d_dot0'], ctx['d_ddot0'])

        z_ref = ctx.get('z_ref')
        if z_ref is not None:
            s_ref = z_ref['s_ref']; s_dot_ref = z_ref['s_dot_ref']
            s_ddot_ref = z_ref['s_ddot_ref']
            d_ref = z_ref['d_ref']; d_dot_ref = z_ref['d_dot_ref']
            d_ddot_ref = z_ref['d_ddot_ref']
        else:
            v_tgt = ctx["v_ref"][0]; s0 = ctx["s0"]; v0 = ctx["s_dot0"]
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

        K00, K01 = omega_s, alpha
        K10, K11 = alpha, omega_d
        K2_00 = K00**2 + K01*K10; K2_01 = K00*K01 + K01*K11
        K2_10 = K10*K00 + K11*K10; K2_11 = K10*K01 + K11**2

        term0 = es**2 + ed**2
        v1_s = es_dot + K00*es + K01*ed
        v1_d = ed_dot + K10*es + K11*ed
        term1 = v1_s**2 + v1_d**2
        v2_s = es_ddot + 2.0*(K00*es_dot + K01*ed_dot) + K2_00*es + K2_01*ed
        v2_d = ed_ddot + 2.0*(K10*es_dot + K11*ed_dot) + K2_10*es + K2_11*ed
        term2 = v2_s**2 + v2_d**2

        return jnp.sum(term0) + jnp.sum(term1) + jnp.sum(term2)
    return agent_cost


# ═══════════════════════════════════════════════════════════════════════
# Game fitness function
# ═══════════════════════════════════════════════════════════════════════

def make_game_fitness(gen, all_init_states, v_targets):
    """Build fitness_fn_j for MPC_G_MS Blocks solver.

    joint_x_flat layout (N_blocks=4):
      [agent0_s(n_free), agent0_d(n_free), agent1_s(n_free), agent1_d(n_free)]
    """
    n_free = gen.n_free
    agent_cost_0 = make_agent_cost(gen)
    agent_cost_1 = make_agent_cost(gen)

    v_ref_0 = jnp.array([v_targets[0]])
    v_ref_1 = jnp.array([v_targets[1]])

    def fitness_fn_j(agent_idx, joint_x_flat, ctx):
        """ctx is a dict with keys: s0_a0, s_dot0_a0, d0_a0, s0_a1, s_dot0_a1, d0_a1"""
        from jax import lax

        theta_0 = joint_x_flat[0 : 2*n_free]
        theta_1 = joint_x_flat[2*n_free : 4*n_free]

        # Build per-agent ctx from the dynamic context dict
        def eval_a0():
            c = {'s0': ctx['s0_a0'], 's_dot0': ctx['s_dot0_a0'], 's_ddot0': 0.0,
                 'd0': ctx['d0_a0'], 'd_dot0': 0.0, 'd_ddot0': 0.0, 'v_ref': v_ref_0}
            return agent_cost_0(theta_0, c)

        def eval_a1():
            c = {'s0': ctx['s0_a1'], 's_dot0': ctx['s_dot0_a1'], 's_ddot0': 0.0,
                 'd0': ctx['d0_a1'], 'd_dot0': 0.0, 'd_ddot0': 0.0, 'v_ref': v_ref_1}
            return agent_cost_1(theta_1, c)

        return lax.cond(agent_idx == 0, eval_a0, eval_a1)

    return fitness_fn_j


# ═══════════════════════════════════════════════════════════════════════
# Warm-start
# ═══════════════════════════════════════════════════════════════════════

def build_game_warmstart(gen, agent_s0, agent_s_dot0, agent_d0, K=3, key=None):
    """Initial GMM mu: each component has different speed (aggressive/neutral/conservative)."""
    n_free = gen.n_free
    N_blocks = 4  # 2 agents × 2 channels
    mu_init = jnp.zeros((N_blocks, K, n_free))

    if key is None:
        key = random.PRNGKey(0)
    speed_factors = jnp.array([1.15, 1.0, 0.85])  # aggressive, neutral, conservative

    for a in range(2):
        for k in range(K):
            v_k = agent_s_dot0[a] * speed_factors[k]
            ctrl_s = agent_s0[a] + v_k * gen.greville[2:gen.n_ctrl]
            ctrl_d = jnp.full(n_free, agent_d0[a])
            # Small random perturbation so components aren't identical
            key, sk = random.split(key)
            noise = random.normal(sk, (n_free,)) * 0.1
            mu_init = mu_init.at[a*2,   k].set(ctrl_s + noise)
            mu_init = mu_init.at[a*2+1, k].set(ctrl_d)

    L_inv = jnp.stack([jnp.eye(n_free)] * (N_blocks * K)).reshape(N_blocks, K, n_free, n_free)
    return mu_init, L_inv


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def run(steps=30, seed=42):
    ref_path = StraightReference()
    gen = FrenetBSplineTrajectory(BASIS / "bspline_basis.npz", ref_path)
    n_free = gen.n_free

    # ── Scenario: two cars lane-change toward center ──
    v_targets = [18.0, 18.0]
    agent_s0 = [0.0, 0.0]
    agent_s_dot0 = [12.0, 12.0]
    agent_d0 = [-3.0, 3.0]   # Agent0 right lane, Agent1 left lane

    init_states = list(zip(agent_s0, agent_s_dot0, agent_d0))

    # States
    states = [
        FrenetState(s=agent_s0[0], s_dot=agent_s_dot0[0], s_ddot=0.0,
                    d=agent_d0[0], d_dot=0.0, d_ddot=0.0, psi=0.0),
        FrenetState(s=agent_s0[1], s_dot=agent_s_dot0[1], s_ddot=0.0,
                    d=agent_d0[1], d_dot=0.0, d_ddot=0.0, psi=0.0),
    ]

    # ── Game solver config ──
    N_blocks = 4
    M_agent = 2
    K = 3
    B = 64; B0 = 20
    T_rne = 200; dt_rne = 0.15
    T_0 = 100
    M_inner = 30
    dims = (n_free, n_free, n_free, n_free)
    block_to_agent = (0, 0, 1, 1)

    fitness_fn = make_game_fitness(gen, init_states, v_targets)

    key = random.PRNGKey(seed)
    mu_init, L_inv_init = build_game_warmstart(gen, agent_s0, agent_s_dot0, agent_d0, K, key=key)

    # Warmup: one RNE call to trigger JIT
    ctx_warm = {'s0_a0': 0.0, 's_dot0_a0': 12.0, 'd0_a0': -3.0,
                's0_a1': 0.0, 's_dot0_a1': 12.0, 'd0_a1': 3.0}
    _ = mmog_igo_rne_blocks_solver(
        random.PRNGKey(999), T_rne, dt_rne, N_blocks, M_agent, K, B, B0,
        dims, T_0, fitness_fn, mu_init, L_inv_init, ctx_warm, M_inner, block_to_agent)

    d_hist = [[], []]
    v_hist = [[], []]
    pi_hist = [[], []]

    for step in range(steps):
        key, sk = random.split(key)
        t0 = time.time()

        game_ctx = {'s0_a0': float(states[0].s), 's_dot0_a0': float(states[0].s_dot),
                    'd0_a0': float(states[0].d),
                    's0_a1': float(states[1].s), 's_dot0_a1': float(states[1].s_dot),
                    'd0_a1': float(states[1].d)}
        final_mu, final_L, final_pi, metrics = mmog_igo_rne_blocks_solver(
            sk, T_rne, dt_rne, N_blocks, M_agent, K, B, B0,
            dims, T_0, fitness_fn, mu_init, L_inv_init, game_ctx, M_inner, block_to_agent)

        ms = (time.time() - t0) * 1000

        # Extract best component per agent (max π)
        for a in range(2):
            best_k = int(jnp.argmax(final_pi[a]))
            ctrl_s = final_mu[a*2, best_k]
            ctrl_d = final_mu[a*2+1, best_k]

            # Evaluate plan → execute t=1 state
            s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
                ctrl_s, ctrl_d,
                states[a].s, states[a].s_dot, states[a].s_ddot,
                states[a].d, states[a].d_dot, states[a].d_ddot)
            st = gen.to_vehicle_states(
                s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot)

            states[a] = execute_perfect_tracking(
                s, d, s_dot, d_dot, s_ddot, d_ddot, st[1, 3])

            d_hist[a].append(float(states[a].d))
            v_hist[a].append(float(states[a].s_dot))
            pi_hist[a].append(np.array(final_pi[a]))

        # Warm-start next step: shift best mu forward
        key, wk = random.split(key)
        mu_init, L_inv_init = build_game_warmstart(
            gen,
            [states[0].s, states[1].s],
            [states[0].s_dot, states[1].s_dot],
            [states[0].d, states[1].d], K, key=wk)

        if step % 5 == 0:
            pi0 = np.array(final_pi[0])
            pi1 = np.array(final_pi[1])
            print(f"step {step:2d} | d0={states[0].d:+.2f} d1={states[1].d:+.2f} "
                  f"v0={states[0].s_dot:.1f} v1={states[1].s_dot:.1f} "
                  f"π0={pi0} π1={pi1} | {ms:.0f}ms")

    # Final summary
    d0 = np.array(d_hist[0]); d1 = np.array(d_hist[1])
    settle0 = next((i for i in range(5, len(d0))
                    if all(abs(d0[j]) < 0.1 for j in range(i, min(i+5, len(d0))))), steps)
    settle1 = next((i for i in range(5, len(d1))
                    if all(abs(d1[j]) < 0.1 for j in range(i, min(i+5, len(d1))))), steps)
    pi0 = np.array(pi_hist[0]); pi1 = np.array(pi_hist[1])
    print(f"\nAgent 0: d={d0[0]:.0f}→{d0[-1]:.2f} settle={settle0*0.1:.1f}s "
          f"v={v_hist[0][0]:.0f}→{v_hist[0][-1]:.1f} "
          f"π std={float(np.std(pi0[:,0])):.3f}")
    print(f"Agent 1: d={d1[0]:.0f}→{d1[-1]:.2f} settle={settle1*0.1:.1f}s "
          f"v={v_hist[1][0]:.0f}→{v_hist[1][-1]:.1f} "
          f"π std={float(np.std(pi1[:,0])):.3f}")
    print(f"π stability: {'STABLE' if max(np.std(pi0[:,0]), np.std(pi1[:,0])) < 0.1 else 'OSCILLATING'}")


if __name__ == "__main__":
    run()
