"""
Diagnose: why MPC_Rblockwise (reuse) produces worse quality than M22.

Runs ONE fixed MPC step on the Cartest B-spline trajectory problem
and tracks per-iteration internal state for both solvers.

Also tests different B_n/B_o splits to find the sweet spot.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax, jax.numpy as jnp
from jax import random, vmap, lax, jit
import numpy as np
import functools

# ── Cartest problem setup ────────────────────────────────────────────
from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.core.reference_path import StraightReference
from Cartest.planning.warmstart import build_initial_mu
from Cartest.planning.cost import make_objective, build_context
from Cartest.planning.constraints import make_constraints
from Cartest.execution.execute import FrenetState
from Cartest.planning.scenarios import get_scenario, make_initial_state, build_obstacle_predictions
from Cartest.planning.costs.default_lyapunov import DEFAULT_CONSTRAINTS

ROOT = Path(__file__).resolve().parents[1] / "Cartest"

# ── Solver internals for non‑jitted diagnostic tracing ───────────────
from gmm_igo.MPCsolverM22 import (
    _safe_spd_projection as _spd_m22,
    _gaussian_log_pdf_l_masked,
)
from gmm_igo.MPC_Rblockwise import (
    _safe_spd_projection as _spd_rb,
    _gaussian_log_pdf,
    _block_mixture_log_pdf,
    _update_block_k,
)

# ══════════════════════════════════════════════════════════════════════

def setup_problem():
    """Build the Cartest problem: gen, cost_fn, ctx, mu_init (one fixed step)."""
    scenario = get_scenario("empty")
    ref_path = scenario["ref_path"]
    gen = FrenetBSplineTrajectory(ROOT / "basis" / "bspline_basis.npz", ref_path)

    road = scenario["road"]
    safety = scenario["safety"]
    lane_hw = road["lane_hw"]
    safe_dist = safety["obs_safe_dist"]
    v_target = scenario["behavior"]["v_target"]
    obs_pos, obs_rad = build_obstacle_predictions(scenario, gen)

    from Constraintdealer.Constran import build as constran_build
    cost_fn = constran_build(
        make_objective(gen, omega_s=1.0, omega_d=4.0, alpha=0.0),
        make_constraints(gen, road, safety, DEFAULT_CONSTRAINTS),
        k_inner=1.0, obj_transform='standard',
    )

    # One fixed initial state
    state = make_initial_state(scenario)
    ctx = build_context(gen, state, v_target, lane_hw, obs_pos, obs_rad)
    mu_init = build_initial_mu(gen, state.s, state.s_dot, state.d)

    return gen, cost_fn, ctx, mu_init, v_target, lane_hw, obs_pos, obs_rad


# ══════════════════════════════════════════════════════════════════════
# Non‑jitted diagnostic step: tracks M22 internals
# ══════════════════════════════════════════════════════════════════════

def trace_m22(key, cost_fn, ctx, mu_init, T, dt, K, B, B0, dims, T_0):
    """Run M22 step-by-step (Python loop) and record diagnostics."""
    M = len(dims)
    D_max = max(dims)
    D_m = dims[0]

    # Init
    L_init = jnp.tile(jnp.eye(D_max), (M, K, 1, 1))
    S_init = vmap(vmap(lambda L: L @ L.T))(L_init[:, :K, :, :])
    v_init = jnp.zeros((M, K - 1))
    mu = mu_init[:, :K, :]
    S = S_init
    v = v_init

    records = []
    keys = random.split(key, T)

    for t_idx in range(T):
        t_val = t_idx + 1  # step counter
        step_key = keys[t_idx]

        # T0 reset
        v = jnp.where((t_val > 0) & ((t_val % T_0) == 0),
                       jnp.zeros_like(v), v)

        # pi from v
        def v_to_pi(v_m):
            exps = jnp.exp(jnp.clip(v_m, -70, 70))
            sum_e = 1.0 + jnp.sum(exps)
            return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
        pi_all = vmap(v_to_pi)(v)  # (M, K)

        # Sample
        key, sk = random.split(step_key)
        def sample_block(m_idx, sub_key):
            comps = random.choice(sub_key, K, p=pi_all[m_idx], shape=(B,))
            def gen_sample(c_idx, s_key):
                cov = jnp.linalg.inv(S[m_idx, c_idx] + jnp.eye(D_max) * 1e-7)
                return random.multivariate_normal(s_key, mu[m_idx, c_idx], cov)
            return vmap(gen_sample)(comps, random.split(sub_key, B))
        samples_m = vmap(sample_block)(jnp.arange(M), random.split(sk, M))
        samples_flat = samples_m.transpose(1, 0, 2).reshape(B, -1)

        # Evaluate
        f_vals = vmap(lambda s: cost_fn(s, ctx))(samples_flat)
        ranks = jnp.argsort(jnp.argsort(f_vals))
        w_hat = jnp.where(ranks < B0, 1.0 / B, 0.0)

        # Update
        def update_block(m_idx):
            D_m_i = dims[m_idx]
            mu_base, S_base = mu[m_idx, K-1], S[m_idx, K-1]

            def upd_core(k_idx, mu_k, S_k):
                S_k = _spd_m22(S_k)
                log_pdf_k = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_k, S_k, D_m_i))(samples_m[m_idx])
                log_pdf_base = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_base, S_base, D_m_i))(samples_m[m_idx])

                def mog_pdf_fn(xi):
                    l_pdfs = vmap(lambda m, s: _gaussian_log_pdf_l_masked(xi, m, s, D_m_i))(mu[m_idx], S[m_idx])
                    return jnp.logaddexp.reduce(jnp.log(pi_all[m_idx] + 1e-15) + l_pdfs)
                log_mog = vmap(mog_pdf_fn)(samples_m[m_idx])

                a_i = jnp.exp(jnp.clip(log_pdf_k - log_mog, -20.0, 20.0))
                b_i = jnp.exp(jnp.clip(log_pdf_base - log_mog, -20.0, 20.0))

                diff = samples_m[m_idx] - mu_k
                def s_grad_fn(d):
                    return S_k @ jnp.outer(d, d) @ S_k - S_k
                sum_S_grad = jnp.sum((w_hat * a_i)[:, None, None] * vmap(s_grad_fn)(diff), axis=0)
                S_new = S_k - dt * sum_S_grad
                S_new = (S_new + S_new.T) / 2.0
                S_new = _spd_m22(S_new)

                grad_mu_terms = (S_k @ diff.T).T
                sum_mu_grad = jnp.sum((w_hat * a_i)[:, None] * grad_mu_terms, axis=0)
                mu_new = mu_k + dt * jnp.linalg.solve(S_new, sum_mu_grad)

                v_delta = jnp.sum(w_hat * (a_i - b_i))
                return mu_new, S_new, v_delta

            new_mu_m, new_S_m, v_deltas = vmap(upd_core, in_axes=(0, 0, 0))(
                jnp.arange(K), mu[m_idx], S[m_idx])
            return new_mu_m, new_S_m, v_deltas[:K-1]

        next_mu, next_S, next_v_deltas = vmap(update_block)(jnp.arange(M))
        next_v = jnp.clip(v + dt * next_v_deltas, -70.0, 70.0)

        # Record
        best_cost = _best_multi_block(next_mu, cost_fn, ctx, M, dims)
        records.append({
            't': t_val, 'best_cost': best_cost,
            'f_min': float(jnp.min(f_vals)), 'f_max': float(jnp.max(f_vals)),
            'pi': np.array(pi_all[0]),
        })

        mu, S, v = next_mu, next_S, next_v

    return records


# ══════════════════════════════════════════════════════════════════════
# Non‑jitted diagnostic step: tracks Rblockwise internals
# ══════════════════════════════════════════════════════════════════════

def trace_rblockwise(key, cost_fn, ctx, mu_init, T, dt, K, B_n, B_o,
                     a_threshold, dims, T_0):
    """Run Rblockwise step-by-step and record per-iteration diagnostics."""
    M = len(dims)
    D_max = max(dims)
    total_D = sum(dims)

    # Init
    S_init = jnp.tile(jnp.eye(D_max) * 4.0, (M, K, 1, 1))
    pi_init = jnp.ones((M, K)) / K
    v_init = vmap(lambda p: jnp.log(p[:-1] / (p[-1] + 1e-10)))(pi_init)

    mu = mu_init[:, :K, :]
    S = S_init
    v = v_init
    prev_mu = mu_init[:, :K, :]
    prev_S = S_init
    prev_pi = pi_init

    # Initial old samples (random normal, matching original init)
    key, sk = random.split(key)
    old_z = random.normal(sk, (B_o, total_D))
    old_f = vmap(lambda z: cost_fn(z, ctx))(old_z)

    records = []
    keys = random.split(key, T)

    for t_idx in range(T):
        t_val = t_idx + 1
        step_key = keys[t_idx]

        is_reset_step = (t_val % T_0 == 0)
        was_reset_prev = ((t_val - 1) % T_0 == 0)

        # pi
        pi_t = vmap(lambda vk: jnp.concatenate([jnp.exp(vk), jnp.array([1.0])]))(v)
        pi_t = vmap(lambda p: p / jnp.sum(p))(pi_t)

        # Sample
        key, sk = random.split(step_key)
        def sample_block(j, d_j):
            c_idx = random.choice(random.split(sk, M)[j], K, shape=(B_n,), p=pi_t[j])
            chol_prec = vmap(jnp.linalg.cholesky)(S[j])
            return vmap(
                lambda i, k: mu[j, i] + jnp.linalg.solve(chol_prec[i].T, random.normal(k, (d_j,)))
            )(c_idx, random.split(sk, B_n))

        new_z_list = [sample_block(j, dims[j]) for j in range(M)]
        new_z = jnp.concatenate(new_z_list, axis=1)
        new_f = vmap(lambda z: cost_fn(z, ctx))(new_z)

        N_total = B_n + B_o
        all_z = jnp.concatenate([new_z, old_z], axis=0)
        all_f = jnp.concatenate([new_f, old_f], axis=0)

        # Omega
        def calc_joint_log_p(z_full, mu_all, S_all, pi_all):
            lp = 0.0; start = 0
            for j in range(M):
                lp += _block_mixture_log_pdf(
                    z_full[start:start+dims[j]], mu_all[j], S_all[j], pi_all[j])
                start += dims[j]
            return lp

        log_p_t = vmap(calc_joint_log_p, in_axes=(0, None, None, None))(all_z, mu, S, pi_t)
        log_p_prev = vmap(calc_joint_log_p, in_axes=(0, None, None, None))(all_z, prev_mu, prev_S, prev_pi)

        omega_old_raw = jnp.exp(jnp.clip(log_p_t[B_n:] - log_p_prev[B_n:], -10, 10))
        omega_old = jnp.where(was_reset_prev, jnp.zeros(B_o), omega_old_raw)
        omega = jnp.concatenate([jnp.ones(B_n), omega_old])

        # w_hat
        sort_idx = jnp.argsort(all_f)
        q_vals = jnp.zeros(N_total).at[sort_idx].set(
            jnp.concatenate([jnp.zeros(1), jnp.cumsum(omega[sort_idx])[:-1]]) / N_total)
        w_hat = (q_vals < a_threshold).astype(jnp.float32)

        # Block updates
        mu_list, S_list, v_list = [], [], []
        start = 0
        new_f_vmap_vals = new_f   # (B_n,)

        for j in range(M):
            z_j_all = all_z[:, start:start + dims[j]]
            upd_vmap = vmap(_update_block_k, in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None))
            m_n, s_n, dv = upd_vmap(
                jnp.arange(K), mu[j], S[j], z_j_all, w_hat,
                pi_t[j], mu[j], S[j], prev_pi[j], prev_mu[j], prev_S[j],
                dt, B_n, B_o)
            vn = lax.cond(is_reset_step,
                          lambda _: jnp.zeros_like(v[j]),
                          lambda _: jnp.clip(v[j] + dt * dv[:-1], -70.0, 70.0), None)
            m_n = jnp.clip(m_n, -50.0, 50.0)
            s_n = vmap(_spd_rb)(s_n)
            mu_list.append(m_n); S_list.append(s_n); v_list.append(vn)
            start += dims[j]

        mu_next = jnp.stack(mu_list); S_next = jnp.stack(S_list)
        v_next = jnp.stack(v_list)

        # ── diagnostics ──
        # Omega stats
        omega_arr = np.array(omega)
        ess = float(jnp.sum(omega)**2 / jnp.maximum(jnp.sum(omega**2), 1e-10))
        n_elite = int(jnp.sum(w_hat))
        new_elite = int(jnp.sum(w_hat[:B_n]))
        old_elite = int(jnp.sum(w_hat[B_n:]))
        omega_old_mean = float(jnp.mean(omega_old_raw))

        # a_tilde stats (block 0, component 0)
        z0 = all_z[:, :dims[0]]
        log_p0 = vmap(lambda z: _gaussian_log_pdf(z, mu[0, 0], S[0, 0]))(z0)
        log_pK = vmap(lambda z: _gaussian_log_pdf(z, mu[0, K-1], S[0, K-1]))(z0)

        def get_ld_all(z_single, idx):
            def ct():
                lpdfs = vmap(lambda m, s: _gaussian_log_pdf(z_single, m, s))(mu[0], S[0])
                return jnp.log(jnp.sum(pi_t[0] * jnp.exp(lpdfs)) + 1e-15)
            def cp():
                lpdfs = vmap(lambda m, s: _gaussian_log_pdf(z_single, m, s))(prev_mu[0], prev_S[0])
                return jnp.log(jnp.sum(prev_pi[0] * jnp.exp(lpdfs)) + 1e-15)
            return lax.cond(idx < B_n, lambda _: ct(), lambda _: cp(), None)
        log_denoms = vmap(get_ld_all, in_axes=(0, 0))(z0, jnp.arange(N_total))
        a_tilde_0 = jnp.exp(jnp.clip(log_p0 - log_denoms, -20.0, 20.0))
        a_tilde_max = float(jnp.max(a_tilde_0))
        a_tilde_max_new = float(jnp.max(a_tilde_0[:B_n]))
        a_tilde_max_old = float(jnp.max(a_tilde_0[B_n:]))

        # S condition numbers
        conds = [float(jnp.linalg.cond(S_next[0, k])) for k in range(K)]

        # Cost
        best_cost = _best_multi_block(mu_next, cost_fn, ctx, M, dims)

        records.append({
            't': t_val, 'best_cost': best_cost,
            'f_min': float(jnp.min(all_f)), 'f_max': float(jnp.max(all_f)),
            'n_elite': n_elite, 'new_elite': new_elite, 'old_elite': old_elite,
            'omega_mean': float(jnp.mean(omega)),
            'omega_max': float(jnp.max(omega)),
            'omega_old_mean': omega_old_mean,
            'ess': ess,
            'a_tilde_max': a_tilde_max,
            'a_tilde_max_new': a_tilde_max_new,
            'a_tilde_max_old': a_tilde_max_old,
            'S_cond_max': max(conds),
            'pi': np.array(pi_t[0]),
        })

        mu, S, v = mu_next, S_next, v_next
        prev_mu, prev_S, prev_pi = mu, S, pi_t
        old_z, old_f = all_z[:B_o], all_f[:B_o]

    return records


# ══════════════════════════════════════════════════════════════════════
# Tracer using jitted solvers (fast, final result only)
# ══════════════════════════════════════════════════════════════════════

def _best_multi_block(mu, cost_fn, ctx, M, dims):
    """Evaluate all K^M component-mean combinations, return best cost."""
    K_ = mu.shape[1]
    best_c = jnp.inf
    for i0 in range(K_):
        parts0 = [mu[0, i0, :dims[0]]]
        if M == 1:
            c = float(cost_fn(parts0[0], ctx))
            if c < best_c: best_c = c
        else:
            for i1 in range(K_):
                parts = parts0 + [mu[1, i1, :dims[1]]]
                x_i = jnp.concatenate(parts)
                c = float(cost_fn(x_i, ctx))
                if c < best_c: best_c = c
    return best_c


def run_m22_jitted(key, cost_fn, ctx, mu_init, T, dt, K, B, B0, dims, T_0):
    from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
    M = len(dims); D_max = max(dims)
    L_init = jnp.tile(jnp.eye(D_max), (M, K, 1, 1))
    v_init = jnp.zeros((M, K - 1))
    t0 = time.time()
    fm, fL, fpi = mmog_igo_optimizer_mpc(
        key, T, dt, M, K, B, B0, dims, T_0,
        cost_fn, mu_init[:, :K, :], L_init, v_init, ctx)
    jax.block_until_ready((fm, fL, fpi))
    elapsed = time.time() - t0
    return _best_multi_block(fm, cost_fn, ctx, M, dims), elapsed


def run_rb_jitted(key, cost_fn, ctx, mu_init, T, dt, K, B_n, B_o,
                  a_threshold, dims, T_0):
    from gmm_igo.MPC_Rblockwise import blockwise_reuse_optimizer
    M = len(dims); D_max = max(dims)
    S_init = jnp.tile(jnp.eye(D_max) * 4.0, (M, K, 1, 1))
    pi_init = jnp.ones((M, K)) / K
    t0 = time.time()
    fm, fS, fpi = blockwise_reuse_optimizer(
        key, T, dt, M, K, B_n, B_o, a_threshold, T_0, dims,
        cost_fn, mu_init[:, :K, :], S_init, pi_init, ctx)
    jax.block_until_ready((fm, fS, fpi))
    elapsed = time.time() - t0
    return _best_multi_block(fm, cost_fn, ctx, M, dims), elapsed


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    gen, cost_fn, ctx, mu_init, v_target, lane_hw, obs_pos, obs_rad = setup_problem()
    dims = (gen.n_free, gen.n_free)  # (9, 9)

    T = 300; dt = 0.3; K = 3; T_0 = 300

    # Common config
    B_total = 64
    B0_m22 = 30
    a_th = B0_m22 / B_total  # ≈ 0.47

    print(f"{'='*90}")
    print(f"Cartest B-spline trajectory — fixed MPC step, T={T}, dt={dt}, K={K}")
    print(f"M22: B={B_total}, B0={B0_m22}")
    print(f"{'='*90}")

    # ── Jitted comparison (fast) ────────────────────────────────────
    print(f"\n── JITTED SOLVER COMPARISON (cost + wall time) ──")
    for label, Bn, Bo in [("M22", 64, 0), ("RB_n32_o32", 32, 32),
                           ("RB_n48_o16", 48, 16), ("RB_n16_o48", 16, 48)]:
        costs = []; times = []
        for trial in range(3):
            key = random.PRNGKey(42 + trial * 100)
            if label == "M22":
                c, elapsed = run_m22_jitted(key, cost_fn, ctx, mu_init, T, dt, K,
                                            B_total, B0_m22, dims, T_0)
            else:
                c, elapsed = run_rb_jitted(key, cost_fn, ctx, mu_init, T, dt, K,
                                           Bn, Bo, a_th, dims, T_0)
            costs.append(c); times.append(elapsed)
        c_arr = np.array(costs); t_arr = np.array(times)
        print(f"  {label:15s}: cost={np.mean(c_arr):.6f}±{np.std(c_arr):.4f}  "
              f"best={np.min(c_arr):.6f}  time={np.mean(t_arr):.3f}s")

    # ── NON-JITTED trace (detailed diagnostics) ──────────────────────
    print(f"\n── INTERNAL STATE TRACE ──")
    print(f"Tracing M22 (B={B_total}) and RB (B_n=32, B_o=32) step-by-step...")

    key = random.PRNGKey(42)
    m22_recs = trace_m22(key, cost_fn, ctx, mu_init, T, dt, K,
                          B_total, B0_m22, dims, T_0)

    key = random.PRNGKey(42)
    rb_recs = trace_rblockwise(key, cost_fn, ctx, mu_init, T, dt, K,
                                32, 32, a_th, dims, T_0)

    # Print every 30 steps
    print(f"\n  {'Step':>4s} | {'M22 cost':>10s} | {'RB cost':>10s} | "
          f"{'omega_μ':>8s} | {'omega_max':>9s} | {'ESS':>6s} | "
          f"{'elite N/O':>10s} | {'a_tilde':>8s} | {'S_cond':>7s}")
    print("  " + "-" * 95)
    for i in range(0, T, 30):
        rm = m22_recs[i]; rr = rb_recs[i]
        elite_str = f"{rr['new_elite']}/{rr['old_elite']}"
        print(f"  {rm['t']:4d} | {rm['best_cost']:10.4f} | {rr['best_cost']:10.4f} | "
              f"{rr['omega_mean']:8.3f} | {rr['omega_max']:9.2f} | {rr['ess']:6.1f} | "
              f"{elite_str:>10s} | {rr['a_tilde_max']:8.2f} | {rr['S_cond_max']:7.1f}")

    # Summary
    m22_final = m22_recs[-1]['best_cost']
    rb_final = rb_recs[-1]['best_cost']
    rb_ess_mean = np.mean([r['ess'] for r in rb_recs])
    rb_atilde_mean = np.mean([r['a_tilde_max'] for r in rb_recs])
    elite_new_frac = np.mean([r['new_elite'] / max(r['n_elite'], 1)
                               for r in rb_recs])

    print(f"\n── SUMMARY ──")
    print(f"  M22 final cost:     {m22_final:.6f}")
    print(f"  RB  final cost:     {rb_final:.6f}")
    print(f"  RB mean ESS:        {rb_ess_mean:.1f} / 64 = {rb_ess_mean/64:.2%}")
    print(f"  RB mean a_tilde_max: {rb_atilde_mean:.2f}")
    print(f"  RB elite new frac:   {elite_new_frac:.2%} (lower = old samples dominate)")
