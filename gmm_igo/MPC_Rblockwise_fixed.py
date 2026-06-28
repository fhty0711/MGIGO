"""
Patched MPC_Rblockwise with stability fixes.

Fixes applied vs original MPC_Rblockwise.py:
  F1: Omega normalization (prevents cumulative omega drift)
  F2: was_reset_prev sets omega_old=0 instead of 1 (correct IS weight)
  F3: Old-z initialized from GMM, not random.normal
  F4: Log-space a_i clipping instead of hard safe_tilde_a cutoff
  F5: S update norm clipping (from MPC_ReuseOldSamples.py)
  F6: MIN_EIG raised to 1e-2 (matching MPC_ReuseOldSamples.py)
"""

import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools

# ======================================================================
# I. 基础数值与概率辅助函数
# ======================================================================

MIN_EIG = 1e-2      # F6: raised from 1e-3
MAX_EIG = 1e3
MAX_S_UPDATE_NORM = 1e4  # F5: gradient clipping

def _safe_spd_projection(S):
    eigvals, eigvecs = jnp.linalg.eigh(S)
    eigvals = jnp.clip(eigvals, MIN_EIG, MAX_EIG)
    return eigvecs @ (eigvals[:, None] * eigvecs.T)

@jit
def _gaussian_log_pdf(xi, mu, S):
    """基于精度矩阵 S 计算对数概率"""
    D = mu.shape[0]
    diff = xi - mu
    mahalanobis_sq = jnp.sum(diff * (S @ diff))
    sign, logdet = jnp.linalg.slogdet(S)
    return 0.5 * (logdet - D * jnp.log(2 * jnp.pi) - mahalanobis_sq)

@jit
def _block_mixture_log_pdf(z_j, mu_j, S_j, pi_j):
    """计算单个分块的混合高斯对数概率"""
    log_pdfs = vmap(lambda m, s: _gaussian_log_pdf(z_j, m, s))(mu_j, S_j)
    return jnp.logaddexp.reduce(jnp.log(pi_j) + log_pdfs)

# ======================================================================
# II. 分块更新步 (核心逻辑) - 修复版
# ======================================================================

@jit
def _update_block_k_fixed(
    k_idx, mu_jk_t, S_jk_t, z_j_all, w_hat,
    pi_j_curr, mu_j_curr, S_j_curr,
    pi_j_prev, mu_j_prev, S_j_prev,
    alpha_t, B_n, B_o
):
    """
    F4: 使用 log-space a_i clipping 替代 hard safe_tilde_a 截断。
    对齐 MPCsolverM22 的做法: a_i = exp(clip(log_pdf_k - log_mog, -20, 20))
    """
    N_total = z_j_all.shape[0]
    sample_indices = jnp.arange(N_total)

    # 1. 分子: log p_k(z)
    log_p_jk_t = vmap(lambda z: _gaussian_log_pdf(z, mu_jk_t, S_jk_t))(z_j_all)

    # 2. 分母: log denom(z)
    def get_log_denom(idx, z_single):
        def compute_t():
            lpdfs = vmap(lambda m, s: _gaussian_log_pdf(z_single, m, s))(mu_j_curr, S_j_curr)
            return jnp.log(jnp.sum(pi_j_curr * jnp.exp(lpdfs)) + 1e-15)
        def compute_prev():
            lpdfs = vmap(lambda m, s: _gaussian_log_pdf(z_single, m, s))(mu_j_prev, S_j_prev)
            return jnp.log(jnp.sum(pi_j_prev * jnp.exp(lpdfs)) + 1e-15)
        return lax.cond(idx < B_n, lambda _: compute_t(), lambda _: compute_prev(), None)

    log_denoms = vmap(get_log_denom, in_axes=(0, 0))(sample_indices, z_j_all)

    # F4: clip in log-space, just like M22's a_i = exp(clip(log_pdf - log_mog, -20, 20))
    log_a_tilde_jk = jnp.clip(log_p_jk_t - log_denoms, -20.0, 20.0)
    a_tilde_jk = jnp.exp(log_a_tilde_jk)

    log_pdf_jK = vmap(lambda z: _gaussian_log_pdf(z, mu_j_curr[-1], S_j_curr[-1]))(z_j_all)
    log_a_tilde_jK = jnp.clip(log_pdf_jK - log_denoms, -20.0, 20.0)
    a_tilde_jK = jnp.exp(log_a_tilde_jK)

    # 不再使用 safe_tilde_a 硬截断 — log-space clipping 已经保证数值安全
    w_block = w_hat

    # 3. 参数更新
    weighted_a = (w_block * a_tilde_jk) / N_total
    diff = z_j_all - mu_jk_t
    S_diff = (S_jk_t @ diff.T).T

    # F5: S update with norm clipping (from MPC_ReuseOldSamples.py)
    term_S = vmap(lambda sd: jnp.outer(sd, sd) - S_jk_t)(S_diff)
    S_update = jnp.sum(weighted_a[:, None, None] * term_S, axis=0)
    S_up_norm = jnp.linalg.norm(S_update)
    S_update = jnp.where(
        S_up_norm > MAX_S_UPDATE_NORM,
        S_update * (MAX_S_UPDATE_NORM / (S_up_norm + 1e-9)),
        S_update,
    )
    S_next = _safe_spd_projection(S_jk_t - alpha_t * S_update)

    grad_mu = jnp.sum(weighted_a[:, None] * S_diff, axis=0)
    mu_next = mu_jk_t + alpha_t * jnp.linalg.solve(S_next, grad_mu)

    delta_pi = jnp.sum(weighted_a) - jnp.sum((w_block * a_tilde_jK) / N_total)

    return mu_next, S_next, delta_pi


# ======================================================================
# III. 迭代步与优化器 - 修复版
# ======================================================================

def _step_fn_reuse_fixed(state, key_input, M, K, B_n, B_o, dt, a_threshold, T_0, fitness_fn, context, dims):
    mu, S, v, prev_mu, prev_S, prev_pi, old_z, old_f, t = state

    is_reset_step = (t % T_0 == 0)
    was_reset_prev = ((t-1) % T_0 == 0)

    # 1. 准备 pi
    pi_t = vmap(lambda vk: jnp.concatenate([jnp.exp(vk), jnp.array([1.0])]))(v)
    pi_t = vmap(lambda p: p / jnp.sum(p))(pi_t)

    # 2. 分块采样
    key, subkey = random.split(key_input)
    def sample_block(j, d_j):
        c_idx = random.choice(random.split(subkey, M)[j], K, shape=(B_n,), p=pi_t[j])
        chol_prec = vmap(jnp.linalg.cholesky)(S[j])
        return vmap(
            lambda i, k: mu[j, i] + jnp.linalg.solve(chol_prec[i].T, random.normal(k, (d_j,)))
        )(c_idx, random.split(subkey, B_n))

    new_z_list = [sample_block(j, dims[j]) for j in range(M)]
    new_z = jnp.concatenate(new_z_list, axis=1)
    new_f = vmap(fitness_fn, in_axes=(0, None))(new_z, context)

    all_z = jnp.concatenate([new_z, old_z], axis=0)
    all_f = jnp.concatenate([new_f, old_f], axis=0)

    # 3. 计算全局 omega
    def calc_joint_log_p(z_full, mu_all, S_all, pi_all):
        lp = 0.0
        start = 0
        for j in range(M):
            lp += _block_mixture_log_pdf(z_full[start:start+dims[j]], mu_all[j], S_all[j], pi_all[j])
            start += dims[j]
        return lp

    log_p_t = vmap(calc_joint_log_p, in_axes=(0, None, None, None))(all_z, mu, S, pi_t)
    log_p_prev = vmap(calc_joint_log_p, in_axes=(0, None, None, None))(all_z, prev_mu, prev_S, prev_pi)

    omega_old = jnp.exp(jnp.clip(log_p_t[B_n:] - log_p_prev[B_n:], -10, 10))
    # F2: after reset, old samples should get zero weight (not 1)
    omega_old = jnp.where(was_reset_prev, jnp.zeros(B_o), omega_old)
    omega = jnp.concatenate([jnp.ones(B_n), omega_old])

    # F1: normalize omega (prevents cumulative drift)
    omega = omega / (jnp.mean(omega) + 1e-10)

    # 4. 排序与 w_hat
    sort_idx = jnp.argsort(all_f)
    q_vals = jnp.zeros(B_n + B_o).at[sort_idx].set(
        jnp.concatenate([jnp.zeros(1), jnp.cumsum(omega[sort_idx])[:-1]]) / (B_n + B_o)
    )
    w_hat = (q_vals < a_threshold).astype(jnp.float32)

    # 5. 并行更新各块
    mu_list, S_list, v_list = [], [], []
    start = 0
    for j in range(M):
        z_j_all = all_z[:, start:start + dims[j]]
        upd_vmap = vmap(_update_block_k_fixed, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None))
        m_n, s_n, dv = upd_vmap(
            jnp.arange(K), mu[j], S[j], z_j_all, w_hat,
            pi_t[j], mu[j], S[j], prev_pi[j], prev_mu[j], prev_S[j],
            dt, B_n, B_o
        )
        # F1+: clip v_deltas to prevent weight explosion
        dv_clipped = jnp.clip(dv[:-1], -10.0, 10.0)
        vn = lax.cond(
            is_reset_step,
            lambda _: jnp.zeros_like(v[j]),
            lambda _: jnp.clip(v[j] + dt * dv_clipped, -70.0, 70.0),
            None,
        )
        m_n = jnp.clip(m_n, -50.0, 50.0)
        s_n = vmap(_safe_spd_projection)(s_n)
        mu_list.append(m_n); S_list.append(s_n); v_list.append(vn)
        start += dims[j]

    mu_next = jnp.stack(mu_list); S_next = jnp.stack(S_list); v_next = jnp.stack(v_list)

    return (mu_next, S_next, v_next, mu, S, pi_t, new_z, new_f, t+1), None


# ======================================================================
# IV. 顶层入口 - 修复版
# ======================================================================

@functools.partial(jit, static_argnames=('T', 'M', 'K', 'B_n', 'B_o', 'T_0', 'dims', 'fitness_fn'))
def blockwise_reuse_optimizer_fixed(key, T, dt, M, K, B_n, B_o, a_threshold, T_0, dims, fitness_fn, initial_mu, initial_S, initial_pi, context):
    total_D = sum(dims)
    v_init = vmap(lambda p: jnp.log(p[:-1] / (p[-1] + 1e-10)))(initial_pi)

    # F3: Initialize old_z from the actual GMM (not random.normal)
    key, subkey = random.split(key)
    def sample_from_gmm(key_j, mu_j, S_j, pi_j, d_j):
        """Sample B_o points from block j's GMM."""
        c_idx = random.choice(key_j, K, shape=(B_o,), p=pi_j)
        chol_prec = vmap(jnp.linalg.cholesky)(S_j)
        return vmap(
            lambda i, k: mu_j[i] + jnp.linalg.solve(chol_prec[i].T, random.normal(k, (d_j,)))
        )(c_idx, random.split(key_j, B_o))

    old_z_blocks = []
    for j in range(M):
        key, subkey = random.split(key)
        old_z_blocks.append(sample_from_gmm(subkey, initial_mu[j], initial_S[j], initial_pi[j], dims[j]))
    old_z = jnp.concatenate(old_z_blocks, axis=1)
    old_f = vmap(fitness_fn, in_axes=(0, None))(old_z, context)

    # 初始状态
    init_state = (initial_mu, initial_S, v_init, initial_mu, initial_S, initial_pi, old_z, old_f, 1)

    loop_body = functools.partial(_step_fn_reuse_fixed, M=M, K=K, B_n=B_n, B_o=B_o, dt=dt,
                                  a_threshold=a_threshold, T_0=T_0, fitness_fn=fitness_fn, context=context, dims=dims)

    final_state, _ = lax.scan(loop_body, init_state, random.split(key, T))

    f_mu, f_S, f_v = final_state[0], final_state[1], final_state[2]
    f_pi = vmap(lambda vk: jnp.concatenate([jnp.exp(vk), jnp.array([1.0])]))(f_v)
    f_pi = vmap(lambda p: p / jnp.sum(p))(f_pi)

    return f_mu, f_S, f_pi
