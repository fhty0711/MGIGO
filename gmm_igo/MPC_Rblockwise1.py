import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools

# ======================================================================
# I. 基础数值与概率辅助函数 (全向量化)
# ======================================================================

MIN_EIG = 1e-3
MAX_EIG = 1e3
SAFE_A_THRESHOLD = 1e0  # 数值合理性阈值

@jit
def _safe_spd_projection_vmap(S):
    """向量化投影：确保精度矩阵特征值在安全范围内"""
    eigvals, eigvecs = jnp.linalg.eigh(S)
    eigvals = jnp.clip(eigvals, MIN_EIG, MAX_EIG)
    return eigvecs @ (eigvals[..., None] * jnp.swapaxes(eigvecs, -1, -2))

@jit
def _gaussian_log_pdf_masked(xi, mu, S, mask):
    """带掩码的高斯对数概率，支持异构分块维度对齐"""
    # xi: (max_D,), mu: (max_D,), S: (max_D, max_D), mask: (max_D,)
    diff = (xi - mu) * mask
    mahalanobis_sq = jnp.sum(diff * (S @ diff))
    
    # 仅累加有效维度的特征值 logdet
    eigvals = jnp.linalg.eigvalsh(S)
    D_eff = jnp.sum(mask).astype(jnp.int32)
    # 取最大的 D_eff 个特征值（对应有效维度的精度）
    logdet = jnp.sum(jnp.log(jnp.clip(eigvals, 1e-12, None)) * (jnp.arange(S.shape[-1]) >= (S.shape[-1] - D_eff)))
    
    return 0.5 * (logdet - D_eff * jnp.log(2 * jnp.pi) - mahalanobis_sq)

@jit
def _mixture_lpdf_vmap(z_m, m_m, s_m, p_m, msk):
    """向量化计算单个分块内所有样本的混合高斯 log_pdf"""
    # z_m: (N_total, max_D), m_m: (K, max_D)
    # 对样本和分量进行双重映射
    lpdfs = vmap(vmap(_gaussian_log_pdf_masked, in_axes=(None, 0, 0, None)), in_axes=(0, None, None, None))(z_m, m_m, s_m, msk)
    return jnp.logaddexp.reduce(lpdfs + jnp.log(p_m), axis=1)

# ======================================================================
# II. 分块参数更新逻辑 (核心向量化步)
# ======================================================================

@functools.partial(jit, static_argnums=(10, 11))
def _update_single_block(
    mu_t, S_t, pi_t, prev_mu, prev_S, prev_pi, 
    z_all_M, w_hat_global, mask, alpha_t, B_n, B_o
):
    """
    更新单个分块 (Block j) 的所有 K 个分量参数
    z_all_M: (N_total, max_D)
    """
    N_total = B_n + B_o
    is_new_mask = jnp.arange(N_total) < B_n

    # 1. 计算每个分量对每个样本的 a_tilde
    def compute_all_a_tilde(z_single, is_new):
        # 计算该样本在当前所有 K 个分量下的 pdf
        lpdfs_curr = vmap(_gaussian_log_pdf_masked, in_axes=(None, 0, 0, None))(z_single, mu_t, S_t, mask)
        # 计算该样本在采样时分布下的 pdf (用于分母)
        def get_denom(p, m, s):
            return jnp.sum(p * jnp.exp(vmap(_gaussian_log_pdf_masked, in_axes=(None, 0, 0, None))(z_single, m, s, mask)))
            
        denom = lax.cond(is_new, lambda _: get_denom(pi_t, mu_t, S_t), lambda _: get_denom(prev_pi, prev_mu, prev_S), None)
        return jnp.exp(lpdfs_curr) / jnp.clip(denom, 1e-12)

    # a_tilde_all_k: (N_total, K)
    a_tilde_all_k = vmap(compute_all_a_tilde)(z_all_M, is_new_mask)

    # 2. 数值防御：基于 a_tilde 范围过滤非法样本
    # 如果该样本在任何一个分量上的响应爆炸，则该样本不参与本块更新
    local_safe_mask = jnp.all(a_tilde_all_k < SAFE_A_THRESHOLD, axis=1)
    w_block = w_hat_global * local_safe_mask.astype(jnp.float32) # (N_total,)

    # 3. 参数更新 (vmap over K)
    def update_k(k_idx):
        weighted_a = (w_block * a_tilde_all_k[:, k_idx]) / N_total
        diff = (z_all_M - mu_t[k_idx]) * mask
        S_diff = (S_t[k_idx] @ diff.T).T
        
        # S 更新
        term_S = vmap(lambda sd: jnp.outer(sd, sd) - S_t[k_idx])(S_diff)
        S_next_k = _safe_spd_projection_vmap(S_t[k_idx] - alpha_t * jnp.sum(weighted_a[:, None, None] * term_S, axis=0))
        
        # mu 更新
        grad_mu = jnp.sum(weighted_a[:, None] * S_diff, axis=0)
        mu_next_k = mu_t[k_idx] + alpha_t * jnp.linalg.solve(S_next_k, grad_mu)
        
        # delta_v 计算所需的权重增量 (使用 a_tilde_K 作为基准)
        dv_k = jnp.sum(weighted_a) - jnp.sum((w_block * a_tilde_all_k[:, -1]) / N_total)
        return mu_next_k, S_next_k, dv_k

    return vmap(update_k)(jnp.arange(mu_t.shape[0]))

# ======================================================================
# III. Step 迭代函数 (Scan Body)
# ======================================================================

def _step_fn(state, key_input, M, K, B_n, B_o, dt, a_threshold, T_0, fitness_fn, context, max_D, masks):
    mu, S, v, prev_mu, prev_S, prev_pi, old_z, old_f, t = state
    
    is_reset_step = (t % T_0 == 0)
    was_reset_prev = ((t-1) % T_0 == 0)
    
    # 1. 映射得到当前 pi (M, K)
    pi_t = vmap(lambda vk: jnp.concatenate([jnp.exp(vk), jnp.array([1.0])]))(v)
    pi_t = vmap(lambda p: p / jnp.sum(p))(pi_t)
    
    # 2. 全向量化采样
    key, subkey = random.split(key_input)
    def sample_op(m, s, p, msk, k):
        idx = random.choice(k, K, shape=(B_n,), p=p)
        L_sigma = vmap(lambda mat: jnp.linalg.cholesky(jnp.linalg.inv(mat)))(s)
        eps = random.normal(random.split(k, 2)[1], (B_n, max_D))
        return vmap(lambda i, e: m[i] + L_sigma[i] @ e)(idx, eps) * msk

    # (M, B_n, max_D)
    new_z_blocks = vmap(sample_op)(mu, S, pi_t, masks, random.split(subkey, M))
    # 拼接成全维度样本 (B_n, total_D)
    new_z_full = new_z_blocks.transpose(1, 0, 2).reshape(B_n, -1)
    new_f = vmap(fitness_fn, in_axes=(0, None))(new_z_full, context)
    
    all_z_full = jnp.concatenate([new_z_full, old_z], axis=0)
    all_f = jnp.concatenate([new_f, old_f], axis=0)
    
    # 3. 计算全局 omega (无循环)
    # 拆分样本回分块视角 (M, N_total, max_D)
    all_z_split = all_z_full.reshape(N_total := B_n+B_o, M, max_D).transpose(1, 0, 2)
    
    log_p_t = jnp.sum(vmap(_mixture_lpdf_vmap)(all_z_split, mu, S, pi_t, masks), axis=0)
    log_p_prev = jnp.sum(vmap(_mixture_lpdf_vmap)(all_z_split, prev_mu, prev_S, prev_pi, masks), axis=0)
    
    omega_old = jnp.exp(jnp.clip(log_p_t[B_n:] - log_p_prev[B_n:], -10, 10))
    # 重置保护：如果前一步重置，本步不复用旧样本
    omega_old = jnp.where(was_reset_prev, jnp.zeros(B_o), omega_old)
    omega = jnp.concatenate([jnp.ones(B_n), omega_old])
    
    
    # 4. 全局选择 q_vals
    sort_idx = jnp.argsort(all_f)
    q_vals = jnp.zeros(B_n + B_o).at[sort_idx].set(jnp.cumsum(omega[sort_idx]) / (B_n + B_o))
    w_hat_global = (q_vals <= a_threshold).astype(jnp.float32)
    
    # 5. 向量化更新所有块
    # mu_next: (M, K, max_D), S_next: (M, K, max_D, max_D), dv: (M, K)
    mu_next, S_next, dv = vmap(_update_single_block, in_axes=(0,0,0,0,0,0,0,None,0,None,None,None))(
        mu, S, pi_t, prev_mu, prev_S, prev_pi, all_z_split, w_hat_global, masks, dt, B_n, B_o
    )
    
    # 权重更新与重置处理
    v_next = v + dt * dv[:, :-1]
    v_next = jnp.where(is_reset_step, jnp.zeros_like(v), jnp.clip(v_next, -70, 70))
    
    return (mu_next, S_next, v_next, mu, S, pi_t, all_z_full[:B_o], all_f[:B_o], t+1), None

# ======================================================================
# IV. 顶层接口
# ======================================================================

@functools.partial(jit, static_argnames=('T', 'M', 'K', 'B_n', 'B_o', 'T_0', 'dims', 'fitness_fn'))
def blockwise_reuse_optimizer(
    key, T, dt, M, K, B_n, B_o, a_threshold, T_0, dims, fitness_fn, initial_mu, initial_S, initial_pi, context
):
    max_D = max(dims)
    dims_arr = jnp.array(dims)
    masks = (jnp.arange(max_D)[None, :] < dims_arr[:, None]).astype(jnp.float32)
    
    # 准备初始 v (M, K-1)
    v_init = vmap(lambda p: jnp.log(p[:-1] / (p[-1] + 1e-10)))(initial_pi)
    
    key, subkey = random.split(key)
    # 初始旧样本
    old_z = random.normal(subkey, (B_o, sum(dims)))
    old_f = vmap(fitness_fn, in_axes=(0, None))(old_z, context)
    
    init_state = (initial_mu, initial_S, v_init, initial_mu, initial_S, initial_pi, old_z, old_f, 1)
    
    loop_body = functools.partial(_step_fn, M=M, K=K, B_n=B_n, B_o=B_o, dt=dt, 
                                  a_threshold=a_threshold, T_0=T_0, fitness_fn=fitness_fn, 
                                  context=context, max_D=max_D, masks=masks)
    
    final_state, _ = lax.scan(loop_body, init_state, random.split(key, T))
    
    f_mu, f_S, f_v = final_state[0], final_state[1], final_state[2]
    f_pi = vmap(lambda vk: jnp.concatenate([jnp.exp(vk), jnp.array([1.0])]))(f_v)
    f_pi = vmap(lambda p: p / jnp.sum(p))(f_pi)
    
    return f_mu, f_S, f_pi