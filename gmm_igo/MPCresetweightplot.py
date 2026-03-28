# MPCsolverM2_v3.py - 严格对齐 Algorithm 3 & 4 并支持历史记录记录
import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools

# ======================================================================
# I. 核心辅助函数 (信息几何范式)
# ======================================================================

@jit
def _logsumexp(a, axis=None):
    """数值稳定的 log(sum(exp(a)))"""
    return jnp.logaddexp.reduce(a, axis=axis)

@jit
def _gaussian_log_pdf_l_masked(xi, mu, L_inv, D_m):
    """计算对数概率密度，支持异构维度掩码"""
    diff = (xi - mu)
    mask = jnp.arange(xi.shape[0]) < D_m
    diff = diff * mask
    
    y = L_inv @ diff
    mahalanobis_sq = jnp.sum(y**2)
    
    diag_L = jnp.diag(L_inv)
    log_det_S_inv = -2.0 * jnp.sum(jnp.where(mask, jnp.log(diag_L + 1e-12), 0.0))
    
    log_pdf = -0.5 * (D_m * jnp.log(2 * jnp.pi) + log_det_S_inv + mahalanobis_sq)
    return log_pdf

# ======================================================================
# II. 单分量更新逻辑 (对齐 Algorithm 3 步骤 23-34)
# ======================================================================

def _update_step_k_l_single_component(
    k_idx, mu_k_t, L_inv_k_t, samples, elite_weights, 
    pi_all, mu_all, L_inv_all, delta_t,
    mu_baseline, L_inv_baseline, D_m
):
    D_max = mu_k_t.shape[0]
    S_k_t = L_inv_k_t @ L_inv_k_t.T # 精度矩阵
    
    # 1. 计算对数 PDF
    log_pdf_k = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_k_t, L_inv_k_t, D_m))(samples)
    log_pdf_base = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_baseline, L_inv_baseline, D_m))(samples)
    
    def mog_pdf_fn(xi):
        l_pdfs = vmap(lambda m, l: _gaussian_log_pdf_l_masked(xi, m, l, D_m))(mu_all, L_inv_all)
        return _logsumexp(jnp.log(pi_all + 1e-15) + l_pdfs)
    
    log_mog = vmap(mog_pdf_fn)(samples)

    # 2. 计算 IGO 权重项 a_i 和 b_i (对齐 Algorithm 3 Step 26)
    a_i = jnp.exp(jnp.clip(log_pdf_k - log_mog, a_min=-70.0, a_max=70.0))
    b_i = jnp.exp(jnp.clip(log_pdf_base - log_mog, a_min=-70.0, a_max=70.0))
    
    # 3. 均值 mu 更新 (步骤 30)
    diff = (samples - mu_k_t)
    grad_mu = (S_k_t @ diff.T).T
    sum_mu = jnp.sum((elite_weights * a_i)[:, None] * grad_mu, axis=0)
    
    # 4. 精度矩阵 S 更新 (步骤 28)
    Sigma_k = jnp.linalg.inv(S_k_t + jnp.eye(D_max)*1e-6)
    def outer_prod_update(d):
        return Sigma_k @ jnp.outer(d, d) @ Sigma_k - Sigma_k
    
    S_grads = vmap(outer_prod_update)(diff)
    sum_S = jnp.sum((elite_weights * a_i)[:, None, None] * S_grads, axis=0)

    S_new = S_k_t - delta_t * sum_S
    S_new = (S_new + S_new.T) / 2.0 + jnp.eye(D_max) * 1e-6
    L_inv_new = jnp.linalg.cholesky(S_new) # Step 29

    mu_new = mu_k_t + delta_t * jnp.linalg.solve(S_new , sum_mu)
    
    # 5. 权重 v 更新增量 (步骤 34)
    v_update_val = jnp.sum(elite_weights * (a_i - b_i))

    return mu_new, L_inv_new, v_update_val

# ======================================================================
# III. 主优化流程 (集成 Algorithm 4 与 历史记录)
# ======================================================================

def _parallel_step_with_algo4(state, iter_data, M, K, B, B0, dt, dims_arr, T_0, fitness_fn, v_reset, context):
    mu, L_inv, v, z_star = state # z_star 存储上一步的 B0 分位数
    key, idx = iter_data
    
    # 判定是否应用 Algorithm 4 的权重筛选 (重置后的第一步)
    is_after_reset = (idx % T_0 == 1) & (idx > 0)
    
    # Step 33-38: 重置权重逻辑
    v_curr = jnp.where((idx % T_0) == 0, v_reset, v)
    
    def v_to_pi(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_exps = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_exps, jnp.array([1.0 / sum_exps])])
    
    pi_all = vmap(v_to_pi)(v_curr)

    # 1. 采样阶段 (Step 4-13)
    def sample_block(m_idx, sub_key):
        comps = random.choice(sub_key, K, p=pi_all[m_idx], shape=(B,))
        def sample_gaussian(c_idx, s_key):
            m_k = mu[m_idx, c_idx]
            S_k = L_inv[m_idx, c_idx] @ L_inv[m_idx, c_idx].T
            sigma = jnp.linalg.inv(S_k + jnp.eye(S_k.shape[0])*1e-7)
            return m_k + jnp.linalg.cholesky(sigma) @ random.normal(s_key, (m_k.shape[0],))
        return vmap(sample_gaussian)(comps, random.split(sub_key, B))

    keys_m = random.split(key, M)
    samples_m = vmap(sample_block)(jnp.arange(M), keys_m)
    
    # 2. 评价与排序 (Step 14-19)
    samples_flat = samples_m.transpose(1, 0, 2).reshape(B, -1)
    f_vals = vmap(lambda s: fitness_fn(s, context))(samples_flat)
    
    # 记录当前步的最小值用于绘图
    min_fitness_step = jnp.min(f_vals)
    
    # 计算当前精英门槛
    sorted_f = jnp.sort(f_vals)
    current_z_B0 = sorted_f[B0 - 1]
    
    # --- Algorithm 4 权重分配逻辑 ---
    def weight_logic_algo4():
        # 门槛取 min(上步精英, 本步精英)
        threshold = jnp.minimum(z_star, current_z_B0)
        return jnp.where(f_vals <= threshold, 1.0/B, 0.0)

    def weight_logic_algo3():
        ranks = jnp.argsort(jnp.argsort(f_vals))
        return jnp.where(ranks < B0, 1.0/B, 0.0)

    # 仅在重置后的第一步启用 Algorithm 4 策略
    elite_weights = lax.cond(is_after_reset, weight_logic_algo4, weight_logic_algo3)

    # 3. 块并行更新 (Step 21-41)
    def update_m(m_idx):
        D_m = dims_arr[m_idx]
        mu_m = mu[m_idx]; L_m = L_inv[m_idx]; p_m = pi_all[m_idx]
        mu_base = mu_m[K-1]; L_base = L_m[K-1] # 选取第 K 个作为基准
        
        res_mu, res_L, res_v_delta = vmap(
            _update_step_k_l_single_component,
            in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
        )(jnp.arange(K), mu_m, L_m, samples_m[m_idx], elite_weights, p_m, mu_m, L_m, dt, mu_base, L_base, D_m)
        
        return res_mu, res_L, res_v_delta[:K-1]

    new_mu, new_L, v_deltas = vmap(update_m)(jnp.arange(M))
    new_v = jnp.clip(v_curr + dt * v_deltas, -70.0, 70.0)
    
    # 更新 state 并返回当前代最优值
    return (new_mu, new_L, new_v, current_z_B0), min_fitness_step

# ======================================================================
# IV. 顶层入口函数
# ======================================================================

@functools.partial(jit, static_argnums=(1, 3, 4, 5, 7, 9))
def mmog_igo_optimizer_mpc(
    key, T, dt, M, K, B, B0, dims, T_0, 
    fitness_fn_total, initial_mu_k, initial_L_inv_k, initial_v_k, context
):
    dims_array = jnp.array(dims)
    v_reset = jnp.zeros((M, K-1))
    
    # 初始 z_star 设为极大值
    z_star_init = jnp.array(1e15) 
    state = (initial_mu_k[:, :K, :], initial_L_inv_k[:, :K, :, :], v_reset, z_star_init)
    
    keys = random.split(key, T)
    
    # 执行循环并捕获 fitness_history
    final_state, fitness_history = lax.scan(
        functools.partial(
            _parallel_step_with_algo4, 
            M=M, K=K, B=B, B0=B0, dt=dt, 
            dims_arr=dims_array, T_0=T_0, 
            fitness_fn=fitness_fn_total, v_reset=v_reset,
            context=context
        ),
        state, (keys, jnp.arange(T))
    )
    
    final_mu, final_L, final_v, _ = final_state
    
    def v_to_pi_final(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    final_pi = vmap(v_to_pi_final)(final_v)
    
    # 返回 4 个值：均值、Cholesky 因子、权重、适应度历史
    return final_mu, final_L, final_pi, fitness_history