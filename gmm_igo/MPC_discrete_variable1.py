import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools

# ======================================================================
# I. 核心底层辅助函数 (正定矩阵投影及掩码 PDF)
# ======================================================================

MIN_EIG = 1e-2
MAX_EIG = 1e3

@jit
def _safe_spd_projection(S):
    """保证连续变量的精度矩阵 S 维持对称正定"""
    sym = 0.5 * (S + S.T)
    eigvals, eigvecs = jnp.linalg.eigh(sym)
    eigvals = jnp.clip(eigvals, MIN_EIG, MAX_EIG)
    return eigvecs @ (eigvals[:, None] * eigvecs.T)

@jit
def _logsumexp(a, axis=None):
    return jnp.logaddexp.reduce(a, axis=axis)

@jit
def _gaussian_log_pdf_l_masked(xi, mu, S, D_m):
    """基于精度矩阵 S 计算对数概率密度，支持异构控制维度掩码"""
    diff = (xi - mu)
    mask = jnp.arange(xi.shape[0]) < D_m
    diff = diff * mask
    
    mahalanobis_sq = jnp.dot(diff, jnp.dot(S, diff))
    sign, logdet_S = jnp.linalg.slogdet(S)
    
    log_pdf = -0.5 * (D_m * jnp.log(2 * jnp.pi) - logdet_S + mahalanobis_sq)
    return log_pdf

# ======================================================================
# II. 连续与离散 Block 内部解耦更新算子 (接收独立外置步长)
# ======================================================================

def _update_gmm_component_core(
    k_idx, mu_k, S_k, samples, elite_weights, 
    pi_all, mu_all, S_all, alpha_continuous,
    mu_base, S_base, D_m
):
    """
    [连续块算子] 显式接收外置连续更新步长 alpha_continuous
    """
    S_k = _safe_spd_projection(S_k)
    
    log_pdf_k = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_k, S_k, D_m))(samples)
    log_pdf_base = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_base, S_base, D_m))(samples)
    
    def mog_pdf_fn(xi):
        l_pdfs = vmap(lambda m, s: _gaussian_log_pdf_l_masked(xi, m, s, D_m))(mu_all, S_all)
        return _logsumexp(jnp.log(pi_all + 1e-15) + l_pdfs)
    
    log_mog = vmap(mog_pdf_fn)(samples)

    a_i = jnp.exp(jnp.clip(log_pdf_k - log_mog, -70.0, 70.0))
    b_i = jnp.exp(jnp.clip(log_pdf_base - log_mog, -70.0, 70.0))
    
    diff = (samples - mu_k)
    def s_grad_fn(d):
        return S_k @ jnp.outer(d, d) @ S_k - S_k
    
    sum_S_grad = jnp.sum((elite_weights * a_i)[:, None, None] * vmap(s_grad_fn)(diff), axis=0)
    S_new = _safe_spd_projection(S_k - alpha_continuous * sum_S_grad)

    grad_mu_terms = (S_k @ diff.T).T
    sum_mu_grad = jnp.sum((elite_weights * a_i)[:, None] * grad_mu_terms, axis=0)
    mu_new = mu_k + alpha_continuous * jnp.linalg.solve(S_new, sum_mu_grad)
    
    v_delta = jnp.sum(elite_weights * (a_i - b_i))

    return mu_new, S_new, v_delta


@functools.partial(jit, static_argnums=(4,))
def _update_categorical_block_core(v_m, samples_m, w_hat, alpha_discrete, M_categories):
    """
    [修正式离散块算子] 显式接收外置离散更新步长 alpha_discrete
    """
    exps = jnp.exp(jnp.clip(v_m, -70.0, 70.0))
    sum_e = 1.0 + jnp.sum(exps)
    theta = jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    safe_theta = jnp.maximum(theta, 1e-6)
    
    indicator_matrix = (samples_m[:, None] == jnp.arange(M_categories)).astype(jnp.float32)
    
    term_i = indicator_matrix[:, :M_categories-1] / safe_theta[None, :M_categories-1]
    term_M = indicator_matrix[:, -1:] / safe_theta[None, -1:]
    bracket_term = term_i - term_M  
    
    natural_gradient = jnp.sum(w_hat[:, None] * bracket_term, axis=0)
    
    next_v_m = v_m + alpha_discrete * natural_gradient
    return jnp.clip(next_v_m, -70.0, 70.0)

# ======================================================================
# III. 混合动力层级双流更新的单步 Scan 循环逻辑
# ======================================================================

def _step_fn_hybrid(state, iter_data, N, max_M, K, B, B0, alpha_discrete, alpha_continuous, dims_arr, active_modes_arr, T_0, fitness_fn, v_reset, context):
    """
    支持双流解耦进化速率的 Blockwise 统一演化步
    """
    theta_v, mu, S, v, t = state  
    key, _ = iter_data
    
    # 仅在 T_0 周期重置连续内部 GMM 的混合对数几率 v
    v = jnp.where((t % T_0) == 0, v_reset, v)
    
    def logits_to_pi(v_vector):
        exps = jnp.exp(jnp.clip(v_vector, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    pi_all = vmap(vmap(logits_to_pi))(v)             
    theta_all = vmap(logits_to_pi)(theta_v)          

    # 1. 串行层级采样 
    key_discrete, key_continuous = random.split(key)
    
    # 采样步骤 A：抽取离散模式序列
    def sample_modes_all_blocks(sub_key):
        return vmap(lambda p: random.choice(sub_key, max_M, p=p))(theta_all)
        #return vmap(lambda p: random.choice(sub_key, max_M, p=p))(theta_all)
    sampled_modes = vmap(sample_modes_all_blocks)(random.split(key_discrete, B)) 
    #sampled_modes = sampled_modes.at[:, 0].set(0)
    #sampled_modes = sampled_modes.at[:, -1].set(0)

    # 采样步骤 B：依据离散状态抽取连续控制量
    def sample_continuous_flow(b_idx, sub_key):
        def sample_per_block(j_idx, block_key):
            m_chosen = sampled_modes[b_idx, j_idx]
            c_idx = random.choice(block_key, K, p=pi_all[j_idx, m_chosen])
            
            mu_target = mu[j_idx, m_chosen, c_idx]
            S_target = S[j_idx, m_chosen, c_idx]
            cov_target = jnp.linalg.inv(S_target + jnp.eye(S_target.shape[-1]) * 1e-7)
            return random.multivariate_normal(block_key, mu_target, cov_target)
        
        return vmap(sample_per_block)(jnp.arange(N), random.split(sub_key, N))
    
    samples_continuous = vmap(sample_continuous_flow)(jnp.arange(B), random.split(key_continuous, B)) 

    # 2. 全局非马尔可夫轨迹评估与精英权重广播
    def evaluate_hybrid_sample(b_idx):
        flat_discrete = sampled_modes[b_idx].astype(jnp.float32)
        flat_cont = samples_continuous[b_idx].reshape(-1)
        u_combined = jnp.concatenate([flat_discrete, flat_cont])
        return fitness_fn(u_combined, context)
        
    f_vals = vmap(evaluate_hybrid_sample)(jnp.arange(B))
    ranks = jnp.argsort(jnp.argsort(f_vals)) 
    w_hat = jnp.where(ranks < B0, 1.0 / B, 0.0) 

    # 3. 【并行解耦双速率更新】
    def update_single_time_block(j_idx):
        D_m = dims_arr[j_idx]
        M_categories = max_M
        
        # 离散块更新：接收外置离散步长 alpha_discrete
        next_theta_v_j = _update_categorical_block_core(
            theta_v[j_idx], sampled_modes[:, j_idx], w_hat, alpha_discrete, M_categories
        )
        
        # 连续块更新：
        def update_per_mode(m_idx):
            mu_base, S_base = mu[j_idx, m_idx, K-1], S[j_idx, m_idx, K-1]
            block_samples = samples_continuous[:, j_idx, :]
            
            # 条件门控机制掩码
            mode_active_mask = (sampled_modes[:, j_idx] == m_idx).astype(jnp.float32)
            gated_elite_weights = w_hat * mode_active_mask
            
            # 连续高斯更新：接收外置连续步长 alpha_continuous
            new_mu_m, new_S_m, v_deltas = vmap(
                _update_gmm_component_core,
                in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
            )(
                jnp.arange(K), mu[j_idx, m_idx], S[j_idx, m_idx], block_samples, gated_elite_weights, 
                pi_all[j_idx, m_idx], mu[j_idx, m_idx], S[j_idx, m_idx], alpha_continuous, mu_base, S_base, D_m
            )
            return new_mu_m, new_S_m, v_deltas[:K-1]
            
        next_mu_j, next_S_j, next_v_deltas_j = vmap(update_per_mode)(jnp.arange(max_M))
        
        return next_theta_v_j, next_mu_j, next_S_j, next_v_deltas_j

    next_theta_v, next_mu, next_S, next_v_deltas = vmap(update_single_time_block)(jnp.arange(N))
    # 条件 GMM 的权重对数几率更新同样采用 alpha_continuous 进化
    next_v = jnp.clip(v + alpha_continuous * next_v_deltas, -70.0, 70.0)
    
    return (next_theta_v, next_mu, next_S, next_v, t + 1), None

# ======================================================================
# IV. 顶层入口 API (双外置步长接口)
# ======================================================================

@functools.partial(jit, static_argnums=(1, 4, 5, 6, 7, 9, 11, 12))
def mmog_igo_optimizer_mpc(
    key, T, alpha_discrete, alpha_continuous, N, max_M, K, B, B0, dims, active_modes, T_0, 
    fitness_fn_total, initial_theta_logits_k, initial_mu_k, initial_L_inv_k, context
):
    """
    基于时空解耦速率（双步长）条件分块信息几何优化的连续-离散混合轨迹 MPC 求解器
    
    参数:
        alpha_discrete: 外置离散更新步长，建议设小 (如 0.01~0.05) 防止过早锁死模式
        alpha_continuous: 外置连续更新步长，建议设大 (如 0.1~0.3) 加速 GMM 流形收敛
    """
    dims_array = jnp.array(dims)
    active_modes_array = jnp.array(active_modes)
    
    if initial_theta_logits_k.shape[-1] == max_M:
        raw_v_init = initial_theta_logits_k[:, :-1] - initial_theta_logits_k[:, -1:]
    else:
        raw_v_init = initial_theta_logits_k
        
    mask_invalid_modes = jnp.arange(max_M - 1)[None, :] < (active_modes_array[:, None] - 1)
    theta_v_init = jnp.where(mask_invalid_modes, raw_v_init, -1e9)
    
    S_init = vmap(vmap(vmap(lambda L: L @ L.T)))(initial_L_inv_k[:, :, :K, :, :])
    mu_init = initial_mu_k[:, :, :K, :]
    
    v_init = jnp.zeros((N, max_M, K-1))
    v_reset = jnp.zeros((N, max_M, K-1))

    init_state = (theta_v_init, mu_init, S_init, v_init, 0)
    
    loop_fn = functools.partial(
        _step_fn_hybrid, N=N, max_M=max_M, K=K, B=B, B0=B0, 
        alpha_discrete=alpha_discrete, alpha_continuous=alpha_continuous, 
        dims_arr=dims_array, active_modes_arr=active_modes_array, T_0=T_0, 
        fitness_fn=fitness_fn_total, v_reset=v_reset, context=context
    )
    
    final_state, _ = lax.scan(loop_fn, init_state, (random.split(key, T), jnp.arange(T)))
    
    final_theta_v = final_state[0]
    final_mu = final_state[1]
    final_S = final_state[2]
    final_v = final_state[3]
    
    def v_to_pi_final(v_vector):
        exps = jnp.exp(jnp.clip(v_vector, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
        
    final_theta = vmap(v_to_pi_final)(final_theta_v)
    final_pi = vmap(vmap(v_to_pi_final))(final_v)
    
    final_L = vmap(vmap(vmap(jnp.linalg.cholesky)))(final_S)
    
    return final_theta, final_mu, final_L, final_pi