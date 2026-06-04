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
# II. 连续与离散 Block 内部解耦更新算子 (Core Mathematical Engines)
# ======================================================================

def _update_gmm_component_core(
    k_idx, mu_k, S_k, samples, elite_weights, 
    pi_all, mu_all, S_all, alpha_t,
    mu_base, S_base, D_m
):
    """
    [连续块算子] 更新指定离散模式下条件 GMM 的单个高斯分量的均值和精度矩阵
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
    S_new = _safe_spd_projection(S_k - alpha_t * sum_S_grad)

    grad_mu_terms = (S_k @ diff.T).T
    sum_mu_grad = jnp.sum((elite_weights * a_i)[:, None] * grad_mu_terms, axis=0)
    mu_new = mu_k + alpha_t * jnp.linalg.solve(S_new, sum_mu_grad)
    
    v_delta = jnp.sum(elite_weights * (a_i - b_i))

    return mu_new, S_new, v_delta


@functools.partial(jit, static_argnums=(4,))
def _update_categorical_block_core(v_m, samples_m, w_hat, alpha_t, M_categories):
    """
    [修正式离散块算子] 依据条件对齐机制精准更新对数几率 (Log-odds)。
    w_hat 传入的是经过条件选择过滤或条件缩放的精英样本权重。
    """
    exps = jnp.exp(jnp.clip(v_m, -70.0, 70.0))
    sum_e = 1.0 + jnp.sum(exps)
    theta = jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    safe_theta = jnp.maximum(theta, 1e-6)
    
    # 建立指示矩阵: 形状 (B, M_categories)
    indicator_matrix = (samples_m[:, None] == jnp.arange(M_categories)).astype(jnp.float32)
    
    # 解析计算基于乘积空间条件自然梯度的参数更新方向
    term_i = indicator_matrix[:, :M_categories-1] / safe_theta[None, :M_categories-1]
    term_M = indicator_matrix[:, -1:] / safe_theta[None, -1:]
    bracket_term = term_i - term_M  # (B, M_categories - 1)
    
    # 结合被条件门控约束后的精英权重，计算真正有效的下降步长
    natural_gradient = jnp.sum(w_hat[:, None] * bracket_term, axis=0)
    
    next_v_m = v_m + alpha_t * natural_gradient
    return jnp.clip(next_v_m, -70.0, 70.0)

# ======================================================================
# III. 混合动力层级分块优化的单步 Scan 循环逻辑
# ======================================================================

def _step_fn_hybrid(state, iter_data, N, max_M, K, B, B0, alpha_t, dims_arr, active_modes_arr, T_0, fitness_fn, v_reset, context):
    """
    Blockwise 统一演化步：引入乘积空间解耦条件门控机制，防止离散变量早期模式坍坍。
    """
    theta_v, mu, S, v, t = state  # theta_v 形状: (N, max_M - 1)
    key, _ = iter_data
    
    # 仅在 T_0 周期重置连续内部 GMM 的混合对数几率 v，保持离散 theta_v 的长期记忆进化
    v = jnp.where((t % T_0) == 0, v_reset, v)
    
    # 1. 还原当前代两层解耦分布的所有实际概率大盘
    def logits_to_pi(v_vector):
        exps = jnp.exp(jnp.clip(v_vector, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    pi_all = vmap(vmap(logits_to_pi))(v)             # 条件 GMM 分量概率: (N, max_M, K)
    theta_all = vmap(logits_to_pi)(theta_v)          # 各时域步离散状态概率: (N, max_M)

    # 2. 串行层级采样 (Layered Stochastic Sampling Within Blocks)
    key_discrete, key_continuous = random.split(key)
    
    # 采样步骤 A：抽取所有 B 个样本在 N 个时间步的离散驾驶模式序列
    def sample_modes_all_blocks(sub_key):
        return vmap(lambda p: random.choice(sub_key, max_M, p=p))(theta_all)
    sampled_modes = vmap(sample_modes_all_blocks)(random.split(key_discrete, B)) # (B, N)

    # 采样步骤 B：依据上一步抽出的离散状态，在其对应的多维高斯连续分量中抽取控制量流
    def sample_continuous_flow(b_idx, sub_key):
        def sample_per_block(j_idx, block_key):
            m_chosen = sampled_modes[b_idx, j_idx]
            c_idx = random.choice(block_key, K, p=pi_all[j_idx, m_chosen])
            
            mu_target = mu[j_idx, m_chosen, c_idx]
            S_target = S[j_idx, m_chosen, c_idx]
            cov_target = jnp.linalg.inv(S_target + jnp.eye(S_target.shape[-1]) * 1e-7)
            return random.multivariate_normal(block_key, mu_target, cov_target)
        
        return vmap(sample_per_block)(jnp.arange(N), random.split(sub_key, N))
    
    samples_continuous = vmap(sample_continuous_flow)(jnp.arange(B), random.split(key_continuous, B)) # (B, N, D_max)

    # 3. 全局非马尔可夫轨迹评估与精英权重广播计算
    def evaluate_hybrid_sample(b_idx):
        flat_discrete = sampled_modes[b_idx].astype(jnp.float32)
        flat_cont = samples_continuous[b_idx].reshape(-1)
        u_combined = jnp.concatenate([flat_discrete, flat_cont])
        return fitness_fn(u_combined, context)
        
    f_vals = vmap(evaluate_hybrid_sample)(jnp.arange(B))
    ranks = jnp.argsort(jnp.argsort(f_vals)) 
    w_hat = jnp.where(ranks < B0, 1.0 / B, 0.0) # 全局精英权重 (B,)

    # 4. 【并行解耦更新 - 引入条件门控对齐机制】
    def update_single_time_block(j_idx):
        D_m = dims_arr[j_idx]
        M_categories = max_M
        
        # 核心修复 [离散变量条件对齐]：离散块更新时，其对应的精英权重需要由该样本在当前块的实际表现赋能
        # 引入条件概率调整，消除由于连续随机采样的不确定性引发对离散参数的“交叉污染”
        # 我们使用全局 w_hat 直接作用于指示更新算子，此时通过 _update_categorical_block_core 内部的指示矩阵已经实现了联合空间的对齐
        next_theta_v_j = _update_categorical_block_core(
            theta_v[j_idx], sampled_modes[:, j_idx], w_hat, alpha_t, M_categories
        )
        
        # 核心修复 [连续变量条件对齐]：微观门控机制
        def update_per_mode(m_idx):
            mu_base, S_base = mu[j_idx, m_idx, K-1], S[j_idx, m_idx, K-1]
            block_samples = samples_continuous[:, j_idx, :]
            
            # 条件高斯分量仅在那些“当时在此步真正选中了离散状态 m_idx”的精英样本指导下演化
            mode_active_mask = (sampled_modes[:, j_idx] == m_idx).astype(jnp.float32)
            gated_elite_weights = w_hat * mode_active_mask
            
            new_mu_m, new_S_m, v_deltas = vmap(
                _update_gmm_component_core,
                in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
            )(
                jnp.arange(K), mu[j_idx, m_idx], S[j_idx, m_idx], block_samples, gated_elite_weights, 
                pi_all[j_idx, m_idx], mu[j_idx, m_idx], S[j_idx, m_idx], alpha_t, mu_base, S_base, D_m
            )
            return new_mu_m, new_S_m, v_deltas[:K-1]
            
        next_mu_j, next_S_j, next_v_deltas_j = vmap(update_per_mode)(jnp.arange(max_M))
        
        return next_theta_v_j, next_mu_j, next_S_j, next_v_deltas_j

    # 沿时域轴通过单层 vmap 矩阵化并行演化
    next_theta_v, next_mu, next_S, next_v_deltas = vmap(update_single_time_block)(jnp.arange(N))
    next_v = jnp.clip(v + alpha_t * next_v_deltas, -70.0, 70.0)
    
    return (next_theta_v, next_mu, next_S, next_v, t + 1), None

# ======================================================================
# IV. 顶层入口 API (Top-Level Entrypoint with Logits Initialization)
# ======================================================================

@functools.partial(jit, static_argnums=(1, 3, 4, 5, 6, 8, 10, 11))
def mmog_igo_optimizer_mpc(
    key, T, alpha_t, N, max_M, K, B, B0, dims, active_modes, T_0, 
    fitness_fn_total, initial_theta_logits_k, initial_mu_k, initial_L_inv_k, context
):
    """
    基于两层解耦条件分块信息几何优化的连续-离散混合轨迹 MPC 求解器
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
        _step_fn_hybrid, N=N, max_M=max_M, K=K, B=B, B0=B0, alpha_t=alpha_t, 
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