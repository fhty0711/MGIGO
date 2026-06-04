import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools

# ======================================================================
# I. 核心辅助函数
# ======================================================================

MIN_EIG = 1e-2
MAX_EIG = 1e3

def _safe_spd_projection(S):
    """确保协方差矩阵的正定性 (SPD)"""
    eigvals, eigvecs = jnp.linalg.eigh(S)
    eigvals = jnp.clip(eigvals, MIN_EIG, MAX_EIG)
    return eigvecs @ (eigvals[:, None] * eigvecs.T)

@jit
def _logsumexp(a, axis=None):
    return jnp.logaddexp.reduce(a, axis=axis)

@jit
def _gaussian_log_pdf_l_masked(xi, mu, S, D_m):
    """带维度掩码的高斯对数概率密度计算"""
    diff = (xi - mu)
    mask = jnp.arange(xi.shape[0]) < D_m
    diff = diff * mask
    
    mahalanobis_sq = jnp.dot(diff, jnp.dot(S, diff))
    sign, logdet_S = jnp.linalg.slogdet(S)
    
    log_pdf = -0.5 * (D_m * jnp.log(2 * jnp.pi) - logdet_S + mahalanobis_sq)
    return log_pdf

# ======================================================================
# II. 单块独立更新逻辑 (严格对应 Algorithm 4 几何下降)
# ======================================================================

def _update_block_component_core(
    k_idx, mu_k, S_k, samples, elite_weights, 
    pi_all, mu_all, S_all, delta_t,
    mu_base, S_base, D_m
):
    """
    单个 Block 的高斯混合分量流形梯度更新。
    每个 Block 的 mu 和 S 依据分配给它的 elite_weights 独立演化。
    """
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
    S_new = _safe_spd_projection(S_k - delta_t * sum_S_grad)

    S_diff = (S_k @ diff.T).T
    sum_mu_grad = jnp.sum((elite_weights * a_i)[:, None] * S_diff, axis=0)
    mu_new = mu_k + delta_t * jnp.linalg.solve(S_new, sum_mu_grad)
    
    v_delta = jnp.sum(elite_weights * (a_i - b_i))
    return mu_new, S_new, v_delta

# ======================================================================
# III. 算法 6 泛化：异构块联合博弈迭代步
# ======================================================================

def _step_fn_rne_blocks(
    state, iter_data, N_blocks, M_agent, K, B, B0, dt, dims_arr, T_0, 
    fitness_fn_j, v_reset, context, M_inner, block_to_agent_idx
):
    mu, S, v, t = state
    key, _ = iter_data
    
    # 权重周期性重置
    v = jnp.where((t % T_0) == 0, v_reset, v)
    
    # 将自然参数 v 映射为单纯形上的概率 pi
    def v_to_pi(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    pi_all = vmap(v_to_pi)(v)  # (N_blocks, K)

    # 1. 块级并行采样方案 (Blockwise Sampling)
    def sample_all_blocks(count, sub_key):
        def sample_single_block(b_idx, b_key):
            comps = random.choice(b_key, K, p=pi_all[b_idx], shape=(count,))
            def gen_sample(c_idx, s_key):
                cov = jnp.linalg.inv(S[b_idx, c_idx] + jnp.eye(S.shape[-1]) * 1e-7)
                return random.multivariate_normal(s_key, mu[b_idx, c_idx], cov)
            return vmap(gen_sample)(comps, random.split(b_key, count))
        return vmap(sample_single_block)(jnp.arange(N_blocks), random.split(sub_key, N_blocks))

    key_B, key_M = random.split(key)
    samples_B = sample_all_blocks(B, key_B)       # (N_blocks, B, D_max)
    samples_M = sample_all_blocks(M_inner, key_M) # (N_blocks, M_inner, D_max)

    # 2. Agent 级联合蒙特卡洛评估（实现跨块决策边界一致性）
    def evaluate_agent_expected_cost(agent_idx):
        # 建立块遮罩：属于当前 Agent 的块为 True
        block_mask = (block_to_agent_idx == agent_idx)
        
        def compute_f_hat_for_one_zb(b_sim_idx):
            """
            计算当前 Agent 内部所有 Block 在对应的第 b_sim_idx 个动作候选下的联合代价
            """
            def eval_with_context_m(m_idx):
                # 构造博弈联合快照：默认全取背景池 samples_M
                # 形状展开成一维平面以便于外部代价函数解析：(N_blocks * D_max,)
                joint_m = samples_M[:, m_idx, :]
                
                # 【核心替换】：将属于当前 Agent 的所有块的动作，同时替换为当前候选池的第 b_sim_idx 个动作
                # 利用 vmap 索引或 masked select 维持 JAX 静态无分支特性
                joint_m_modified = jnp.where(block_mask[:, None], samples_B[:, b_sim_idx, :], joint_m)
                
                return fitness_fn_j(agent_idx, joint_m_modified.flatten(), context)
            
            f_vals = vmap(eval_with_context_m)(jnp.arange(M_inner))
            return jnp.mean(f_vals)

        # 评估当前 Agent 的 B 个联合候选样本的期望代价
        f_hat = vmap(compute_f_hat_for_one_zb)(jnp.arange(B))
        
        # 依据 Agent 综合代价排序分配 elite 权重
        ranks = jnp.argsort(jnp.argsort(f_hat))
        elite_weights = jnp.where(ranks < B0, 1.0 / B, 0.0)
        return elite_weights, jnp.mean(f_hat)

    # 并行评估所有 Agent 的权重平面 -> 形状: (M_agent, B)
    agent_w_hat, mean_fitness_m = vmap(evaluate_agent_expected_cost)(jnp.arange(M_agent))

    # 【跨块相同核心】：通过查表映射，将 Agent 级的排序权重广播散布到每个 Block 上
    # 散布后 block_w_hat 形状: (N_blocks, B)
    block_w_hat = agent_w_hat[block_to_agent_idx]

    # 3. Blockwise 独立几何更新
    def update_single_block(b_idx):
        D_m = dims_arr[b_idx]
        mu_base, S_base = mu[b_idx, K-1], S[b_idx, K-1]
        
        new_mu_b, new_S_b, v_deltas = vmap(
            _update_block_component_core,
            in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
        )(jnp.arange(K), mu[b_idx], S[b_idx], samples_B[b_idx], block_w_hat[b_idx], 
          pi_all[b_idx], mu[b_idx], S[b_idx], dt, mu_base, S_base, D_m)
        
        return new_mu_b, new_S_b, v_deltas[:K-1]

    next_mu, next_S, next_v_deltas = vmap(update_single_block)(jnp.arange(N_blocks))
    next_v = jnp.clip(v + dt * next_v_deltas, -70.0, 70.0)

    def aggregate_agent_pi(agent_idx):
        mask = (block_to_agent_idx == agent_idx).astype(pi_all.dtype)[:, None]
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        return jnp.sum(mask * pi_all, axis=0) / denom

    agent_pi = vmap(aggregate_agent_pi)(jnp.arange(M_agent))
    metrics = {
        "mu": mu,
        "pi": agent_pi,
        "block_pi": pi_all,
        "mean_fitness": mean_fitness_m,
    }

    return (next_mu, next_S, next_v, t + 1), metrics

# ======================================================================
# IV. 顶层入口
# ======================================================================

@functools.partial(jit, static_argnums=(1, 3, 4, 5, 6, 8, 10, 14))
def mmog_igo_rne_blocks_solver(
    key, T, dt, N_blocks, M_agent, K, B, B0, dims, T_0, 
    fitness_fn_j, initial_mu_k, initial_L_inv_k, context, M_inner, block_to_agent_idx,
    initial_v_k=None
):
    """
    支持异构块分配的博弈 RNE 求解器架构。
    
    参数:
        N_blocks: 系统总块数 (所有 agent 的块数总和)
        M_agent: 独立博弈的 Agent 数量
        block_to_agent_idx: 长度为 N_blocks 的数组，指定各块的 Agent 归属 (如 [0, 0, 1])
        dims: 长度为 N_blocks 的一维数组，指示每个块的遮罩有效维度
        initial_mu_k: 形状为 (N_blocks, K, D_max)
        initial_L_inv_k: 形状为 (N_blocks, K, D_max, D_max)
        initial_v_k: 形状为 (N_blocks, K-1) 的可选初始权重参数；若不传则默认为 0
    """
    dims_array = jnp.array(dims)
    v_reset = jnp.zeros((N_blocks, K-1))
    block_to_agent_array = jnp.array(block_to_agent_idx)
    
    # 基础空间投影初始化
    S_init = vmap(vmap(lambda L: L @ L.T))(initial_L_inv_k[:, :K, :, :])
    mu_init = initial_mu_k[:, :K, :]
    if initial_v_k is None:
        v_init = jnp.zeros((N_blocks, K-1))
    else:
        v_init = jnp.asarray(initial_v_k[:, :K-1])

    state = (mu_init, S_init, v_init, 0)
    
    loop_fn = functools.partial(
        _step_fn_rne_blocks, N_blocks=N_blocks, M_agent=M_agent, K=K, B=B, B0=B0, dt=dt, 
        dims_arr=dims_array, T_0=T_0, fitness_fn_j=fitness_fn_j, v_reset=v_reset, 
        context=context, M_inner=M_inner, block_to_agent_idx=block_to_agent_array
    )
    
    # 执行时间轴扫描
    final_state, metrics_history = lax.scan(loop_fn, state, (random.split(key, T), jnp.arange(T)))
    
    # 结果提取与 Cholesky 还原
    final_mu = final_state[0]
    final_L = vmap(vmap(jnp.linalg.cholesky))(final_state[1])
    
    def v_to_pi_final(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    final_pi = vmap(v_to_pi_final)(final_state[2])
    
    return final_mu, final_L, final_pi, metrics_history