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
    eigvals, eigvecs = jnp.linalg.eigh(S)
    eigvals = jnp.clip(eigvals, MIN_EIG, MAX_EIG)
    return eigvecs @ (eigvals[:, None] * eigvecs.T)

@jit
def _logsumexp(a, axis=None):
    return jnp.logaddexp.reduce(a, axis=axis)

@jit
def _gaussian_log_pdf_l_masked(xi, mu, S, D_m):
    diff = (xi - mu)
    mask = jnp.arange(xi.shape[0]) < D_m
    diff = diff * mask
    
    mahalanobis_sq = jnp.dot(diff, jnp.dot(S, diff))
    sign, logdet_S = jnp.linalg.slogdet(S)
    
    log_pdf = -0.5 * (D_m * jnp.log(2 * jnp.pi) - logdet_S + mahalanobis_sq)
    return log_pdf

# ======================================================================
# II. 单分量更新逻辑 (严格对应 Alg 4 的更新步)
# ======================================================================

def _update_component_core(
    k_idx, mu_k, S_k, samples, elite_weights, 
    pi_all, mu_all, S_all, delta_t,
    mu_base, S_base, D_m
):
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
# III. 算法 6 核心：严格博弈迭代步
# ======================================================================
# ======================================================================
# III. 算法 6 核心：严格博弈迭代步
# ======================================================================

def _step_fn_rne(state, iter_data, M_agent, K, B, B0, dt, dims_arr, T_0, fitness_fn_j, v_reset, context, M_inner):
    mu, S, v, t = state
    key, _ = iter_data
    
    # 权重重置
    v = jnp.where((t % T_0) == 0, v_reset, v)
    
    def v_to_pi(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    pi_all = vmap(v_to_pi)(v)

    # 1. 每个 Agent 独立采样自己的 B 个样本 z_b^{(i)}
    def sample_single_agent(agent_idx, sub_key):
        comps = random.choice(sub_key, K, p=pi_all[agent_idx], shape=(B,))
        def gen_sample(c_idx, s_key):
            cov = jnp.linalg.inv(S[agent_idx, c_idx] + jnp.eye(S.shape[-1]) * 1e-7)
            return random.multivariate_normal(s_key, mu[agent_idx, c_idx], cov)
        return vmap(gen_sample)(comps, random.split(sub_key, B))

    key_sample, key_inner = random.split(key)
    samples = vmap(sample_single_agent)(jnp.arange(M_agent), random.split(key_sample, M_agent))  
    # samples.shape = (M_agent, B, D_max)

    # 2. 【修正后的严格 Inner Monte Carlo】
    def evaluate_expected_cost(agent_idx, sub_key):
        zb_samples = samples[agent_idx]                    # (B, D_max) 当前 agent 的样本
        
        def compute_f_hat_for_one_zb(zb_i, zb_key):
            """对单个 z_b^{(i)} 做 M_inner 次 Inner MC"""
            def sample_one_opponent_set(m_key):
                """只采样其他 agent，固定当前 zb_i"""
                def sample_other(other_idx, o_key):
                    def fixed_branch(_):
                        return zb_i

                    def sampled_branch(key_and_idx):
                        key, idx = key_and_idx
                        comp = random.choice(key, K, p=pi_all[idx])
                        cov = jnp.linalg.inv(S[idx, comp] + jnp.eye(S.shape[-1]) * 1e-7)
                        return random.multivariate_normal(key, mu[idx, comp], cov)

                    return lax.cond(
                        other_idx == agent_idx,
                        fixed_branch,
                        sampled_branch,
                        operand=(o_key, other_idx),
                    )
                
                # 对所有 agent 采样（但当前 agent 被固定）
                all_actions = vmap(sample_other)(jnp.arange(M_agent), random.split(m_key, M_agent))
                return all_actions.flatten()               # 展平为 fitness_fn_j 需要的格式
            
            # 做 M_inner 次独立采样
            keys = random.split(zb_key, M_inner)
            joint_samples = vmap(sample_one_opponent_set)(keys)
            
            # 计算期望代价
            f_vals = vmap(lambda s: fitness_fn_j(agent_idx, s, context))(joint_samples)
            return jnp.mean(f_vals)

        # 对该 agent 的 B 个样本分别计算期望代价
        f_hat = vmap(compute_f_hat_for_one_zb)(zb_samples, random.split(sub_key, B))
        
        # 排序并分配精英权重
        ranks = jnp.argsort(jnp.argsort(f_hat))
        return jnp.where(ranks < B0, 1.0 / B, 0.0)

    # 为每个 agent 计算权重
    w_hat_m = vmap(evaluate_expected_cost)(jnp.arange(M_agent), random.split(key_inner, M_agent))

    # 3. Blockwise 更新（保持不变）
    def update_block(m_idx):
        D_m = dims_arr[m_idx]
        mu_base, S_base = mu[m_idx, K-1], S[m_idx, K-1]
        
        new_mu_m, new_S_m, v_deltas = vmap(
            _update_component_core,
            in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
        )(jnp.arange(K), mu[m_idx], S[m_idx], samples[m_idx], w_hat_m[m_idx], 
          pi_all[m_idx], mu[m_idx], S[m_idx], dt, mu_base, S_base, D_m)
        
        return new_mu_m, new_S_m, v_deltas[:K-1]

    next_mu, next_S, next_v_deltas = vmap(update_block)(jnp.arange(M_agent))
    next_v = jnp.clip(v + dt * next_v_deltas, -70.0, 70.0)
    
    return (next_mu, next_S, next_v, t + 1), None

# ======================================================================
# IV. 顶层入口
# ======================================================================

@functools.partial(jit, static_argnums=(1, 3, 4, 5, 7, 9, 13))
def mmog_igo_rne_solver(
    key, T, dt, M_agent, K, B, B0, dims, T_0, 
    fitness_fn_j, initial_mu_k, initial_L_inv_k, context, M_inner
):
    """
    STRICT IMPLEMENTATION OF ALGORITHM 6:
    Randomized Nash Equilibrium Solver using Blockwise MGIGO with explicit Inner MC.
    """
    dims_array = jnp.array(dims)
    v_reset = jnp.zeros((M_agent, K-1))
    
    S_init = vmap(vmap(lambda L: L @ L.T))(initial_L_inv_k[:, :K, :, :])
    mu_init = initial_mu_k[:, :K, :]
    v_init = jnp.zeros((M_agent, K-1)) 

    state = (mu_init, S_init, v_init, 0)
    
    loop_fn = functools.partial(
        _step_fn_rne, M_agent=M_agent, K=K, B=B, B0=B0, dt=dt, 
        dims_arr=dims_array, T_0=T_0, 
        fitness_fn_j=fitness_fn_j, v_reset=v_reset, context=context, M_inner=M_inner
    )
    
    final_state, _ = lax.scan(loop_fn, state, (random.split(key, T), jnp.arange(T)))
    
    final_mu = final_state[0]
    final_L = vmap(vmap(jnp.linalg.cholesky))(final_state[1])
    
    def v_to_pi_final(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    final_pi = vmap(v_to_pi_final)(final_state[2])
    
    return final_mu, final_L, final_pi