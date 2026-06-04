# MPCsolver_differentobjects.py - Fully Fixed Pure JAX Multi-Objective IGO Solver

import jax
import jax.numpy as jnp
from jax import vmap, random, lax
import functools

MIN_EIG = 1e-2
MAX_EIG = 1e3
LOG_CLIP_VALUE = 80.0 

@jax.jit
def _safe_spd_projection(S):
    """保证精度矩阵的对称正定性 (SPD)"""
    eigvals, eigvecs = jnp.linalg.eigh(S)
    eigvals = jnp.clip(eigvals, MIN_EIG, MAX_EIG)
    return eigvecs @ (eigvals[:, None] * eigvecs.T)

@jax.jit
def _gaussian_log_pdf_l(xi, mu, L_inv):
    """单个高斯分量的对数密度计算"""
    D = mu.shape[0]
    diff = xi - mu
    y = L_inv @ diff
    mahalanobis_sq = jnp.sum(y**2)
    log_det_S_inv = -2 * jnp.sum(jnp.log(jnp.diag(L_inv)))
    return -0.5 * (D * jnp.log(2 * jnp.pi) + log_det_S_inv + mahalanobis_sq)

_vmap_gaussian_log_pdf_l_k = vmap(_gaussian_log_pdf_l, in_axes=(None, 0, 0))

@jax.jit
def _mixture_log_pdf_l(xi, mu_k, L_inv_k, pi_k):
    """防上溢的高性能混合对数密度计算"""
    log_pdfs_k = _vmap_gaussian_log_pdf_l_k(xi, mu_k, L_inv_k)
    log_pi_k = jnp.log(pi_k + 1e-20)
    log_weighted_pdfs = log_pi_k + log_pdfs_k
    max_log = jnp.max(log_weighted_pdfs)
    return max_log + jnp.log(jnp.sum(jnp.exp(log_weighted_pdfs - max_log)))

@jax.jit
def _sample_from_component_l(idx, key_sample, mu_k_all, L_inv_k_all):
    """从指定的 GMM 组件中采样样本"""
    mu_k = mu_k_all[idx]
    S_k = L_inv_k_all[idx] @ L_inv_k_all[idx].T
    L_Sigma = jnp.linalg.cholesky(jnp.linalg.inv(S_k))
    z = random.normal(key_sample, shape=(mu_k.shape[0],))
    return mu_k + L_Sigma @ z

@jax.jit
def _get_component_elite_weights_from_scores(f_xi, B, B_0):
    """基于非抽样排序的归一化精英权重提取计算"""
    ranks = jnp.argsort(jnp.argsort(f_xi))
    return jnp.where(ranks < B_0, 1.0, 0.0) / B

# ----------------------------------------------------------------------
# 核心演化算子：向量化并行单步更新
# ----------------------------------------------------------------------
@jax.jit
def _update_step_k_single(
    k_idx, mu_k_t, L_inv_k_t, samples, elite_weights_k, elite_weights_global,
    pi_k_all, mu_k_all, L_inv_k_all, delta_t, mu_K_t, L_inv_K_t
):
    S_k_t = L_inv_k_t @ L_inv_k_t.T
    
    log_norm_pdf_k = vmap(_gaussian_log_pdf_l, in_axes=(0, None, None))(samples, mu_k_t, L_inv_k_t)
    log_norm_pdf_K = vmap(_gaussian_log_pdf_l, in_axes=(0, None, None))(samples, mu_K_t, L_inv_K_t)
    log_mog_xi = vmap(_mixture_log_pdf_l, in_axes=(0, None, None, None))(samples, mu_k_all, L_inv_k_all, pi_k_all)

    a_i = jnp.exp(jnp.clip(log_norm_pdf_k - log_mog_xi, a_max=LOG_CLIP_VALUE))
    b_i = jnp.exp(jnp.clip(log_norm_pdf_K - log_mog_xi, a_max=LOG_CLIP_VALUE))
    
    # 诉求实现：各个高斯子分量更新照各个分量负责的 fitness 对应的精英权重(elite_weights_k)排
    scaled_a_i = a_i * elite_weights_k
    diff = samples - mu_k_t
    S_diff = (S_k_t @ diff.T).T
    S_diff_outer = vmap(lambda x: jnp.outer(x, x))(S_diff)
    
    S_update_term_i = S_diff_outer - S_k_t[None, :, :]
    sum_S_update = jnp.sum(scaled_a_i[:, None, None] * S_update_term_i, axis=0)
    S_k_t_plus_1 = _safe_spd_projection(S_k_t - delta_t * sum_S_update)
    L_inv_k_t_plus_1 = jnp.linalg.cholesky(S_k_t_plus_1)

    weighted_diff_sum = jnp.sum(scaled_a_i[:, None] * S_diff, axis=0)
    mu_update_term = jnp.linalg.solve(S_k_t_plus_1, weighted_diff_sum)
    mu_k_t_plus_1 = mu_k_t + delta_t * mu_update_term

    # 诉求实现：mixture weights (v_k) 更新由全局仲裁函数的全局排 (elite_weights_global) 决定
    v_update_sum = jnp.sum(elite_weights_global * (a_i - b_i))
    return mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum


_vmap_update_step = vmap(_update_step_k_single, in_axes=(0, 0, 0, None, 0, None, None, None, None, None, None, None))

def _iteration_step(state, key_input, B, B_0, K, delta_t, fitness_fns_list, global_fitness_fn, context):
    """单步迭代演进"""
    mu_k_t, L_inv_k_t, v_k_t = state
    key, subkey = random.split(key_input)
    
    pi_k_pre = jnp.exp(v_k_t)
    pi_K_t = 1.0 / (1.0 + jnp.sum(pi_k_pre))
    pi_k_t_all = jnp.concatenate([pi_k_pre * pi_K_t, jnp.array([pi_K_t])])
    
    comp_indices = random.choice(subkey, K, shape=(B,), p=pi_k_t_all)
    sample_keys = random.split(subkey, B)
    samples = vmap(_sample_from_component_l, in_axes=(0, 0, None, None))(comp_indices, sample_keys, mu_k_t, L_inv_k_t)

    # 1. 计算全局决策精英权重（用于混合权重更新）
    global_scores = vmap(global_fitness_fn, in_axes=(0, None))(samples, context)
    elite_weights_global = _get_component_elite_weights_from_scores(global_scores, B, B_0)
    
    # 2. 纯 JAX 多目标独立分量并行路由评估 (利用内层 vmap 处理静态多目标函数元组的分支路由)
    k_indices = jnp.arange(K)
    
    def eval_single_objective_k(k):
        # 核心修复点：通过显式的内部评估函数打破 lax.scan 对静态外层 PjitFunction 对象的捕获僵局
        scores_k = vmap(lambda s: lax.switch(k, fitness_fns_list, s, context))(samples)
        return _get_component_elite_weights_from_scores(scores_k, B, B_0)

    elite_weights_k_all = vmap(eval_single_objective_k)(k_indices)
    
    # 3. 向量化并行更新状态
    mu_k_t_plus_1, L_inv_k_t_plus_1, v_update_sum_k = _vmap_update_step(
        k_indices, mu_k_t, L_inv_k_t, samples, elite_weights_k_all, elite_weights_global,
        pi_k_t_all, mu_k_t, L_inv_k_t, delta_t, mu_k_t[-1], L_inv_k_t[-1]
    )
    
    v_update_safe = v_update_sum_k[:K-1]
    v_update_norm = jnp.linalg.norm(v_update_safe)
    v_update_safe = jnp.where(v_update_norm > 10.0, v_update_safe * (10.0 / v_update_norm), v_update_safe)
    v_k_t_plus_1 = jnp.clip(v_k_t + delta_t * v_update_safe, a_max=70.0)
    
    return (mu_k_t_plus_1, L_inv_k_t_plus_1, v_k_t_plus_1), None

def igo_mog_optimizer_impl(key, T, delta_t, K, B, B_0, fitness_fns_list, global_fitness_fn, initial_mu_k, initial_L_inv_k, initial_pi_k, context):
    v_k_0 = jnp.log(initial_pi_k[:-1] / initial_pi_k[-1])
    initial_state = (initial_mu_k, initial_L_inv_k, v_k_0)
    
    # 显式将多目标函数序列转化为元组，绑定为 scan 之外的环境常量
    bound_step = functools.partial(
        _iteration_step, 
        B=B, B_0=B_0, K=K, delta_t=delta_t, 
        fitness_fns_list=tuple(fitness_fns_list), 
        global_fitness_fn=global_fitness_fn, 
        context=context
    )
    
    final_state, _ = lax.scan(bound_step, initial_state, random.split(key, T))
    
    final_pi_pre = jnp.exp(final_state[2])
    final_pi_K = 1.0 / (1.0 + jnp.sum(final_pi_pre))
    final_pi_all = jnp.concatenate([final_pi_pre * final_pi_K, jnp.array([final_pi_K])])
    return final_state[0], final_state[1], final_pi_all

# 正确声明静态参数
igo_mog_optimizer = jax.jit(
    igo_mog_optimizer_impl, 
    static_argnames=('T', 'delta_t', 'K', 'B', 'B_0', 'fitness_fns_list', 'global_fitness_fn')
)