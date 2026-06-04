import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools

# ======================================================================
# I. 核心更新算子 (Core Blockwise Categorical Updates)
# ======================================================================

@functools.partial(jit, static_argnums=(4,))
def _update_categorical_block_core(v_m, samples_m, w_hat, alpha_t, M_categories):
    """
    更新单个离散分块 (Block j) 的对数几率参数 (Log-odds)
    """
    # 1. 通过对数几率还原当前块所有类别的真实概率 theta (隐含最后一项作为分母基准 M)
    exps = jnp.exp(jnp.clip(v_m, -70.0, 70.0))
    sum_e = 1.0 + jnp.sum(exps)
    theta = jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    # 2. 防御性截断，防止大盘收敛时某些冷门类别的概率彻底为 0 导致分母爆炸 (NaN)
    safe_theta = jnp.maximum(theta, 1e-6)
    
    # 3. 【核心对齐】构建指示掩码矩阵 (B, M_categories)
    # 行代表样本 b，列代表类别 i。当样本 b 在该块选择了类别 i 时，对应位置为 1.0
    indicator_matrix = (samples_m[:, None] == jnp.arange(M_categories)).astype(jnp.float32)
    
    # 4. 计算公式右侧括号内的两项：(I_{cb=i} / theta_i) - (I_{cb=M} / theta_M)
    term_i = indicator_matrix[:, :M_categories-1] / safe_theta[None, :M_categories-1]
    term_M = indicator_matrix[:, -1:] / safe_theta[None, -1:]
    
    bracket_term = term_i - term_M # 形状: (B, M_categories - 1)
    
    # 5. 乘以全局精英权重 w_hat 并对样本维度 (axis=0) 进行求和汇聚
    natural_gradient = jnp.sum(w_hat[:, None] * bracket_term, axis=0)
    
    # 6. 沿自然梯度方向更新对数几率
    next_v_m = v_m + alpha_t * natural_gradient
    return jnp.clip(next_v_m, -70.0, 70.0)


def _discrete_step_fn(state, iter_data, M_blocks, M_categories, B, B0, dt, fitness_fn, context):
    """
    离散求解器的单步迭代循环 (Scan Loop)
    """
    v, t = state  # v 形状: (M_blocks, M_categories - 1)
    key, _ = iter_data
    
    # 1. 计算当前所有块的概率大盘 (M_blocks, M_categories)
    def v_to_pi(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70.0, 70.0))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    pi_all = vmap(v_to_pi)(v)

    # 2. 采样：各个 Block 根据自己的概率独立独立抽取离散状态
    def sample_discrete_block(m_idx, sub_key):
        return random.choice(sub_key, M_categories, p=pi_all[m_idx], shape=(B,))

    # samples_m 形状: (M_blocks, B)
    samples_m = vmap(sample_discrete_block)(jnp.arange(M_blocks), random.split(key, M_blocks))
    
    # 3. 评价与全局精英权重计算 (纵向切片并转置，对齐轨迹轴)
    samples_flat = samples_m.transpose(1, 0) # 形状: (B, M_blocks)
    
    # 计算当前批次所有样本的适应度
    f_vals = vmap(lambda s: fitness_fn(s, context))(samples_flat) 
    
    # 硬筛选排序机制（完全保留你 MoG 求解器的精英机制）
    ranks = jnp.argsort(jnp.argsort(f_vals)) 
    w_hat = jnp.where(ranks < B0, 1.0 / B, 0.0)

    # 4. 块并行更新 (Blockwise 并行化更新对数几率)
    next_v = vmap(
        lambda m_idx: _update_categorical_block_core(
            v[m_idx], samples_m[m_idx], w_hat, dt, M_categories
        )
    )(jnp.arange(M_blocks))
    
    # 顺便把这一代的最优得分和平均得分记录下来方便监控验证
    best_f = jnp.min(f_vals)
    mean_f = jnp.mean(f_vals)
    metrics = (best_f, mean_f)
    
    return (next_v, t + 1), metrics

# ======================================================================
# II. 顶层入口 (Top-level Optimizer API)
# ======================================================================

@functools.partial(jit, static_argnums=(1, 2, 3, 4, 5, 7))
def categorical_igo_optimizer_mpc(
    key, T, M_blocks, M_categories, B, B0, dt,
    fitness_fn_total, context, initial_theta_logits_k=None
):
    """
    基于信息几何优化的纯离散范畴分布 MPC 求解器顶层入口
    """
    # 初始化对数几率；若外部传入完整的 theta logits，则转换为内部的 (M_categories - 1) 维表示
    if initial_theta_logits_k is None:
        initial_v = jnp.zeros((M_blocks, M_categories - 1))
    else:
        if initial_theta_logits_k.shape[-1] == M_categories:
            initial_v = initial_theta_logits_k[:, :-1] - initial_theta_logits_k[:, -1:]
        elif initial_theta_logits_k.shape[-1] == M_categories - 1:
            initial_v = initial_theta_logits_k
        else:
            raise ValueError(
                "initial_theta_logits_k must have last dimension M_categories or M_categories - 1"
            )

    state = (initial_v, 0)
    
    loop_fn = functools.partial(
        _discrete_step_fn, M_blocks=M_blocks, M_categories=M_categories, 
        B=B, B0=B0, dt=dt, fitness_fn=fitness_fn_total, context=context
    )
    
    # 执行时间级优化循环
    final_state, history = lax.scan(loop_fn, state, (random.split(key, T), jnp.arange(T)))
    
    # 提取最终更新完的概率大盘
    def v_to_pi_final(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70.0, 70.0))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    
    final_pi = vmap(v_to_pi_final)(final_state[0])
    
    return final_state[0], final_pi, history