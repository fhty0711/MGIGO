import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools

# ======================================================================
# I. 核心更新算子 (Core Autoregressive Natural Gradient Updates)
# ======================================================================

@functools.partial(jit, static_argnums=(4, 5))
def _update_plackett_luce_core(eta, samples_pi, w_hat, dt, n_cities, B):
    """
    严格按照 IGO 原始公式 (+) 和 T_ij 统计量定义更新二维转移对数几率矩阵 eta
    """
    # 1. 核心对齐：计算每个样本内所有 (i, j) 是否满足 “j 紧跟在 i 后面”
    def compute_single_T(pi):
        # pi 是访问顺序，通过 argsort 拿到每个城市在排列中的“时间步”
        inversion_idx = jnp.argsort(pi)
        pos_i = inversion_idx[:, None]
        pos_j = inversion_idx[None, :]
        
        # 严格执行你的物理定义：j 紧跟在 i 后面，即时间步恰好大 1
        T_mat = (pos_j == pos_i + 1).astype(jnp.float32)
        return T_mat

    # 批量并行计算所有样本的 T 矩阵，形状: (B, n_cities, n_cities)
    T_all = vmap(compute_single_T)(samples_pi)
    
    # 2. 计算大盘经验期望 \hat{\mu}_ij
    mu_hat = jnp.mean(T_all, axis=0)
    
    # 3. 计算括号内的统计量残差项 (B, n_cities, n_cities)
    bracket_term = T_all - mu_hat[None, :, :]
    
    # 4. 乘以全局精英权重 w_hat 并沿样本轴聚合 (利用 Einsum 避免写循环)
    natural_gradient = jnp.einsum('b,bij->ij', w_hat, bracket_term)
    
    # 5. 严格捍卫 IGO 原始公式：使用 (+) 号更新
    next_eta = eta + dt * natural_gradient
    
    # 防御性截断，防止对数几率溢出导致不稳定性
    return jnp.clip(next_eta, -50.0, 50.0)


def _plackett_luce_step_fn(state, iter_data, n_cities, B, B0, dt, fitness_fn, context):
    """
    自回归转移流形求解器的单步迭代循环 (Scan Loop)
    """
    eta, t = state  # eta 形状: (n_cities, n_cities), 记录 i 到 j 的转移对数偏好
    key, _ = iter_data
    
    # ======================================================================
    # 【核心对齐】序贯遮蔽采样 (Sequential Masked Sampling)
    # ======================================================================
    # 初始状态：规定所有样本（B个粒子）全部固定从 0 号城市出发
    start_city = jnp.zeros((B,), dtype=jnp.int32)
    # 记录已访问城市掩码 (B, n_cities)，将 0 号城市标记为已访问 (1.0)
    init_visited = jnp.zeros((B, n_cities)).at[:, 0].set(1.0)
    # 记录最终路径矩阵 (B, n_cities)，把第 0 步填入城市 0
    init_trajectory = jnp.zeros((B, n_cities), dtype=jnp.int32).at[:, 0].set(0)
    
    def sampling_step(carry, step_idx):
        current_cities, visited, traj, sub_key = carry
        
        # 依靠 JAX 索引一枪头刷出所有样本当前城市对应的转移几率行: (B, n_cities)
        log_logits = eta[current_cities, :]
        
        # 施加遮蔽惩罚：已经去过的城市，转移几率直接扣掉一个大数（等价于概率降为0）
        masked_logits = log_logits - visited * 1e9
        
        # 注入标准 Gumbel 噪声进行高并发分类采样 (Categorical Sampling)
        split_key = random.fold_in(sub_key, step_idx)
        gumbel = -jnp.log(-jnp.log(random.uniform(split_key, (B, n_cities)) + 1e-10) + 1e-10)
        next_cities = jnp.argmax(masked_logits + gumbel, axis=-1)  # 形状: (B,)
        
        # 更新状态参数
        next_visited = visited.at[jnp.arange(B), next_cities].set(1.0)
        next_traj = traj.at[:, step_idx].set(next_cities)
        
        return (next_cities, next_visited, next_traj, sub_key), None

    # 用 lax.scan 串行走完剩下的 n_cities - 1 步，在 GPU 上对 B 维进行超高并发展开
    scan_carry = (start_city, init_visited, init_trajectory, key)
    final_carry, _ = lax.scan(sampling_step, scan_carry, jnp.arange(1, n_cities))
    samples_pi = final_carry[2] # 提取最终生成的合法无重复城市路径矩阵 (B, n_cities)
    
    # ======================================================================
    # II. 精英筛选与更新 (完全对齐原版最小化机制)
    # ======================================================================
    # 计算当前批次所有排列的 TSP 代价 (正的长度)
    f_vals = vmap(lambda s: fitness_fn(s, context))(samples_pi)
    
    # 经典的双重 argsort 机制：筛选 f_vals 最小（路径最短）的前 B0 个
    ranks = jnp.argsort(jnp.argsort(f_vals))
    w_hat = jnp.where(ranks < B0, 1.0 / B, 0.0)
    
    # 调用核心算子，沿着纯正的 IGO 自然梯度流更新二维流形
    next_eta = _update_plackett_luce_core(eta, samples_pi, w_hat, dt, n_cities, B)
    
    # 记录监控指标
    best_f = jnp.min(f_vals)
    mean_f = jnp.mean(f_vals)
    metrics = (best_f, mean_f)
    
    return (next_eta, t + 1), metrics

# ======================================================================
# III. 顶层入口 (Top-level Optimizer API)
# ======================================================================

@functools.partial(jit, static_argnums=(1, 2, 3, 4, 5, 6))
def plackett_luce_igo_optimizer_tsp(
    key, T, n_cities, B, B0, dt, fitness_fn_total, context, initial_eta=None
):
    """
    基于序贯自回归转移能量模型构建的纯正 IGO Flow TSP 求解器
    """
    # 初始化：若未传入，执行全 0 最大熵冷启动
    if initial_eta is None:
        eta_init = jnp.zeros((n_cities, n_cities))
    else:
        eta_init = initial_eta
        
    state = (eta_init, 0)
    
    loop_fn = functools.partial(
        _plackett_luce_step_fn, n_cities=n_cities,
        B=B, B0=B0, dt=dt, fitness_fn=fitness_fn_total, context=context
    )
    
    # 编译执行全并行的时域级优化循环
    final_state, history = lax.scan(loop_fn, state, (random.split(key, T), jnp.arange(T)))
    
    return final_state[0], history