import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import time

# 导入包含 Algorithm 4 修改后的 MPC 求解器
from gmm_igo.MPCresetweightplot import mmog_igo_optimizer_mpc

# ======================================================================
# 1. Levy 适应度函数 (对齐 IGO 采样和最终评估)
# ======================================================================
@jit
def mpc_levy_fitness(z_flattened, context):
    """
    Levy 函数。全局最优解在 x = [1, 1, ..., 1] 处，f(x) = 0。
    """
    is_batched = z_flattened.ndim > 1
    x = z_flattened
    w = 1.0 + (x - 1.0) / 4.0
    
    if is_batched:
        term1 = jnp.sin(jnp.pi * w[:, 0])**2
        term3 = (w[:, -1] - 1.0)**2 * (1.0 + jnp.sin(2.0 * jnp.pi * w[:, -1])**2)
        wi = w[:, :-1]
        inner_sum = jnp.sum((wi - 1.0)**2 * (1.0 + 10.0 * jnp.sin(jnp.pi * wi + 1.0)**2), axis=1)
        return term1 + inner_sum + term3
    else:
        term1 = jnp.sin(jnp.pi * w[0])**2
        term3 = (w[-1] - 1.0)**2 * (1.0 + jnp.sin(2.0 * jnp.pi * w[-1])**2)
        wi = w[:-1]
        inner_sum = jnp.sum((wi - 1.0)**2 * (1.0 + 10.0 * jnp.sin(jnp.pi * wi + 1.0)**2))
        return term1 + inner_sum + term3

# ======================================================================
# 2. 初始化逻辑
# ======================================================================
def init_mpc_params(key, M, K, D_MAX):
    key_mu, key_L = random.split(key)
    # 在 [-10, 10] 范围内初始化均值
    initial_mu = random.uniform(key_mu, (M, K, D_MAX), minval=-10.0, maxval=10.0)
    # 初始化 Cholesky 因子 (控制初始搜索步长)
    initial_L_inv = jnp.tile(jnp.eye(D_MAX) * 1.5, (M, K, 1, 1))
    initial_v = jnp.zeros((M, K - 1))
    return initial_mu, initial_L_inv, initial_v

# ======================================================================
# 3. 主程序 (验证 Algorithm 4)
# ======================================================================
def main():
    # --- 参数配置 ---
    M_BLOCKS = 10      # 10个块
    K_COMP = 15        # 每个块的分量数
    D_MAX = 5          # 每块维度
    DIMS_TUPLE = (5,) * M_BLOCKS # 总计 50 维
    
    T_RUN = 1000       # 迭代次数
    DELTA_T = 0.4      # 学习率
    B_SAMPLES = 1000     # 采样数
    B_0_ELITE = 400     # 精英样本
    T_0_RESTART = 100  # 重启频率
    
    key = random.PRNGKey(42)
    key_init, key_solve = random.split(key)

    print(f">>> 启动 Levy 优化测试 (Algorithm 4): 总维度 = {M_BLOCKS * D_MAX}")
    init_mu, init_L, init_v = init_mpc_params(key_init, M_BLOCKS, K_COMP, D_MAX)
    current_context = jnp.array([0.0]) 

    # --- 执行优化 ---
    start_t = time.perf_counter()
    
    # 注意：v3 版本返回 4 个值，此处解包 history
    final_mu, final_L, final_pi, _ = mmog_igo_optimizer_mpc(
        key=key_solve, 
        T=T_RUN, 
        dt=DELTA_T, 
        M=M_BLOCKS, 
        K=K_COMP, 
        B=B_SAMPLES, 
        B0=B_0_ELITE, 
        dims=DIMS_TUPLE, 
        T_0=T_0_RESTART,
        fitness_fn_total=mpc_levy_fitness,
        initial_mu_k=init_mu, 
        initial_L_inv_k=init_L, 
        initial_v_k=init_v,
        context=current_context
    )
    
    final_mu.block_until_ready()
    duration = time.perf_counter() - start_t

    # --- 结果解析 ---
    # 提取每个块中概率（权重）最大的分量索引
    best_comp_indices = jnp.argmax(final_pi, axis=1) 
    
    def get_mu(m_idx, k_idx):
        return final_mu[m_idx, k_idx]
    best_means = vmap(get_mu)(jnp.arange(M_BLOCKS), best_comp_indices) 
    
    # 计算最终收敛点的适应度
    best_z = best_means.ravel()
    final_fitness = mpc_levy_fitness(best_z, current_context)

    # --- 打印结果 ---
    print("\n================ 求解结果 (Algorithm 4) ================")
    print(f"最终函数值 (Fitness): {final_fitness:.10f} (理论最优值应接近 0)")
    print(f"计算耗时: {duration:.4f}s")
    
    # 抽样打印前两个块的结果
    for m in range(min(2, M_BLOCKS)):
        comp_idx = best_comp_indices[m]
        print(f"块 {m} (最佳分量 {comp_idx}) 均值: {best_means[m]}")
    
    print("...")
    print(f"全维度均值偏离最优解(1.0)的范数: {jnp.linalg.norm(best_z - 1.0):.6f}")
    print("========================================================")

if __name__ == "__main__":
    main()