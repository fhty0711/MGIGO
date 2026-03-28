import jax
import jax.numpy as jnp
from jax import random
import time

# 导入修改后的求解器
from gmm_igo.MPCresetweight import mmog_igo_optimizer_mpc

# ======================================================================
# 1. 适应度函数 (Styblinski-Tang)
# ======================================================================
@jax.jit
def mpc_styblinski_tang_fitness(z_flattened, context):
    """
    Styblinski-Tang 函数: 全局最优解在每个维度约为 -2.9035, 函数值为 -39.166 * Dim
    注意: IGO-GMM Algorithm 3 倾向于极小化 f(z) (见 Step 18: increasing order)
    """
    term = z_flattened**4 - 16.0 * z_flattened**2 + 5.0 * z_flattened 
    return jnp.sum(term)

# ======================================================================
# 2. 初始化逻辑
# ======================================================================
def init_mpc_params(key, M, K, D_MAX):
    key_mu, _ = random.split(key)
    # 初始化均值 mu: 在 [-4, 4] 随机分布
    initial_mu = random.uniform(key_mu, (M, K, D_MAX), minval=-4.0, maxval=4.0)
    # 初始化 Cholesky 因子: 单位阵 * sqrt(2.0)
    L_template = jnp.eye(D_MAX) * jnp.sqrt(3.0)
    initial_L_inv = jnp.tile(L_template, (M, K, 1, 1))
    # 权重对数项
    initial_v = jnp.zeros((M, K - 1))
    return initial_mu, initial_L_inv, initial_v

# ======================================================================
# 3. 执行测试
# ======================================================================
def run_validation():
    # --- 参数配置 ---
    SEED = 42
    T_RUN = 1000         # 运行 500 轮，涵盖一次 T0=250 的重置
    DELTA_T = 0.2       # 学习率
    M_BLOCKS = 8        # 块数量
    K_COMP = 10          # 混合分量数
    DIMS_TUPLE = (8, 8, 8, 8, 8, 8, 8, 8) 
    D_MAX = max(DIMS_TUPLE)
    
    B_SAMPLES = 100     # 样本数
    B_0_ELITE = 45      # 精英样本数
    T_0_RESTART = 250   # 重置周期
    
    key = random.PRNGKey(SEED)
    key_init, key_solve = random.split(key)

    print(f"--- Algorithm 4 验证测试 ---")
    print(f"配置: Blocks={M_BLOCKS}, T0={T_0_RESTART}, Total_Dim={sum(DIMS_TUPLE)}")
    
    init_mu, init_L, init_v = init_mpc_params(key_init, M_BLOCKS, K_COMP, D_MAX)
    current_context = jnp.array([0.0]) 

    start_t = time.perf_counter()
    
    # 调用包含 Algo 4 的求解器
    final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        key=key_solve,
        T=T_RUN,
        dt=DELTA_T,
        M=M_BLOCKS,
        K=K_COMP,
        B=B_SAMPLES,
        B0=B_0_ELITE,
        dims=DIMS_TUPLE,
        T_0=T_0_RESTART,
        fitness_fn_total=mpc_styblinski_tang_fitness,
        initial_mu_k=init_mu,
        initial_L_inv_k=init_L,
        initial_v_k=init_v,
        context=current_context
    )
    
    final_mu.block_until_ready()
    duration = time.perf_counter() - start_t

    # --- 结果分析 ---
    print(f"\n计算耗时: {duration:.4f} 秒")
    
    # Styblinski-Tang 单维最优解
    GLOBAL_OPT_X = -2.903534
    TOTAL_DIM = sum(DIMS_TUPLE)
    THEORETICAL_MIN = -39.166 * TOTAL_DIM
    
    # 提取权重最高的分量作为最终解
    best_pi_indices = jnp.argmax(final_pi, axis=1)
    
    found_values = []
    print("\n优化结果 (检查各块均值是否接近 -2.903):")
    for m in range(M_BLOCKS):
        idx = best_pi_indices[m]
        mu_res = final_mu[m, idx, :DIMS_TUPLE[m]]
        found_values.append(mu_res)
        print(f"Block {m} (Comp {idx}): {mu_res} | Pi: {final_pi[m, idx]:.4f}")

    # 计算最终点的适应度
    final_solution_flat = jnp.concatenate([val for val in found_values])
    final_f = mpc_styblinski_tang_fitness(final_solution_flat, current_context)
    
    print("-" * 50)
    print(f"理论全局最小值: {THEORETICAL_MIN:.4f}")
    print(f"优化后最终函数值: {final_f:.4f}")
    
    # 验证是否收敛
    if final_f < THEORETICAL_MIN + 5.0:
        print(">> 结论: 算法成功收敛到全局最优附近。")
    else:
        print(">> 结论: 尚未完全收敛，可能需要更多迭代或调整步长。")

if __name__ == "__main__":
    run_validation()