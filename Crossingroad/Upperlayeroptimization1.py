from jax import jit, vmap
import jax.numpy as jnp
from solver.MPCsolverM22 import mmog_igo_optimizer_mpc
from Costcomputation import fitness_fn_total

# 2. 初始化 (符合 M22 的维度要求 [M, K, D])
M_agents = 3
K_comps = 4
D_per_agent = 3

init_mu_m22 = jnp.zeros((M_agents, K_comps, D_per_agent))
# 初始化均值：[t_in, v_pass, k]
init_mu_m22 = init_mu_m22 + jnp.array([5.0, 8.0, 1.0]) 

# 初始 Cholesky 因子 [M, K, D, D]
init_L_m22 = jnp.tile(jnp.eye(D_per_agent), (M_agents, K_comps, 1, 1)) * 0.2

@jit
def full_fitness_fn(joint_samples, context):
    # ... 映射轨迹得到 all_masks ...
    
    # 1. 安全项 (重叠面积 * 时间核)
    j_safety = vmap(evaluator.compute_cost)(...) 
    
    # 2. 效率项 (目标速度追踪)
    v_diff = s_trajs_dot - context['v_ref'] 
    j_speed = jnp.sum(jnp.square(v_diff), axis=-1)
    
    # 3. 舒适项 (加速度/抖动)
    j_accel = jnp.sum(jnp.square(jnp.diff(s_trajs_dot)), axis=-1)
    
    # 4. 边界约束 (逻辑惩罚)
    # 如果参数超限，返回极大的 Cost
    j_limit = check_limits(samples) 

    # 总代价值
    total_cost = (lambda_s * j_safety + 
                  lambda_v * j_speed + 
                  lambda_a * j_accel + 
                  j_limit)
    return total_cost

# 执行优化
final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
    key=random.PRNGKey(0),
    T=100, dt=0.1, M=M_agents, K=K_comps, B=200, B0=80,
    dims=[3, 3, 3], # 每个块的维度
    T_0=10, # 每 10 步重置一次权重，强制探索
    fitness_fn_total=full_fitness_fn,
    initial_mu_k=init_mu_m22,
    initial_L_inv_k=init_L_m22,
    initial_v_k=None, # 内部会自动初始化为 0
    context={
        'evaluator': evaluator_instance,
        's_in_list': s_in_list,
        'v_0_list': v_0_list,
        'weight_matrix': jnp.ones((M_agents, M_agents))
    }
)