import jax
import jax.numpy as jnp
from jax import random
from gmm_igo.MPCsolver import igo_mog_optimizer

# 1. 包装 Fitness 函数
def fitness_fn_unified(joint_sample, context):
    """
    joint_sample: [N * 3] 维向量 (t_in_1, v_pass_1, k_1, t_in_2, ...)
    """
    evaluator = context['evaluator']
    N = evaluator.num_agents
    # 重塑为 [N, 3] 矩阵供评估器使用
    joint_z = joint_sample.reshape(N, 3)
    
    # 调用您的勒贝格冲突计算逻辑
    cost = evaluator.compute_cost(
        joint_z, 
        context['s_in_list'], 
        context['v_0_list'], 
        context['weight_matrix']
    )
    return cost

# 2. 初始化与执行
N_agents = 3
total_dim = N_agents * 3
K_components = 4 # 建议 3-5 个分量以覆盖不同的通行顺序

# 设定初始均值 (例如所有车都尝试在 5s 进入冲突区)
initial_mu = jnp.tile(jnp.array([5.0, 10.0, 1.0]), N_agents) 
initial_mu_k = jnp.repeat(initial_mu[None, :], K_components, axis=0)
# 给不同分量加一点扰动，鼓励探索不同模态
initial_mu_k = initial_mu_k.at[1, 0].set(3.0) # 分量1假设车1先走

initial_L_inv = jnp.repeat(jnp.eye(total_dim)[None, :, :], K_components, axis=0) * 0.5
initial_pi = jnp.ones(K_components) / K_components

# 执行优化
final_mu, final_L, final_pi = igo_mog_optimizer(
    key=random.PRNGKey(42),
    T=50, delta_t=0.1, K=K_components, B=512, B_0=64,
    fitness_fn=fitness_fn_unified,
    initial_mu_k=initial_mu_k,
    initial_L_inv_k=initial_L_inv,
    initial_pi_k=initial_pi,
    context={
        'evaluator': evaluator_instance,
        's_in_list': s_in_list,
        'v_0_list': v_0_list,
        'weight_matrix': jnp.eye(N_agents) # 示例权重
    }
)