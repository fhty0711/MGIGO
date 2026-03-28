import jax
import jax.numpy as jnp
from jax import random, jit
import functools

# 假设你已经有了 MPCsolverM22.py 中的 mmog_igo_optimizer_mpc
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc


# ======================================================================
# 上层专用 cost 函数（functional 形式，包含 belief 加权）
# ======================================================================

@jit
def upper_stage_cost(zeta_flat, context):
    """
    zeta_flat: shape (D,) 例如 [A_ego, t_ego, A2, t2, A3, t3]
    context: dict 包含 belief, t_free, tau_conf, R_ref, L_turn 等
    """
    # 解包 zeta
    A_ego, t_ego = zeta_flat[0], zeta_flat[1]
    A2, t2         = zeta_flat[2], zeta_flat[3]
    A3, t3         = zeta_flat[4], zeta_flat[5]

    # 个体项（Ego）
    J_ind_ego = (t_ego - context['t_free_ego'])**2 + context['gamma'] * A_ego**2

    # 转弯最小时间约束（functional 思想）
    R_ego = context['R_ref_ego'] + context['beta_A'] * A_ego
    t_min_ego = context['L_turn_ego'] / jnp.sqrt(context['a_lat_max'] * R_ego)
    J_dyn_ego = context['w_dyn'] * jnp.maximum(0.0, t_min_ego - t_ego)**2

    J_ind = J_ind_ego + J_dyn_ego   # 可扩展其他代理，但上层主要关注 ego

    # 交互项：对其他车 belief 加权期望
    J_inter = 0.0

    # 与 agent2 的交互（示例）
    for m in range(3):  # agent2 有 3 个模式
        p = context['belief_agent2'][m]
        A2_m, t2_m = context['base_zeta_agent2'][m]   # base 值
        d_t = t_ego - t2_m
        d_A = A_ego - A2_m
        J_ij = jnp.exp(-context['beta'] * (d_t**2 + context['alpha'] * d_A**2))
        J_inter += p * J_ij

    # 与 agent3 同理（省略重复代码）
    # ...

    return J_ind + context['lambda_'] * J_inter


# ======================================================================
# 上层主函数
# ======================================================================

def upper_layer_igo_gmm(
    key,                  # jax.random.PRNGKey
    belief,               # dict: {'agent2': array[3], 'agent3': array[3]}
    ego_state,            # dict: {'pos':..., 'v':..., 'yaw':...}
    precomp,              # dict: t_free, tau_conf, R_ref, L_turn, ...
    zeta_prev=None,       # 上一次结果，用于 warm start
    T=400,                 # 迭代步数（上层低维，40~80 足够）
    dt=0.15,              # 学习率步长（上层可稍大）
    M=3,                  # 单块（上层维度低，不需要多块）
    K=6,                  # 混合高斯分量数（上层常用 4~8）
    B=64,                 # 样本数（上层低维，32~64 够用）
    B0=25,                 # 精英数
    dims=[6],             # 总维度（这里假设 6 维）
    T_0=100,               # 重置周期
):
    """
    上层 IGO-GMM 主入口
    返回：最优 zeta_ego (A_ego, t_ego), 完整 zeta, 最终 cost 等
    """

    # --------------------------------------------------
    # 1. 准备 context（传给 fitness_fn）
    # --------------------------------------------------
    context = {
        'belief_agent2': belief['agent2'],
        'belief_agent3': belief['agent3'],
        't_free_ego': precomp['t_free_ego'],   # 当前速度修正后的自由流时间
        'L_turn_ego': precomp['L_turn_ego'],
        'R_ref_ego': precomp['R_ref_ego'],
        'a_lat_max': 3.0,
        'beta_A': 0.8,          # 半径随 A 变化系数
        'gamma': 0.5,
        'w_dyn': 3.0,
        'lambda_': 0.8,
        'beta': 1.2,
        'alpha': 0.6,
        # base zeta for each mode of other agents
        'base_zeta_agent2': precomp['base_zeta_agent2'],  # list of tuples
        'base_zeta_agent3': precomp['base_zeta_agent3'],
    }

    # fitness_fn = upper_stage_cost 的 partial（带 context）
    fitness_fn = functools.partial(upper_stage_cost, context=context)

    # --------------------------------------------------
    # 2. 生成 warm start 初始均值 mu_init
    # --------------------------------------------------
    D = dims[0]  # 总维度，例如 6
    key_init, key_opt = random.split(key)

    # 基础初始点（可以从 zeta_prev 或 free-flow 开始）
    mu_base = jnp.zeros(D)
    if zeta_prev is not None:
        mu_base = jnp.array([
            zeta_prev['A_ego'], zeta_prev['t_ego'],
            zeta_prev['A2'],    zeta_prev['t2'],
            zeta_prev['A3'],    zeta_prev['t3']
        ])

    # 生成多样初始均值（K 个分量）
    mu_init = mu_base[None, :] + random.normal(key_init, (K, D)) * 0.4  # 小扰动

    # 初始精度矩阵（对角大方差）
    L_inv_init = jnp.eye(D)[None, None, :, :] * 2.0   # shape (M=1, K, D, D)
    v_init = jnp.zeros((M, K-1))                     # 混合权重 logit

    # --------------------------------------------------
    # 3. 调用你的自制求解器
    # --------------------------------------------------
    mu_opt, L_opt, pi_opt = mmog_igo_optimizer_mpc(
        key=key_opt,
        T=T,                    # 迭代次数
        dt=dt,
        M=M,                    # 通常 1
        K=K,
        B=B,
        B0=B0,
        dims=dims,
        T_0=T_0,
        fitness_fn_total=fitness_fn,
        initial_mu_k=mu_init[None, ...],   # 加 M=1 维度
        initial_L_inv_k=L_inv_init,
        initial_v_k=v_init,
        context=context                    # 虽然求解器不直接用，但保持接口一致
    )

    # --------------------------------------------------
    # 4. 提取结果
    # --------------------------------------------------
    # 取权重最大的分量作为最终结果
    best_k = jnp.argmax(pi_opt[0])   # M=1
    zeta_opt_flat = mu_opt[0, best_k, :]

    result = {
        'zeta_ego':   (zeta_opt_flat[0], zeta_opt_flat[1]),
        'zeta_agent2':(zeta_opt_flat[2], zeta_opt_flat[3]),
        'zeta_agent3':(zeta_opt_flat[4], zeta_opt_flat[5]),
        'pi_opt':     pi_opt[0],               # 混合权重
        'J_final':    fitness_fn(zeta_opt_flat),
        'belief':     belief.copy(),
    }

    return result