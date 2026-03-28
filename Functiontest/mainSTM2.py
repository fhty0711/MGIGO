import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import time

# 导入修改后的求解器 (请确保按照上述微调了 scan 的返回值)
from gmm_igo.MPCresetweightplot import mmog_igo_optimizer_mpc

def mpc_styblinski_tang_fitness(z_flattened, context):
    term = z_flattened**4 - 16.0 * z_flattened**2 + 5.0 * z_flattened 
    return jnp.sum(term)

def init_mpc_params(key, M, K, D_MAX):
    key_mu, _ = random.split(key)
    initial_mu = random.uniform(key_mu, (M, K, D_MAX), minval=-4.0, maxval=4.0)
    L_template = jnp.eye(D_MAX) * jnp.sqrt(2.0)
    initial_L_inv = jnp.tile(L_template, (M, K, 1, 1))
    return initial_mu, initial_L_inv, jnp.zeros((M, K - 1))

def run_and_plot():
    # 参数配置
    T_RUN = 1000
    T_0_RESTART = 200  # 设置较短的周期以观察多次重置
    M_BLOCKS = 8
    K_COMP = 10
    DIMS_TUPLE = (8, 8, 8, 8,8,8,8,8)
    D_MAX = max(DIMS_TUPLE)
    
    key = random.PRNGKey(42)
    init_mu, init_L, init_v = init_mpc_params(key, M_BLOCKS, K_COMP, D_MAX)
    
    print("正在运行优化并记录数据...")
    mu, L, pi, history = mmog_igo_optimizer_mpc(
        key=random.PRNGKey(0), T=T_RUN, dt=0.2, M=M_BLOCKS, K=K_COMP,
        B=100, B0=40, dims=DIMS_TUPLE, T_0=T_0_RESTART,
        fitness_fn_total=mpc_styblinski_tang_fitness,
        initial_mu_k=init_mu, initial_L_inv_k=init_L, initial_v_k=init_v,
        context=jnp.array([0.0])
    )

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Best Fitness (min f(z))', color='blue', linewidth=1.5)
    
    # 标记重置点
    for r in range(T_0_RESTART, T_RUN, T_0_RESTART):
        plt.axvline(x=r, color='red', linestyle='--', alpha=0.5, label='Reset Point' if r==T_0_RESTART else "")
        plt.text(r, plt.ylim()[1], 'Reset', color='red', rotation=90, verticalalignment='top')

    plt.title('Algorithm 4: Fitness Convergence with Reset Weighting')
    plt.xlabel('Iteration')
    plt.ylabel('Styblinski-Tang Fitness')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # 保存并显示
    plt.savefig('convergence_curve.png')
    print("曲线图已保存为 convergence_curve.png")
    plt.show()

if __name__ == "__main__":
    run_and_plot()