import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 导入你的双层分块混合系统优化器
#from gmm_igo.MPC_discrete_variables import mmog_igo_optimizer_mpc
from gmm_igo.MPC_discrete_variable1 import mmog_igo_optimizer_mpc
# ======================================================================
# 1. 简单离散-连续混合适应度函数：F(z, x) = f_z(x)
# ======================================================================
@jax.jit
def simple_mode_fitness(u_combined, ctx):
    """
    z 是离散变量。
    这里使用 3 个离散模式：
      z=0 -> f1(x) = x1^2 + x2^2
      z=1 -> f2(x) = (x1-2)^2 + (x2-2)^2
      z=2 -> f3(x) = sin(2*pi*x1) + cos(2*pi*x2)
    """
    del ctx

    N = 1
    D = 2

    sampled_modes = u_combined[:N].astype(jnp.int32)
    samples_continuous = u_combined[N:].reshape(N, D)

    z = sampled_modes[0]
    x = samples_continuous[0]

    f1 = jnp.sum(x ** 2)
    f2 = jnp.sum((x - 2.0) ** 2)
    f3 = jnp.sin( x[0]) + jnp.cos(x[1])

    return jnp.where(z == 0, f1, jnp.where(z == 1, f2, f3))

# ======================================================================
# 2. 初始化函数（离散与连续都按混合求解器接口准备）
# ======================================================================
def init_hybrid_params(key, N, max_M, K, D_MAX):
    key_mu, _ = random.split(key)

    initial_theta_logits = jnp.zeros((N, max_M))
    initial_mu = random.uniform(
        key_mu, (N, max_M, K, D_MAX), minval=-2.0, maxval=2.0
    )
    initial_L_inv = jnp.tile(jnp.eye(D_MAX)*0.5, (N, max_M, K, 1, 1))

    return initial_theta_logits, initial_mu, initial_L_inv

# ======================================================================
# 3. 运行长程混合目标测试
# ======================================================================
def run_hybrid_long_range_test():
    SEED = 999
    N_BLOCKS = 1
    MAX_M = 3
    K_COMP = 6
    DIMS_TUPLE = (2,)
    ACTIVE_MODES = (3,)

    T_TOTAL = 500
    ALPHA_DISCRETE = 0.1
    ALPHA_CONTINUOUS = 0.2
    B_SAMPLES = 1000
    B_0_ELITE = 200
    T_0_RESTART = 100
    
    key = random.PRNGKey(SEED)
    key_init, key_solve = random.split(key)
    
    init_theta_logits, init_mu, init_L = init_hybrid_params(
        key_init, N_BLOCKS, MAX_M, K_COMP, 2
    )
    current_context = jnp.array([0.0])
    
    print("="*60)
    print("🔍 启动长程混合目标测试")
    print(f"时域步数 N={N_BLOCKS}, 模态数 M={MAX_M}, GMM 分量 K={K_COMP}")
    print(f"样本量 B={B_SAMPLES}, 精英样本数 B0={B_0_ELITE}, 重置周期 T0={T_0_RESTART}")
    print("目标函数: z=0 -> x1^2+x2^2, z=1 -> (x1-2)^2+(x2-2)^2, z=2 -> sin(2πx1)+cos(2πx2)")
    print("="*60)
    
    start_t = time.perf_counter()
    final_theta, final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        key=key_solve,
        T=T_TOTAL,
        alpha_discrete=ALPHA_DISCRETE,
        alpha_continuous=ALPHA_CONTINUOUS,
        N=N_BLOCKS,
        max_M=MAX_M,
        K=K_COMP,
        B=B_SAMPLES,
        B0=B_0_ELITE,
        dims=DIMS_TUPLE,
        active_modes=ACTIVE_MODES,
        T_0=T_0_RESTART,
        fitness_fn_total=simple_mode_fitness,
        initial_theta_logits_k=init_theta_logits,
        initial_mu_k=init_mu, initial_L_inv_k=init_L,
        context=current_context
    )
    final_theta.block_until_ready()
    duration = time.perf_counter() - start_t
    
    print(f"检测运行完成，耗时: {duration:.4f} 秒\n")
    print("🎯 --- 长程混合目标收敛结果分析 ---")
    print("-" * 60)
    print("【单块决策】最终模态选择概率:")
    print(f"  -> z=0 / f1(x): {final_theta[0, 0]:.4f}")
    print(f"  -> z=1 / f2(x): {final_theta[0, 1]:.4f}")
    print(f"  -> z=2 / f3(x): {final_theta[0, 2]:.4f}")

    for m in range(MAX_M):
        best_comp = jnp.argmax(final_pi[0, m])
        mu_m = final_mu[0, m, best_comp]
        print(f"  └─ 模态 {m} 下最佳 GMM 均值: ({mu_m[0]:.2f}, {mu_m[1]:.2f})" )
        print(f"     对应的最佳连续解适应度: {simple_mode_fitness(jnp.concatenate([jnp.array([m]), mu_m]), current_context):.4f}")
    print("-" * 60)


@jax.jit
def simulate_contact_sequence_kinematics(discrete_modes, continuous_params):
    """
    为不同接触模式定义分段运动学模型。
    每段轨迹长度约 1s，每段用 20 个连续变量描述 20 个子步的控制轨迹。
    返回: 轨迹状态序列 + 各项物理指标
    """
    num_segments = 3
    substeps_per_segment = 20
    dt = 1.0 / substeps_per_segment

    # 每个离散状态对应一套运动学模型参数。
    # 这样三段轨迹虽然都拥有 20 个连续变量，但每段的动力学形式不同。
    mode_max_accel = jnp.array([0.8, 2.5, 3.2, 4.0, 1.2])
    mode_stability = jnp.array([0.95, 0.85, 0.72, 0.60, 0.90])
    mode_energy_factor = jnp.array([0.6, 1.0, 1.35, 1.8, 0.75])

    seg_target_vx = jnp.array([
        [0.35, 0.95, 1.25, 1.45, 0.65],
        [0.45, 1.10, 1.45, 1.60, 0.75],
        [0.55, 1.20, 1.55, 1.70, 0.85],
    ])
    seg_target_vy = jnp.array([
        [0.00, 0.02, -0.02, 0.03, 0.00],
        [0.00, 0.03, -0.03, 0.04, 0.01],
        [0.00, 0.04, -0.04, 0.05, 0.02],
    ])
    seg_drive_gain = jnp.array([0.45, 0.60, 0.72])
    seg_lateral_gain = jnp.array([0.18, 0.24, 0.30])
    seg_drag = jnp.array([0.05, 0.10, 0.16])
    seg_energy_bias = jnp.array([0.35, 0.45, 0.60])

    def scan_segment(carry, seg_idx):
        x, y, vx_curr, vy_curr = carry
        m = discrete_modes[seg_idx]
        segment_controls = continuous_params[seg_idx]
        vx_ref = seg_target_vx[seg_idx, m]
        vy_ref = seg_target_vy[seg_idx, m]
        drive_gain = seg_drive_gain[seg_idx]
        lateral_gain = seg_lateral_gain[seg_idx]
        drag = seg_drag[seg_idx]
        energy_bias = seg_energy_bias[seg_idx]

        def scan_substep(sub_carry, sub_idx):
            x_s, y_s, vx_s, vy_s = sub_carry
            u = segment_controls[sub_idx]

            # 三段采用不同的动力学模型:
            # segment 0: 直接驱动 + 轻微阻尼
            # segment 1: 非线性饱和驱动 + 速度回归
            # segment 2: 更强横向耦合 + 额外阻力
            vx_cmd = vx_ref + drive_gain * jnp.tanh(u)
            vy_cmd = vy_ref + lateral_gain * jnp.sin(u)

            def dynamics_model_0(args):
                x_i, y_i, vx_i, vy_i = args
                ax_i = jnp.clip(vx_cmd - vx_i - drag * vx_i, -mode_max_accel[m], mode_max_accel[m])
                ay_i = jnp.clip(vy_cmd - vy_i - drag * vy_i, -mode_max_accel[m] * 0.8, mode_max_accel[m] * 0.8)
                return ax_i, ay_i

            def dynamics_model_1(args):
                x_i, y_i, vx_i, vy_i = args
                ax_i = jnp.clip(jnp.tanh(vx_cmd - vx_i) * (1.0 + 0.5 * jnp.abs(vy_i)), -mode_max_accel[m], mode_max_accel[m])
                ay_i = jnp.clip(jnp.tanh(vy_cmd - vy_i) * (0.8 + 0.2 * jnp.abs(vx_i)), -mode_max_accel[m] * 0.8, mode_max_accel[m] * 0.8)
                return ax_i, ay_i

            def dynamics_model_2(args):
                x_i, y_i, vx_i, vy_i = args
                ax_i = jnp.clip(vx_cmd - vx_i - drag * vx_i + 0.15 * jnp.sin(y_i + u), -mode_max_accel[m], mode_max_accel[m])
                ay_i = jnp.clip(vy_cmd - vy_i - drag * vy_i + 0.10 * jnp.cos(x_i - u), -mode_max_accel[m] * 0.8, mode_max_accel[m] * 0.8)
                return ax_i, ay_i

            ax, ay = lax.switch(
                seg_idx,
                (dynamics_model_0, dynamics_model_1, dynamics_model_2),
                (x_s, y_s, vx_s, vy_s),
            )

            vx_new = vx_s + ax * dt
            vy_new = vy_s + ay * dt
            x_new = x_s + vx_new * dt
            y_new = y_s + vy_new * dt

            stability_penalty = (1.0 - mode_stability[m]) * (0.2 + jnp.abs(u)) + 0.05 * seg_idx * jnp.abs(vy_new)
            energy_term = mode_energy_factor[m] * (energy_bias + u ** 2 + 0.5 * vx_new ** 2 + 0.2 * vy_new ** 2)

            new_sub_carry = (x_new, y_new, vx_new, vy_new)
            aux = jnp.array([x_new, y_new, vx_new, vy_new, stability_penalty, energy_term])
            return new_sub_carry, aux

        final_segment_state, segment_info = lax.scan(
            scan_substep,
            (x, y, vx_curr, vy_curr),
            jnp.arange(substeps_per_segment),
        )

        segment_positions = segment_info[:, :4]
        segment_stability = segment_info[:, 4]
        segment_energy = segment_info[:, 5]
        return final_segment_state, (segment_positions, segment_stability, segment_energy)

    final_state, traj_info = lax.scan(scan_segment, (0.0, 0.0, 0.0, 0.0), jnp.arange(num_segments))

    positions = traj_info[0].reshape(num_segments * substeps_per_segment, 4)
    stability_penalties = traj_info[1].reshape(-1)
    energy_terms = traj_info[2].reshape(-1)

    return {
        'final_pos': final_state[:2],
        'final_vel': final_state[2:],
        'positions': positions,
        'stability_penalties': stability_penalties,
        'energy_terms': energy_terms,
        'mode_energy_factor': mode_energy_factor[discrete_modes]
    }


@jax.jit
def contact_sequence_fitness(u_combined, context):
    """基于运动学模拟的真实接触序列 cost"""
    del context
    
    num_segments = 3
    continuous_dim = 20
    discrete_modes = u_combined[:num_segments].astype(jnp.int32)
    continuous_params = u_combined[num_segments:].reshape(num_segments, continuous_dim)
    
    # === 运动学模拟 ===
    sim = simulate_contact_sequence_kinematics(discrete_modes, continuous_params)
    
    # 目标：三段总计向前推进约 5 米，侧向接近 0
    desired_final_x = 5.0
    desired_final_y = 0.0
    segment_targets = jnp.array([0.45, 1.15, 1.45, 1.65, 0.75])
    
    # ====================== Cost 计算 ======================
    # 1. 终点跟踪误差
    pos_error = (sim['final_pos'][0] - desired_final_x)**2 * 2.0 + \
                (sim['final_pos'][1] - desired_final_y)**2 * 5.0
    
    # 2. 分段连续 cost：每段 20 个连续变量共同决定该段轨迹
    segment_velocity_cost = 0.0
    segment_smooth_cost = 0.0
    for seg in range(num_segments):
        seg_controls = continuous_params[seg]
        m = discrete_modes[seg]
        segment_velocity_cost = segment_velocity_cost + jnp.mean((seg_controls - segment_targets[m]) ** 2)
        segment_smooth_cost = segment_smooth_cost + jnp.mean((seg_controls[1:] - seg_controls[:-1]) ** 2)
    
    # 3. 稳定性
    stability_cost = jnp.mean(sim['stability_penalties']) * 18.0
    
    # 4. 能量消耗（不同模式不同系数）
    energy_cost = jnp.mean(sim['energy_terms'])
    
    # 5. 模式切换平滑
    #mode_diff = jnp.abs(discrete_modes[1:] - discrete_modes[:-1])
    #switch_cost = jnp.sum(mode_diff) * 4.0
    #big_jump = jnp.sum(mode_diff > 2) * 20.0
    
    # 6. 总 Cost
    total_cost = (
        12.0 * pos_error +
        12.0 * segment_velocity_cost +
        6.0  * segment_smooth_cost +
        18.0 * stability_cost +
        6.0  * energy_cost 
    )
    
    return total_cost


def run_contact_sequence_test():
    SEED = 2026
    N_BLOCKS = 3
    MAX_M = 5
    K_COMP = 3
    D_CONT_PER_STEP = 20

    T_TOTAL = 500
    ALPHA_DISCRETE = 0.1
    ALPHA_CONTINUOUS = 0.15
    B_SAMPLES = 500
    B_0_ELITE = 100
    T_0_RESTART = 100
    key = random.PRNGKey(SEED)
    key_init, key_solve = random.split(key)

    init_theta_logits, init_mu, init_L = init_hybrid_params(
        key_init, N_BLOCKS, MAX_M, K_COMP, D_CONT_PER_STEP
    )
    current_context = jnp.array([0.0])

    print("=" * 60)
    print("🔍 启动接触序列目标测试")
    print(f"时域步数 N={N_BLOCKS}, 模态数 M={MAX_M}, GMM 分量 K={K_COMP}")
    print(f"样本量 B={B_SAMPLES}, 精英样本数 B0={B_0_ELITE}, 重置周期 T0={T_0_RESTART}")
    print("目标函数: 三段轨迹的终点跟踪 + 分段连续 cost + 稳定性 + 能耗 + 切换平滑")
    print("每个时域块都有 5 种接触模式，连续变量为 20 维分段轨迹控制量")
    print("=" * 60)

    start_t = time.perf_counter()
    final_theta, final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        key=key_solve,
        T=T_TOTAL,
        alpha_discrete=ALPHA_DISCRETE,
        alpha_continuous=ALPHA_CONTINUOUS,
        N=N_BLOCKS,
        max_M=MAX_M,
        K=K_COMP,
        B=B_SAMPLES,
        B0=B_0_ELITE,
        dims=(D_CONT_PER_STEP,) * N_BLOCKS,
        active_modes=(MAX_M,) * N_BLOCKS,
        T_0=T_0_RESTART,
        fitness_fn_total=contact_sequence_fitness,
        initial_theta_logits_k=init_theta_logits,
        initial_mu_k=init_mu,
        initial_L_inv_k=init_L,
        context=current_context,
    )
    final_theta.block_until_ready()
    duration = time.perf_counter() - start_t

    best_modes = jnp.argmax(final_theta, axis=1)
    print(f"检测运行完成，耗时: {duration:.4f} 秒\n")
    print("🎯 --- 接触序列目标优化结果分析 ---")
    print("-" * 60)
    print("【时域决策】每步最优接触模式:")
    mode_labels = ("Static", "Trot", "Bound", "Gallop", "Crawl")
    selected_controls = []

    for step in range(N_BLOCKS):
        mode_id = int(best_modes[step])
        best_comp = int(jnp.argmax(final_pi[step, mode_id]))
        mu_step = final_mu[step, mode_id, best_comp]
        selected_controls.append(mu_step)
        print(
            f"  t={step:02d} -> mode={mode_id} ({mode_labels[mode_id]}), "
            f"p={final_theta[step, mode_id]:.4f}"
        )
        print(
            f"       best GMM 均值控制: {mu_step}, "
        )

    full_sequence = jnp.concatenate(
        [
            best_modes.astype(jnp.float32),
            jnp.stack(selected_controls, axis=0).reshape(-1),
        ]
    )
    total_fitness = float(contact_sequence_fitness(full_sequence, current_context))
    print(f"整体序列 fitness: {total_fitness:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    run_contact_sequence_test()