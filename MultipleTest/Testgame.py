import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import random, vmap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 确保路径正确
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 假设你的目录结构中 MPC_G_S_V 在 gmm_igo 包下
from gmm_igo.MPC_G_S_V import mmog_igo_rne_solver

# --- 超参数配置 (保持用户调整后的参数) ---
SEED = 42
T = 300         
DT = 0.15        
M = 2           
K = 3           
B = 60          
B0 = 20         
T0 = 100         
HORIZON = 10    
M_inner = 30    
T_mpc = 20
DIMS = (HORIZON, HORIZON) 
dt_m = 1.0  # 用户定义的动力学步长
DT_C = 0.05  # 内部积分的时间步长 (更细粒度)
SUB_STEPS = int(dt_m / DT_C)  # 每个控制周期内的积分步数
MAX_ACC = 3.0
MAX_SPEED = 20.0

# --- 向量化动力学 ---

def get_axis_motion(init_pos_scalar, init_vel_scalar, acc_sequence):
    """
    严格的一维积分逻辑。
    确保 init_vel_scalar 是一个标量。
    """
    acc = jnp.clip(acc_sequence, -MAX_ACC, MAX_ACC)
    acc_high_res = jnp.repeat(acc, SUB_STEPS)
    
    # 速度积分：v = v0 + sum(a*dt)
    vel_incremental = jnp.cumsum(acc_high_res) * DT_C
    vel_high_res = jnp.clip(init_vel_scalar + vel_incremental, 0.0, MAX_SPEED)
    
    # 位移积分：p = p0 + sum(v*dt)
    pos_high_res = init_pos_scalar + jnp.cumsum(vel_high_res) * DT_C
    return pos_high_res, vel_high_res, acc_high_res

def get_high_res_trajectory(init_pos, init_vel, acc_sequence, agent_idx):
    """
    修正点：根据 agent_idx 提取正确的初速标量
    Agent 0: 水平运动 (x轴), 关心 vx (index 0)
    Agent 1: 垂直运动 (y轴), 关心 vy (index 1)
    """
    total_steps = HORIZON * SUB_STEPS
    if agent_idx == 0:
        # Agent 0 运动在 x 轴，取 init_pos[0] 和 init_vel[0]
        pos_high_res, _, _ = get_axis_motion(init_pos[0], init_vel[0], acc_sequence)
        return jnp.stack([pos_high_res, jnp.full((total_steps,), init_pos[1])], axis=1)
    else:
        # Agent 1 运动在 y 轴，取 init_pos[1] 和 init_vel[1]
        pos_high_res, _, _ = get_axis_motion(init_pos[1], init_vel[1], acc_sequence)
        return jnp.stack([jnp.full((total_steps,), init_pos[0]), pos_high_res], axis=1)

def compute_mpc_cost(joint_sample, context, agent_idx):
    actions = joint_sample.reshape((M, HORIZON))
    
    # 修正点：显式传递 agent_idx 确保动力学匹配
    traj0_high = get_high_res_trajectory(context["init_pos"][0], context["init_vel"][0], actions[0], 0)
    traj1_high = get_high_res_trajectory(context["init_pos"][1], context["init_vel"][1], actions[1], 1)
    
    my_traj = traj0_high if agent_idx == 0 else traj1_high
    my_goal = context["goals"][agent_idx]
    my_actions = actions[agent_idx]

    # 1. 目标代价
    dist_to_goal = jnp.sum((my_traj - my_goal)**2, axis=1)
    running_state_cost = jnp.mean(dist_to_goal)
    control_cost = 0.1 * jnp.mean(my_actions**2) # 略微提高 control cost 增加稳定性

    # 2. 碰撞代价 (Soft)
    dist_between = jnp.sqrt(jnp.sum((traj0_high - traj1_high)**2, axis=1) + 1e-6)
    collision_cost = jnp.sum(jnp.where(dist_between < 3.0, 5.0, 0.0))

    # 3. 冲突区代价 (十字路口核心)
    in_zone_0 = (traj0_high[:, 0] > -5.0) & (traj0_high[:, 0] < 5.0)
    in_zone_1 = (traj1_high[:, 1] > -5.0) & (traj1_high[:, 1] < 5.0)
    conflict_cost = jnp.sum((in_zone_0 & in_zone_1).astype(jnp.float32) * 500.0)

    # 4. 终端代价
    terminal_cost = 100.0 * jnp.sum((my_traj[-1] - my_goal)**2)

    return running_state_cost + control_cost + collision_cost + conflict_cost + terminal_cost

def propagate_one_mpc_step(init_pos, init_vel, executed_controls):
    """
    修正点：严格根据轴向更新位置和速度向量
    """
    # Agent 0 (x轴)
    p0_x, v0_x, _ = get_axis_motion(init_pos[0, 0], init_vel[0, 0], jnp.array([executed_controls[0]]))
    next_pos_0 = init_pos[0].at[0].set(p0_x[-1])
    next_vel_0 = init_vel[0].at[0].set(v0_x[-1])
    
    # Agent 1 (y轴)
    p1_y, v1_y, _ = get_axis_motion(init_pos[1, 1], init_vel[1, 1], jnp.array([executed_controls[1]]))
    next_pos_1 = init_pos[1].at[1].set(p1_y[-1])
    next_vel_1 = init_vel[1].at[1].set(v1_y[-1])
    
    return jnp.stack([next_pos_0, next_pos_1]), jnp.stack([next_vel_0, next_vel_1])

# --- 辅助函数 ---
def map_action_to_physical(raw_action):
    return jnp.clip(raw_action, -MAX_ACC, MAX_ACC)

def make_warm_start_mu(best_action, k_count, key):
    tail_value = random.uniform(key, (), minval=-3.0, maxval=3.0)
    shifted_action = jnp.concatenate([best_action[1:], jnp.array([tail_value])])
    return jnp.repeat(shifted_action[None, :], k_count, axis=0)

def make_identity_l_inv(m_count, k_count, horizon):
    return jnp.stack([jnp.eye(horizon)] * (m_count * k_count)).reshape(m_count, k_count, horizon, horizon)

def fitness_fn_agent0(idx, joint_sample, context):
    return compute_mpc_cost(joint_sample, context, agent_idx=0)

def fitness_fn_agent1(idx, joint_sample, context):
    return compute_mpc_cost(joint_sample, context, agent_idx=1)


def select_min_cost_component(agent_idx, final_mu, current_context, other_actions):
    candidate_costs = []
    for comp_idx in range(final_mu.shape[1]):
        joint_actions = []
        for other_idx in range(M):
            if other_idx == agent_idx:
                joint_actions.append(final_mu[comp_idx])
            else:
                joint_actions.append(other_actions[other_idx])
        joint_sample = jnp.stack(joint_actions).reshape(-1)
        candidate_costs.append(compute_mpc_cost(joint_sample, current_context, agent_idx))

    candidate_costs = jnp.array(candidate_costs)
    best_k = int(jnp.argmin(candidate_costs))
    return best_k, candidate_costs


def visualize_macro_vs_micro_convergence(macro_pi_history, micro_metrics_sample, sample_mpc_step=5):
    """
    完美的双层博弈诊断图：
    左侧展示宏观 Receding Horizon 过程中意图权重的流变；
    右侧展示在博弈最激烈的一步中，微观 RNE 求解器内部的真实纳什均衡收敛路径。
    """
    import numpy as np
    output_path = Path(__file__).with_name("game_convergence_diagnostics.png")
    
    macro_pi_history = np.array(macro_pi_history) # Shape: (T_mpc, M, K)
    micro_pi = jax.device_get(micro_metrics_sample["pi"]) # Shape: (T_inner_iterations, M, K)
    micro_cost = jax.device_get(micro_metrics_sample["mean_fitness"]) # Shape: (T_inner_iterations, M)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    
    # ======= 第一层：左侧宏观 Receding Horizon 意图流变图 =======
    macro_steps = np.arange(T_mpc)
    for i in range(M):
        ax = axes[i, 0]
        for k in range(K):
            ax.plot(macro_steps, macro_pi_history[:, i, k], marker='o', linestyle='-', linewidth=2, label=f"Component {k}")
        ax.axvline(sample_mpc_step, color='red', linestyle='--', alpha=0.7, label=f"解剖切片点 (Step {sample_mpc_step})")
        ax.set_title(f"Agent i finally did pi_k at (Receding Horizon)")
        ax.set_xlabel("MPC Time Step")
        ax.set_ylabel("Final Probability ($\pi_k$)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # ======= 第二层：右侧微观 RNE 求解器收敛诊断（切片解剖） =======
    micro_iters = np.arange(micro_pi.shape[0])
    
    # 右上：解剖点步内两个 Agent 的多模态权重演化
    ax_micro_pi = axes[0, 1]
    for k in range(K):
        ax_micro_pi.plot(micro_iters, micro_pi[:, 0, k], linestyle='-', linewidth=1.8, label=f"A0-Comp {k}")
        ax_micro_pi.plot(micro_iters, micro_pi[:, 1, k], linestyle='--', linewidth=1.8, label=f"A1-Comp {k}")
    ax_micro_pi.set_title(f"What happend to pi_k at Step {sample_mpc_step}? (RNE Solver Internal)")
    ax_micro_pi.set_xlabel("RNE Solver Internal Iterations ($t$)")
    ax_micro_pi.set_ylabel("Probability ($\pi_k$)")
    ax_micro_pi.grid(True, alpha=0.3)
    ax_micro_pi.legend(loc="upper right", ncol=2)

    # 右下：解剖点步内博弈期望代价收敛情况（验证纳什驻点）
    ax_micro_cost = axes[1, 1]
    ax_micro_cost.plot(micro_iters, micro_cost[:, 0], color="tab:blue", linewidth=2.2, label="Agent 0 Expected Cost")
    ax_micro_cost.plot(micro_iters, micro_cost[:, 1], color="tab:orange", linewidth=2.2, label="Agent 1 Expected Cost")
    ax_micro_cost.set_title(f"Expected Cost Convergence at {sample_mpc_step}")
    ax_micro_cost.set_xlabel("RNE Solver Internal Iterations ($t$)")
    ax_micro_cost.set_ylabel("Evaluated Expected Cost")
    ax_micro_cost.grid(True, alpha=0.3)
    ax_micro_cost.legend(loc="best")

    fig.suptitle("GMM-RNE Information Theoretic Game Convergence Diagnostics", fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"already saved to {output_path}")


def visualize_results(history_positions, history_velocities, history_accelerations, goals):
    # (保持你原有轨迹物理分析代码不动)
    output_path = Path(__file__).with_name("testgame_trajectory.png")
    gif_path = Path(__file__).with_name("testgame_trajectory.gif")
    history_np = jax.device_get(history_positions)
    history_vel_np = jax.device_get(history_velocities)
    history_acc_np = jax.device_get(history_accelerations)
    goals_np = jax.device_get(goals)
    all_dists = jnp.linalg.norm(history_positions[:, 0] - history_positions[:, 1], axis=1)
    min_dist = jnp.min(all_dists)
    min_dist_t = jnp.argmin(all_dists) * dt_m
    time_steps_np = jnp.arange(history_np.shape[0]) * dt_m
    all_dists_np = jax.device_get(all_dists)

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    ax_traj = fig.add_subplot(grid[:, 0])
    ax_dist = fig.add_subplot(grid[:, 1])
    ax_pos0 = fig.add_subplot(grid[0, 2])
    ax_pos1 = fig.add_subplot(grid[1, 2])

    ax_traj.plot(history_np[:, 0, 0], history_np[:, 0, 1], label="Agent 0 path", linewidth=2.2)
    ax_traj.plot(history_np[:, 1, 0], history_np[:, 1, 1], label="Agent 1 path", linewidth=2.2)
    ax_traj.scatter(history_np[0, :, 0], history_np[0, :, 1], c=["tab:blue", "tab:orange"], marker="o", s=70, label="Start")
    ax_traj.scatter(goals_np[:, 0], goals_np[:, 1], c=["tab:blue", "tab:orange"], marker="*", s=140, label="Goal")
    ax_traj.set_title("High-res trajectories")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis("equal")
    ax_traj.legend(loc="best")

    ax_dist.plot(time_steps_np, all_dists_np, color="tab:green", linewidth=2.2)
    ax_dist.axhline(1.0, color="tab:red", linestyle="--")
    ax_dist.set_title("Separation over MPC steps")
    ax_dist.grid(True, alpha=0.3)

    ax_pos0.plot(time_steps_np, history_np[:, 0, 0], color="tab:blue", linewidth=2.2)
    ax_pos0.set_title("Agent 0 Position")
    ax_pos0.grid(True, alpha=0.3)

    ax_pos1.plot(time_steps_np, history_np[:, 1, 1], color="tab:orange", linewidth=2.2)
    ax_pos1.set_title("Agent 1 Position")
    ax_pos1.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved trajectory plot to: {output_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=160, constrained_layout=True)
    ax_traj, ax_speed = axes

    ax_traj.set_facecolor("#f7f3ea")
    x_min = float(jnp.min(history_np[:, :, 0]))
    x_max = float(jnp.max(history_np[:, :, 0]))
    y_min = float(jnp.min(history_np[:, :, 1]))
    y_max = float(jnp.max(history_np[:, :, 1]))
    x_pad = max((x_max - x_min) * 0.15, 10.0)
    y_pad = max((y_max - y_min) * 0.15, 10.0)
    ax_traj.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_traj.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")
    ax_traj.set_title("Testgame Trajectory Over Time")
    ax_traj.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

    colors = ["tab:blue", "tab:orange"]
    labels = ["Agent 0", "Agent 1"]
    path_lines = []
    path_markers = []
    for color, label in zip(colors, labels):
        (line,) = ax_traj.plot([], [], color=color, linewidth=2.4, label=label)
        (marker,) = ax_traj.plot([], [], color=color, marker="o", markersize=8)
        path_lines.append(line)
        path_markers.append(marker)

    ax_traj.scatter(goals_np[:, 0], goals_np[:, 1], c=colors, marker="*", s=140, label="Goal")
    ax_traj.legend(loc="best")

    speed_0 = history_vel_np[:, 0, 0]
    speed_1 = history_vel_np[:, 1, 1]
    time_axis = jnp.arange(history_np.shape[0]) * dt_m
    ax_speed.set_facecolor("#f7f3ea")
    ax_speed.set_xlabel("Time [s]")
    ax_speed.set_ylabel("Speed [m/s]")
    ax_speed.set_title("Agent Speeds")
    ax_speed.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax_speed.set_xlim(float(time_axis[0]), float(time_axis[-1]))
    speed_min = float(jnp.minimum(jnp.min(speed_0), jnp.min(speed_1)))
    speed_max = float(jnp.maximum(jnp.max(speed_0), jnp.max(speed_1)))
    speed_pad = max((speed_max - speed_min) * 0.15, 2.0)
    ax_speed.set_ylim(speed_min - speed_pad, speed_max + speed_pad)
    (speed_line_0,) = ax_speed.plot([], [], color=colors[0], linewidth=2.4, label="Agent 0 speed")
    (speed_line_1,) = ax_speed.plot([], [], color=colors[1], linewidth=2.4, label="Agent 1 speed")
    (speed_marker_0,) = ax_speed.plot([], [], color=colors[0], marker="o", markersize=7)
    (speed_marker_1,) = ax_speed.plot([], [], color=colors[1], marker="o", markersize=7)
    ax_speed.legend(loc="best")

    def init():
        for line, marker in zip(path_lines, path_markers):
            line.set_data([], [])
            marker.set_data([], [])
        speed_line_0.set_data([], [])
        speed_line_1.set_data([], [])
        speed_marker_0.set_data([], [])
        speed_marker_1.set_data([], [])
        return (*path_lines, *path_markers, speed_line_0, speed_line_1, speed_marker_0, speed_marker_1)

    def update(frame_idx):
        for agent_idx in range(2):
            traj = history_np[: frame_idx + 1, agent_idx]
            path_lines[agent_idx].set_data(traj[:, 0], traj[:, 1])
            path_markers[agent_idx].set_data([traj[-1, 0]], [traj[-1, 1]])

        speed_line_0.set_data(time_axis[: frame_idx + 1], speed_0[: frame_idx + 1])
        speed_line_1.set_data(time_axis[: frame_idx + 1], speed_1[: frame_idx + 1])
        speed_marker_0.set_data([time_axis[frame_idx]], [speed_0[frame_idx]])
        speed_marker_1.set_data([time_axis[frame_idx]], [speed_1[frame_idx]])
        return (*path_lines, *path_markers, speed_line_0, speed_line_1, speed_marker_0, speed_marker_1)

    anim = FuncAnimation(fig, update, frames=history_np.shape[0], init_func=init, interval=220, blit=True)
    anim.save(gif_path, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"Saved trajectory gif to: {gif_path}")

# --- 主仿真循环 ---

def run_simulation():
    init_pos = jnp.array([[-100.0, 0.0], [0.0, -100.0]]) 
    goals = jnp.array([[100.0, 0.0], [0.0, 100.0]])    
    current_vel = jnp.array([[10.0, 0.0], [0.0, 10.0]])

    key = random.PRNGKey(SEED)
    f_fns = (fitness_fn_agent0, fitness_fn_agent1)

    current_pos = init_pos
    current_mu_init = random.normal(key, (M, K, HORIZON)) * 3.0
    current_L_inv_init = make_identity_l_inv(M, K, HORIZON)
    
    executed_history = [current_pos]
    executed_velocity_history = [current_vel]
    executed_acceleration_history = [jnp.zeros((M,))]

    # 新增专门的宏观容器，用来收集每一个全局 MPC 步最终收敛出的 pi 权重
    macro_pi_collected = []
    
    # 选定第 6 步（通常是两车进入时空交叉区域碰撞代价激发、博弈对抗最激烈的时候）进行微观切片记录
    CHOSEN_SLICE_STEP = 6
    micro_metrics_sample = None

    for mpc_step in range(T_mpc):
        key, solve_key = random.split(key)
        current_context = {"init_pos": current_pos, "init_vel": current_vel, "goals": goals}

        final_mu, final_L, final_pi, metrics_history = mmog_igo_rne_solver(
            solve_key, T, DT, M, K, B, B0, DIMS, T0,
            f_fns, current_mu_init, current_L_inv_init, current_context, M_inner
        )

        # 核心保存点：记录宏观演化和特定步的微观切片
        macro_pi_collected.append(final_pi) 
        if mpc_step == CHOSEN_SLICE_STEP:
            micro_metrics_sample = metrics_history

        executed_controls = []
        next_mu_blocks = []
        chosen_sequences = [final_mu[i, 0] for i in range(M)]

        for i in range(M):
            key, warm_key = random.split(key)
            best_k, component_costs = select_min_cost_component(i, final_mu[i], current_context, chosen_sequences)
            best_action = final_mu[i, best_k]
            chosen_sequences[i] = best_action
            ctrl = map_action_to_physical(best_action[0])
            executed_controls.append(ctrl)
            next_mu_blocks.append(make_warm_start_mu(best_action, K, warm_key))

        current_pos, current_vel = propagate_one_mpc_step(current_pos, current_vel, jnp.array(executed_controls))
        executed_history.append(current_pos)
        executed_velocity_history.append(current_vel)
        executed_acceleration_history.append(jnp.array(executed_controls))
        current_mu_init = jnp.stack(next_mu_blocks)
        
        print(f"Step {mpc_step:02d} | A0 Pos: {current_pos[0,0]:.2f} | A1 Pos: {current_pos[1,1]:.2f} | Ctrls: {executed_controls}")

    executed_history = jnp.stack(executed_history)
    executed_velocity_history = jnp.stack(executed_velocity_history)
    executed_acceleration_history = jnp.stack(executed_acceleration_history)

    # 1. 绘制物理空间轨迹
    visualize_results(executed_history, executed_velocity_history, executed_acceleration_history, goals)
    
    # 2. 调用全新的双层架构收敛性绘制函数
    if micro_metrics_sample is not None:
        visualize_macro_vs_micro_convergence(macro_pi_collected, micro_metrics_sample, sample_mpc_step=CHOSEN_SLICE_STEP)

if __name__ == "__main__":
    run_simulation()