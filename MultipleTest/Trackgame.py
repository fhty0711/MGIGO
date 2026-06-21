import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import lax, random, jit, vmap
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 确保路径指向公共基础库
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MultipleTest.Transformation import (
    stable_single_car_step,
    stable_lon_car_step,
    pairwise_footprint_overlap_cost,
    low_pass_filter_sequence,
    static_predict_horizon_dense_rollout, # 导入重构后的稠密全轨迹生成算子
    MAX_ACC,
    MAX_STEER,
    WHEEL_BASE,
    LR
)
from gmm_igo.MPC_G_MS import _step_fn_rne_blocks

# ======================================================================
# 1. 严格固化所有博弈与仿真维数常数 (高速步长设定)
# ======================================================================
SEED = 42
T = 300
DT = 0.5              
DT_C = 0.05            # 微观积分步长
CONTROL_HORIZON = 12   

M_AGENT = 3
N_BLOCKS = 4
K = 3
B = 60
B0 = 25
T_0 = 300
M_inner = 30
N_MPC_STEPS = 25

ROAD_LENGTH = 400.0
LANE_WIDTH = 3.5
UPPER_LANE_CENTER = 3.5
LOWER_LANE_CENTER = 0.0

VEHICLE_LENGTH = 5.0
VEHICLE_WIDTH = 2.0
SAFE_GAP = 3.0  

BLOCK_TO_AGENT = jnp.array([0, 0, 1, 2], dtype=jnp.int32)
BLOCK_DIMS = jnp.array([CONTROL_HORIZON, CONTROL_HORIZON, CONTROL_HORIZON, CONTROL_HORIZON], dtype=jnp.int32)

TAU_ACC = 0.25      
TAU_STEER = 0.20    

def _decode_joint_sample(joint_sample_flat):
    return joint_sample_flat.reshape((N_BLOCKS, CONTROL_HORIZON))


def _dense_horizon_pair_collision_cost(traj_a, traj_b):
    """
    对包含全量微观积分步的状态序列进行密集的矢量化碰撞度量。
    输入张量形状为 (140, 6)，输出为全时域测度连续的标量碰撞代价。
    """
    poses_a = traj_a[:, :2]
    poses_b = traj_b[:, :2]
    psis_a = traj_a[:, 3]
    psis_b = traj_b[:, 3]
    return jnp.sum(
        vmap(pairwise_footprint_overlap_cost, in_axes=(0, 0, 0, 0, None, None, None))(
            poses_a, poses_b, psis_a, psis_b, VEHICLE_LENGTH, VEHICLE_WIDTH, SAFE_GAP
        )
    )

@jit
def static_propagate_system(current_states, ego_ctrl, front_acc, rear_acc):
    sub_steps = int(round(DT / DT_C))
    ego, front, rear = current_states[0], current_states[1], current_states[2]

    def loop_body(i, carry):
        e, f, r = carry
        next_e = stable_single_car_step(e, ego_ctrl[0], ego_ctrl[1], DT_C)
        next_f = stable_lon_car_step(f, front_acc, DT_C)
        next_r = stable_lon_car_step(r, rear_acc, DT_C)
        return next_e, next_f, next_r

    final_ego, final_front, final_rear = lax.fori_loop(0, sub_steps, loop_body, (ego, front, rear))
    return jnp.stack([final_ego, final_front, final_rear])


# ======================================================================
# 3. 数学修正后的博弈 Cost 体系 (全生命周期稠密黎曼求和)
# ======================================================================
def _ego_cost(joint_sample_flat, context_arr):
    current_states = context_arr[0:18].reshape(3, 6)
    v_ref          = context_arr[18]
    blocks = _decode_joint_sample(joint_sample_flat)
    
    dense_trajectory = static_predict_horizon_dense_rollout(
        current_states, blocks[0], blocks[1], blocks[2], blocks[3]
    )

    ego_traj, front_traj, rear_traj = dense_trajectory[:, 0], dense_trajectory[:, 1], dense_trajectory[:, 2]
    ego_v, ego_y, ego_psi = ego_traj[:, 2], ego_traj[:, 1], ego_traj[:, 3]
    ego_steer = ego_traj[:, 5]
    ego_beta = jnp.arctan(LR * jnp.tan(ego_steer) / WHEEL_BASE)
    ego_vy = ego_v * jnp.sin(ego_psi + ego_beta)
    y_target = UPPER_LANE_CENTER

    # 状态积分与平滑开销
    state_cost = jnp.sum((3.0 * (ego_v - v_ref) ** 2 + 10.0 * (ego_y - y_target) ** 2 + 5.0 * ego_vy ** 2 + 10.8 * ego_psi ** 2) * DT_C)
    control_cost = 0.5 * jnp.sum(blocks[0] ** 2) + 0.5 * jnp.sum(blocks[1] ** 2)
    smooth_cost = 1.0 * jnp.sum(jnp.diff(blocks[0]) ** 2) + 1.0 * jnp.sum(jnp.diff(blocks[1]) ** 2)
    terminal_cost = 50.0 * (ego_y[-1] - y_target) ** 2 + 65.0 * ego_vy[-1] ** 2 + 65.0 * ego_psi[-1] ** 2
    
    asymmetry_cost = (
        jnp.sum(jnp.where(ego_y - y_target > 0.5, 50.0 , 0.0) * DT_C) + 
        jnp.sum(jnp.where(ego_y - y_target < -0.5, 15.0 , 0.0) * DT_C) + 
        jnp.sum(jnp.where(ego_y < LOWER_LANE_CENTER - 1.0, 100.0 , 0.0) * DT_C) + 
        jnp.sum(jnp.where(ego_y > UPPER_LANE_CENTER + 1.0, 100.0 , 0.0) * DT_C)
    )

    # Ego 负全责：全轨迹 120 步无死角碰撞检测
    total_collisions = _dense_horizon_pair_collision_cost(ego_traj, front_traj) + \
                       _dense_horizon_pair_collision_cost(ego_traj, rear_traj)
    collision_cost = 100.0 * total_collisions
    
    return state_cost + control_cost + collision_cost + smooth_cost + terminal_cost + asymmetry_cost


def _front_cost(joint_sample_flat, context_arr):
    current_states = context_arr[0:18].reshape(3, 6)
    v_ref          = context_arr[19]
    blocks = _decode_joint_sample(joint_sample_flat)
    
    dense_trajectory = static_predict_horizon_dense_rollout(
        current_states, blocks[0], blocks[1], blocks[2], blocks[3]
    )
    front_traj, ego_traj, rear_traj = dense_trajectory[:, 1], dense_trajectory[:, 0], dense_trajectory[:, 2]
    
    state_cost = jnp.sum((3.0 * (front_traj[:, 2] - v_ref) ** 2) * DT_C)
    phys_front_acc = MAX_ACC * jnp.tanh(blocks[2])
    control_cost = 2.0 * jnp.sum(phys_front_acc ** 2) + 2.0 * jnp.sum(jnp.diff(phys_front_acc) ** 2)
    terminal_cost = 100.0 * (front_traj[-1, 2] - v_ref) ** 2 
    
    # 【完美实现设想】: front 与 ego 的碰撞检测强制切片为 [:2]
    # 无论后面发生什么惨烈碰撞，front 都不在乎；它只在乎眼前 0.1s 内 ego 有没有压到它的车头
    collision_cost = 100.0 * _dense_horizon_pair_collision_cost(front_traj[:2], ego_traj[:2])
    
    return state_cost + control_cost + collision_cost + terminal_cost


def _rear_cost(joint_sample_flat, context_arr):
    current_states = context_arr[0:18].reshape(3, 6)
    v_ref          = context_arr[20]
    blocks = _decode_joint_sample(joint_sample_flat)
    
    dense_trajectory = static_predict_horizon_dense_rollout(
        current_states, blocks[0], blocks[1], blocks[2], blocks[3]
    )
    rear_traj, ego_traj, front_traj = dense_trajectory[:, 2], dense_trajectory[:, 0], dense_trajectory[:, 1]
    
    state_cost = jnp.sum((3.0 * (rear_traj[:, 2] - v_ref) ** 2) * DT_C)
    phys_rear_acc = MAX_ACC * jnp.tanh(blocks[3])
    control_cost = 2.0 * jnp.sum(phys_rear_acc ** 2) + 2.0 * jnp.sum(jnp.diff(phys_rear_acc) ** 2)
    terminal_cost = 100.0 * (rear_traj[-1, 2] - v_ref) ** 2
    
    collision_to_ego = _dense_horizon_pair_collision_cost(rear_traj[:2], ego_traj[:2])
    
    dx_longitudinal = front_traj[:, 0] - rear_traj[:, 0]
    
    safety_clearance = VEHICLE_LENGTH  # 5.0 + 3.0 = 8.0米
    dist_to_hazard = jnp.maximum(dx_longitudinal - safety_clearance, 0.1)
    longitudinal_barrier = jnp.sum((20.0 / dist_to_hazard) * DT_C)
    
    # 如果后车极其恶劣地彻底超越/碾压了前车（dx <= 0），施加巨大的惩罚基底
    overtake_punishment = jnp.sum(jnp.where(dx_longitudinal <= 0.0, 500.0, 0.0) * DT_C)


    
    collision_cost = 100.0 * collision_to_ego + longitudinal_barrier + overtake_punishment
    return state_cost + control_cost + collision_cost + terminal_cost

@jit
def fitness_fn_j_jax(agent_idx, joint_sample_flat, context_arr):
    return lax.switch(
        agent_idx,
        (lambda s, c: _ego_cost(s, c), lambda s, c: _front_cost(s, c), lambda s, c: _rear_cost(s, c)),
        joint_sample_flat,
        context_arr,
    )


@jit
def pure_jax_game_rollout(solve_key, mu_init, S_init, v_init, context_arr):
    v_reset_internal = jnp.zeros((N_BLOCKS, K - 1))
    
    def loop_wrapper(state, iter_data):
        return _step_fn_rne_blocks(
            state, iter_data, N_blocks=N_BLOCKS, M_agent=M_AGENT, K=K, B=B, B0=B0, dt=0.15,
            dims_arr=BLOCK_DIMS, T_0=T_0, fitness_fn_j=fitness_fn_j_jax, v_reset=v_reset_internal,
            context=context_arr, M_inner=M_inner, block_to_agent_idx=BLOCK_TO_AGENT
        )
        
    init_state = (mu_init, S_init, v_init, 0)
    final_state, _ = lax.scan(loop_wrapper, init_state, (random.split(solve_key, T), jnp.arange(T)))
    return final_state[0]

@jit
def _select_block_wise_best_components(final_mu, context_arr):
    def eval_block_k(b_idx, k_idx):
        b0 = jnp.where(b_idx == 0, final_mu[0, k_idx], final_mu[0, 0])
        b1 = jnp.where(b_idx == 1, final_mu[1, k_idx], final_mu[1, 0])
        b2 = jnp.where(b_idx == 2, final_mu[2, k_idx], final_mu[2, 0])
        b3 = jnp.where(b_idx == 3, final_mu[3, k_idx], final_mu[3, 0])
        joint_sample = jnp.concatenate([b0, b1, b2, b3])
        
        agent_idx = BLOCK_TO_AGENT[b_idx]
        return fitness_fn_j_jax(agent_idx, joint_sample, context_arr)

    all_block_costs = vmap(lambda b: vmap(lambda k: eval_block_k(b, k))(jnp.arange(K)))(jnp.arange(N_BLOCKS))
    return jnp.argmin(all_block_costs, axis=1)

@jit
def _generate_static_warm_start_mu(final_mu, best_block_ks, key):
    best_seqs = jnp.stack([
        final_mu[0, best_block_ks[0]],
        final_mu[1, best_block_ks[1]],
        final_mu[2, best_block_ks[2]],
        final_mu[3, best_block_ks[3]]
    ])
    shifted_bests = jnp.concatenate([best_seqs[:, 1:], jnp.zeros((N_BLOCKS, 1))], axis=1)
    noise = random.normal(key, shape=(N_BLOCKS, K, CONTROL_HORIZON)) * 0.15
    base_mu = jnp.expand_dims(shifted_bests, axis=1)
    k_mask = (jnp.arange(K) == 0)[None, :, None]
    return jnp.where(k_mask, base_mu, base_mu + noise)


def _compute_plot_limits(history_positions, margin_ratio=0.15, min_margin=10.0):
    xy = jnp.asarray(history_positions)
    x_min = float(jnp.min(xy[:, :, 0]))
    x_max = float(jnp.max(xy[:, :, 0]))
    y_min = float(jnp.min(xy[:, :, 1]))
    y_max = float(jnp.max(xy[:, :, 1]))

    x_span = max(x_max - x_min, min_margin)
    y_span = max(y_max - y_min, min_margin)
    x_pad = max(x_span * margin_ratio, min_margin * 0.5)
    y_pad = max(y_span * margin_ratio, min_margin * 0.5)

    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)


def _visualize_trajectory(history_positions, save_png="trackgame_trajectory.png", save_gif="trackgame_trajectory.gif"):
    history_positions = jnp.asarray(history_positions)
    x_min, x_max, y_min, y_max = _compute_plot_limits(history_positions)

    colors = ["#d1495b", "#1f77b4", "#2ca02c"]
    labels = ["ego", "front", "rear"]

    fig, ax = plt.subplots(figsize=(11, 8), dpi=160)
    ax.set_facecolor("#f7f3ea")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trackgame Trajectory")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

    for lane_y in [LOWER_LANE_CENTER, UPPER_LANE_CENTER]:
        ax.axhline(lane_y, color="#8f7a63", linestyle=":", linewidth=1.0, alpha=0.45)

    for agent_idx, color, label in zip(range(3), colors, labels):
        traj = jnp.asarray(history_positions[:, agent_idx])
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.2, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=42, marker="o", zorder=4)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=55, marker="s", zorder=5)

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_png, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=160)
    ax.set_facecolor("#f7f3ea")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trackgame Trajectory Over Time")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

    for lane_y in [LOWER_LANE_CENTER, UPPER_LANE_CENTER]:
        ax.axhline(lane_y, color="#8f7a63", linestyle=":", linewidth=1.0, alpha=0.45)

    path_lines = []
    path_markers = []
    for color, label in zip(colors, labels):
        (line,) = ax.plot([], [], color=color, linewidth=2.4, label=label)
        (marker,) = ax.plot([], [], color=color, marker="o", markersize=8)
        path_lines.append(line)
        path_markers.append(marker)

    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    ax.legend(loc="best")

    def init():
        for line, marker in zip(path_lines, path_markers):
            line.set_data([], [])
            marker.set_data([], [])
        step_text.set_text("")
        return (*path_lines, *path_markers, step_text)

    def update(frame_idx):
        for agent_idx in range(3):
            traj = jnp.asarray(history_positions[: frame_idx + 1, agent_idx])
            path_lines[agent_idx].set_data(traj[:, 0], traj[:, 1])
            path_markers[agent_idx].set_data([traj[-1, 0]], [traj[-1, 1]])
        step_text.set_text(f"step = {frame_idx:02d}")
        return (*path_lines, *path_markers, step_text)

    anim = FuncAnimation(
        fig,
        update,
        frames=history_positions.shape[0],
        init_func=init,
        interval=220,
        blit=True,
    )
    anim.save(save_gif, writer=PillowWriter(fps=5))
    plt.close(fig)

# ======================================================================
# 5. 闭环仿真大轮盘
# ======================================================================
def run_simulation():
    key = random.PRNGKey(SEED)

    # 包含执行器状态的 6 维车辆初始状态 [x, y, v, psi, curr_acc, curr_steer]
    init_ego   = jnp.array([15.0, 0.0, 15.0, 0.0, 0.0, 0.0])       
    init_front = jnp.array([17.0, 3.5, 10.0, 0.0, 0.0, 0.0])   
    init_rear  = jnp.array([13.0, 3.5, 10.0, 0.0, 0.0, 0.0])    
    current_states = jnp.stack([init_ego, init_front, init_rear])

    current_mu_init = random.normal(key, (N_BLOCKS, K, CONTROL_HORIZON)) * 0.5
    
    static_S_identity = jnp.stack([jnp.eye(CONTROL_HORIZON)] * (N_BLOCKS * K)).reshape(N_BLOCKS, K, CONTROL_HORIZON, CONTROL_HORIZON)
    static_v_reset    = jnp.zeros((N_BLOCKS, K - 1))
    
    prev_ego_acc, prev_ego_steer = jnp.array(0.0), jnp.array(0.0)
    prev_front_acc, prev_rear_acc = jnp.array(0.0), jnp.array(0.0)

    history_positions = [current_states[:, :2]]

    for mpc_step in range(N_MPC_STEPS):
        key, solve_key, warm_key = random.split(key, 3)
        
        context_arr = jnp.concatenate([
            current_states.flatten(),             
            jnp.array([17.5, 20.0 ,17.5])        
        ])

        t_step_start = time.time()

        final_mu = pure_jax_game_rollout(
            solve_key,
            mu_init=current_mu_init,
            S_init=static_S_identity,  
            v_init=static_v_reset,     
            context_arr=context_arr
        )

        best_block_ks = _select_block_wise_best_components(final_mu, context_arr)
        
        t_step_end = time.time()

        ego_acc_seq   = low_pass_filter_sequence(final_mu[0, best_block_ks[0]], alpha=0, init_value=prev_ego_acc)
        ego_steer_seq = low_pass_filter_sequence(final_mu[1, best_block_ks[1]], alpha=0, init_value=prev_ego_steer)
        front_acc_seq = low_pass_filter_sequence(final_mu[2, best_block_ks[2]], alpha=0, init_value=prev_front_acc)
        rear_acc_seq  = low_pass_filter_sequence(final_mu[3, best_block_ks[3]], alpha=0, init_value=prev_rear_acc)

        prev_ego_acc, prev_ego_steer = ego_acc_seq[0], ego_steer_seq[0]
        prev_front_acc, prev_rear_acc = front_acc_seq[0], rear_acc_seq[0]

        # 推进真实物理步进
        current_states = static_propagate_system(
            current_states,
            jnp.array([ego_acc_seq[0], ego_steer_seq[0]]),
            front_acc_seq[0],
            rear_acc_seq[0]
        )
        history_positions.append(current_states[:, :2])

        current_mu_init = _generate_static_warm_start_mu(final_mu, best_block_ks, warm_key)

        print(f"步数 {mpc_step:02d} | 自车 X={current_states[0,0]:.1f} Y={current_states[0,1]:.2f} V={current_states[0,2]:.1f} | 实际延时后Acc={current_states[0,4]:.2f} | 求解时间 {t_step_end - t_step_start:.2f}s")

    _visualize_trajectory(jnp.stack(history_positions))
    print("轨迹可视化已保存: trackgame_trajectory.png, trackgame_trajectory.gif")


if __name__ == "__main__":
    run_simulation()