# Highspeed_M22.py - 2026 量产级 MPC：MPCsolverM22.py 分块版（纵横 8+8 维）
# 保持原版所有 EPF、RSS、Minimax 约束，完全兼容！

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import functools
from gmm_igo.MPCsolverM2 import mmog_igo_optimizer_mpc  # ← 导入你的 M22 求解器
from scipy.interpolate import BSpline
import numpy as onp
from matplotlib.animation import FuncAnimation

# ====================== 1. 全局配置（完全不变） ======================
DT = 0.1
HORIZON = 100
TOTAL_TIME = HORIZON * DT
POLY_ORDER = 5

NUM_CONTROL_POINTS_FULL = 10
NUM_CONTROL_POINTS_OPT = 8
DIM_S = NUM_CONTROL_POINTS_OPT      # 纵向维度：8
DIM_L = NUM_CONTROL_POINTS_OPT      # 横向维度：8
BLOCK_DIMS = (DIM_S, DIM_L)         # M22 分块维度 [8,8]
TOTAL_DIM = DIM_S + DIM_L           # 拼接后仍为 16 维

LANE_WIDTH = 3.7
V_MAX = 35.0; V_MIN = 10.0
V_TARGET_TERMINAL = 30.0
X_TARGET = 3000.0

# Minimax EPF 惩罚（原版不变）
C_CRITICAL_EPF = 1e2
C_CRITICAL_BOUND = 1e5
C_KINEMATIC_EPF = 1.0
MAX_KINEMATIC_RATIO = jnp.tan(jnp.radians(15.0))
L_MAX_BOUNDARY = 1.3
TAU, A_BRAKE, A_ACCEL_MIN = 0.5, 4.0, -2.0
MARGIN_LEADING, MARGIN_LAT = 1.0, 0.4

def create_5_tap_filter_matrix(N):
    F = onp.zeros((N, N))
    W = onp.array([1, 26, 66, 26, 1]) / 120.0
    for i in range(N):
        if i == 0 or i == N - 1:
            F[i, i] = 1.0; continue
        for r in range(-2, 3):
            q_idx_raw = i + r
            weight = W[r + 2]
            j = q_idx_raw
            j_final = abs(j) if j < 0 else (2 * (N - 1) - j if j >= N else j)
            F[i, j_final] += weight
    return jnp.array(F, dtype=jnp.float32)

F_MATRIX = create_5_tap_filter_matrix(NUM_CONTROL_POINTS_FULL)

INTERNAL_KNOTS_COUNT = NUM_CONTROL_POINTS_FULL - POLY_ORDER
KNOT_DELTA = TOTAL_TIME / INTERNAL_KNOTS_COUNT
internal_knots = onp.arange(1, INTERNAL_KNOTS_COUNT + 1) * KNOT_DELTA
knots = onp.concatenate([onp.zeros(POLY_ORDER + 1), internal_knots, onp.full(POLY_ORDER, TOTAL_TIME)])
t_eval = onp.arange(HORIZON) * DT

def compute_basis_matrix(knots, t_eval, k, nu):
    basis = onp.zeros((len(t_eval), NUM_CONTROL_POINTS_FULL))
    for i in range(NUM_CONTROL_POINTS_FULL):
        c = onp.zeros(NUM_CONTROL_POINTS_FULL); c[i] = 1.0
        spl = BSpline(knots, c, k=k, extrapolate=True)
        basis[:, i] = spl(t_eval, nu=nu)
    return jnp.array(basis, dtype=jnp.float32)

N5_BASIS, N5_PRIME, N5_DOUBLE_PRIME, N5_TRIPLE_PRIME = [
    compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=i) for i in range(4)
]

# ====================== 3. theta_to_trajectory（支持分块输入，原逻辑不变） ======================
@jit
def theta_to_trajectory(theta, ctx):
    """theta: (16,) 拼接向量，内部自动拆分"""
    s_cur, l_cur, ds_cur = ctx['s_cur'], ctx['l_cur'], ctx['ds_cur']
    
    theta_s = theta[:DIM_S]     # 前 8 维：纵向 Qs_opt
    theta_l = theta[DIM_L:]     # 后 8 维：横向 Ql_opt
    
    # 锚点（原版不变）
    s_anchor_0, s_anchor_1 = s_cur, s_cur + ds_cur * DT
    l_anchor_0, l_anchor_1 = l_cur, l_cur
    Qs_anchors = jnp.array([s_anchor_0, s_anchor_1])
    Ql_anchors = jnp.array([l_anchor_0, l_anchor_1])
    
    Qs_full = jnp.concatenate([Qs_anchors, theta_s])
    Ql_full = jnp.concatenate([Ql_anchors, theta_l])
    
    Ps, Pl = F_MATRIX @ Qs_full, F_MATRIX @ Ql_full
    
    s_traj = N5_BASIS @ Ps; l_traj = N5_BASIS @ Pl
    s_dot = N5_PRIME @ Ps; l_dot = N5_PRIME @ Pl
    s_ddot = N5_DOUBLE_PRIME @ Ps; l_ddot = N5_DOUBLE_PRIME @ Pl
    s_dddot = N5_TRIPLE_PRIME @ Ps; l_dddot = N5_TRIPLE_PRIME @ Pl
    
    traj = jnp.stack([s_traj, l_traj * LANE_WIDTH], axis=1)
    return traj, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot

# ====================== 4. 代价函数（原版完全保留，Minimax EPF + RSS 避障） ======================
@jit
def overtake_cost(theta, ctx):
    traj, s_traj, l_traj, s_dot, l_dot, s_ddot, l_ddot, s_dddot, l_dddot = theta_to_trajectory(theta, ctx)
    
    lead_boxes, V_lead_array, Y_target, X_target = ctx['lead_boxes'], ctx['V_lead_array'], ctx['Y_target'], ctx['X_target']
    
    # 动力学项（原版）
    y_traj = l_traj * LANE_WIDTH
    a_mag = jnp.sqrt(s_ddot**2 + (l_ddot * LANE_WIDTH)**2)
    
    cost_stage_deviation = jnp.sum(10.0 * (y_traj - Y_target)**2 + 0.0 * (s_traj - X_target)**2)
    cost_term = 10.0 * (y_traj[-1] - Y_target)**2 + 50.0 * (s_traj[-1] - X_target)**2 + 50.0 * (s_dot[-1] - V_TARGET_TERMINAL)**2
    cost_acc = 10.0 * jnp.sum(jnp.maximum(0.0, a_mag - 5.0)**2)
    cost_lat = 0.1 * jnp.sum((l_ddot * LANE_WIDTH)**2)
    cost_jerk = 10.0 * jnp.sum(s_dddot**2 + (l_dddot * LANE_WIDTH)**2)
    
    # EPF 约束（原版）
    kinematic_ratio = LANE_WIDTH * jnp.abs(l_dot) / (s_dot + 1e-6)
    violation_kin = jnp.maximum(0.0, kinematic_ratio - MAX_KINEMATIC_RATIO)
    cost_EPF_kinematic = C_KINEMATIC_EPF * jnp.sum(violation_kin)
    
    # RSS 避碰（原版）
    N = lead_boxes.shape[0]
    V_ego_batch = jnp.tile(s_dot, (N, 1)); V_lead_batch = V_lead_array.reshape(N, 1)
    V_diff = V_ego_batch - V_lead_batch; V_diff_pos = jnp.maximum(0.0, V_diff)
    margin_lon = V_ego_batch * TAU + V_diff_pos**2 / (2.0 * (A_BRAKE - A_ACCEL_MIN))
    s_traj_batch = jnp.tile(s_traj, (N, 1))
    
    lead_s_center = (lead_boxes[:, 0, 0] + lead_boxes[:, 1, 0]) / 2.0
    s_lead_front, s_lead_rear = lead_boxes[:, 1, 0].reshape(N, 1), lead_boxes[:, 0, 0].reshape(N, 1)
    s_ego_front, s_ego_rear = s_traj_batch + 2.4, s_traj_batch - 2.4
    
    clearance_trailing = s_lead_rear - s_ego_front
    clearance_leading = s_ego_rear - s_lead_front
    violation_lon_trailing = jnp.maximum(0.0, margin_lon - clearance_trailing)
    violation_lon_leading = jnp.maximum(0.0, MARGIN_LEADING - clearance_leading)
    ego_is_trailing = (s_traj_batch < lead_s_center.reshape(N, 1)).astype(jnp.float32)
    violation_lon = violation_lon_trailing * ego_is_trailing + violation_lon_leading * (1.0 - ego_is_trailing)
    
    lead_y_center = (lead_boxes[:, 0, 1] + lead_boxes[:, 1, 1]) / 2.0
    Y_SEP_ACT_BOXES = jnp.abs(lead_y_center.reshape(N, 1) - jnp.tile(y_traj, (N, 1))) - 0.95 - 0.95
    violation_lat = jnp.maximum(0.0, MARGIN_LAT - Y_SEP_ACT_BOXES)
    is_close_lon = (jnp.abs(s_traj_batch - lead_s_center.reshape(N, 1)) < 20.0).astype(jnp.float32)
    
    # 车道边界 + Minimax EPF（原版关键！）
    violation_boundary = jnp.maximum(0.0, jnp.abs(l_traj) - L_MAX_BOUNDARY)
    cost_EPF_Critical_Minimax = jnp.maximum(
        C_CRITICAL_EPF * (violation_lon + violation_lat * is_close_lon),
        C_CRITICAL_BOUND * violation_boundary
    )
    cost_noway = jnp.sum(cost_EPF_Critical_Minimax)
    
    return (cost_term + cost_stage_deviation + cost_acc + cost_lat + cost_jerk + 
            cost_EPF_kinematic + cost_noway)

# ====================== 5. 主循环（集成 M22 分块优化器） ======================
def run_overtake_demo_realtime():
    print("🚀 2026 M22 分块 MPC —— 实时绘图模式启动！")
    key = jax.random.PRNGKey(0)

    # --- 1. 初始化物理状态 ---
    robot_s, robot_l, robot_ds = 0.0, 0.0, 40.0
    lead_car_states = [
        {'s': 100.0, 'l': 0.0, 'v': 20.0}, {'s': 150.0, 'l': -1.0, 'v': 21.0},
    ]

    # --- 2. 初始化 M22 分块优化器参数 (保持状态持久化以实现热启动) ---
    M, K, B, B0 = 2, 3, 60, 25      
    T_0 = 50                        
    D_max = 8                       
    
    # 初始均值 mu_init (M=2, K=8, D=8)
    Qs0_full = onp.arange(NUM_CONTROL_POINTS_FULL) * KNOT_DELTA * 20.0 / KNOT_DELTA
    theta0_s = jnp.array(Qs0_full[2:]).reshape(1, 1, -1)  # 纵向块
    theta0_l = jnp.zeros((1, 1, D_max))                   # 横向块
    
    mu_init = jnp.concatenate([
        jnp.tile(theta0_s, (1, K, 1)),           
        jnp.tile(theta0_l, (1, K, 1))
    ], axis=0)
    
    # 初始精度矩阵与权重比率
    L_inv_init = jnp.tile(jnp.eye(D_max).reshape(1, 1, D_max, D_max) * 2.0, (M, K, 1, 1))
    v_init = jnp.zeros((M, K-1))

    # --- 3. 绘图准备 ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 7))
    history = []

    for t in range(80):
        if robot_s >= X_TARGET: break

        key, subkey = jax.random.split(key)
        
        # 更新障碍物状态
        current_lead_boxes, V_lead_array = [], []
        for car in lead_car_states:
            car['s'] += car['v'] * DT
            s_center, y_center = car['s'], car['l'] * LANE_WIDTH
            current_lead_boxes.append([[s_center - 2.4, y_center - 0.95], [s_center + 2.4, y_center + 0.95]])
            V_lead_array.append(car['v'])
        
        ctx = {
            's_cur': robot_s, 'l_cur': robot_l, 'ds_cur': robot_ds,
            'Y_target': 0.8, 'X_target': X_TARGET,
            'lead_boxes': jnp.array(current_lead_boxes),
            'V_lead_array': jnp.array(V_lead_array)
        }

        # --- 4. 运行分块优化器 ---
        # 初始步数较多以建立分布，后续步数减少以保证实时性
        steps = 1000 if t == 0 else 400
        t0 = time.time()
        mu_k, L_k, pi_k = mmog_igo_optimizer_mpc(
            subkey, steps, 0.15, M, K, B, B0, BLOCK_DIMS, T_0,
            overtake_cost, mu_init, L_inv_init, v_init, ctx
        )
        t1 = time.time()

        # 提取当前最佳组合
        best_idx_s = jnp.argmax(pi_k[0])
        best_idx_l = jnp.argmax(pi_k[1])
        best_theta = jnp.concatenate([mu_k[0, best_idx_s], mu_k[1, best_idx_l]])
        
        # 物理执行
        traj, s_traj, l_traj, s_dot, _, _, _, _, _ = theta_to_trajectory(best_theta, ctx)
        robot_ds_next = jnp.clip(s_dot[1], V_MIN, V_MAX)
        shift_s = robot_ds_next * DT
        robot_s += shift_s
        robot_l = l_traj[1]
        robot_ds = robot_ds_next
        history.append(jnp.array([robot_s, robot_l * LANE_WIDTH]))

        # --- 5. 分块热启动 (Warm-Start) ---
        # 纵向块随车移动，横向块保持探索
        mu_init = mu_k.at[0].add(shift_s).at[1].add(jax.random.uniform(key, (K, D_max))*0.0)
        L_inv_init = L_k 

        # --- 6. 实时绘图渲染 ---
        if t % 1 == 0:
            ax.cla()
            ax.set_xlim(robot_s - 30, robot_s + 200)
            ax.set_ylim(-15, 15)
            
            # 道路背景
            for y in [-11.1, -7.4, -3.7, 0, 3.7, 7.4, 11.1]:
                ax.axhline(y, color='gray', ls='--', alpha=0.3)
            ax.axhline(L_MAX_BOUNDARY * LANE_WIDTH, color='black', lw=2)
            ax.axhline(-L_MAX_BOUNDARY * LANE_WIDTH, color='black', lw=2)

            # 绘制物体
            for box in current_lead_boxes:
                ax.add_patch(Rectangle((box[0][0], box[0][1]), 6.0, 1.4, color='red', alpha=0.8))
            ax.add_patch(Rectangle((robot_s-2.4, robot_l*LANE_WIDTH-0.95), 6.0, 1.4, color='green'))
            
            # 绘制轨迹
            hist_np = onp.array(history)
            ax.plot(hist_np[:,0], hist_np[:,1], 'b-', lw=3, alpha=0.6, label='History')
            ax.plot(s_traj, l_traj*LANE_WIDTH, 'cyan', lw=4, label='M22 Plan')
            
            ax.set_title(f"Step {t} | S:{robot_s:.1f}m | V:{robot_ds:.1f}m/s | OPT:{(t1-t0)*1000:.0f}ms")
            ax.legend(loc='upper right')
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_overtake_demo_realtime()