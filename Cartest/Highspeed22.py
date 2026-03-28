import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from gmm_igo.MPCsolver import igo_mog_optimizer 
from scipy.interpolate import BSpline
import numpy as onp

# ====================== 1. 全局配置 ======================
DT = 0.1
HORIZON = 100
TOTAL_TIME = HORIZON * DT 
POLY_ORDER = 5       

# --- 关键维度 ---
NUM_CONTROL_POINTS_FULL = 10 
NUM_CONTROL_POINTS_OPT = 8  

# ！！！新增：1个隐变量用于车道选择决策！！！
# 范围建议：[-1.5, 1.5]，对应右车道(-1), 中车道(0), 左车道(1)
NUM_LATENT_VARS = 1 

# 总优化维度 = 纵向CP + 横向CP + 隐变量
TOTAL_DIM = 2 * NUM_CONTROL_POINTS_OPT + NUM_LATENT_VARS

# --- 关键常数 ---
LANE_WIDTH = 3.7   
V_MAX = 35.0; V_MIN = 10.0 
V_TARGET_TERMINAL = 25.0 
X_TARGET = 3000.0 

# ！！！新增：道路曲率配置！！！
# 0.0015 对应半径 ~666m 的左转弯道
ROAD_CURVATURE = 0.0015 

# EPF 惩罚系数
C_CRITICAL_EPF = 1e4 
C_CRITICAL_BOUND = 1e5
C_KINEMATIC_EPF = 1.0 
MAX_KINEMATIC_RATIO = jnp.tan(jnp.radians(15.0)) 

L_MAX_BOUNDARY = 1.3  

# RSS 常数
TAU = 0.5            
A_BRAKE = 4.0        
A_ACCEL_MIN = -2.0   
MARGIN_LEADING = 1.0 
MARGIN_LAT = 1.0     

# ====================== 2. 构造 5-Tap 滤波器矩阵 F ======================
def create_5_tap_filter_matrix(N):
    F = onp.zeros((N, N))
    W = onp.array([1, 26, 66, 26, 1]) / 120.0
    for i in range(N):
        if i == 0 or i == N - 1:
            F[i, i] = 1.0
            continue
        for r in range(-2, 3):
            q_idx_raw = i + r
            weight = W[r + 2]
            j = q_idx_raw
            if j < 0: j_final = abs(j)
            elif j >= N: j_final = 2 * (N - 1) - j
            else: j_final = j
            F[i, j_final] += weight
    return jnp.array(F, dtype=jnp.float32)

F_MATRIX = create_5_tap_filter_matrix(NUM_CONTROL_POINTS_FULL)

# ====================== 3. B-spline 基函数 ======================
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

N5_BASIS = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=0) 
N5_PRIME = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=1) 
N5_DOUBLE_PRIME = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=2) 
N5_TRIPLE_PRIME = compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=3) 

# ====================== 3.5 Frenet 转 Cartesian 核心函数 ======================
@jit
def frenet_to_cartesian_batch(s, l, kappa=ROAD_CURVATURE):
    """
    将 Frenet (s, l) 转换为 Cartesian (x, y) 和航向角 theta
    支持恒定曲率模型
    """
    # 避免除以零
    safe_kappa = jnp.where(jnp.abs(kappa) < 1e-6, 1e-6, kappa)
    R = 1.0 / safe_kappa
    
    # 参考线上的点 (假设起点在原点，切线沿X轴)
    angle = s * safe_kappa
    x_ref = R * jnp.sin(angle)
    y_ref = R * (1.0 - jnp.cos(angle))
    theta_ref = angle
    
    # 直线情况近似
    is_straight = jnp.abs(kappa) < 1e-5
    x_r = jnp.where(is_straight, s, x_ref)
    y_r = jnp.where(is_straight, jnp.zeros_like(s), y_ref)
    theta_r = jnp.where(is_straight, jnp.zeros_like(s), theta_ref)
    
    # 叠加横向偏移
    l_meters = l * LANE_WIDTH
    x = x_r - l_meters * jnp.sin(theta_r)
    y = y_r + l_meters * jnp.cos(theta_r)
    
    return x, y, theta_r

# ====================== 4. 轨迹生成 (含动力学转换 & 隐变量解析) ======================
@jit
def theta_to_trajectory(theta, ctx):
    s_cur = ctx['s_cur']; l_cur = ctx['l_cur']; ds_cur = ctx['ds_cur']
    
    # ！！！解析优化变量！！！
    # 前 8 个是纵向 CP
    Qs_opt = theta[:NUM_CONTROL_POINTS_OPT]
    # 中间 8 个是横向 CP
    Ql_opt = theta[NUM_CONTROL_POINTS_OPT : 2*NUM_CONTROL_POINTS_OPT]
    # ！！！最后一个是车道决策隐变量！！！
    target_lane_decision = theta[-1] 
    
    s_anchor_0 = s_cur; l_anchor_0 = l_cur
    s_anchor_1 = s_cur + ds_cur * DT; l_anchor_1 = l_cur
    
    Qs_full = jnp.concatenate([jnp.array([s_anchor_0, s_anchor_1]), Qs_opt])
    Ql_full = jnp.concatenate([jnp.array([l_anchor_0, l_anchor_1]), Ql_opt])

    Ps = F_MATRIX @ Qs_full; Pl = F_MATRIX @ Ql_full
    
    # Frenet 域计算
    s_traj = N5_BASIS @ Ps; l_traj = N5_BASIS @ Pl
    s_dot = N5_PRIME @ Ps; l_dot = N5_PRIME @ Pl
    s_ddot = N5_DOUBLE_PRIME @ Ps; l_ddot = N5_DOUBLE_PRIME @ Pl
    s_dddot = N5_TRIPLE_PRIME @ Ps; l_dddot = N5_TRIPLE_PRIME @ Pl

    # 转换为笛卡尔坐标
    x_traj, y_traj, theta_traj = frenet_to_cartesian_batch(s_traj, l_traj)
    
    # 计算笛卡尔速度和加速度
    vx = jnp.gradient(x_traj, DT)
    vy = jnp.gradient(y_traj, DT)
    v_mag = jnp.sqrt(vx**2 + vy**2)
    
    ax = jnp.gradient(vx, DT)
    ay = jnp.gradient(vy, DT)
    a_mag = jnp.sqrt(ax**2 + ay**2)
    
    jx = jnp.gradient(ax, DT)
    jy = jnp.gradient(ay, DT)

    # 侧向加速度投影
    sin_t = jnp.sin(theta_traj); cos_t = jnp.cos(theta_traj)
    a_lat_cart = -ax * sin_t + ay * cos_t 

    return s_traj, l_traj, s_dot, l_dot, x_traj, y_traj, v_mag, a_mag, a_lat_cart, jx, jy, theta_traj, vx, vy, target_lane_decision

# ====================== 5. 代价函数 (多圆盘避障 & 隐变量引导) ======================
@jit
def overtake_cost(theta, ctx):
    # 1. 生成轨迹
    s_traj, l_traj, s_dot, l_dot, x_traj, y_traj, v_mag, a_mag, a_lat, jx, jy, theta_traj, v_x, v_y, target_lane_decision = \
        theta_to_trajectory(theta, ctx)
    
    lead_boxes = ctx['lead_boxes'] 
    X_target = ctx['X_target'] 
    
    # --- 1. 基础动力学代价 ---
    cost_dynamics = 50.0 * (s_traj[-1] - X_target)**2 + \
                    50.0 * (v_mag[-1] - V_TARGET_TERMINAL)**2 + \
                    10.0 * jnp.sum(jnp.maximum(0.0, a_mag - 5.0)**2) + \
                    5.0 * jnp.sum(a_lat**2) + \
                    10.0 * jnp.sum(jx**2 + jy**2)
    
    # ============================================================
    # Part A: Frenet 宏观决策层 (基于隐变量的风险评估)
    # ============================================================
    N_obs = lead_boxes.shape[0]
    obs_s = lead_boxes[:, 0, 0] 
    obs_l = lead_boxes[:, 0, 1] 
    
    s_traj_batch = s_traj.reshape(1, HORIZON)
    l_traj_batch = l_traj.reshape(1, HORIZON) # 补全定义
    obs_s_batch = obs_s.reshape(N_obs, 1)
    
    # 纵向接近度
    is_front = jax.nn.sigmoid(2.0 * (obs_s_batch - s_traj_batch))
    dist_weight = jnp.exp(-0.05 * jnp.abs(obs_s_batch - s_traj_batch))
    risk_factor = is_front * dist_weight 
    
    # 3个车道的风险评估
    LANE_CENTERS = jnp.array([-1.0, 0.0, 1.0]) 
    obs_l_batch = obs_l.reshape(N_obs, 1)
    lanes_batch = LANE_CENTERS.reshape(1, 3)
    lane_occupancy = jnp.exp(-2.0 * (obs_l_batch - lanes_batch)**2) 
    
    risk_per_lane = jnp.sum(
        lane_occupancy.reshape(N_obs, 3, 1) * risk_factor.reshape(N_obs, 1, HORIZON),
        axis=(0, 2)
    ) * 1000.0
    
    # 隐变量选择器
    k_sharp = 10.0
    w_right  = 0.5 * (jnp.tanh(k_sharp * (target_lane_decision - (-1.0) + 0.4)) - jnp.tanh(k_sharp * (target_lane_decision - (-1.0) - 0.4)))
    w_center = 0.5 * (jnp.tanh(k_sharp * (target_lane_decision - (0.0) + 0.4)) - jnp.tanh(k_sharp * (target_lane_decision - (0.0) - 0.4)))
    w_left   = 0.5 * (jnp.tanh(k_sharp * (target_lane_decision - (1.0) + 0.4)) - jnp.tanh(k_sharp * (target_lane_decision - (1.0) - 0.4)))
    
    cost_lane_risk = w_right * risk_per_lane[0] + \
                     w_center * risk_per_lane[1] + \
                     w_left * risk_per_lane[2]

    # ============================================================
    # Part B: 笛卡尔微观安全层 (基于圆盘的物理碰撞检测)
    # ============================================================
    # 相比 Frenet 矩形，这能处理弯道下的真实几何重叠
    
    # 1. 准备数据
    lead_s_center = (lead_boxes[:, 0, 0] + lead_boxes[:, 1, 0]) / 2.0
    lead_l_center = lead_boxes[:, 0, 1] 
    obs_x_c, obs_y_c, obs_theta = frenet_to_cartesian_batch(lead_s_center, lead_l_center)
    
    # 2. 车辆几何 (3圆盘)
    CAR_CIRCLE_R = 1.0 
    CIRCLE_OFFSETS = jnp.array([1.5, 0.0, -1.5]) 
    
    # 自车圆心 [3, 1, N_steps]
    ego_cos = jnp.cos(theta_traj.reshape(1, HORIZON))
    ego_sin = jnp.sin(theta_traj.reshape(1, HORIZON))
    ego_circles_x = x_traj.reshape(1, HORIZON) + CIRCLE_OFFSETS.reshape(3, 1) * ego_cos
    ego_circles_y = y_traj.reshape(1, HORIZON) + CIRCLE_OFFSETS.reshape(3, 1) * ego_sin
    
    # 障碍物圆心 [N_obs, 3, 1]
    obs_cos = jnp.cos(obs_theta.reshape(N_obs, 1))
    obs_sin = jnp.sin(obs_theta.reshape(N_obs, 1))
    obs_circles_x = obs_x_c.reshape(N_obs, 1) + CIRCLE_OFFSETS.reshape(1, 3) * obs_cos
    obs_circles_y = obs_y_c.reshape(N_obs, 1) + CIRCLE_OFFSETS.reshape(1, 3) * obs_sin
    
    # 3. 计算距离矩阵 [3(ego), 3(obs), N_obs, N_steps]
    # 广播维度对齐
    ex = ego_circles_x.reshape(3, 1, 1, HORIZON)
    ey = ego_circles_y.reshape(3, 1, 1, HORIZON)
    ox = obs_circles_x.reshape(1, 3, N_obs, 1)
    oy = obs_circles_y.reshape(1, 3, N_obs, 1)
    
    dist_sq = (ex - ox)**2 + (ey - oy)**2
    dist = jnp.sqrt(dist_sq + 1e-6)
    
    # 4. 碰撞判据
    SAFE_DIST = 2.0 * CAR_CIRCLE_R + 0.2 
    violation_cart = jnp.maximum(0.0, SAFE_DIST - dist)
    cost_collision_cart = jnp.sum(violation_cart) * 1e5 # 极高权重

    # ============================================================
    # Part C: 其他约束
    # ============================================================
    
    # 边界 Cost
    violation_boundary = jnp.maximum(0.0, jnp.abs(l_traj) - L_MAX_BOUNDARY)
    cost_boundary = C_CRITICAL_BOUND * jnp.sum(violation_boundary)
    
    # 决策一致性 & 正则化
    cost_consistency = 10.0 * jnp.mean((l_traj - target_lane_decision)**2)
    dist_to_int = jnp.minimum(jnp.abs(target_lane_decision - 0), 
                  jnp.minimum(jnp.abs(target_lane_decision - 1), jnp.abs(target_lane_decision + 1)))
    cost_reg = 10.0 * dist_to_int

    # ！！！融合所有 Cost ！！！
    # cost_lane_risk: 负责选路 (Frenet 逻辑)
    # cost_collision_cart: 负责保命 (Cartesian 物理)
    return cost_dynamics + cost_lane_risk + cost_boundary + cost_collision_cart + cost_consistency + cost_reg
# ====================== 6. 主循环 ======================
def run_overtake_demo():
    print(f"2025 Curved MPC | Curvature: {ROAD_CURVATURE}")
    key = jax.random.PRNGKey(0)

    robot_s = 0.0; robot_l = 0.0; robot_ds = 20.0 
    
    # 障碍物 (s, l, v)
    lead_car_states = [
        {'s': 100.0, 'l': 0.0, 'v': 20.0},  # 障碍车 1: 中车道, 20 m/s
        {'s': 150.0, 'l': -1.0, 'v': 21.0}, # 障碍车 2: 右车道, 25 m/s
        {'s': 200.0, 'l': 1.0, 'v': 22.0},   # 障碍车 3: 左车道, 30 m/s 
    ]
    
    # 初始化优化变量
    Qs0_full = onp.arange(NUM_CONTROL_POINTS_FULL) * KNOT_DELTA * 10.0 / KNOT_DELTA
    Ql0_full = onp.zeros(NUM_CONTROL_POINTS_FULL)
    
    # ！！！新增：初始化隐变量为 0 (当前车道)！！！
    latent_init = jnp.array([0.0]) 
    
    theta0 = jnp.concatenate([
        jnp.array(Qs0_full[2:]), 
        jnp.array(Ql0_full[2:]),
        latent_init
    ])
    
    mu_k = jnp.stack([theta0] * 3); 
    L_inv_k = jnp.stack([jnp.eye(TOTAL_DIM) * 1.5] * 3); 
    pi_k = jnp.ones(3) / 3.0

    # 强制使用独立窗口
    import matplotlib
    try:
        matplotlib.use('TkAgg') 
    except:
        pass

    plt.ion(); fig, ax = plt.subplots(figsize=(12, 8)); history = []

    rx_init, ry_init, _ = frenet_to_cartesian_batch(jnp.array([robot_s]), jnp.array([robot_l]))
    history.append([float(rx_init[0]), float(ry_init[0])])

    for t in range(500): 
        if robot_s >= X_TARGET: break

        key, subkey = jax.random.split(key)
        
        # 更新障碍物 (考虑曲率修正)
        current_lead_boxes = []
        V_lead_array = []
        for car in lead_car_states:
            l_dist = car['l'] * LANE_WIDTH
            curvature_factor = 1.0 - l_dist * ROAD_CURVATURE
            if abs(curvature_factor) < 0.1: curvature_factor = 0.1
            
            s_increment = (car['v'] * DT) / curvature_factor
            car['s'] += s_increment
            
            lead_box = [[car['s'] - 2.4, car['l']], [car['s'] + 2.4, car['l']]]
            current_lead_boxes.append(lead_box)
            V_lead_array.append(car['v'])
        
        ctx = {
            's_cur': robot_s, 'l_cur': robot_l, 'ds_cur': robot_ds, 
            'lead_boxes': jnp.array(current_lead_boxes), 
            'X_target': X_TARGET, 'V_lead_array': jnp.array(V_lead_array)
        }

        # 优化
        t0 = time.time()
        steps = 3000 if t == 0 else 300
        mu_k, L_inv_k, pi_k = igo_mog_optimizer(subkey, steps, 0.15, 3, 60, 25, overtake_cost, mu_k, L_inv_k, pi_k, ctx)
        t1 = time.time()
        
        best_theta = mu_k[jnp.argmax(pi_k)]
        s_traj, l_traj, s_dot, l_dot, x_traj, y_traj, v_mag, _, _, _, _, theta_traj, _, _, decision_val = theta_to_trajectory(best_theta, ctx)
        
        # 执行
        robot_ds = jnp.clip(s_dot[1], V_MIN, V_MAX)
        robot_s += robot_ds * DT
        robot_l = l_traj[1]
        
        # 记录
        rx, ry, _ = frenet_to_cartesian_batch(jnp.array([robot_s]), jnp.array([robot_l]))
        cx_val = float(rx[0]); cy_val = float(ry[0])
        
        if np.isnan(cx_val) or np.isinf(cx_val):
            print(f"Error: NaN/Inf detected at step {t}. Stopping.")
            break
        history.append([cx_val, cy_val])
        
        print(f"Step {t:03d} | S: {robot_s:6.1f}m | V: {robot_ds:4.1f}m/s | Dec: {float(decision_val):.2f} | Comp: {(t1-t0)*1000:4.0f}ms")

        # Warm Start (含隐变量)
        shift_s = robot_ds * DT
        Qs_old = best_theta[:NUM_CONTROL_POINTS_OPT]
        Ql_old = best_theta[NUM_CONTROL_POINTS_OPT : 2*NUM_CONTROL_POINTS_OPT]
        latent_old = best_theta[-1:] 
        
        key_perturb, key = jax.random.split(key) # 确保每次扰动是新的
        PERTURBATION_STD = 0.1 # 小扰动的标准差 (l 单位，约 5cm)

        random_perturbation = jax.random.normal(key_perturb, shape=(NUM_CONTROL_POINTS_OPT,)) * PERTURBATION_STD

        mu_k = mu_k.at[:, :NUM_CONTROL_POINTS_OPT].set(Qs_old + shift_s)
        mu_k = mu_k.at[:, NUM_CONTROL_POINTS_OPT : 2*NUM_CONTROL_POINTS_OPT].set(Ql_old+ random_perturbation)
        mu_k = mu_k.at[:, -1].set(latent_old)

        # --- 可视化 ---
        if t % 1 == 0: 
            ax.cla()
            cx, cy = history[-1]
            ax.set_xlim(cx - 20, cx + 100)
            ax.set_ylim(cy - 30, cy + 30)
            ax.set_aspect('equal')
            
            # 绘制道路
            s_plot = jnp.linspace(robot_s - 50, robot_s + 150, 200)
            for l_line in [-1.5, -0.5, 0.5, 1.5]:
                lx, ly, _ = frenet_to_cartesian_batch(s_plot, jnp.full_like(s_plot, l_line))
                ax.plot(lx, ly, 'k--', alpha=0.3)
            
            bx_p, by_p, _ = frenet_to_cartesian_batch(s_plot, jnp.full_like(s_plot, L_MAX_BOUNDARY))
            bx_n, by_n, _ = frenet_to_cartesian_batch(s_plot, jnp.full_like(s_plot, -L_MAX_BOUNDARY))
            ax.plot(bx_p, by_p, 'k-', lw=2)
            ax.plot(bx_n, by_n, 'k-', lw=2)

            # 绘制障碍物 (修正旋转)
            for car in lead_car_states:
                ox, oy, otheta = frenet_to_cartesian_batch(jnp.array([car['s']]), jnp.array([car['l']]))
                ox_val = float(ox[0]); oy_val = float(oy[0]); theta_val = float(otheta[0])
                
                rect = Rectangle((ox_val - 2.4, oy_val - 0.95), 4.8, 1.9, angle=0.0, color='red', alpha=0.8)
                t_trans = plt.matplotlib.transforms.Affine2D().rotate_around(ox_val, oy_val, theta_val) + ax.transData
                rect.set_transform(t_trans)
                ax.add_patch(rect)

            # 绘制自车
            ego_theta = float(theta_traj[0])
            ego_rect = Rectangle((cx-2.4, cy-0.95), 4.8, 1.9, angle=0.0, color='green', alpha=0.9)
            t_ego = plt.matplotlib.transforms.Affine2D().rotate_around(cx, cy, ego_theta) + ax.transData
            ego_rect.set_transform(t_ego)
            ax.add_patch(ego_rect)

            # 轨迹
            hist_arr = np.array(history)
            if len(hist_arr) > 1:
                recent_hist = hist_arr[-50:] if len(hist_arr) > 50 else hist_arr
                ax.plot(recent_hist[:,0], recent_hist[:,1], 'b-', lw=2, label='History')
            ax.plot(x_traj, y_traj, 'c-', lw=3, label='Plan')
            
            ax.set_title(f"Curved MPC (k={ROAD_CURVATURE}) | V={robot_ds:.1f} | Dec={float(decision_val):.2f}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
    plt.show()
if __name__ == '__main__':
    run_overtake_demo()