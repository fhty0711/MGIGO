import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gmm_igo.MPCsolverM2 import mmog_igo_optimizer_mpc 
from scipy.interpolate import BSpline
import numpy as onp

# ====================== 1. 核心物理配置 ======================
DT = 0.1
HORIZON = 100
TOTAL_TIME = HORIZON * DT
POLY_ORDER = 5
NUM_CONTROL_POINTS_FULL = 10
BLOCK_DIMS = (8, 8, 8) 

LANE_WIDTH = 3.7
V_MAX, V_MIN, V_TARGET = 35.0, 10.0, 30.0
LOOK_AHEAD_TIME = 10.0
L_MAX_BOUNDARY = 1.3 # 允许偏离车道中心的最大比例
ROAD_BOUNDARY_Y = L_MAX_BOUNDARY * LANE_WIDTH

# ====================== 2. 矩阵预计算 ======================
def create_5_tap_filter_matrix(N):
    F = onp.zeros((N, N))
    W = onp.array([1, 26, 66, 26, 1]) / 120.0
    for i in range(N):
        if i == 0 or i == N - 1: F[i, i] = 1.0; continue
        for r in range(-2, 3):
            j = i + r
            if 0 <= j < N: F[i, j] += W[r + 2]
    return jnp.array(F, dtype=jnp.float32)

F_MATRIX = create_5_tap_filter_matrix(NUM_CONTROL_POINTS_FULL)
n_internal = NUM_CONTROL_POINTS_FULL - POLY_ORDER - 1
tau_knots = onp.concatenate([onp.zeros(POLY_ORDER+1), onp.linspace(0,1,n_internal+2)[1:-1], onp.ones(POLY_ORDER+1)])
tau_eval = onp.linspace(0, 1, HORIZON)

def compute_basis(knots, t_eval, k, nu):
    basis = onp.zeros((len(t_eval), NUM_CONTROL_POINTS_FULL))
    for i in range(NUM_CONTROL_POINTS_FULL):
        c = onp.zeros(NUM_CONTROL_POINTS_FULL); c[i] = 1.0
        spl = BSpline(knots, c, k=k)
        basis[:, i] = spl(t_eval, nu=nu)
    return jnp.array(basis, dtype=jnp.float32)

N5_BASIS = compute_basis(tau_knots, tau_eval, POLY_ORDER, 0)
N5_PRIME = compute_basis(tau_knots, tau_eval, POLY_ORDER, 1)

# ====================== 3. 时空重参数化与 Cost ======================
@jit
def theta_to_trajectory(theta, ctx):
    ts, tl, tu = theta[0:8], theta[8:16], theta[16:24]
    
    # 几何 Gamma(tau)
    Qs = jnp.concatenate([jnp.array([ctx['s_cur'], ctx['s_cur']+ctx['ds_cur']*DT]), ts])
    Ql = jnp.concatenate([jnp.array([ctx['l_cur'], ctx['l_cur']]), tl])
    Ps, Pl = F_MATRIX @ Qs, F_MATRIX @ Ql
    
    # 时间映射 alpha(t)
    rho = jnp.exp(N5_BASIS @ jnp.concatenate([jnp.zeros(2), tu]))
    cum_rho = jnp.cumsum(rho * DT)
    alpha = (cum_rho - cum_rho[0]) / (cum_rho[-1] - cum_rho[0])
    dot_alpha = rho / (cum_rho[-1] / 1.0)
    
    # 物理采样
    s_t = jnp.interp(alpha, tau_eval, N5_BASIS @ Ps)
    l_t = jnp.interp(alpha, tau_eval, N5_BASIS @ Pl)
    v_t = jnp.interp(alpha, tau_eval, N5_PRIME @ Ps) * dot_alpha
    
    return s_t, l_t, v_t, alpha

@jit
def game_cbf_cost(theta, ctx):
    s_t, l_t, v_t, alpha = theta_to_trajectory(theta, ctx)
    
    # 1. 沿循车道 Cost (Lane Keeping)
    # 目标是维持在 Y_target，通常为 0 (中心线)
    cost_lane = 50.0 * jnp.sum((l_t * LANE_WIDTH - ctx['Y_target'])**2)
    
    # 2. 目标点与速度追踪
    cost_target =  30.0 * jnp.sum((v_t - V_TARGET)**2)
    
    # 3. 时空 CBF (对预测冲突点进行时序判断)
    # O_k(t) 预测
    t_axis = jnp.linspace(0, TOTAL_TIME, HORIZON)
    s_obs = ctx['lead_s'][:, None] + ctx['lead_v'][:, None] * t_axis[None, :]
    l_obs = ctx['lead_l'][:, None] * LANE_WIDTH
    
    # 相对位置与时空椭圆 h(x, t)
    rel_s = s_t[None, :] - s_obs
    rel_l = (l_t * LANE_WIDTH)[None, :] - l_obs
    h_k = (rel_s**2 / 5.0**2) + (rel_l**2 / 2.0**2) - 1.0
    
    # CBF 导数驱动博弈
    v_rel = v_t[None, :] - ctx['lead_v'][:, None]
    dot_h = (2.0 * rel_s / 25.0) * v_rel # 简化纵向导数
    
    cbf_violation = jnp.maximum(0.0, -(dot_h + 0.8 * h_k))
    cost_cbf = 15.0 * jnp.sum(cbf_violation**2)
    
    return  cost_lane + cost_target + cost_cbf

# ====================== 2. 仿真与绘图增强 ======================
def run_enhanced_visualization():
    key = jax.random.PRNGKey(42)
    robot_s, robot_l, robot_ds = 0.0, 0.0, 25.0
    
    # M=3, K=3 初始化
    mu_init = jnp.zeros((3, 3, 8))
    # S 块初始化
    s_vals = jnp.linspace(robot_ds*DT*2.5, robot_ds*10.0, 8)
    mu_init = mu_init.at[0].set(s_vals)
    # L 块：核心拓扑初始化 (左绕 0.8, 直行 0, 右绕 -0.8)
    mu_init = mu_init.at[1, 0].set(jnp.zeros(8))      # Straight
    mu_init = mu_init.at[1, 1].set(jnp.full(8, 0.8))  # Left Bypass
    mu_init = mu_init.at[1, 2].set(jnp.full(8, -0.8)) # Right Bypass
    
    L_inv_init = jnp.tile(jnp.eye(8) * 5.0, (3, 3, 1, 1))
    v_init = jnp.zeros((3, 2))

    lead_cars = [{'s': 110.0, 'l': 0.0, 'v': 16.0}]

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))

    for t in range(150):
        key, sub_k = jax.random.split(key)
        for c in lead_cars: c['s'] += c['v'] * DT
        
        ctx = {
            's_cur': robot_s, 'l_cur': robot_l, 'ds_cur': robot_ds,
            'Y_target': 0.0, 'X_target': robot_s + 300.0,
            'lead_s': jnp.array([c['s'] for c in lead_cars]),
            'lead_l': jnp.array([c['l'] for c in lead_cars]),
            'lead_v': jnp.array([c['v'] for c in lead_cars])
        }

        mu_k, L_k, pi_k = mmog_igo_optimizer_mpc(
            sub_k, 400, 0.15, 3, 3, 60, 25, BLOCK_DIMS, 50,
            game_cbf_cost, mu_init, L_inv_init, v_init, ctx
        )

        # 渲染前置逻辑
        best_mode = jnp.argmax(pi_k[1]) # 选取概率最大的横向分量
        best_theta = jnp.concatenate([mu_k[i, jnp.argmax(pi_k[i])] for i in range(3)])
        s_p, l_p, v_p, _ = theta_to_trajectory(best_theta, ctx)
        
        # 执行
        v_cmd = jnp.clip(v_p[1], 10.0, 35.0)
        robot_s += v_cmd * DT
        robot_l = l_p[1]
        robot_ds = v_cmd
        mu_init = mu_k.at[0].add(v_cmd * DT)

        # --- 绘图部分 ---
        if t % 1 == 0:
            ax.cla()
            # 动态视点：始终跟随自车
            ax.set_xlim(robot_s - 30, robot_s + 180)
            ax.set_ylim(-12, 12)

            # 1. 绘制道路边界 (Road Boundaries)
            ax.axhline(ROAD_BOUNDARY_Y, color='black', lw=3, label='Boundary')
            ax.axhline(-ROAD_BOUNDARY_Y, color='black', lw=3)
            
            # 2. 绘制车道线 (Lane Lines)
            # 绘制中心线以及上下车道分割线
            for y_off in [-LANE_WIDTH, 0, LANE_WIDTH]:
                ax.axhline(y_off, color='gray', ls='--', alpha=0.5)

            # 3. 绘制障碍物
            for c in lead_cars:
                rect = Rectangle((c['s']-2.4, c['l']*LANE_WIDTH-0.95), 6.0, 1.0, 
                                 color='red', alpha=0.7, zorder=3)
                ax.add_patch(rect)

            # 4. 绘制所有候选拓扑轨迹 (展示 M22 的搜索过程)
            colors = ['gray', 'blue', 'purple']
            labels = ['Straight', 'Left Bypass', 'Right Bypass']
            for k in range(3):
                # 提取每个模式的均值轨迹
                m_theta = jnp.concatenate([mu_k[0, jnp.argmax(pi_k[0])], mu_k[1, k], mu_k[2, jnp.argmax(pi_k[2])]])
                ms, ml, _, _ = theta_to_trajectory(m_theta, ctx)
                alpha_val = 1.0 if k == best_mode else 0.2
                ax.plot(ms, ml*LANE_WIDTH, color=colors[k], lw=2, alpha=alpha_val, label=labels[k])

            # 5. 绘制自车
            ax.add_patch(Rectangle((robot_s-2.4, robot_l*LANE_WIDTH-0.95), 6.0, 1.0, 
                                   color='green', zorder=5))
            
            ax.set_title(f"M=3 Spatio-Temporal Game | Speed: {robot_ds:.1f} m/s")
            ax.legend(loc='upper right', fontsize='small')
            plt.pause(0.01)

if __name__ == "__main__":
    run_enhanced_visualization()