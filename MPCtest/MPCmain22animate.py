import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import functools

# 直接从你的库中导入
from gmm_igo.MPCsolver import igo_mog_optimizer

# ==============================================================================
# 1. 配置参数
# ==============================================================================
CONFIG = {
    'dt': 0.1,
    'horizon': 12,        
    'dim': 2,
    'n_components': 2,    
    'pop_size': 60,       
    'elite_size': 25,     
    'opt_steps': 100,     
    'warmup_steps': 1000,
    'max_v': 2.8,         
    'max_w': 2.0,         
    'obs_radius': 3.0,    
    'safe_margin': 0.1,   
}

TOTAL_DIM = CONFIG['horizon'] * CONFIG['dim']
N_TASKS = 3 

# ==============================================================================
# 2. 核心逻辑：定义代价函数与并行算子
# ==============================================================================

parallel_igo_solver = jax.vmap(
    igo_mog_optimizer,
    in_axes=(0, None, None, None, None, None, None, 0, 0, 0, {
        'current_state': 0, 
        'target_pos': 0, 
        'obs_traj': None, 
        'obs_radius': None,
        'safe_distance': None,
        'strategy_id': 0, 
    })
)

@jax.jit
def mpc_cost_fn(flat_actions, context):
    raw_actions = flat_actions.reshape((CONFIG['horizon'], CONFIG['dim']))
    vs = jnp.tanh(raw_actions[:, 0]) * CONFIG['max_v']
    ws = jnp.tanh(raw_actions[:, 1]) * CONFIG['max_w']
    x0, y0, theta0 = context['current_state']
    
    thetas = theta0 + jnp.cumsum(ws * CONFIG['dt'])
    thetas_prev = jnp.concatenate([jnp.array([theta0]), thetas[:-1]])
    trajectory = jnp.stack([
        x0 + jnp.cumsum(vs * jnp.cos(thetas_prev) * CONFIG['dt']),
        y0 + jnp.cumsum(vs * jnp.sin(thetas_prev) * CONFIG['dt'])
    ], axis=1)

    obs_traj = context['obs_traj'] 
    safe_dist = context['safe_distance']
    
    # 时空距离计算 (H, 1, 2) vs (H, N_obs, 2)
    diff = trajectory[:, None, :] - obs_traj
    dists = jnp.linalg.norm(diff, axis=-1)
    instant_risk_base = 1e3 * jnp.maximum(0, safe_dist - dists)

    s_id = context['strategy_id']
    risk_0 = jnp.sum(instant_risk_base, axis=1)
    
    # 策略 1: 偏向左绕 (通过局部坐标系的 y 轴正向赋予更高代价，强迫向右或向左避障)
    cos_t = jnp.cos(thetas_prev)[:, None]
    sin_t = jnp.sin(thetas_prev)[:, None]
    local_y = -diff[..., 0] * sin_t + diff[..., 1] * cos_t
    left_mask = jnp.where(local_y > 0, 15.0, 1.0) 
    risk_1 = jnp.sum(instant_risk_base * left_mask, axis=1)
    
    # 策略 2: 时间衰减 (只看眼前，不看远期)
    time_decay = jnp.exp(-0.5 * jnp.arange(CONFIG['horizon']))
    risk_2 = jnp.sum(instant_risk_base, axis=1) * time_decay

    selected_instant = jax.lax.switch(s_id, [
        lambda _: risk_0,
        lambda _: risk_1,
        lambda _: risk_2
    ], None)

    cum_risk = jnp.flip(jnp.cumsum(jnp.flip(selected_instant)))
    dist_to_target = jnp.linalg.norm(trajectory - context['target_pos'], axis=1)
    cum_dist = jnp.flip(jnp.cumsum(jnp.flip(dist_to_target)))

    return jnp.sum(cum_risk) * 6000.0 + jnp.sum(cum_dist) * 1.5 + jnp.sum(jnp.diff(ws)**2) * 2.0

# ==============================================================================
# 3. 混合障碍物预测逻辑 (4x4 桶阵)
# ==============================================================================

def generate_4x4_grid():
    """生成初始网格位置"""
    xs = jnp.linspace(4.0, 20.0, 4)
    ys = jnp.linspace(4.0, 20.0, 4)
    xx, yy = jnp.meshgrid(xs, ys)
    return jnp.stack([xx.ravel(), yy.ravel()], axis=1)

def predict_mixed_trajectories(t_start, obs_base, horizon, dt):
    """
    偶数索引静止，奇数索引做圆周运动
    """
    t_steps = t_start + jnp.arange(horizon) * dt
    N = obs_base.shape[0]
    
    # 动态掩码
    dynamic_mask = (jnp.arange(N) % 2 == 1)
    
    # 圆周运动轨迹
    angles = 1.3 * t_steps[:, None] + (jnp.arange(N) * 0.5)
    circ_x = 2.2 * jnp.cos(angles)
    circ_y = 2.2 * jnp.sin(angles)
    
    offsets = jnp.where(dynamic_mask[None, :, None], 
                       jnp.stack([circ_x, circ_y], axis=-1), 
                       0.0)
    
    return obs_base[None, :, :] + offsets

# ==============================================================================
# 4. 主循环
# ==============================================================================

def get_trajectory_np(robot_state, action_seq):
    u = action_seq.reshape(CONFIG['horizon'], CONFIG['dim'])
    vs, ws = np.tanh(u[:, 0]) * CONFIG['max_v'], np.tanh(u[:, 1]) * CONFIG['max_w']
    x, y, th = robot_state
    pts = []
    for i in range(CONFIG['horizon']):
        x += vs[i] * np.cos(th) * CONFIG['dt']
        y += vs[i] * np.sin(th) * CONFIG['dt']
        th += ws[i] * CONFIG['dt']
        pts.append([x, y])
    return np.array(pts)

def run_simulation():
    key = jax.random.PRNGKey(42)
    K = CONFIG['n_components']
    
    # 初始化
    obs_base = generate_4x4_grid()
    mu_k_batch = jax.random.normal(key, (N_TASKS, K, TOTAL_DIM)) * 0.1
    L_inv_k_batch = jnp.stack([jnp.stack([jnp.eye(TOTAL_DIM) * 2.0 for _ in range(K)]) for _ in range(N_TASKS)])
    pi_k_batch = jnp.ones((N_TASKS, K)) / K
    
    robot_state = jnp.array([0.0, 0.0, 0.0])
    target_final = jnp.array([24.0, 24.0])

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for t in range(500):
        current_time = t * CONFIG['dt']
        # 获取预测时域内的所有障碍物位置
        obs_trajs = predict_mixed_trajectories(current_time, obs_base, CONFIG['horizon'], CONFIG['dt'])
        current_obs_pos = obs_trajs[0]

        # 并行 MPC 求解
        batch_context = {
            'current_state': jnp.stack([robot_state] * N_TASKS),
            'target_pos': jnp.stack([target_final] * N_TASKS),
            'obs_traj': obs_trajs,
            'obs_radius': CONFIG['obs_radius'],
            'safe_distance': CONFIG['obs_radius'] + CONFIG['safe_margin'],
            'strategy_id': jnp.array([0, 1, 2])
        }

        key, subkey = jax.random.split(key)
        mu_k_batch, L_inv_k_batch, pi_k_batch = parallel_igo_solver(
            jax.random.split(subkey, N_TASKS), 
            CONFIG['warmup_steps'] if t==0 else CONFIG['opt_steps'], 
            0.1, K, CONFIG['pop_size'], CONFIG['elite_size'], 
            mpc_cost_fn, mu_k_batch, L_inv_k_batch, pi_k_batch, batch_context
        )

        # 决策与执行
        eval_scores = [float(mpc_cost_fn(mu_k_batch[i, jnp.argmax(pi_k_batch[i])], 
                       {**batch_context, 'strategy_id': 0, 'current_state': robot_state, 'target_pos': target_final})) 
                       for i in range(N_TASKS)]
        best_idx = np.argmin(eval_scores)
        
        best_u_seq = mu_k_batch[best_idx, jnp.argmax(pi_k_batch[best_idx])].reshape(CONFIG['horizon'], CONFIG['dim'])
        v_e = jnp.tanh(best_u_seq[0, 0]) * CONFIG['max_v']
        w_e = jnp.tanh(best_u_seq[0, 1]) * CONFIG['max_w']
        
        robot_state = jnp.array([
            robot_state[0] + v_e * jnp.cos(robot_state[2]) * CONFIG['dt'],
            robot_state[1] + v_e * jnp.sin(robot_state[2]) * CONFIG['dt'],
            robot_state[2] + w_e * CONFIG['dt']
        ])

        # 可视化
        if t % 1 == 0:
            ax.cla()
            ax.set_xlim(-2, 28); ax.set_ylim(-2, 28); ax.set_aspect('equal')
            
            # 画障碍物
            for i in range(obs_base.shape[0]):
                is_dynamic = (i % 2 == 1)
                color = 'red' if is_dynamic else 'gray'
                ax.add_patch(Circle(current_obs_pos[i], CONFIG['obs_radius'], color=color, alpha=0.3))
                # 绘制预测轨迹线
                ax.plot(obs_trajs[:, i, 0], obs_trajs[:, i, 1], color=color, lw=0.5, alpha=0.5)

            # 画不同策略的路径
            task_colors = ['blue', 'green', 'purple']
            for i in range(N_TASKS):
                traj_np = get_trajectory_np(robot_state, mu_k_batch[i, jnp.argmax(pi_k_batch[i])])
                ax.plot(traj_np[:,0], traj_np[:,1], color=task_colors[i], 
                        lw=3.0 if i==best_idx else 1.0, alpha=0.8 if i==best_idx else 0.3)
            
            ax.plot(target_final[0], target_final[1], 'g*', ms=15)
            ax.set_title(f"Step {t} | Winner: Task {best_idx} | Mixed 4x4 Obstacles")
            plt.pause(0.01)

        # 解平移
        mu_reshaped = mu_k_batch.reshape(N_TASKS, CONFIG['n_components'], CONFIG['horizon'], CONFIG['dim'])
        mu_shifted = jnp.roll(mu_reshaped, shift=-1, axis=2)
        mu_shifted = mu_shifted.at[:, :, -1, :].set(0.0)
        mu_k_batch = mu_shifted.reshape(N_TASKS, CONFIG['n_components'], -1)

if __name__ == "__main__":
    run_simulation()