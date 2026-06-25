import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cma  # 导入 pycma

# 复用你提供的 CONFIG
CONFIG = {
    'dt': 0.1,
    'horizon': 12,
    'dim': 2,
    'max_v': 2.5,
    'max_w': 1.0,
    'obs_spacing': 4.0,
    'obs_radius': 1.9,
    'safe_margin': 0.0,
}
TOTAL_DIM = CONFIG['horizon'] * CONFIG['dim']

# ==============================================================================
# 2. 核心代价函数 (保持 JAX 加速)
# ==============================================================================
@jax.jit
def mpc_cost_fn(flat_actions, context):
    raw_actions = flat_actions.reshape((CONFIG['horizon'], CONFIG['dim']))
    vs = jnp.tanh(3*raw_actions[:, 0]) * CONFIG['max_v']
    ws = jnp.tanh(3*raw_actions[:, 1]) * CONFIG['max_w']
    
    x0, y0, theta0 = context['current_state']
    target = context['target_pos']
    dt = CONFIG['dt']
    
    thetas = theta0 + jnp.cumsum(ws * dt)
    thetas_prev = jnp.concatenate([jnp.array([theta0]), thetas[:-1]])
    
    dx = vs * jnp.cos(thetas_prev) * dt
    dy = vs * jnp.sin(thetas_prev) * dt
    
    trajectory = jnp.stack([x0 + jnp.cumsum(dx), y0 + jnp.cumsum(dy)], axis=1)
    
    dist_to_target = jnp.linalg.norm(trajectory - target, axis=1)
    cost_track = jnp.sum(dist_to_target) * 2.0
    cost_final = jnp.linalg.norm(trajectory[-1] - target) * 50.0
    
    obs_pos = context['obs_pos']
    safe_dist = context['safe_distance']
    
    diff = trajectory[:, None, :] - obs_pos[None, :, :]
    min_dists = jnp.min(jnp.linalg.norm(diff, axis=-1), axis=-1)
    
    cost_obstacle = 600.0 * jnp.sum(jnp.maximum(0.0, safe_dist - min_dists))
    
    cost_smooth = jnp.sum(jnp.diff(ws)**2) * 1.5 + jnp.sum(jnp.diff(vs)**2) * 0.5
    cost_v = jnp.sum(vs**2) * 1.0 + jnp.sum(ws**2) * 0.5
    
    return cost_track + cost_final + cost_obstacle + cost_smooth + cost_v

# ==============================================================================
# 3. 辅助函数
# ==============================================================================
def generate_grid_obstacles(rows, cols, spacing, start_x, start_y):
    x = jnp.linspace(start_x, start_x + (cols-1)*spacing, cols)
    y = jnp.linspace(start_y, start_y + (rows-1)*spacing, rows)
    xx, yy = jnp.meshgrid(x, y)
    return jnp.stack([xx.ravel(), yy.ravel()], axis=1)


# ==============================================================================
# 4. 主循环 (CMA-ES 版本)
# ==============================================================================
def run_cma_es_comparison():
    print("Starting CMA-ES Comparison Simulation...")
    key = jax.random.PRNGKey(42)
    
    # 障碍物初始化
    obs_initial_pos = generate_grid_obstacles(4, 4, CONFIG['obs_spacing'], 2.5, 2.5)
    obs_num = obs_initial_pos.shape[0]
    key, subkey = jax.random.split(key)
    obs_phases = jax.random.uniform(subkey, (obs_num, 2)) * 2 * jnp.pi
    
    # CMA-ES 初始化参数
    # 因为 CMA-ES 是单峰的，我们只需一个初始均值
    x_current = np.zeros(TOTAL_DIM) 
    sigma0 = 0.3  # 初始步长
    
    robot_state = jnp.array([0.0, 0.0, 0.0])
    target_final = jnp.array([18.0, 15.0])
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))

    for t in range(180):
        # 更新环境
        offsets = 0.25 * jnp.stack([jnp.sin(t * 0.1 + obs_phases[:, 0]), jnp.cos(t * 0.12 + obs_phases[:, 1])], axis=1)
        current_obs_pos = obs_initial_pos + offsets
        target_pos = target_final + jnp.array([jnp.sin(t*0.5), jnp.cos(t*0.2)]) * 1.2
        
        context_data = {
            'current_state': robot_state, 'target_pos': target_pos,
            'obs_pos': current_obs_pos, 'obs_radius': CONFIG['obs_radius'],
            'safe_distance': CONFIG['obs_radius'] + CONFIG['safe_margin']
        }

        # 1. 调用 CMA-ES 求解器
        # 我们限制迭代次数以模拟实时性需求 (maxiter)
        # popsize 设为 60 与你的 MGIGO 一致
        res = cma.fmin(
            lambda x: float(mpc_cost_fn(jnp.array(x), context_data)),
            x_current,
            sigma0,
            options={
                'popsize': 60,
                'maxiter': 5, # 限制迭代次数，模拟实时环境
                'verb_disp': 0,
                'bounds': [-1, 1]
            }
        )
        x_best = res[0]
        
        # 2. 执行控制
        best_u_seq = jnp.array(x_best).reshape(CONFIG['horizon'], CONFIG['dim'])
        v_exec = jnp.tanh(3*best_u_seq[0, 0]) * CONFIG['max_v']
        w_exec = jnp.tanh(3*best_u_seq[0, 1]) * CONFIG['max_w']
        
        # 更新机器人状态
        new_theta = robot_state[2] + w_exec * CONFIG['dt']
        new_x = robot_state[0] + v_exec * jnp.cos(robot_state[2]) * CONFIG['dt']
        new_y = robot_state[1] + v_exec * jnp.sin(robot_state[2]) * CONFIG['dt']
        robot_state = jnp.array([new_x, new_y, new_theta])
        
        # 3. 热启动 (Simple Shift)
        x_reshaped = x_best.reshape(CONFIG['horizon'], CONFIG['dim'])
        x_shifted = np.roll(x_reshaped, shift=-1, axis=0)
        x_shifted[-1, :] = jax.random.uniform(subkey, (2,)) * 2 - 1  
        x_current = x_shifted.flatten()

        # --- 绘图 ---
        ax.cla()
        ax.set_xlim(-2, 22); ax.set_ylim(-2, 22); ax.set_aspect('equal')
        for i in range(obs_num):
            ax.add_patch(Circle(current_obs_pos[i], CONFIG['obs_radius'], color='red', alpha=0.3))
        
        # 绘制 CMA-ES 预测轨迹
        c_vs, c_ws = jnp.tanh(best_u_seq[:, 0]) * CONFIG['max_v'], jnp.tanh(best_u_seq[:, 1]) * CONFIG['max_w']
        c_thetas = robot_state[2] + jnp.cumsum(c_ws * CONFIG['dt'])
        c_thetas_pre = jnp.concatenate([jnp.array([robot_state[2]]), c_thetas[:-1]])
        c_traj = jnp.stack([
            robot_state[0] + jnp.cumsum(c_vs * jnp.cos(c_thetas_pre) * CONFIG['dt']),
            robot_state[1] + jnp.cumsum(c_vs * jnp.sin(c_thetas_pre) * CONFIG['dt'])
        ], axis=1)
        ax.plot(c_traj[:, 0], c_traj[:, 1], color='blue', lw=4.5, label='CMA-ES Prediction')
        
        ax.plot(target_pos[0], target_pos[1], 'g*', ms=12)
        ax.set_title(f"Step {t} | Solver: CMA-ES")
        plt.pause(0.01)

if __name__ == "__main__":
    run_cma_es_comparison()