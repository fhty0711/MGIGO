import jax.numpy as jnp
from functools import partial
from jax import jit, lax

# ======================================================================
# 车辆物理边界限值与几何参数 (笛卡尔空间常数)
# ======================================================================
WHEEL_BASE = 2.8
LR = 1.4            
MAX_SPEED = 30.0
MAX_ACC = 3.0       
MAX_STEER = 0.12    

TAU_ACC = 0.25      
TAU_STEER = 0.20    
MAX_CENTRIPETAL_ACC = 2.0 

dt_step, dt_c = 0.5, 0.05
sub_steps = int(round(dt_step / dt_c))

@jit
def stable_single_car_step(state, target_raw_acc, target_raw_steer, dt):
    x, y, v, psi, curr_acc, curr_steer = state[0], state[1], state[2], state[3], state[4], state[5]
    
    t_acc = MAX_ACC * jnp.tanh(target_raw_acc)
    t_steer = MAX_STEER * jnp.tanh(target_raw_steer)
    
    next_acc = curr_acc + (t_acc - curr_acc) / TAU_ACC * dt
    next_acc = jnp.clip(next_acc, -MAX_ACC, MAX_ACC)
    next_steer = curr_steer + (t_steer - curr_steer) / TAU_STEER * dt
    next_steer = jnp.clip(next_steer, -MAX_STEER, MAX_STEER)
    
    beta = jnp.arctan(LR * jnp.tan(next_steer) / WHEEL_BASE)
    
    next_v = v + next_acc * dt
    next_v = jnp.clip(next_v, 0.0, MAX_SPEED)
    
    psi_dot_raw = next_v * jnp.cos(beta) * jnp.tan(next_steer) / WHEEL_BASE
    v_safe = jnp.maximum(next_v, 1e-3)
    max_psi_dot = MAX_CENTRIPETAL_ACC / v_safe
    next_psi_dot = jnp.clip(psi_dot_raw, -max_psi_dot, max_psi_dot)
    
    next_psi = psi + next_psi_dot * dt
    next_x = x + next_v * jnp.cos(next_psi + beta) * dt
    next_y = y + next_v * jnp.sin(next_psi + beta) * dt
    
    return jnp.array([next_x, next_y, next_v, next_psi, next_acc, next_steer])


@jit
def stable_lon_car_step(state, target_raw_acc, dt):
    x, y, v, psi, curr_acc, curr_steer = state[0], state[1], state[2], state[3], state[4], state[5]
    t_acc = MAX_ACC * jnp.tanh(target_raw_acc)
    next_acc = curr_acc + (t_acc - curr_acc) / TAU_ACC * dt
    next_acc = jnp.clip(next_acc, -MAX_ACC, MAX_ACC)
    next_v = v + next_acc * dt
    next_v = jnp.clip(next_v, 0.0, MAX_SPEED)
    next_x = x + next_v * dt
    return jnp.array([next_x, y, next_v, psi, next_acc, curr_steer])


@jit
def static_predict_horizon_dense_rollout(init_states, ego_acc_seq, ego_steer_seq, front_acc_seq, rear_acc_seq):
    """
    数学修正后的全轨迹稠密生成器。
    将原本 14 步的宏观序列按照分段常值（Zero-Order Hold）平铺至 140 步的微观时间网格上，
    返回形状为 (140, 3, 6) 的全量化状态时空轨迹，杜绝宏观采样点之间的测度丢失。
    """
    
    
    # 向量零阶保持上采样：通过重复元素将 (14,) 扩展为 (140,)
    ego_acc_dense = jnp.repeat(ego_acc_seq, sub_steps)
    ego_steer_dense = jnp.repeat(ego_steer_seq, sub_steps)
    front_acc_dense = jnp.repeat(front_acc_seq, sub_steps)
    rear_acc_dense = jnp.repeat(rear_acc_seq, sub_steps)
    
    total_micro_steps = ego_acc_dense.shape[0]
    ego_init, front_init, rear_init = init_states[0], init_states[1], init_states[2]

    def micro_step_fn(carry, idx):
        ego, front, rear = carry
        next_e = stable_single_car_step(ego, ego_acc_dense[idx], ego_steer_dense[idx], dt_c)
        next_f = stable_lon_car_step(front, front_acc_dense[idx], dt_c)
        next_r = stable_lon_car_step(rear, rear_acc_dense[idx], dt_c)
        current_step_states = jnp.stack([next_e, next_f, next_r])
        return (next_e, next_f, next_r), current_step_states

    _, dense_trajectory = lax.scan(micro_step_fn, (ego_init, front_init, rear_init), jnp.arange(total_micro_steps))
    return dense_trajectory


@jit
def _propagate_system(current_states, ego_ctrl, front_acc, rear_acc, dt_step=0.5, dt_c=0.05):
    sub_steps = int(round(dt_step / dt_c))
    ego, front, rear = current_states[0], current_states[1], current_states[2]

    def loop_body(i, carry):
        e, f, r = carry
        next_e = stable_single_car_step(e, ego_ctrl[0], ego_ctrl[1], dt_c)
        next_f = stable_lon_car_step(f, front_acc, dt_c)
        next_r = stable_lon_car_step(r, rear_acc, dt_c)
        return next_e, next_f, next_r

    final_ego, final_front, final_rear = lax.fori_loop(0, sub_steps, loop_body, (ego, front, rear))
    return jnp.stack([final_ego, final_front, final_rear])


@jit
def low_pass_filter_sequence(current_sequence, alpha=0.0, init_value=0.0):
    def scan_fn(carry, x):
        next_val = alpha * carry + (1.0 - alpha) * x
        return next_val, next_val
    _, filtered_seq = lax.scan(scan_fn, init_value, current_sequence)
    return filtered_seq


@jit
def pairwise_footprint_overlap_cost(pos_a, pos_b, psi_a, psi_b, length=5.0, width=2.0, safe_gap=3.0):
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    
    cos_psi = jnp.cos(psi_a)
    sin_psi = jnp.sin(psi_a)
    
    # 投影到自车车体坐标系
    rx = dx * cos_psi + dy * sin_psi
    ry = -dx * sin_psi + dy * cos_psi
    
    eff_len = length + safe_gap
    eff_wid = width 
    
    # 椭圆/矩形边界安全范围度量
    overlap_indicator = (rx / eff_len)**2 + (ry / eff_wid)**2
    
    # 直接使用 jnp.where：如果进入边界(value < 1.0)触发大惩罚常数，否则无视
    return jnp.where(overlap_indicator < 1.0, 10.0, 0.0) 