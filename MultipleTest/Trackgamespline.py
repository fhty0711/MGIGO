"""Spline-based trackgame utilities built on the offline B-spline basis."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
from jax import jit, lax, vmap

from MultipleTest.bsplineplanner import BSplineTrajectoryGenerator
from MultipleTest.Transformation import pairwise_footprint_overlap_cost
from MultipleTest.Transformerspline import TRAJ_GEN, trajectory_to_vehicle_states

# 物理与几何常数
WHEEL_BASE = 2.8
LR = 1.4
MAX_SPEED = 30.0
MAX_ACC = 3.0
MAX_STEER = 0.12

LOWER_LANE_CENTER = 0.0
UPPER_LANE_CENTER = 3.5
VEHICLE_LENGTH = 5.0
VEHICLE_WIDTH = 2.0
SAFE_GAP = 3.0

DEFAULT_BASIS_PATH = Path(__file__).resolve().parents[1] / "Cartest" / "bspline_basis.npz"
DEFAULT_N_AGENTS = 3

T = TRAJ_GEN.B.shape[0]        # 140
N_CTRL = TRAJ_GEN.B.shape[1]  # 10
DT_C = float(TRAJ_GEN.dt)


@jit
def decode_joint_sample(joint_sample_flat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """将优化器采样得到的单一平铺一维向量 [3 * n_ctrl * 2] 解包为三辆车的控制点矩阵 [n_ctrl, 2]."""
    # 结构: [Ego_x..., Ego_y..., Front_x..., Front_y..., Rear_x..., Rear_y...]
    pts_per_agent = N_CTRL * 2
    ego_part = joint_sample_flat[0 : pts_per_agent].reshape(N_CTRL, 2)
    front_part = joint_sample_flat[pts_per_agent : 2 * pts_per_agent].reshape(N_CTRL, 2)
    rear_part = joint_sample_flat[2 * pts_per_agent : 3 * pts_per_agent].reshape(N_CTRL, 2)
    return ego_part, front_part, rear_part



@jit
def evaluate_joint_trajectory(joint_sample_flat: jnp.ndarray, context_arr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """计算联合控制点对应的时空全量密集轨迹。"""
    ego_cps, front_cps, rear_cps = decode_joint_sample(joint_sample_flat)

    ep, ev, ea = TRAJ_GEN.evaluate(ego_cps)
    fp, fv, fa = TRAJ_GEN.evaluate(front_cps)
    rp, rv, ra = TRAJ_GEN.evaluate(rear_cps)


    ego_traj = trajectory_to_vehicle_states(ep, ev, ea, wheel_base=WHEEL_BASE, lr=LR)
    front_traj = trajectory_to_vehicle_states(fp, fv, fa, wheel_base=WHEEL_BASE, lr=LR)
    rear_traj = trajectory_to_vehicle_states(rp, rv, ra, wheel_base=WHEEL_BASE, lr=LR)

    joint_states = jnp.stack([ego_traj, front_traj, rear_traj], axis=0)
    return ego_traj, front_traj, rear_traj, joint_states


@jit
def dense_pair_collision_cost(traj_a: jnp.ndarray, traj_b: jnp.ndarray) -> float:
    """使用 vmap 沿全时空网格 T 轴并行评估两车脚印重叠度。"""

    def single_step_cost(state_a, state_b):
        return pairwise_footprint_overlap_cost(
            state_a[:2], state_b[:2], state_a[3], state_b[3],
            length=VEHICLE_LENGTH, width=VEHICLE_WIDTH
        )

    return jnp.sum(vmap(single_step_cost)(traj_a, traj_b))


@jit
def _initial_state_cost(traj: jnp.ndarray, target_init_state: jnp.ndarray) -> float:
    """强约束锚定惩罚：确保 B-spline 轨迹的起点与当前真实物理状态锁死。"""
    init_spline = traj[0, :4]
    init_target = target_init_state[:4]
    return jnp.sum((init_spline - init_target) ** 2)


@jit
def _ego_cost(joint_sample_flat: jnp.ndarray, context_arr: jnp.ndarray) -> float:
    current_states = context_arr.reshape(3, 6)
    ego_traj, front_traj, rear_traj, _ = evaluate_joint_trajectory(joint_sample_flat)

    # 目标行为导向 (开在下车道中心，保持期望速度 18m/s)
    y_ego = ego_traj[:, 1]
    v_ego = ego_traj[:, 2]
    state_cost = 2.0 * jnp.mean((y_ego - LOWER_LANE_CENTER) ** 2) + 0.5 * jnp.mean((v_ego - 18.0) ** 2)

    # 物理超限软惩罚
    control_cost = (
        1.0 * jnp.mean(jnp.maximum(0.0, jnp.abs(ego_traj[:, 4]) - MAX_ACC) ** 2) +
        10.0 * jnp.mean(jnp.maximum(0.0, jnp.abs(ego_traj[:, 5]) - MAX_STEER) ** 2)
    )

    # 多车交互时空冲突
    collision_cost = 50.0 * (
        dense_pair_collision_cost(ego_traj, front_traj) +
        dense_pair_collision_cost(ego_traj, rear_traj)
    )

    # 初始状态强制硬锚定成本
    init_cost = 100.0 * _initial_state_cost(ego_traj, current_states[0])

    return state_cost + control_cost + collision_cost + init_cost


@jit
def _front_cost(joint_sample_flat: jnp.ndarray, context_arr: jnp.ndarray) -> float:
    current_states = context_arr.reshape(3, 6)
    ego_traj, front_traj, rear_traj, _ = evaluate_joint_trajectory(joint_sample_flat)

    y_front = front_traj[:, 1]
    v_front = front_traj[:, 2]
    # 前车偏好在上车道以 15m/s 巡航
    state_cost = 2.0 * jnp.mean((y_front - UPPER_LANE_CENTER) ** 2) + 0.5 * jnp.mean((v_front - 15.0) ** 2)

    control_cost = 1.0 * jnp.mean(jnp.maximum(0.0, jnp.abs(front_traj[:, 4]) - MAX_ACC) ** 2)
    collision_cost = 50.0 * (
        dense_pair_collision_cost(front_traj, ego_traj) +
        dense_pair_collision_cost(front_traj, rear_traj)
    )

    init_cost = 100.0 * _initial_state_cost(front_traj, current_states[1])
    return state_cost + control_cost + collision_cost + init_cost


@jit
def _rear_cost(joint_sample_flat: jnp.ndarray, context_arr: jnp.ndarray) -> float:
    current_states = context_arr.reshape(3, 6)
    ego_traj, front_traj, rear_traj, _ = evaluate_joint_trajectory(joint_sample_flat)

    y_rear = rear_traj[:, 1]
    v_rear = rear_traj[:, 2]
    # 后车试图在下车道以 22m/s 的高速度追赶
    state_cost = 2.0 * jnp.mean((y_rear - LOWER_LANE_CENTER) ** 2) + 0.5 * jnp.mean((v_rear - 22.0) ** 2)

    control_cost = 1.0 * jnp.mean(jnp.maximum(0.0, jnp.abs(rear_traj[:, 4]) - MAX_ACC) ** 2)
    collision_cost = 50.0 * (
        dense_pair_collision_cost(rear_traj, ego_traj) +
        dense_pair_collision_cost(rear_traj, front_traj)
    )

    init_cost = 100.0 * _initial_state_cost(rear_traj, current_states[2])
    return state_cost + control_cost + collision_cost + init_cost


@jit
def fitness_fn_j_jax(agent_idx, joint_sample_flat, context_arr):
    return lax.switch(
        agent_idx,
        (
            lambda s, c: _ego_cost(s, c),
            lambda s, c: _front_cost(s, c),
            lambda s, c: _rear_cost(s, c),
        ),
        joint_sample_flat,
        context_arr,
    )


@jit
def batch_fitness_fn(agent_idx: jnp.ndarray, joint_samples: jnp.ndarray, context_arr: jnp.ndarray):
    return vmap(lambda sample: fitness_fn_j_jax(agent_idx, sample, context_arr))(joint_samples)


__all__ = [
    "DEFAULT_BASIS_PATH",
    "DEFAULT_N_AGENTS",
    "DT_C",
    "N_CTRL",
    "T",
    "TRAJ_GEN",
    "batch_fitness_fn",
    "decode_joint_sample",
    "dense_pair_collision_cost",
    "evaluate_joint_trajectory",
    "fitness_fn_j_jax",
    "_initial_state_cost",
]
