"""Spline-based physical transformation helpers.

This module uses the offline basis artifact generated in ``Cartest/spline.py``
and the lightweight ``BSplineTrajectoryGenerator`` wrapper to convert control
points into continuous trajectories and vehicle states.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
from jax import jit

from MultipleTest.bsplineplanner import BSplineTrajectoryGenerator


WHEEL_BASE = 2.8
LR = 1.4
MAX_SPEED = 30.0
MAX_ACC = 3.0
MAX_STEER = 0.12
TAU_ACC = 0.25
TAU_STEER = 0.20
MAX_CENTRIPETAL_ACC = 2.0

DEFAULT_BASIS_PATH = Path(__file__).resolve().parents[1] / "Cartest" / "bspline_basis.npz"


@lru_cache(maxsize=1)
def get_trajectory_generator(basis_path: str | Path = DEFAULT_BASIS_PATH) -> BSplineTrajectoryGenerator:
    """Load the offline basis once and reuse it across all calls."""
    return BSplineTrajectoryGenerator(Path(basis_path))


TRAJ_GEN = get_trajectory_generator()


@jit
def trajectory_to_vehicle_states(
    pos: jnp.ndarray, vel: jnp.ndarray, acc: jnp.ndarray, wheel_base: float = WHEEL_BASE, lr: float = LR
) -> jnp.ndarray:
    """从 B-spline 的一、二阶时空导数反解出严格符合双轮单车动力学几何的车辆状态 [T, 6].

    状态分量: [x, y, v, psi, a_long, steer]
    """
    v = jnp.linalg.norm(vel, axis=-1)
    v_norm = v + 1e-6

    # 1. B-spline 导数方向为实际质心运动方向 (Course Angle = psi + beta)
    course_angle = jnp.arctan2(vel[..., 1], vel[..., 0])

    # 2. 计算几何曲率 kappa
    curvature = (vel[..., 0] * acc[..., 1] - vel[..., 1] * acc[..., 0]) / (v**3 + 1e-6)

    # 3. 根据几何关系反解质心侧偏角 beta: sin(beta) = lr * kappa
    beta = jnp.arcsin(jnp.clip(curvature * lr, -0.9, 0.9))

    # 4. 解耦解出车身绝对姿态角 psi (Heading Angle)
    psi = course_angle - beta

    # 5. 根据几何关系反解前轮转向角 steer: tan(steer) = (L / lr) * tan(beta)
    steer = jnp.arctan((wheel_base / lr) * jnp.tan(beta))
    steer = jnp.clip(steer, -MAX_STEER, MAX_STEER)

    # 6. 纵向加速度为加速度在速度切线方向的投影
    v_dir = vel / v_norm[..., None]
    a_long = jnp.sum(acc * v_dir, axis=-1)
    a_long = jnp.clip(a_long, -MAX_ACC, MAX_ACC)

    return jnp.stack([pos[..., 0], pos[..., 1], v, psi, a_long, steer], axis=-1)


@jit
def evaluate_trajectory_batch(
    control_points_batch: jnp.ndarray, basis_path: str | Path = DEFAULT_BASIS_PATH
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """批量计算控制点的时空导数。"""
    return TRAJ_GEN.evaluate_batch(control_points_batch)


@jit
def ego_trajectory_to_vehicle_states(
    control_points_batch: jnp.ndarray, wheel_base: float = WHEEL_BASE
) -> jnp.ndarray:
    """便捷打包包装器，直接将批量的 [B, n_ctrl, 2] 控制点转化为全状态矩阵 [B, T, 6]."""
    pos, vel, acc = TRAJ_GEN.evaluate_batch(control_points_batch)
    # 利用 vmap 作用于时间序列上
    return jnp.vectorize(
        lambda p, v, a: trajectory_to_vehicle_states(p, v, a, wheel_base=wheel_base),
        signature="(t,2),(t,2),(t,2)->(t,6)"
    )(pos, vel, acc)


@jit
def physical_constraint_cost(
    control_points: jnp.ndarray,
    wheel_base: float = WHEEL_BASE,
    max_speed: float = MAX_SPEED,
    max_acc: float = MAX_ACC,
    max_steer: float = MAX_STEER,
) -> float:
    pos, vel, acc = TRAJ_GEN.evaluate(control_points)
    states = trajectory_to_vehicle_states(pos, vel, acc, wheel_base=wheel_base)

    v = states[..., 2]
    a_long = states[..., 4]
    
    # 引入高阶惩罚：一旦超速或超加速度，惩罚以 4 次方暴涨，产生陡峭的屏障效应
    v_violation = jnp.maximum(0.0, jnp.abs(v) - max_speed)
    a_violation = jnp.maximum(0.0, jnp.abs(a_long) - max_acc)
    
    v_cost = jnp.mean(v_violation ** 2 + 10.0 * (v_violation ** 4))
    a_cost = jnp.mean(a_violation ** 2 + 50.0 * (a_violation ** 4))
    
    # 强烈惩罚控制点的空间跳跃度（差分惩罚项）
    smooth_cost = 50.0 * jnp.mean(jnp.sum(jnp.diff(control_points, axis=0) ** 2, axis=-1))

    return v_cost + a_cost + smooth_cost

__all__ = [
    "BSplineTrajectoryGenerator",
    "DEFAULT_BASIS_PATH",
    "TRAJ_GEN",
    "evaluate_trajectory_batch",
    "ego_trajectory_to_vehicle_states",
    "get_trajectory_generator",
    "physical_constraint_cost",
    "trajectory_to_vehicle_states",
]
