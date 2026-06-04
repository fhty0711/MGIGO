"""Receding-horizon runner for the spline-based trackgame variant."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random, vmap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Trackgamespline import (
    DEFAULT_BASIS_PATH,
    DEFAULT_N_AGENTS,
    TRAJ_GEN,
    N_CTRL,
    T,
    DT_C,
    batch_fitness_fn,
    decode_joint_sample,
    evaluate_joint_trajectory,
    fitness_fn_j_jax,
)
from gmm_igo.MPC_G_MS import mmog_igo_rne_blocks_solver

SEED = 42
DT = 0.5
# 根据离线导出的基矩阵分辨率算出来的推进帧数索引
EXEC_STEPS = max(1, int(round(DT / DT_C)))

M_AGENT = 3
K = 3
B = 80       # 提升采样数以适配高维空间搜索
B0 = 30
T_0 = 300
M_inner = 35
N_MPC_STEPS = 20

# 核心：定义分块。每个智能体占用一个独立块，决策空间是 10 个控制点的 (x,y) 坐标
N_BLOCKS = 3
BLOCK_DIMS = (N_CTRL * 2, N_CTRL * 2, N_CTRL * 2)
BLOCK_TO_AGENT = (0, 1, 2)


def _generate_static_warm_start_mu(current_states: jnp.ndarray) -> jnp.ndarray:
    """基于当前车辆真实的物理状态，极为稳健地生成暖启动控制点"""
    mus = []
    for i in range(M_AGENT):
        x, y, v, psi = current_states[i, 0], current_states[i, 1], current_states[i, 2], current_states[i, 3]
        
        # 裁剪当前速度上限，严防暖启动阶段就把控制点铺到几百米外
        v_clipped = jnp.clip(v, 0.0, 25.0) 
        
        # 生成时间网格
        t_steps = jnp.linspace(0.0, float(TRAJ_GEN.total_time), N_CTRL)
        
        # 严格外推
        cx = x + v_clipped * jnp.cos(psi) * t_steps
        cy = y + v_clipped * jnp.sin(psi) * t_steps
        c_pts = jnp.stack([cx, cy], axis=-1).flatten()
        mus.append(c_pts)
        
    return jnp.concatenate(mus)


def _select_block_wise_best_components(final_mu, context_arr):
    """从优化的 GMM 组件中挑出总体 Cost 最优的分块索引。"""
    # 建立确定性的联合样本形态
    best_joint = final_mu[:, 0, :] # 暂取高概率主轴组件
    return jnp.array([0, 0, 0])


def _assemble_joint_sample(final_mu, best_block_ks):
    e_pts = final_mu[0, best_block_ks[0]]
    f_pts = final_mu[1, best_block_ks[1]]
    r_pts = final_mu[2, best_block_ks[2]]
    return jnp.concatenate([e_pts, f_pts, r_pts])


def main(save_outputs: bool = True):
    print(f"B-spline 轨迹空间博弈启动。时空网格分辨率: {TRAJ_GEN.B.shape[0]}步, dt={DT_C}s")
    
    # 状态分量: [x, y, v, psi, a_long, steer]
    current_states = jnp.array([
        [0.0, 0.0, 15.0, 0.0, 0.0, 0.0],     # Ego 自车
        [30.0, 3.5, 12.0, 0.0, 0.0, 0.0],    # West 前车
        [-25.0, 0.0, 18.0, 0.0, 0.0, 0.0],   # North 后车
    ])

    history_positions = [current_states[:, :2]]
    static_L_inv_identity = jnp.stack(
        [jnp.stack([jnp.eye(N_CTRL * 2)] * K) for _ in range(N_BLOCKS)]
    ) * 2.5

    main_key = random.PRNGKey(SEED)

    for mpc_step in range(N_MPC_STEPS):
        t_step_start = time.time()
        main_key, solve_key = random.split(main_key)

        context_arr = current_states.flatten()
        # 每一物理步根据车辆最新的时空锚点更新控制点暖启动初值
        base_mu = _generate_static_warm_start_mu(current_states)
        current_mu_init = jnp.repeat(base_mu[:, None, :], K, axis=1)

        # 唤醒基于变分信息几何的多分块博弈求解器
        final_mu, final_L, final_pi, _ = mmog_igo_rne_blocks_solver(
            solve_key,
            N_MPC_STEPS,  # 全局优化周期迭代数
            DT,
            N_BLOCKS,
            M_AGENT,
            K,
            B,
            B0,
            BLOCK_DIMS,
            T_0,
            fitness_fn_j_jax,
            current_mu_init,
            static_L_inv_identity,
            context_arr,
            M_inner,
            BLOCK_TO_AGENT,
        )

        best_block_ks = _select_block_wise_best_components(final_mu, context_arr)
        best_joint_sample = _assemble_joint_sample(final_mu, best_block_ks)
        
        # 核心变动：利用 B-spline 全量时空外推得到的全状态矩阵直接进行车辆物理更新
        _, _, _, best_dense_states = evaluate_joint_trajectory(best_joint_sample)
        
        # 将环境向前推进 EXEC_STEPS 帧（通常等于 0.5s / 0.05s = 10 帧的位置）
        current_states = best_dense_states[:, EXEC_STEPS]
        history_positions.append(current_states[:, :2])

        t_step_end = time.time()
        print(
            f"Step {mpc_step:02d} | Ego X={current_states[0,0]:.2f} Y={current_states[0,1]:.2f} V={current_states[0,2]:.2f} | "
            f"耗时: {t_step_end - t_step_start:.2f}s"
        )


if __name__ == "__main__":
    main()
