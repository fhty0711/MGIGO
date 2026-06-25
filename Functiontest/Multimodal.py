"""
Multimodal Cost Optimization with IGO Solver
============================================
探索多模态cost的自动寻优：不同模态（目标）各自有不同的约束（硬约束/概率约束），
通过饱和嵌套实现自动模态选择与竞争。

总cost结构:
    total = saturate( saturate(f1(z)) + saturate(f2(z)) )

其中 f1(z) 和 f2(z) 各自包含内部的约束饱和嵌套结构（硬约束/概率约束）。
求解器通过 GMM 在联合 cost 景观上自动探索，天然地"选择"可达的模态。

自动驾驶隐喻：
- 不同传感器模态（camera/lidar/radar）有不同的不确定性特征
- Speed vs Safety：激进驾驶 vs 保守驾驶
- 多任务竞争：同时导航到多个可能目标，自动选择最可行的
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from jax import random, vmap, lax
import tqdm

# ── Import shared infrastructure from RobustConstraints ────────────────────
from RobustConstraints import (
    saturate,
    solver,
    generate_animation_data,
    render_iteration_animation,
    save_contour_plot,
    plot_3D_surface,
    calculate_countours,
    exp_fn,
)


# ═══════════════════════════════════════════════════════════════════════════
# Multimodal Cost Functions
# ═══════════════════════════════════════════════════════════════════════════
#
# 每个 cost_multimodal_* 遵循统一结构:
#   cost_A = saturate( penalty_A or saturate(f_obj_A) )
#   cost_B = saturate( penalty_B or saturate(f_obj_B) )
#   return saturate(cost_A + cost_B)
#
# 配套的 _modA / _modB 函数用于分解可视化。


# ── Scenario 0: Constraint-Driven Modality Selection ────────────────────
# Modality A 被硬约束限制（x+y ≤ 4），Modality B 无约束。
# 求解器应自动偏好无约束的 B。


@jax.jit
def _hard_select_modA(x):
    """Modality A: 目标 (-4, 3)，硬约束 x+y <= 4"""
    x1, x2 = x[0], x[1]
    f_obj = ((x1 + 4.0) ** 2 + (x2 - 4.0) ** 2) / 10.0
    viol = jnp.maximum(0.0, x1 + x2 + 1.0)  # x+y > -1 → 违反
    return saturate(
        jnp.where(viol > 0, viol + 1.5, saturate(f_obj))
    )


@jax.jit
def _hard_select_modB(x):
    """Modality B: 目标 (4, -3)，无约束"""
    x1, x2 = x[0], x[1]
    f_obj = ((x1 - 4.0) ** 2 + (x2 + 4.0) ** 2) / 10.0
    viol = jnp.maximum(0.0, x1 + x2 - 1.0)  # x+y > -2 → 违反（更宽松的约束，模拟弱约束）s
    return saturate(
        jnp.where(viol > 0, viol + 1.5, saturate(f_obj))
    )


@jax.jit
def cost_multimodal_hard_select(x, ctx=None):
    """约束驱动的模态选择: A有硬约束(x+y≤-4), B无约束

    期望: 求解器偏好无约束的 B (在 (4,-3) 附近)，
    因为 A 的可行域限制了目标达成。
    """
    return saturate(_hard_select_modA(x) + _hard_select_modB(x))


# ── Scenario 1: Probabilistic Sensor Fusion ────────────────────────────
# 模拟 Camera (中等噪声) vs Lidar (低噪声) 传感器融合。
# 每个模态有各自的概率约束（不同的不确定性分布）。


@jax.jit
def _prob_sensors_modA(x):
    """Camera模态: 目标(-3,3), P(x²+y² ≤ 25 | ξ_cam) ≥ 0.9"""
    x1, x2 = x[0], x[1]
    M = 100
    key = random.PRNGKey(0)
    # Camera: 中等噪声 N(0, 0.8²)
    xi = random.normal(key, (M,)) * 0.8
    d2 = (x1 + xi) ** 2 + (x2 + xi) ** 2
    q = jnp.quantile(d2, 0.90)
    viol = jnp.maximum(0.0, q - 25.0)
    f_obj = ((x1 + 3.0) ** 2 + (x2 - 3.0) ** 2) / 10.0
    return saturate(
        jnp.where(viol > 0, viol + 1.5, saturate(f_obj))
    )


@jax.jit
def _prob_sensors_modB(x):
    """Lidar模态: 目标(3,-3), P((x-2)²+(y-2)² ≤ 16 | ξ_lidar) ≥ 0.9"""
    x1, x2 = x[0], x[1]
    M = 100
    key = random.PRNGKey(1)  # 不同 seed，不同不确定性
    # Lidar: 更低噪声 N(0, 0.3²)，但双峰分布（模拟镜面反射）
    k1, k2 = random.split(key)
    mix = random.bernoulli(k1, 0.15, (M,))  # 15% 异常
    xi = jnp.where(
        mix,
        random.normal(k2, (M,)) * 1.5,  # 异常: 高噪声
        random.normal(random.fold_in(k2, 1), (M,)) * 0.3,  # 正常: 低噪声
    )
    d2 = (x1 - 2.0 + xi) ** 2 + (x2 - 2.0 + xi) ** 2
    q = jnp.quantile(d2, 0.90)
    viol = jnp.maximum(0.0, q - 16.0)
    f_obj = ((x1 - 3.0) ** 2 + (x2 + 3.0) ** 2) / 10.0
    return saturate(
        jnp.where(viol > 0, viol + 1.5, saturate(f_obj))
    )


@jax.jit
def cost_multimodal_prob_sensors(x, ctx=None):
    """概率传感器融合: Camera(中等噪声) vs Lidar(低噪声+异常值)

    两个模态有不同的不确定性特征，求解器需自动权衡。
    Lidar 噪声低但有 15% 异常，Camera 噪声中等但稳定。
    total = saturate(cost_cam + cost_lidar)
    """
    return saturate(_prob_sensors_modA(x) + _prob_sensors_modB(x))


# ── Scenario 2: Three-Way Competition ──────────────────────────────────
# 三个模态竞争: 各自有不同的目标和约束难度


@jax.jit
def _threeway_modA(x):
    """Modality A: 目标 (-5, 5), 硬约束 x >= -3"""
    x1, x2 = x[0], x[1]
    viol = jnp.maximum(0.0, -3.0 - x1)  # x1 < -3 → 违反
    f_obj = ((x1 + 5.0) ** 2 + (x2 - 5.0) ** 2) / 10.0
    return saturate(
        jnp.where(viol > 0, viol + 1.5, saturate(f_obj))
    )


@jax.jit
def _threeway_modB(x):
    """Modality B: 目标 (5, 5), 无约束"""
    x1, x2 = x[0], x[1]
    f_obj = ((x1 - 5.0) ** 2 + (x2 - 5.0) ** 2) / 10.0
    return saturate(f_obj)


@jax.jit
def _threeway_modC(x):
    """Modality C: 目标 (0, -6), 概率约束: P(x²+y² ≤ 40 | ξ) ≥ 0.9"""
    x1, x2 = x[0], x[1]
    M = 80
    key = random.PRNGKey(4)
    xi = random.normal(key, (M,)) * 0.7
    d2 = (x1 + xi) ** 2 + (x2 + xi) ** 2
    q = jnp.quantile(d2, 0.90)
    viol = jnp.maximum(0.0, q - 40.0)
    f_obj = (x1**2 + (x2 + 6.0) ** 2) / 10.0
    return saturate(
        jnp.where(viol > 0, viol + 1.5, saturate(f_obj))
    )


@jax.jit
def cost_multimodal_three_way(x, ctx=None):
    """三模态竞争: A(硬约束), B(无约束), C(概率约束)

    total = saturate(cost_A + cost_B + cost_C)
    三个不同难度的模态同时竞争，求解器自动选择最可行的。
    """
    return saturate(
        _threeway_modA(x) + _threeway_modB(x) + _threeway_modC(x)
    )


# ═══════════════════════════════════════════════════════════════════════════
# Decomposition & Enhanced Visualization
# ═══════════════════════════════════════════════════════════════════════════


def plot_multimodal_decomposition(
    cost_total,
    cost_modality_list,
    names,
    output_path,
    x1_lim=(-8.0, 8.0),
    x2_lim=(-8.0, 8.0),
    resolution=120,
):
    """绘制多模态 cost 的分解图：总 cost + 各模态单独 cost 的 3D 表面对比。

    Parameters
    ----------
    cost_total : callable
        总 cost 函数 cost(x, ctx)
    cost_modality_list : list of callable
        各模态的单独 cost 函数 [modA(x), modB(x), ...]
    names : list of str
        各子图标题 ["Total: A+B", "Modality A", "Modality B", ...]
    output_path : Path or str
        输出 PNG 路径
    """
    n_mods = len(cost_modality_list)
    n_cols = n_mods + 1  # +1 for total

    x1 = np.linspace(start=x1_lim[0], stop=x1_lim[1], num=resolution)
    x2 = np.linspace(start=x2_lim[0], stop=x2_lim[1], num=resolution)
    xx, yy = np.meshgrid(x1, x2)
    grid = jnp.stack(
        [jnp.asarray(xx).reshape(-1), jnp.asarray(yy).reshape(-1)], axis=1
    )

    # 计算所有 surfaces
    z_total = np.asarray(vmap(lambda p: cost_total(p, None))(grid)).reshape(xx.shape)
    z_mods = []
    for c in cost_modality_list:
        z_mods.append(
            np.asarray(vmap(lambda p: c(p))(grid)).reshape(xx.shape)
        )

    all_z = [z_total] + z_mods

    fig = plt.figure(figsize=(5.5 * n_cols, 4.5), dpi=150)
    for i, (z, name) in enumerate(zip(all_z, names)):
        ax = fig.add_subplot(1, n_cols, i + 1, projection="3d")
        ax.plot_surface(xx, yy, z, cmap="viridis", edgecolor="none", alpha=0.9)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("cost")
        # 统一 z 轴范围以便比较
        z_finite = z[np.isfinite(z)]
        if z_finite.size > 0:
            ax.set_zlim(np.min(z_finite), np.max(z_finite))

    # fig.suptitle("Multimodal Cost Decomposition", fontsize=13, y=1.01)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved decomposition plot: {output_path}")
    return output_path


def plot_multimodal_contour_comparison(
    cost_total,
    cost_modality_list,
    labels,
    output_path,
    x1_lim=(-8.0, 8.0),
    x2_lim=(-8.0, 8.0),
    resolution=200,
):
    """绘制总 cost 和模态单独 cost 的等值线对比（2D 叠加）。

    总 cost 用填充等值线，各模态的最低点用标记标出。
    """
    n_mods = len(cost_modality_list)
    n_cols = min(n_mods + 1, 3)
    n_rows = int(np.ceil((n_mods + 1) / n_cols))

    x1 = np.linspace(start=x1_lim[0], stop=x1_lim[1], num=resolution)
    x2 = np.linspace(start=x2_lim[0], stop=x2_lim[1], num=resolution)
    xx, yy = np.meshgrid(x1, x2)
    grid = jnp.stack(
        [jnp.asarray(xx).reshape(-1), jnp.asarray(yy).reshape(-1)], axis=1
    )

    all_costs = [cost_total] + cost_modality_list
    all_labels = ["Total"] + labels
    all_z = []
    for idx, c in enumerate(all_costs):
        if idx == 0:
            # Total cost: takes (x, ctx)
            zz = np.asarray(vmap(lambda p: c(p, None))(grid))
        else:
            # Modality costs: take only (x,)
            zz = np.asarray(vmap(lambda p: c(p))(grid))
        all_z.append(zz.reshape(xx.shape))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        dpi=150,
        squeeze=False,
    )

    for idx, (ax, z, label) in enumerate(
        zip(axes.flat, all_z, all_labels)
    ):
        if idx >= len(all_z):
            ax.set_visible(False)
            continue
        cf = ax.contourf(xx, yy, z, levels=24, cmap="viridis", alpha=0.85)
        ax.contour(xx, yy, z, levels=12, colors="black", linewidths=0.4, alpha=0.5)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(x1_lim)
        ax.set_ylim(x2_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    # fig.suptitle("Multimodal Cost Contour Comparison", fontsize=13)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved contour comparison: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Run a single multimodal test (contour + animation)
# ═══════════════════════════════════════════════════════════════════════════


def run_multimodal_test(
    cost,
    cost_name,
    mu0_list,
    output_root,
    l0=0.5,
    iter_tot=600,
    k=3,
    dt=0.15,
    b=300,
    b0=120,
    t0=100,
    seed=80,
    fps=15,
):
    """运行单个多模态测试：保存等值线图 + 生成 IGO 动画。

    Parameters
    ----------
    cost : callable
        总 cost 函数
    cost_name : str
        用于文件命名的名称
    mu0_list : list of [x1, x2]
        各分量的初始位置（利于探索不同模态）
    output_root : Path
        输出目录
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 等值线图
    save_contour_plot(output_root, cost, cost_name=cost_name)

    # 3D 表面图
    try:
        plot_3D_surface(cost, x1_lim=(-8.0, 8.0), x2_lim=(-8.0, 8.0), resolution=100)
        surface_path = output_root / f"{cost_name}_surface.png"
        plt.savefig(surface_path)
        plt.close()
        print(f"Saved surface plot: {surface_path}")
    except Exception as e:
        print(f"[warn] 3D surface plot failed for {cost_name}: {e}")

    # IGO 动画
    print(f"\n{'='*60}")
    print(f"Running IGO solver for: {cost_name}")
    print(f"{'='*60}")
    frame_data, all_f_vals = generate_animation_data(
        total_iterations=iter_tot,
        frame_stride=5,
        seed=seed,
        m=1,
        dims=(2,),
        k=k,
        dt=dt,
        b=b,
        b0=b0,
        t0=t0,
        cost=cost,
        sample_count=100,
        mu0_list=mu0_list,
        l0=l0,
    )
    render_iteration_animation(
        output_root=output_root,
        frame_data=frame_data,
        all_f_vals=all_f_vals,
        cost=cost,
        cost_name=cost_name,
        fps=fps,
    )
    return frame_data, all_f_vals


# ═══════════════════════════════════════════════════════════════════════════
# Main: run all multimodal tests
# ═══════════════════════════════════════════════════════════════════════════

# 所有多模态测试的注册表
MULTIMODAL_TESTS = [
    {
        "name": "multimodal_hard_select",
        "cost": cost_multimodal_hard_select,
        "mod_costs": [_hard_select_modA, _hard_select_modB],
        "mod_names": [
            "Total: A+B",
            "Mod A (x+y<=4)",
            "Mod B (unconstrained)",
        ],
        "mu0_list": [
            [0.0, 1.0],   # near Mod A (constrained)
            [1.0, 0.0],   # near Mod B (free)
            [5.0, 5.0],    # middle
        ],
        "description": "Constraint-driven: A constrained(x+y<=4), B free -> solver prefers B",
    },
    {
        "name": "multimodal_prob_sensors",
        "cost": cost_multimodal_prob_sensors,
        "mod_costs": [_prob_sensors_modA, _prob_sensors_modB],
        "mod_names": [
            "Total: Camera+Lidar",
            "Camera (sig=0.8 stable)",
            "Lidar (sig=0.3, 15% outlier)",
        ],
        "mu0_list": [
            [6.0, 2.0],   # near Camera target
            [6.0, 2.0],   # near Lidar target
            [0.0, -6.0],    # middle
        ],
        "description": "Sensor fusion: Camera(moderate stable noise) vs Lidar(low noise + outliers)",
    },
    {
        "name": "multimodal_three_way",
        "cost": cost_multimodal_three_way,
        "mod_costs": [_threeway_modA, _threeway_modB, _threeway_modC],
        "mod_names": [
            "Total: A+B+C",
            "A (x>=-3, target -5,5)",
            "B (unconstrained, target 5,5)",
            "C (prob, target 0,-6)",
        ],
        "mu0_list": [
            [0.0, 0.0],   # near A (hard constrained)
            [0.0, 0.0],    # near B (free)
            [0.0, 0.0],   # near C (probabilistic)
        ],
        "description": "Three-way competition: A(hard) vs B(free) vs C(probabilistic)",
    },
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multimodal Cost Optimization with IGO"
    )
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help=f"Test index (0-{len(MULTIMODAL_TESTS)-1})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory",
    )
    args = parser.parse_args()

    output_root = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "Multimodal_test"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    matplotlib.use("Agg")

    if 0 <= args.test < len(MULTIMODAL_TESTS):
        test = MULTIMODAL_TESTS[args.test]
        print(f"Test [{args.test}]: {test['name']} — {test['description']}")

        # Decomposition plots
        try:
            plot_multimodal_decomposition(
                cost_total=test["cost"],
                cost_modality_list=test["mod_costs"],
                names=test["mod_names"],
                output_path=output_root / f"{test['name']}_decomposition_3d.png",
            )
            plot_multimodal_contour_comparison(
                cost_total=test["cost"],
                cost_modality_list=test["mod_costs"],
                labels=test["mod_names"][1:],
                output_path=output_root / f"{test['name']}_contour_comparison.png",
            )
        except Exception as e:
            print(f"[warn] Decomposition failed: {e}")

        # IGO solver + animation
        run_multimodal_test(
            cost=test["cost"],
            cost_name=test["name"],
            mu0_list=test["mu0_list"],
            output_root=output_root,
            l0=0.5,
            iter_tot=600,
            k=3,
        )
    else:
        print(f"Invalid test index: {args.test}. Choose 0-{len(MULTIMODAL_TESTS)-1}")

    print(f"\n{'='*60}")
    print(f"Done. Outputs: {output_root}")
    print(f"{'='*60}")
