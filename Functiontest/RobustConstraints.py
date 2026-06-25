import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from matplotlib import animation
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from pathlib import Path
from jax import random, vmap, lax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from utils import _safe_color_limits, _blend_with_previous_if_nonfinite, _ellipse_within_limit, add_cov_ellipse
from utilss import _safe_eigh_numpy, build_joint_component_candidates, get_best_component_stats, sample_joint_from_solver

@jax.jit
def saturate(x: jnp.ndarray, k: float = 1.0):
    x = x / k
    return k * x / jnp.sqrt(1.0 + x ** 2)


@jax.jit
def cost_robust1(x, ctx=None):
    # 非凸不确定性集 (LMI无法处理): xi in [-3,-2] U [1,2]
    # 约束: (x1+xi)^2 + x2^2 <= 4.5  for all xi
    # 目标: 3个basin在可行域边界附近
    x1 = x[0]; x2 = x[1]
    N = 40
    xi_lo = jnp.linspace(-3.0, -2.0, N//2)
    xi_hi = jnp.linspace( 1.0,  2.0, N//2)
    xi_all = jnp.concatenate([xi_lo, xi_hi])

    def body(carry, xi):
        viol = jnp.maximum(0.0, (x1 + xi)**2 + x2**2 - 10.0)
        return jnp.maximum(carry, viol), None

    worst_viol, _ = lax.scan(body, 0.0, xi_all)

    # 多峰目标: 3个basin, 在可行域(透镜: x1~[-0.1,1.0])内不同区域
    f1 = (x1 - 0.1)**2 + (x2 - 1.2)**2
    f2 = (x1 - 0.7)**2 + (x2 + 1.1)**2
    f3 = (x1 - 0.4)**2 + (x2 - 0.0)**2
    f_obj = jnp.minimum(jnp.minimum(f1, f2), f3) / 3.0

    res = saturate(
        jnp.where(worst_viol > 0, worst_viol + 1.5,
            saturate(f_obj))
    )
    return res


@jax.jit # 多重RO+优先级: 安全(外) > 精度(内) > 目标
def cost_robust2(x, ctx=None):
    x1 = x[0]; x2 = x[1]
    N = 40
    # g1 (外层, 安全优先): xi in [-3,-2] U [1,2], (x1+xi)^2+x2^2 <= 10
    xi1 = jnp.concatenate([jnp.linspace(-3.0, -2.0, N//2),
                            jnp.linspace( 1.0,  2.0, N//2)])
    def body1(c, xi):
        v = jnp.maximum(0.0, (x1+xi)**2 + x2**2 - 10.0)
        return jnp.maximum(c, v), None
    viol1, _ = lax.scan(body1, 0.0, xi1)

    # g2 (内层, 精度): xi in [-0.5,0.5], (x1+xi)^2+x2^2 <= 3
    xi2 = jnp.linspace(-0.5, 0.5, N//2)
    def body2(c, xi):
        v = jnp.maximum(0.0, (x1+xi)**2 + x2**2 - 3.0)
        return jnp.maximum(c, v), None
    viol2, _ = lax.scan(body2, 0.0, xi2)

    # 目标: 3个basin, 在精度约束可行域内
    f1 = (x1 - 0.3)**2 + (x2 - 0.8)**2
    f2 = (x1 + 0.2)**2 + (x2 + 0.6)**2
    f3 = (x1 - 0.5)**2 + (x2 + 0.3)**2
    f_obj = jnp.minimum(jnp.minimum(f1, f2), f3) / 3.0

    res = saturate(
        jnp.where(viol1 > 0, viol1 + 1.5,
            saturate(jnp.where(viol2 > 0, viol2 + 1.5,
                saturate(f_obj))))
    )
    return res

# 在此处添加新的 cost 函数

# ======================================================================
# solver 包装器
# ======================================================================

def solver(key, T, dt, M, K, B, B0, dims, T_0, cost, mu, L_inv, v, context):
    """Wrapper around mmog_igo_optimizer_mpc (MPCsolverM22)."""
    final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        key, T, dt, M, K, B, B0, dims, T_0,
        fitness_fn_total=cost,
        initial_mu_k=mu,
        initial_L_inv_k=L_inv,
        initial_v_k=v,
        context=context,
    )
    # 从 pi 反算 v，保证循环调用间 v 状态不丢失
    final_v = jnp.log(jnp.clip(final_pi[:, :-1], 1e-20, None)
                      / jnp.clip(final_pi[:, -1:], 1e-20, None))
    return final_mu, final_L, final_pi, final_v

# ======================================================================
def plot_3D_surface(cost, x1_lim=(-8.0, 8.0), x2_lim=(-8.0, 8.0), resolution=100):
    x1 = np.linspace(start=x1_lim[0], stop=x1_lim[1], num=resolution)
    x2 = np.linspace(start=x2_lim[0], stop=x2_lim[1], num=resolution)
    xx, yy = np.meshgrid(x1, x2)
    grid_points = jnp.stack([jnp.asarray(xx).reshape(-1), jnp.asarray(yy).reshape(-1)], axis=1)
    zz = np.asarray(vmap(lambda p: cost(p, None))(grid_points)).reshape(xx.shape)

    fig = plt.figure(figsize=(8.0, 6.0), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none')
    ax.set_title('Cost Surface')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('cost')
    fig.tight_layout()
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

# plot_3D_surface(cost_sat_hierarchical, x1_lim=(-8.0, 8.0), x2_lim=(-8.0, 8.0), resolution=100)

def exp_fn(x, ctx, cost, dt=0.05):
    return jnp.exp(- cost(x, ctx) * dt)

def calculate_countours(cost, x1_lim=(-8.0, 8.0), x2_lim=(-8.0, 8.0), resolution=280):
    x1 = np.linspace(start=x1_lim[0], stop=x1_lim[1], num=resolution)
    x2 = np.linspace(start=x2_lim[0], stop=x2_lim[1], num=resolution)
    xx, yy = np.meshgrid(x1, x2)
    grid_points = jnp.stack([jnp.asarray(xx).reshape(-1), jnp.asarray(yy).reshape(-1)], axis=1)
    zz = np.asarray(vmap(lambda p: exp_fn(p, None, cost))(grid_points)).reshape(xx.shape)
    return xx, yy, zz

def generate_animation_data(
    total_iterations=500,
    frame_stride=5,
    seed=5,
    m=1,
    dims=(2,),
    k=3,
    dt=0.15,
    b=100,
    b0=40,
    t0=100,
    cost=None,
    sample_count=100,
    **kwargs,
):
    d_max = max(dims)
    if not (m == 1 and d_max == 2):
        raise ValueError("Animation currently supports single-block 2D only (m=1, dims=(2,)).")

    key = random.PRNGKey(seed)
    key, key_init = random.split(key)
    mu0_list = kwargs.get("mu0_list", None)   # [(x1,x2), ...] 每分量各一个起点
    mu0 = kwargs.get("mu0", None)             # 单个起点, 所有分量共用
    if mu0_list is not None:
        # 每个分量放不同位置, 加微小抖动
        initial_mu = random.uniform(key_init, shape=(m, k, d_max), minval=-0.1, maxval=0.1)
        mu0_arr = jnp.asarray(mu0_list).reshape(m, k, d_max)
        initial_mu = initial_mu + mu0_arr
    elif mu0 is not None:
        initial_mu = random.uniform(key_init, shape=(m, k, d_max), minval=-0.5, maxval=0.5)
        for i in range(m):
            for j in range(k):
                initial_mu = initial_mu.at[i, j].set(jnp.asarray(mu0) + initial_mu[i, j])
    else:
        initial_mu = random.uniform(key_init, shape=(m, k, d_max), minval=-8.0, maxval=8.0)
    l0 = kwargs.get("l0", None)
    if l0 is not None:
        initial_l_inv = jnp.tile(jnp.eye(d_max, dtype=jnp.float32)[None, None, :, :], (m, k, 1, 1)) * l0
    else:
        initial_l_inv = jnp.tile(jnp.eye(d_max, dtype=jnp.float32)[None, None, :, :], (m, k, 1, 1)) * 1.0
    initial_v = jnp.zeros((m, k - 1), dtype=jnp.float32)

    frame_data = []
    all_f_vals = []
    cur_mu = initial_mu
    cur_l_inv = initial_l_inv
    cur_v = initial_v
    warned_nonfinite_frame = False
    iter_list = list(range(0, total_iterations, frame_stride))
    if iter_list[-1] != total_iterations:
        iter_list.append(total_iterations)
    for t in tqdm.tqdm(iter_list, desc="IGO solver iterations"):
        key_solver = random.fold_in(key, 100000 + t)
        key_plot = random.fold_in(key, 100000 + t)

        final_mu, final_l_inv, final_pi, final_v = solver(key_solver, frame_stride, dt, m, k, b, b0, dims, t0, cost, cur_mu, cur_l_inv, cur_v, None)

        final_mu_np = _blend_with_previous_if_nonfinite(final_mu, cur_mu, clip_abs=1e3)
        final_l_inv_np = _blend_with_previous_if_nonfinite(final_l_inv, cur_l_inv, clip_abs=1e3)
        final_v_np = _blend_with_previous_if_nonfinite(final_v, cur_v, clip_abs=70.0)
        final_pi_np = np.asarray(final_pi)

        _, best_mu_np, best_cov_np, _ = get_best_component_stats(final_mu_np, final_l_inv_np, cost)

        raw_samples = sample_joint_from_solver(
            key_plot, final_mu_np, final_l_inv_np, final_pi_np, sample_count
        )
        samples_np = np.asarray(raw_samples)
        f_vals_np = np.asarray(vmap(lambda s: cost(s, None))(samples_np))

        finite_mask = np.isfinite(samples_np).all(axis=1) & np.isfinite(f_vals_np)
        if np.any(finite_mask):
            samples_np = samples_np[finite_mask]
            f_vals_np = f_vals_np[finite_mask]
        else:
            if not warned_nonfinite_frame:
                print(f"[warn] all samples non-finite at iter={t}")
                warned_nonfinite_frame = True
            samples_np = np.zeros((1, 2))
            f_vals_np = np.zeros((1,))

        means_np, covs_np = build_joint_component_candidates(final_mu_np, final_l_inv_np)

        frame_data.append({
            "t": t,
            "samples": samples_np,
            "f_vals": f_vals_np,
            "means": np.asarray(means_np),
            "covs": np.asarray(covs_np),
            "best_mu": best_mu_np,
            "best_cov": best_cov_np,
        })
        all_f_vals.append(f_vals_np)
        if t % t0 == 0:
            cur_mu, cur_l_inv, cur_v = final_mu_np, initial_l_inv, initial_v
        else:
            cur_mu, cur_l_inv, cur_v = final_mu_np, final_l_inv_np, final_v_np

    return frame_data, all_f_vals

def render_iteration_animation(output_root, frame_data, all_f_vals, cost, cost_name="cost", fps=15):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    f_all = np.concatenate(all_f_vals, axis=0)
    vmin, vmax = _safe_color_limits(f_all)

    xx, yy, zz = calculate_countours(cost)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    contour = ax.contour(xx, yy, zz, levels=32, cmap="Blues")
    ax.clabel(contour, inline=True, fontsize=7)

    first = frame_data[0]
    sc = ax.scatter(
        first["samples"][:, 0], first["samples"][:, 1], c=first["f_vals"],
        cmap="viridis_r", vmin=vmin, vmax=vmax, s=20, alpha=0.7, linewidths=0.0, zorder=1,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cost")

    mean_scatter = ax.scatter([], [], c="tab:orange", s=36, marker="D", edgecolors="black", linewidths=0.5, zorder=4)
    best_scatter = ax.scatter([], [], c="gold", s=180, marker="*", edgecolors="black", linewidths=0.9, zorder=6)
    title = ax.set_title(f"IGO optimization animation (iter={first['t']})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.2)
    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(-8.0, 8.0)
    ax.set_aspect("equal", adjustable="box")
    
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, alpha=0.8, label="samples"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="tab:orange", markeredgecolor="black", markersize=7, label="component means"),
        Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=1.2, label="3σ ellipses"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="black", markersize=11, label="best mean"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="best 3σ"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.tight_layout()

    def _compute_ellipse_params(mean_xy, cov_xy, n_std=1.0):
        """使用 utilss._safe_eigh_numpy 和 utils._ellipse_within_limit 计算椭圆参数."""
        eigvals, eigvecs = _safe_eigh_numpy(cov_xy)
        valid = _ellipse_within_limit(mean_xy, eigvals)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        w = 2.0 * n_std * np.sqrt(max(eigvals[0], 1e-12))
        h = 2.0 * n_std * np.sqrt(max(eigvals[1], 1e-12))
        a = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        return tuple(mean_xy), w, h, a, valid

    n_comp = first["means"].shape[0]
    comp_ellipses = []
    for i in range(n_comp):
        e = add_cov_ellipse(
            ax, first["means"][i], first["covs"][i], n_std=3.0,
            edgecolor="tab:orange", linewidth=1.2, alpha=0.8, linestyle="-", zorder=3,
        )
        if e is None:
            # 超出限制时创建一个不可见的占位椭圆
            e = Ellipse(xy=(0, 0), width=0.1, height=0.1, angle=0,
                        edgecolor="tab:orange", facecolor="none",
                        linewidth=1.2, alpha=0.8, linestyle="-", zorder=3)
            e.set_visible(False)
            ax.add_patch(e)
        comp_ellipses.append(e)

    best_e = add_cov_ellipse(
        ax, first["best_mu"], first["best_cov"], n_std=3.0,
        edgecolor="black", linewidth=1.8, alpha=0.85, linestyle="--", zorder=5,
    )
    if best_e is None:
        best_e = Ellipse(xy=(0, 0), width=0.1, height=0.1, angle=0,
                         edgecolor="black", facecolor="none",
                         linewidth=1.8, alpha=0.85, linestyle="--", zorder=5)
        best_e.set_visible(False)
        ax.add_patch(best_e)
    best_ellipse = best_e

    def update(frame_idx):
        fd = frame_data[frame_idx]
        title.set_text(f"IGO optimization animation (iter={fd['t']})")

        sc.set_offsets(fd["samples"])
        sc.set_array(fd["f_vals"])

        mean_scatter.set_offsets(fd["means"])
        best_scatter.set_offsets([fd["best_mu"]])

        for i in range(n_comp):
            center, w, h, a, valid = _compute_ellipse_params(fd["means"][i], fd["covs"][i], n_std=3.0)
            e = comp_ellipses[i]
            e.set_center(center)
            e.width = w
            e.height = h
            e.angle = a
            e.set_visible(valid)

        b_center, b_w, b_h, b_a, b_valid = _compute_ellipse_params(fd["best_mu"], fd["best_cov"], n_std=3.0)
        best_ellipse.set_center(b_center)
        best_ellipse.width = b_w
        best_ellipse.height = b_h
        best_ellipse.angle = b_a
        best_ellipse.set_visible(b_valid)

        return [sc, mean_scatter, best_scatter, best_ellipse, title] + comp_ellipses

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_data), interval=150, blit=False, repeat=False
    )

    gif_path = output_root / (cost_name + ".gif")
    ani.save(str(gif_path), writer="pillow", fps=fps)
    mp4_path = gif_path

    plt.close(fig)

    print(f"Saved animation: {mp4_path}")
    return mp4_path

def save_contour_plot(output_root, cost, cost_name="cost"):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    xx, yy, zz = calculate_countours(cost)
    plt.figure(figsize=(5.0, 5.0), dpi=300)
    contour = plt.contour(xx, yy, zz, levels=32, cmap="Blues")
    plt.clabel(contour, inline=True, fontsize=7)
    plt.title(f"Cost contour: {cost_name}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.2)
    plt.xlim(-8.0, 8.0)
    plt.ylim(-8.0, 8.0)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    # plt.tight_layout()
    contour_path = output_root / f"{cost_name}_contour.png"
    plt.savefig(contour_path)
    plt.close()
    print(f"Saved contour plot: {contour_path}")
    return contour_path

def run_robust_test(cost, cost_name, mu0, l0, iter_tot=600):
    output_root = Path("Robust_test")
    output_root.mkdir(parents=True, exist_ok=True)
    save_contour_plot(output_root, cost, cost_name=cost_name)
    frame_data, all_f_vals = generate_animation_data(
        total_iterations=iter_tot, frame_stride=5, seed=42,
        m=1, dims=(2,), k=5, dt=0.15, b=300, b0=120, t0=600,
        cost=cost, sample_count=100, mu0=mu0, l0=l0,
    )
    render_iteration_animation(output_root=output_root, frame_data=frame_data,
                               all_f_vals=all_f_vals, cost=cost, cost_name=cost_name, fps=15)

if __name__ == "__main__":
    cost = cost_robust2
    cost_name = "cost_robust2"
    output_root = Path("Robust_test")
    output_root.mkdir(parents=True, exist_ok=True)
    save_contour_plot(output_root, cost, cost_name=cost_name)

    # 每分量各放一个方位
    mu0_list = [
        [-6.0,  4.0],   # comp0: 上左, 近 f1(0.1,1.2)
        [ 6.0,  4.0],   # comp1: 上右
        [-6.0, -4.0],   # comp2: 下左, 近 f2(0.7,-1.1)
        [ 6.0, -4.0],   # comp3: 下右
        [ 5.0,  5.0],   # comp4: 原点, 近 f3(0.4,0)
    ]
    frame_data, all_f_vals = generate_animation_data(
        total_iterations=600, frame_stride=5, seed=42,
        m=1, dims=(2,), k=5, dt=0.15, b=300, b0=120, t0=600,
        cost=cost, sample_count=100, mu0_list=mu0_list, l0=4.0,
    )
    render_iteration_animation(output_root=output_root, frame_data=frame_data,
                               all_f_vals=all_f_vals, cost=cost, cost_name=cost_name, fps=15)