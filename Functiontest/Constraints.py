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
from jax import random, vmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from utils import _safe_color_limits, _blend_with_previous_if_nonfinite, _ellipse_within_limit, add_cov_ellipse
from utilss import _safe_eigh_numpy, build_joint_component_candidates, get_best_component_stats, sample_joint_from_solver

@jax.jit
def saturate(x: jnp.ndarray, k: float = 1.0):
    x = x / k
    return k * x / jnp.sqrt(1.0 + x ** 2)


@jax.jit
def cost_cos_cos(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = - jnp.cos(x1 * 3.14/4) * jnp.cos(x2 * 3.14/4)
    return res

@jax.jit
def cost_quadratic_half_plane(x, ctx):
    x1 = x[0]
    x2 = x[1]
    # res = 2.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2
    # res /= 10.0
    res = jnp.where(x1 - x2 < 2, 
                    -5.0 * (x1 - x2 - 2) + 20.0, 
                    saturate(2.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2))
    return res

@jax.jit
def cost_quadratic1(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = 2.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2
    res /= 10.0
    return res

@jax.jit
def cost_quadratic2(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = 20.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2
    res /= 10.0
    return res

@jax.jit
def cost_quadratic_linear_constraint(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = jnp.where(jnp.abs(x1 - x2 - 2) > 0.3, 
                    5.0 * jnp.abs(x1 - x2 - 2) + 20.0,
                    (2.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2) / 10.0)
    return res

@jax.jit
def cost_circle_constraint(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = jnp.where(jnp.abs(x1**2 + (x2 + 1)**2 - 16.0) > 3, 
                    5.0 * (jnp.abs(x1**2 + (x2 + 1)**2 - 16)) + 10.0, 
                    (2.0 * (x1 + x2 - 2) ** 2 + (x1 - x2) ** 2) / 10.0)
    return res

# 饱和嵌套的方法论：假设原始目标函数为f(z). 罚函数分别是g1(z) 到g_M(z). 
# 那么饱和嵌套：\sigma_1( \sigma_2( ... \sigma_M(\sigma_f(f(z)) + g_M(z)) + g_{M-1}(z)) + ... ) + g_1(z))
# 我们可以通过调整不同的饱和嵌套顺序来模拟到底哪个约束先满足。
@jax.jit
def cost_sat_hierarchical(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = saturate(jnp.where(x2 + x1 < -4, - (x2 + x1 + 4) + 1.5,
                    saturate(jnp.where(x2 - x1 > 0, 
                    1.5 + (x2 - x1), saturate(((x1-6)**2 + (x2+4) ** 2) / 50.0)), k=1.0)))
    return res
@jax.jit #交换嵌套饱和顺序，模拟另一个约束先满足的情况
def cost_sat_hierarchical2(x, ctx):
    x1 = x[0]
    x2 = x[1]
    res = saturate( jnp.where(x2 - x1 > 0, 
                    1.5 + (x2 - x1), saturate(jnp.where(x2 + x1 < -4, - (x2 + x1 + 4) + 1.5,
                    saturate(((x1-6)**2 + (x2+4) ** 2) / 50.0, k=1.0)), k=1.0)))
    return res

@jax.jit 
def cost_sat_hierarchical3(x, ctx):
    x1 = x[0]
    x2 = x[1]
    # 最外层 g3: x1 <= -2.5  (必须满足)
    # 中层   g2: x2 - x1 <= 0
    # 内层   g1: x1 + x2 >= -4
    # 最内层 f: minimize distance to (0,0)
    res = saturate(
        jnp.where(x1 > -2.5, (x1 + 2.5) + 1.5,   # g3 penalty
            saturate(
                jnp.where(x2 - x1 > 0, 1.5 + (x2 - x1),  # g2
                    saturate(
                        jnp.where(x2 + x1 < -4, -(x2 + x1 + 4) + 1.5,  # g1
                            saturate(((x1-0)**2 + (x2+0) ** 2) / 50.0)  # f
                        )
                    )
                )
            )
        )
    )
    return res

@jax.jit # 机会约束: min x^2+y^2 s.t. P(x+y+ξ<=0)>=0.9, ξ~N(0,1). 通用MC: 分位数法
def cost_sat_hierarchical4(x, ctx):
    x1 = x[0]
    x2 = x[1]
    # P(x+ξ<=0)>=0.9  ⇔  Q_{0.9}(x+ξ) <= 0  (90%分位≤0则至少90%满足)
    M = 30
    key = random.PRNGKey(0)
    xi = random.normal(key, shape=(M,))
    q = jnp.quantile(x1 + x2 + xi, 0.95)  # 0.9-分位数: q<=0 则约束满足
    res = saturate(
        jnp.where(q > 0,
            q + 1.5,
            saturate(x1**2 + x2**2)
        )
    )
    return res

@jax.jit # 机会约束+离散分布: min x^2+y^2 s.t. P(x+y+ξ<=0)>=0.9, ξ~Categorical
def cost_sat_hierarchical5(x, ctx):
    x1 = x[0]
    x2 = x[1]
    # ξ ~ Categorical({-3,-1,0,1,3}, probs={0.1,0.2,0.4,0.2,0.1})
    M = 100
    key = random.PRNGKey(0)
    xi_vals = jnp.array([-3., -1., 0., 1., 3.])
    xi_probs = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
    xi = random.choice(key, xi_vals, shape=(M,), p=xi_probs)
    # P(ξ<=1)=0.9, 所以 Q_{0.9}(ξ)≈1, 约束 Q_{0.95}(x+y+ξ)<=0  ⇔ P(x+y+ξ<=0)>=0.95
    q = jnp.quantile(x1 + x2 + xi, 0.95)
    res = saturate(
        jnp.where(q > 0, q + 1.5, saturate(x1**2 + x2**2))
    )
    return res

@jax.jit # 乘法噪声: min x^2+y^2 s.t. P(ξ1*x^2+ξ2*y^2<=10)>=0.9, ξ1,ξ2 独立离散
def cost_sat_hierarchical6(x, ctx):
    x1 = x[0]
    x2 = x[1]
    # 非凸环形可行域: P(1 <= (x+ξ)^2 <= 9) >= 0.9
    # ξ ~ 双峰混合正态: 0.5*N(-1,0.3)+0.5*N(+1,0.3) → 约束不可行，求解器自动 min violation
    M = 100
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    mix = random.bernoulli(k1, 0.5, (M,))
    xi1 = jnp.where(mix,
        random.normal(k2, (M,))*0.3 - 1.0,
        random.normal(random.fold_in(k2,1), (M,))*0.3 + 1.0)
    k3, k4 = random.split(random.fold_in(k2,2))
    mix2 = random.bernoulli(k3, 0.5, (M,))
    xi2 = jnp.where(mix2,
        random.normal(k4, (M,))*0.3 - 1.0,
        random.normal(random.fold_in(k4,1), (M,))*0.3 + 1.0)
    d2 = (x1+xi1)**2 + (x2+xi2)**2
    q_lo = jnp.quantile(d2, 0.05)   # 下界: Q_{0.05}>=1
    q_hi = jnp.quantile(d2, 0.95)   # 上界: Q_{0.95}<=9
    violation = jnp.maximum(0.0, jnp.maximum(1.0 - q_lo, q_hi - 9.0))
    res = saturate(
        jnp.where(violation > 0, violation + 1.5,
            saturate(((x1-2)**2 + (x2-2)**2)/5.0))
    )
    return res

@jax.jit # 嵌套冲突机会约束: 外层=待在圆内(r<=2), 内层=待在圆外(r>=1), 目标(3,3)
def cost_sat_hierarchical7(x, ctx):
    x1 = x[0]; x2 = x[1]
    M = 100; key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    mix = random.bernoulli(k1, 0.5, (M,))
    xi1 = jnp.where(mix, random.normal(k2,(M,))*0.3-1.0, random.normal(random.fold_in(k2,1),(M,))*0.3+1.0)
    k3,k4 = random.split(random.fold_in(k2,2))
    mix2 = random.bernoulli(k3,0.5,(M,))
    xi2 = jnp.where(mix2, random.normal(k4,(M,))*0.3-1.0, random.normal(random.fold_in(k4,1),(M,))*0.3+1.0)
    d = jnp.sqrt((x1+xi1)**2 + (x2+xi2)**2)
    r_out = jnp.quantile(d, 0.95)  # 外层: r<=2
    r_in  = jnp.quantile(d, 0.05)  # 内层: r>=1
    viol_out = jnp.maximum(0.0, r_out - 2.0)
    viol_in  = jnp.maximum(0.0, 1.0 - r_in)
    res = saturate(
        jnp.where(viol_out > 0, viol_out + 1.5,
            saturate(jnp.where(viol_in > 0, viol_in + 1.5,
                saturate(((x1-3)**2 + (x2-3)**2)/5.0))))
    )
    return res

@jax.jit # 嵌套冲突机会约束: 外层=待在圆外(r>=1), 内层=待在圆内(r<=2)  (交换优先级)
def cost_sat_hierarchical8(x, ctx):
    x1 = x[0]; x2 = x[1]
    M = 100; key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    mix = random.bernoulli(k1, 0.5, (M,))
    xi1 = jnp.where(mix, random.normal(k2,(M,))*0.3-1.0, random.normal(random.fold_in(k2,1),(M,))*0.3+1.0)
    k3,k4 = random.split(random.fold_in(k2,2))
    mix2 = random.bernoulli(k3,0.5,(M,))
    xi2 = jnp.where(mix2, random.normal(k4,(M,))*0.3-1.0, random.normal(random.fold_in(k4,1),(M,))*0.3+1.0)
    d = jnp.sqrt((x1+xi1)**2 + (x2+xi2)**2)
    r_out = jnp.quantile(d, 0.95)
    r_in  = jnp.quantile(d, 0.05)
    viol_in  = jnp.maximum(0.0, 1.0 - r_in)
    viol_out = jnp.maximum(0.0, r_out - 2.0)
    res = saturate(
        jnp.where(viol_in > 0, viol_in + 1.5,
            saturate(jnp.where(viol_out > 0, viol_out + 1.5,
                saturate(((x1-3)**2 + (x2-3)**2)/5.0))))
    )
    return res

@jax.jit # consistent planning: 安全优先(不掉下悬崖), ξ重尾90%N(0,0.5)+10%N(-6,1)
def cost_sat_hierarchical9(x, ctx):
    x1 = x[0]; x2 = x[1]
    M = 200; key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    tail = random.bernoulli(k1, 0.1, (M,))
    xi = jnp.where(tail, random.normal(k2,(M,))*1.0-6.0,
                         random.normal(random.fold_in(k2,1),(M,))*0.5)
    # 安全(外层): P(x2+ξ>=0)>=0.95  ⇔ Q_{0.05}(x2+ξ)>=0
    q_safe = jnp.quantile(x2 + xi, 0.01)
    viol_safe = jnp.maximum(0.0, -q_safe)
    # 舒适(内层): P(||x-(5,0)||+ξ<=3)>=0.9  ⇔ Q_{0.9}(||x-(5,0)||+ξ)<=3
    dist = jnp.sqrt((x1-5)**2 + x2**2)
    q_comfort = jnp.quantile(dist + xi, 0.90)
    viol_comfort = jnp.maximum(0.0, q_comfort - 3.0)
    res = saturate(
        jnp.where(viol_safe > 0, viol_safe + 1.5,
            saturate(jnp.where(viol_comfort > 0, viol_comfort + 1.5,
                saturate(((x1-5)**2 + x2**2)/10.0))))
    )
    return res

@jax.jit # consistent planning: 舒适优先, 安全在后 (交换)
def cost_sat_hierarchical10(x, ctx):
    x1 = x[0]; x2 = x[1]
    M = 200; key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    tail = random.bernoulli(k1, 0.1, (M,))
    xi = jnp.where(tail, random.normal(k2,(M,))*1.0-6.0,
                         random.normal(random.fold_in(k2,1),(M,))*0.5)
    dist = jnp.sqrt((x1-5)**2 + x2**2)
    q_comfort = jnp.quantile(dist + xi, 0.90)
    viol_comfort = jnp.maximum(0.0, q_comfort - 3.0)
    q_safe = jnp.quantile(x2 + xi, 0.01)
    viol_safe = jnp.maximum(0.0, -q_safe)
    res = saturate(
        jnp.where(viol_comfort > 0, viol_comfort + 1.5,
            saturate(jnp.where(viol_safe > 0, viol_safe + 1.5,
                saturate(((x1-5)**2 + x2**2)/10.0))))
    )
    return res

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
    cost=cost_cos_cos,
    sample_count=100,
    **kwargs,
):
    d_max = max(dims)
    if not (m == 1 and d_max == 2):
        raise ValueError("Animation currently supports single-block 2D only (m=1, dims=(2,)).")

    key = random.PRNGKey(seed)
    key, key_init = random.split(key)
    mu0 = kwargs.get("mu0", None)
    if mu0 is not None:
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

TEST = [
    {"name": "cos_cos_test", "cost": cost_cos_cos},
    {"name": "quadratic_half_plane_test", "cost": cost_quadratic_half_plane},
    {"name": "quadratic_linear_test", "cost": cost_quadratic_linear_constraint},
    {"name": "quadratic_test1", "cost": cost_quadratic1},
    {"name": "quadratic_test2", "cost": cost_quadratic2},
    {"name": "quadratic_circle_test", "cost": cost_circle_constraint},
    {"name": "sat_hierarchical_test", "cost": cost_sat_hierarchical},
    {"name": "sat_hierarchical2_test", "cost": cost_sat_hierarchical2},
    {"name": "sat_hierarchical3_test", "cost": cost_sat_hierarchical3},
    {"name": "sat_hierarchical4_test", "cost": cost_sat_hierarchical4},
    {"name": "sat_hierarchical5_test", "cost": cost_sat_hierarchical5},
    {"name": "sat_hierarchical6_test", "cost": cost_sat_hierarchical6},
    {"name": "sat_hierarchical7_test", "cost": cost_sat_hierarchical7},
    {"name": "sat_hierarchical8_test", "cost": cost_sat_hierarchical8},
    {"name": "sat_hierarchical9_test", "cost": cost_sat_hierarchical9},
    {"name": "sat_hierarchical10_test", "cost": cost_sat_hierarchical10},
]

def save_all_contours(output_root):
    for test in TEST:
        save_contour_plot(output_root, test["cost"], cost_name=test["name"])

def save_all_surface_plots(output_root: Path):
    for test in TEST:
        plot_3D_surface(test["cost"], x1_lim=(-8.0, 8.0), x2_lim=(-8.0, 8.0), resolution=100)
        surface_path = output_root / (test['name'] + "_surface.png")
        plt.savefig(surface_path)
        plt.close()
        print(f"Saved surface plot: {surface_path}")

def sat_test():
    
    test_idx = 14
    output_root = Path("output_igo_test")
    output_root.mkdir(parents=True, exist_ok=True)
    save_contour_plot(output_root, TEST[test_idx]["cost"], cost_name=TEST[test_idx]["name"])
    # breakpoint()
    k = 5
    dt = 0.15
    b = 300
    b0 = 120
    t0 = 600
    iter_tot = 600
    frame_data, all_f_vals = generate_animation_data(
        total_iterations=iter_tot, frame_stride=5, seed=42,
        m=1, dims=(2,), k=k, dt=dt, b=b, b0=b0, t0=t0, cost=TEST[test_idx]["cost"], sample_count=100,
        mu0=(-2, -4), l0=2.0,
    )
    render_iteration_animation(output_root=output_root, frame_data=frame_data, all_f_vals=all_f_vals, cost=TEST[test_idx]["cost"], cost_name=TEST[test_idx]["name"], fps=15)

def test_igo(test_idx=0):
    output_root = Path("output_igo_test")
    output_root.mkdir(parents=True, exist_ok=True)
    k = 5
    dt = 0.15
    b = 300
    b0 = 120
    t0 = 600
    iter_tot = 600
    frame_data, all_f_vals = generate_animation_data(
        total_iterations=iter_tot, frame_stride=5, seed=42,
        m=1, dims=(2,), k=k, dt=dt, b=b, b0=b0, t0=t0, cost=TEST[test_idx]["cost"], sample_count=100,
    )
    render_iteration_animation(output_root=output_root, frame_data=frame_data, all_f_vals=all_f_vals, cost=TEST[test_idx]["cost"], cost_name=TEST[test_idx]["name"], fps=15)

def main_igo_test_all():
    for idx in range(len(TEST)):
        test_igo(test_idx=idx)

if __name__ == "__main__":
    sat_test()