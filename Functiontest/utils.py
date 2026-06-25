import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from utilss import _safe_eigh_numpy, ELLIPSE_ABS_LIMIT, build_joint_component_candidates, get_best_component_stats, sample_joint_from_solver
from jax import vmap
import jax.numpy as jnp

def _safe_color_limits(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax

def _ellipse_within_limit(mean_xy, eigvals, abs_limit=ELLIPSE_ABS_LIMIT):
    mean_arr = np.asarray(mean_xy, dtype=np.float64)
    eig_arr = np.asarray(eigvals, dtype=np.float64)
    if not np.all(np.isfinite(mean_arr)) or not np.all(np.isfinite(eig_arr)):
        return False
    return (np.max(np.abs(mean_arr)) <= abs_limit) and (np.max(np.abs(eig_arr)) <= abs_limit)

def _blend_with_previous_if_nonfinite(curr, prev, clip_abs=None):
    curr_arr = np.asarray(curr)
    prev_arr = np.asarray(prev)
    finite_mask = np.isfinite(curr_arr)
    blended = np.where(finite_mask, curr_arr, prev_arr)
    if clip_abs is not None:
        blended = np.clip(blended, -clip_abs, clip_abs)
    return blended

def add_cov_ellipse(
    ax, mean, cov, n_std=2.0, edgecolor="tab:orange", linewidth=2.0, alpha=1.0,
    label=None, linestyle="-", zorder=2,
):
    eigvals, eigvecs = _safe_eigh_numpy(cov)
    if not _ellipse_within_limit(mean, eigvals):
        return None

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width = 2.0 * n_std * np.sqrt(max(eigvals[0], 1e-12))
    height = 2.0 * n_std * np.sqrt(max(eigvals[1], 1e-12))
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle, edgecolor=edgecolor,
        facecolor="none", linewidth=linewidth, alpha=alpha, label=label,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(ellipse)
    return ellipse

def plot_result(iter_count, mu, l_inv, pi, output_dir, plot_key, fitness_fn):
    x1 = np.linspace(-8.0, 8.0, 300)
    x2 = np.linspace(-8.0, 8.0, 300)
    xx, yy = np.meshgrid(x1, x2)
    grid_points = jnp.stack(
        [jnp.asarray(xx).reshape(-1), jnp.asarray(yy).reshape(-1)], axis=1
    )
    zz = np.asarray(vmap(lambda p: fitness_fn(p, None))(grid_points)).reshape(xx.shape)

    sample_count = 201
    samples = sample_joint_from_solver(plot_key, mu, l_inv, pi, sample_count)
    f_vals = vmap(lambda s: fitness_fn(s, None))(samples)

    samples_np = np.asarray(samples)
    f_vals_np = np.asarray(f_vals)
    finite_mask = np.isfinite(samples_np).all(axis=1) & np.isfinite(f_vals_np)
    if np.any(finite_mask):
        samples_np = samples_np[finite_mask]
        f_vals_np = f_vals_np[finite_mask]
    else:
        samples_np = np.zeros((1, 2))
        f_vals_np = np.zeros((1,))

    c_vmin, c_vmax = _safe_color_limits(f_vals_np)

    best_comp, best_mu, best_cov, _ = get_best_component_stats(mu, l_inv, fitness_fn)
    all_means, all_covs = build_joint_component_candidates(mu, l_inv)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    contour = ax.contour(xx, yy, zz, levels=30, cmap="Blues")
    ax.clabel(contour, inline=True, fontsize=8)

    sc = ax.scatter(
        samples_np[:, 0], samples_np[:, 1], c=f_vals_np, cmap="viridis_r",
        vmin=c_vmin, vmax=c_vmax, s=20, alpha=0.65, linewidths=0.0, zorder=1
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fitness")

    k_comp = all_means.shape[0]
    for k_idx in range(k_comp):
        comp_mean = np.asarray(all_means[k_idx])
        comp_cov = np.asarray(all_covs[k_idx])

        add_cov_ellipse(
            ax, comp_mean, comp_cov, n_std=1.0, edgecolor="tab:orange",
            linewidth=1.2, linestyle="-", alpha=0.8,
        )
        ax.scatter(
            comp_mean[0], comp_mean[1], c="tab:orange", s=36, marker="D",
            edgecolors="black", linewidths=0.5, zorder=4
        )

    ax.scatter(
        best_mu[0], best_mu[1], c="gold", s=180, marker="*",
        edgecolors="black", linewidths=0.9, zorder=6
    )
    add_cov_ellipse(
        ax, best_mu, best_cov, n_std=1.0, edgecolor="black",
        linewidth=1.8, linestyle="--", alpha=0.8, zorder=5
    )

    ax.set_title(f"IGO solver test (T={iter_count})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, alpha=0.8, label="all samples"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="tab:orange", markeredgecolor="black", markersize=7, label="component means"),
        Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=1.2, label="1σ ellipses"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="black", markersize=11, label="best mean"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="best 1σ"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    output_path = output_dir / f"igo_iter_{iter_count:03d}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def plot_best_solution_progress(iter_list, all_mu_hist, all_sigma_hist, output_dir):
    x_axis = np.asarray(iter_list)
    mu_hist = np.asarray(all_mu_hist)
    sigma_hist = np.asarray(all_sigma_hist)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)
    n_comp = mu_hist.shape[1]
    cmap = plt.get_cmap("tab20")

    for comp_idx in range(n_comp):
        color = cmap(comp_idx % 20)
        axes[0].plot(
            x_axis, mu_hist[:, comp_idx, 0], color=color, linewidth=1.6,
            marker="o", markersize=3.5, label=f"comp {comp_idx}"
        )
        axes[0].fill_between(
            x_axis, mu_hist[:, comp_idx, 0] - sigma_hist[:, comp_idx, 0],
            mu_hist[:, comp_idx, 0] + sigma_hist[:, comp_idx, 0],
            color=color, alpha=0.12
        )
        axes[1].plot(
            x_axis, mu_hist[:, comp_idx, 1], color=color, linewidth=1.6,
            marker="s", markersize=3.5, label=f"comp {comp_idx}"
        )
        axes[1].fill_between(
            x_axis, mu_hist[:, comp_idx, 1] - sigma_hist[:, comp_idx, 1],
            mu_hist[:, comp_idx, 1] + sigma_hist[:, comp_idx, 1],
            color=color, alpha=0.12
        )

    axes[0].set_title("All components: mu and sigma band")
    axes[0].set_ylabel("x1")
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("x2")
    axes[1].set_xlabel("iteration count")
    axes[1].grid(alpha=0.25)

    if n_comp <= 10:
        axes[0].legend(loc="best", fontsize=8, ncol=2)
    else:
        axes[0].legend(loc="best", fontsize=7, ncol=3)

    output_path = output_dir / "igo_best_solution_progress.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def plot_compare_best_progress(iter_list, best_mu_hist_a, best_mu_hist_b, output_root):
    x_axis = np.asarray(iter_list)
    mu_a = np.asarray(best_mu_hist_a)
    mu_b = np.asarray(best_mu_hist_b)

    fig, ax = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    ax[0].plot(x_axis, mu_a[:, 0], color="tab:blue", linewidth=2.0, marker="o", label="single block (2D)")
    ax[0].plot(x_axis, mu_b[:, 0], color="tab:green", linewidth=2.0, marker="s", label="two blocks (1D+1D)")
    ax[0].set_ylabel("best x1")
    ax[0].grid(alpha=0.25)
    ax[0].legend(loc="best", fontsize=9)

    ax[1].plot(x_axis, mu_a[:, 1], color="tab:red", linewidth=2.0, marker="o", label="single block (2D)")
    ax[1].plot(x_axis, mu_b[:, 1], color="tab:purple", linewidth=2.0, marker="s", label="two blocks (1D+1D)")
    ax[1].set_ylabel("best x2")
    ax[1].set_xlabel("iteration count")
    ax[1].grid(alpha=0.25)
    ax[1].legend(loc="best", fontsize=9)

    fig.suptitle("Best solution comparison: single-block vs two-block")
    fig.tight_layout()
    out_path = output_root / "igo_best_solution_compare.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)