"""Trajectory optimization demo for `bsplinetraj.py` using `MPCsolverM22.py`.

The optimizer only handles free variables. `bsplinetraj.py` reconstructs the
fixed prefix internally and returns the full trajectory for evaluation.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jax import random

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bsplinetraj import BSplineTrajectoryGenerator
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc


DT = 0.15
T_RUN = 500
M = 3
K = 3
B = 60
B0 = 25
T0 = 100

V_TARGET = 18.0
LANE_SHIFT = 3.5
ACC_LIMIT = 3.0

W_GOAL = 100.0
W_TRACK = 25.0
W_SPEED = 6.0
W_ACC = 5.0
W_JERK = 1.0
W_THETA_SMOOTH = 0.0
T_MAX = 20.0


def _load_augmented_basis(source_path: Path, dt: float, total_time: float) -> Path:
	"""Attach `dt` and `total_time` metadata to the basis file expected by the generator."""
	data = np.load(source_path)
	payload = {key: data[key] for key in data.files}
	payload["dt"] = np.array(dt, dtype=np.float64)
	payload["total_time"] = np.array(total_time, dtype=np.float64)
	temp_file = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
	temp_file.close()
	np.savez(temp_file.name, **payload)
	return Path(temp_file.name)


def _build_initial_state(x0: float, y0: float, vx0: float, vy0: float, ax0: float, ay0: float) -> jnp.ndarray:
	return jnp.asarray([x0, y0, vx0, vy0, ax0, ay0], dtype=jnp.float32)


def _split_decision_vector(samples: jnp.ndarray, n_free: int, n_theta: int, block_dim: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	x_block = samples[:n_free]
	y_block = samples[n_free : 2 * n_free]
	theta_block = samples[2 * block_dim : 3 * block_dim]
	return x_block, y_block, theta_block[:n_theta]


def create_fitness_fn(
	traj_gen: BSplineTrajectoryGenerator,
	x0_state: jnp.ndarray,
	total_time_max: float,
	goal_pos: jnp.ndarray,
	n_free: int,
	n_theta: int,
	block_dim: int,
):
	"""Create the optimizer objective used by MPCsolverM22.py."""

	def fitness_fn(samples, context):
		block_x, block_y, theta_latent = _split_decision_vector(samples, n_free, n_theta, block_dim)
		traj = traj_gen.evaluate.__wrapped__(traj_gen, block_x, block_y, theta_latent, x0_state, total_time_max)

		pos = jnp.stack(traj["pos"], axis=-1)
		vel = jnp.stack(traj["vel"], axis=-1)
		acc = jnp.stack(traj["acc"], axis=-1)
		jerk = jnp.stack(traj["jerk"], axis=-1)

		speed = jnp.linalg.norm(vel, axis=-1)
		acc_mag = jnp.linalg.norm(acc, axis=-1)
		jerk_mag = jnp.linalg.norm(jerk, axis=-1)

		pos_err = jnp.sum((pos[-1] - goal_pos) ** 2)
		track_cost = jnp.sum((pos[:, 0] - context["x_ref"]) ** 2 + (pos[:, 1] - context["y_ref"]) ** 2)
		speed_cost = jnp.sum((speed - context["v_ref"]) ** 2)
		acc_cost = jnp.sum(jnp.maximum(acc_mag - ACC_LIMIT, 0.0) ** 2)
		jerk_cost = jnp.sum(jerk_mag ** 2)
		theta_cost = jnp.sum(jnp.diff(theta_latent) ** 2)

		return (
			W_GOAL * pos_err
			+ W_TRACK * track_cost
			+ W_SPEED * speed_cost
			+ W_ACC * acc_cost
			+ W_JERK * jerk_cost
			+ W_THETA_SMOOTH * theta_cost
		)

	return fitness_fn


def _make_initial_params(
	base_x: jnp.ndarray,
	base_y: jnp.ndarray,
	base_theta: jnp.ndarray,
	m_blocks: int,
	k_components: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""Build initial mean, precision factor, and mixture logits for the solver."""
	d = base_x.shape[0]
	base_blocks = jnp.stack([base_x, base_y, base_theta], axis=0)
	mu_init = jnp.tile(base_blocks[:, None, :], (1, k_components, 1))
	L_inv_init = jnp.tile(jnp.eye(d, dtype=jnp.float32)[None, None, :, :], (m_blocks, k_components, 1, 1))
	v_init = jnp.zeros((m_blocks, k_components - 1), dtype=jnp.float32)
	return mu_init.astype(jnp.float32), L_inv_init.astype(jnp.float32), v_init


def _summarize_solution_details(best_sample: jnp.ndarray, total_time_max: float, n_free: int, n_theta: int, block_dim: int) -> dict:
	"""Decode the optimizer sample into the optimized free control points and time allocation."""
	block_x, block_y, theta_latent = _split_decision_vector(best_sample, n_free, n_theta, block_dim)
	theta_full = jnp.concatenate([jnp.array([0.0], dtype=theta_latent.dtype), theta_latent])
	weights = jax.nn.softmax(theta_full)
	dt_segments = weights * total_time_max

	return {
		"block_x": block_x,
		"block_y": block_y,
		"theta_latent": theta_latent,
		"theta_full": theta_full,
		"weights": weights,
		"dt_segments": dt_segments,
	}


def _evaluate_final_solution(
	generator: BSplineTrajectoryGenerator,
	best_sample: jnp.ndarray,
	x0_state: jnp.ndarray,
	total_time_max: float,
	n_free: int,
	n_theta: int,
	block_dim: int,
):
	block_x, block_y, theta_latent = _split_decision_vector(best_sample, n_free, n_theta, block_dim)
	return generator.evaluate.__wrapped__(generator, block_x, block_y, theta_latent, x0_state, total_time_max)


def _assemble_best_sample(mu_k: jnp.ndarray, pi_k: jnp.ndarray) -> tuple[jnp.ndarray, tuple[int, int, int]]:
	"""Pick the best component from each block and concatenate them into one flat decision vector."""
	best_indices = tuple(int(jnp.argmax(pi_k[m_idx])) for m_idx in range(mu_k.shape[0]))
	best_sample = jnp.concatenate([mu_k[m_idx, best_indices[m_idx]] for m_idx in range(mu_k.shape[0])], axis=0)
	return best_sample, best_indices


def _plot_result(output_path: Path, traj, goal_pos: jnp.ndarray, total_time_max: float) -> None:
	pos = np.stack([np.asarray(traj["pos"][0]), np.asarray(traj["pos"][1])], axis=-1)
	vel = np.stack([np.asarray(traj["vel"][0]), np.asarray(traj["vel"][1])], axis=-1)
	acc = np.stack([np.asarray(traj["acc"][0]), np.asarray(traj["acc"][1])], axis=-1)
	jerk = np.stack([np.asarray(traj["jerk"][0]), np.asarray(traj["jerk"][1])], axis=-1)
	traj_t = np.linspace(0.0, total_time_max, pos.shape[0])

	speed = np.linalg.norm(vel, axis=-1)
	acc_mag = np.linalg.norm(acc, axis=-1)
	jerk_mag = np.linalg.norm(jerk, axis=-1)

	fig, axes = plt.subplots(4, 1, figsize=(11, 16), constrained_layout=True)
	axes[0].plot(pos[:, 0], pos[:, 1], lw=2.0, color="#1f77b4", label="optimized")
	axes[0].scatter([float(goal_pos[0])], [float(goal_pos[1])], color="#d62728", s=40, label="goal")
	axes[0].set_title("Optimized trajectory")
	axes[0].set_xlabel("x")
	axes[0].set_ylabel("y")
	axes[0].axis("equal")
	axes[0].grid(True, alpha=0.25)
	axes[0].legend(loc="best")

	axes[1].plot(traj_t, speed, color="#2ca02c", lw=2.0)
	axes[1].set_title("Speed")
	axes[1].set_xlabel("time")
	axes[1].set_ylabel("m/s")
	axes[1].grid(True, alpha=0.25)

	axes[2].plot(traj_t, acc_mag, color="#ff7f0e", lw=2.0)
	axes[2].axhline(ACC_LIMIT, color="#d62728", ls="--", lw=1.2, label="acc limit")
	axes[2].set_title("Acceleration magnitude")
	axes[2].set_xlabel("time")
	axes[2].set_ylabel("m/s^2")
	axes[2].grid(True, alpha=0.25)
	axes[2].legend(loc="best")

	axes[3].plot(traj_t, jerk_mag, color="#9467bd", lw=2.0)
	axes[3].set_title("Jerk magnitude")
	axes[3].set_xlabel("time")
	axes[3].set_ylabel("m/s^3")
	axes[3].grid(True, alpha=0.25)

	fig.suptitle("Simple1 trajectory optimization", fontsize=14)
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def run_simple_demo(steps: int = 300, seed: int = 0, plot: bool = True):
	basis_path = Path(__file__).with_name("bspline_basis.npz")
	total_time_max = T_MAX
	augmented_basis_path = _load_augmented_basis(basis_path, dt=DT, total_time=total_time_max)

	try:
		generator = BSplineTrajectoryGenerator(augmented_basis_path)
		n_free = generator.N - 3
		n_theta = generator.N - 1
		block_dim = n_theta

		x0, y0 = 0.0, -3.0
		vx0, vy0, ax0, ay0 = 20.0, 0.0, 0.0, 0.0
		x0_state = _build_initial_state(x0, y0, vx0, vy0, ax0, ay0)
		goal_pos = jnp.asarray([V_TARGET * total_time_max, LANE_SHIFT], dtype=jnp.float32)

		base_x, base_y, base_theta = generator.splineguess(x0_state, total_time_max, goal_pos)
		initial_ctrl, initial_theta_latent, initial_theta_full, initial_weights, initial_dt_segments, initial_dt_du = generator.splineguess_full(
			x0_state,
			total_time_max,
			goal_pos,
			theta_latent=base_theta,
		)
		base_x_block = jnp.zeros((block_dim,), dtype=jnp.float32)
		base_y_block = jnp.zeros((block_dim,), dtype=jnp.float32)
		base_x_block = base_x_block.at[:n_free].set(base_x)
		base_y_block = base_y_block.at[:n_free].set(base_y)
		mu_init, L_inv_init, v_init = _make_initial_params(base_x_block, base_y_block, base_theta, M, K)

		print("Initial state [x, y, vx, vy, ax, ay]:")
		print(np.array2string(np.asarray(x0_state), precision=4, separator=", "))
		print("Initial full control points [P0..PN] (x, y):")
		print(np.array2string(np.asarray(initial_ctrl), precision=4, separator=", "))
		print("Initial theta latent:")
		print(np.array2string(np.asarray(initial_theta_latent), precision=4, separator=", "))
		print("Initial theta full:")
		print(np.array2string(np.asarray(initial_theta_full), precision=4, separator=", "))
		print("Initial per-segment time allocation:")
		for seg_idx, (weight, dt_seg) in enumerate(zip(np.asarray(initial_weights), np.asarray(initial_dt_segments))):
			print(f"  segment {seg_idx:02d}: weight={float(weight):.6f}, dt={float(dt_seg):.6f}")

		x_ref = jnp.linspace(x0, float(goal_pos[0]), generator.T)
		y_ref = jnp.full((generator.T,), float(goal_pos[1]))
		v_ref = jnp.full((generator.T,), V_TARGET)
		context = {
			"x_ref": x_ref,
			"y_ref": y_ref,
			"v_ref": v_ref,
			"goal_pos": goal_pos,
			"v_target": V_TARGET,
			"acc_limit": ACC_LIMIT,
			"lane_shift": LANE_SHIFT,
		}

		fitness_fn = create_fitness_fn(generator, x0_state, total_time_max, goal_pos, n_free, n_theta, block_dim)

		key = random.PRNGKey(seed)

		start_time = time.perf_counter()
		mu_k, L_k, pi_k = mmog_igo_optimizer_mpc(
			key,
			steps,
			DT,
			M,
			K,
			B,
			B0,
			(n_free, n_free, n_theta),
			T0,
			fitness_fn,
			mu_init,
			L_inv_init,
			v_init,
			context,
		)
		mu_k.block_until_ready()
		solve_s = time.perf_counter() - start_time

		best_sample, best_indices = _assemble_best_sample(mu_k, pi_k)
		traj = _evaluate_final_solution(generator, best_sample, x0_state, total_time_max, n_free, n_theta, block_dim)
		solution = _summarize_solution_details(best_sample, total_time_max, n_free, n_theta, block_dim)
		final_cost = float(fitness_fn(best_sample, context))

		print(f"Optimization finished in {solve_s:.3f} s")
		print(f"Best component indices: {best_indices}")
		print(f"Final cost: {final_cost:.4f}")
		print(f"Final pos: ({float(traj['pos'][0][-1]):.3f}, {float(traj['pos'][1][-1]):.3f})")
		print(f"Final speed: {float(jnp.linalg.norm(jnp.stack(traj['vel'], axis=-1)[-1])):.3f} m/s")
		print("Optimized free control points block_x:")
		print(np.array2string(np.asarray(solution["block_x"]), precision=4, separator=", "))
		print("Optimized free control points block_y:")
		print(np.array2string(np.asarray(solution["block_y"]), precision=4, separator=", "))
		print("Optimized theta latent:")
		print(np.array2string(np.asarray(solution["theta_latent"]), precision=4, separator=", "))
		print("Per-segment time allocation:")
		for seg_idx, (weight, dt_seg) in enumerate(zip(np.asarray(solution["weights"]), np.asarray(solution["dt_segments"]))):
			print(f"  segment {seg_idx:02d}: weight={float(weight):.6f}, dt={float(dt_seg):.6f}")

		if plot:
			output_path = Path(__file__).with_name("simple1_trajectory.png")
			_plot_result(output_path, traj, goal_pos, total_time_max)
			print(f"saved plot to {output_path}")

		return {
			"mu": mu_k,
			"L": L_k,
			"pi": pi_k,
			"best_sample": best_sample,
			"trajectory": traj,
			"final_cost": final_cost,
		}
	finally:
		if augmented_basis_path.exists():
			augmented_basis_path.unlink()


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Trajectory optimization with MPCsolverM22")
	parser.add_argument("--steps", type=int, default=T_RUN)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--no-plot", action="store_true")
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	run_simple_demo(steps=args.steps, seed=args.seed, plot=not args.no_plot)