"""最小化的 B-spline warm start demo.

这个版本只做一件事：用 `bsplinetraj.py` 里的离线 basis 矩阵，
把一条单车 B-spline 轨迹丢给 `gmm_igo.MPCsolverM22.mmog_igo_optimizer_mpc`
做 warm start 追踪。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from jax import jit, random

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bsplinetraj import BSplineTrajectoryGenerator
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc


DT = 0.1
M = 1
K = 3
B = 64
B0 = 20
T0 = 250
IGO_STEPS = 500

V_TARGET = 18.0
W_POS = 25.0
W_VEL = 8.0
W_ACC = 0.2
W_STEER = 0.3
W_TERMINAL = 60.0

ROAD_Y = 0.0


def _build_nominal_control_points(robot_x: float, robot_y: float, robot_v: float, n_ctrl: int, total_time: float) -> jnp.ndarray:
	"""生成一个直线参考轨迹的控制点初值。"""
	x_ctrl = jnp.linspace(robot_x, robot_x + V_TARGET * total_time, n_ctrl)
	y_ctrl = jnp.full((n_ctrl,), robot_y if abs(robot_y) < 1e-6 else ROAD_Y)

	# 给首段速度一个更自然的坡度，避免完全平直导致 warm start 太弱。
	x_ctrl = x_ctrl.at[1].set(robot_x + robot_v * total_time / max(n_ctrl - 1, 1))
	return jnp.stack([x_ctrl, y_ctrl], axis=1)


def _make_initial_params(key: jax.Array, base_theta: jnp.ndarray, D: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""构造 solver 所需的 warm start 初值。"""
	mu_init = jnp.stack(
		[
			base_theta,
			base_theta,
			base_theta
		],
		axis=0,
	)[None, ...]

	L_inv_init = jnp.eye(D, dtype=jnp.float32) * 1.0
	L_inv_init = jnp.tile(L_inv_init[None, None, :, :], (M, K, 1, 1))
	v_init = jnp.zeros((M, K - 1), dtype=jnp.float32)
	return mu_init.astype(jnp.float32), L_inv_init, v_init


def _make_fitness_fn(generator: BSplineTrajectoryGenerator):
	B_mat = generator.B
	dB_mat = generator.dB
	d2B_mat = generator.d2B

	@jit
	def fitness_fn_total(samples_overall, context):
		ctrl_pts_free = samples_overall.reshape((generator.n_ctrl - 2, 2))
		pos, vel, acc = generator.evaluate(ctrl_pts_free, context["robot_x0"], context["robot_v0"])

		speed = jnp.linalg.norm(vel, axis=-1)
		speed_safe = speed + 1e-6
		speed_dir = vel / speed_safe[:, None]
		a_long = jnp.sum(acc * speed_dir, axis=-1)
		curvature = (vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]) / (speed**3 + 1e-6)
		steer = jnp.arctan(curvature * context["wheel_base"])

		cost_track = W_POS * jnp.sum((pos[:, 0] - context["x_ref"]) ** 2 + (pos[:, 1] - context["y_ref"]) ** 2)
		cost_speed = W_VEL * jnp.sum((speed - context["v_ref"]) ** 2)
		cost_smooth = W_ACC * jnp.sum(a_long**2) + W_STEER * jnp.sum(steer**2)
		cost_terminal = W_TERMINAL * (
			(pos[-1, 0] - context["x_ref"][-1]) ** 2
			+ (pos[-1, 1] - context["y_ref"][-1]) ** 2
			+ 0.3 * (speed[-1] - context["v_ref"][-1]) ** 2
		)

		return cost_track + cost_speed + cost_smooth + cost_terminal

	return fitness_fn_total

def _shift_control_points(ctrl_pts, v_target, dt_step):
	"""将控制点沿 X 方向平移，并补上一个前视点。"""
	shifted = jnp.zeros_like(ctrl_pts)
	shifted = shifted.at[:-1].set(ctrl_pts[1:])
	last_pt = ctrl_pts[-1]
	new_pt = last_pt + jnp.array([v_target * dt_step, 0.0])
	return shifted.at[-1].set(new_pt)

def run_simple_demo(steps: int = 100, seed: int = 0, plot: bool = True):
	basis_path = Path(__file__).with_name("bspline_basis.npz")
	generator = BSplineTrajectoryGenerator(basis_path)
	fitness_fn_total = _make_fitness_fn(generator)
	output_dir = Path(__file__).resolve().parent
	gif_path = output_dir / "simple_bspline_demo.gif"
	data_path = output_dir / "simple_bspline_demo_data.npz"

	key = random.PRNGKey(seed)
	robot_x, robot_y, robot_v = 0.0, -3.0, 12.0
	robot_vx, robot_vy = robot_v, 0.0

	base_ctrl_pts = _build_nominal_control_points(robot_x, robot_y, robot_v, generator.n_ctrl, generator.total_time)
	base_theta = base_ctrl_pts[2:].reshape(-1)
	mu_init, L_inv_init, v_init = _make_initial_params(key, base_theta, base_theta.shape[0])

	if plot:
		plt.ion()
		fig, ax = plt.subplots(figsize=(12, 5))

	history_x = [robot_x]
	history_y = [robot_y]
	frame_data = []

	for step in range(steps):
		key, subkey = random.split(key)

		# 1. 动态上下文更新
		x_ref = jnp.linspace(robot_x, robot_x + V_TARGET * generator.total_time, generator.T)
		y_ref = jnp.full((generator.T,), ROAD_Y)
		v_ref = jnp.full((generator.T,), V_TARGET)

		context = {
			"x_ref": x_ref,
			"y_ref": y_ref,
			"v_ref": v_ref,
			"robot_x0": jnp.array([robot_x, robot_y], dtype=jnp.float32),
			"robot_v0": jnp.array([robot_vx, robot_vy], dtype=jnp.float32),
			"wheel_base": 2.8,
		}

		# 2. 求解器运算
		t0 = time.time()
		mu_k, L_k, pi_k = mmog_igo_optimizer_mpc(
			subkey, IGO_STEPS, 0.15, M, K, B, B0,
			(base_theta.shape[0],), T0,
			fitness_fn_total, mu_init, L_inv_init, v_init, context,
		)
		mu_k.block_until_ready()
		solve_ms = (time.time() - t0) * 1000.0

		# 3. 提取最优轨迹并执行
		best_idx = int(jnp.argmax(pi_k[0]))
		best_theta = mu_k[0, best_idx]
		ctrl_pts_free = best_theta.reshape(generator.n_ctrl - 2, 2)
		pos, vel, acc = generator.evaluate(ctrl_pts_free, jnp.array([robot_x, robot_y], dtype=jnp.float32), jnp.array([robot_vx, robot_vy], dtype=jnp.float32))
		states = generator.to_vehicle_states(pos, vel, acc)

		# 状态向前滚动（执行一步，即仿真时钟前进了一步）
		robot_x = float(pos[1, 0])
		robot_y = float(pos[1, 1])
		robot_v = float(jnp.clip(states[1, 2], 5.0, 35.0))
		robot_vx = float(vel[1, 0])
		robot_vy = float(vel[1, 1])

		history_x.append(robot_x)
		history_y.append(robot_y)

		frame_data.append(
			{
				"history_x": np.asarray(history_x, dtype=np.float32),
				"history_y": np.asarray(history_y, dtype=np.float32),
				"planned_x": np.asarray(pos[:, 0], dtype=np.float32),
				"planned_y": np.asarray(pos[:, 1], dtype=np.float32),
				"ref_x": np.asarray(x_ref, dtype=np.float32),
				"ref_y": np.asarray(y_ref, dtype=np.float32),
				"robot_v": np.float32(robot_v),
				"solve_ms": np.float32(solve_ms),
			}
		)

		dt_step = generator.dt
		shifted_base_pts = _shift_control_points(ctrl_pts_free, V_TARGET, dt_step)
		
		# 将平移后的、物理意义正确的控制点重新展平，作为下一帧优化的均值中心
		shifted_base = shifted_base_pts.reshape(-1)
		mu_init, L_inv_init, v_init = _make_initial_params(key, shifted_base, shifted_base.shape[0])


	if plot:
		data = {
			"history_x": np.asarray(history_x, dtype=np.float32),
			"history_y": np.asarray(history_y, dtype=np.float32),
			"frames": frame_data,
		}
		np.savez(data_path, history_x=data["history_x"], history_y=data["history_y"], frames=np.array(frame_data, dtype=object))

		def _render_frame(frame_idx):
			frame = frame_data[frame_idx]
			ax.cla()
			ax.plot(frame["history_x"], frame["history_y"], "g-", lw=2, label="executed")
			ax.plot(frame["planned_x"], frame["planned_y"], "b--", lw=2, label="planned")
			ax.plot(frame["ref_x"], frame["ref_y"], "k:", lw=2, label="reference")
			ax.set_title(
				f"Simple B-spline MPC | step={frame_idx} | v={float(frame['robot_v']):.1f} m/s | solve={float(frame['solve_ms']):.0f} ms"
			)
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			ax.set_aspect("equal", adjustable="box")
			ax.grid(True, alpha=0.25)
			ax.legend(loc="upper left")
			ax.set_xlim(min(frame["history_x"].min(), frame["ref_x"].min()) - 10.0, frame["planned_x"].max() + 10.0)
			ax.set_ylim(-5.0, 5.0)
			return ax.lines

		anim = FuncAnimation(fig, _render_frame, frames=len(frame_data), interval=80, blit=False)
		anim.save(gif_path, writer=PillowWriter(fps=12))
		plt.close(fig)
		print(f"saved data to {data_path}")
		print(f"saved gif to {gif_path}")


def _parse_args():
	parser = argparse.ArgumentParser(description="Simple B-spline MPC warm start demo")
	parser.add_argument("--steps", type=int, default=100)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--no-plot", action="store_true")
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	run_simple_demo(steps=args.steps, seed=args.seed, plot=not args.no_plot)


