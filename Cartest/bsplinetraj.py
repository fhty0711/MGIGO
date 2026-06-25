import jax
import jax.numpy as jnp
import numpy as np

class BSplineTrajectoryGenerator:
    def __init__(self, basis_path):
        data = np.load(basis_path)
        self.B = jnp.array(data["B"])     # [T, N]
        self.dB = jnp.array(data["dB"])   # [T, N]
        self.d2B = jnp.array(data["d2B"]) # [T, N]
        self.d3B = jnp.array(data["d3B"]) # [T, N]
        self.N = self.B.shape[1]
        self.T = self.B.shape[0]

    def _time_weights(self, theta_latent):
        theta_full = jnp.concatenate([jnp.array([0.0], dtype=theta_latent.dtype), theta_latent])
        return theta_full, jax.nn.softmax(theta_full)

    def _build_fixed_prefix(self, x0_state, total_time_max, theta_latent, l=2.5, l_r=1.2):
        x0, y0, vx0, vy0, ax0, ay0 = x0_state
        theta_full, weights = self._time_weights(theta_latent)

        du = 1.0 / (self.N - 1)
        dt_segments = weights * total_time_max
        dt_du = dt_segments / du
        dt_du_0 = jnp.clip(dt_du[0], 0.1, total_time_max)

        k = 4.0
        p1_x = x0 + (dt_du_0 / k) * vx0
        p1_y = y0 + (dt_du_0 / k) * vy0

        a_geom_x = ax0 * (dt_du_0**2)
        a_geom_y = ay0 * (dt_du_0**2)

        p2_x = 2 * p1_x - x0 + a_geom_x / (k * (k - 1))
        p2_y = 2 * p1_y - y0 + a_geom_y / (k * (k - 1))

        return theta_full, weights, dt_segments, dt_du, p1_x, p1_y, p2_x, p2_y

    def splineguess(self, x0_state, total_time_max, goal_pos, theta_latent=None):
        """
        修正后的初始猜测逻辑：
        1. 使用 evaluate 同步的 _build_fixed_prefix 确保物理前缀完全一致。
        2. 将自由控制点 P3..PN 从 P2 位置向目标位置插值，确保 C2 连续性。
        """
        if theta_latent is None:
            theta_latent = jnp.zeros((self.N - 1,), dtype=jnp.float32)
        else:
            theta_latent = jnp.asarray(theta_latent, dtype=jnp.float32)

        # 1. 获取与 evaluate 完全一致的物理边界点 P1, P2
        theta_full, weights, dt_segments, dt_du, p1_x, p1_y, p2_x, p2_y = self._build_fixed_prefix(
            jnp.asarray(x0_state, dtype=jnp.float32),
            total_time_max,
            theta_latent,
        )

        # 2. 从 P2 开始生成自由控制点，而不是从 x0 开始
        free_count = self.N - 3
        
        # 构造平滑插值路径：从 P2 到 Goal
        # progress 归一化到 [0, 1]，0 对应 P2，1 对应 Goal
        progress = jnp.linspace(0.0, 1.0, free_count, dtype=jnp.float32)
        
        # 使用 sigmoid 形状的插值，平滑衔接 P2 的切线方向
        smooth = progress * progress * (3.0 - 2.0 * progress)

        goal_pos = jnp.asarray(goal_pos, dtype=jnp.float32)
        
        # 此时 x_free[0] 等于 P2，确保了物理上的位置连续
        x_free = p2_x + (goal_pos[0] - p2_x) * smooth
        y_free = p2_y + (goal_pos[1] - p2_y) * smooth

        return x_free, y_free, theta_latent

    def splineguess_full(self, x0_state, total_time_max, goal_pos, theta_latent=None):
        """Build the full initial control polygon [P0..PN] for inspection or debugging."""
        x_free, y_free, theta_latent = self.splineguess(x0_state, total_time_max, goal_pos, theta_latent=theta_latent)
        theta_full, weights, dt_segments, dt_du, p1_x, p1_y, p2_x, p2_y = self._build_fixed_prefix(
            jnp.asarray(x0_state, dtype=jnp.float32),
            total_time_max,
            theta_latent,
        )

        x0, y0, *_ = jnp.asarray(x0_state, dtype=jnp.float32)
        full_X = jnp.concatenate([jnp.array([x0, p1_x, p2_x], dtype=jnp.float32), x_free])
        full_Y = jnp.concatenate([jnp.array([y0, p1_y, p2_y], dtype=jnp.float32), y_free])

        return jnp.stack([full_X, full_Y], axis=-1), theta_latent, theta_full, weights, dt_segments, dt_du

    @jax.jit
    def evaluate(self, block_X, block_Y, theta_latent, x0_state, total_time_max, l=2.5, l_r=1.2):
        """
        block_X, block_Y: [N-3] 维自由控制点 (对应 P3...P_{N-1})
        theta_latent: [N-1] 维 logits
        x0_state: (x0, y0, vx0, vy0, ax0, ay0)
        """
        x0, y0, vx0, vy0, ax0, ay0 = x0_state

        # 1. 锚定时间分配：固定第一个权重对应 theta=0
        theta_full, weights, dt_segments, dt_du, p1_x, p1_y, p2_x, p2_y = self._build_fixed_prefix(
            x0_state, total_time_max, theta_latent, l=l, l_r=l_r
        )
        
        # 4. 拼装完整控制点序列 [N]
        full_X = jnp.concatenate([jnp.array([x0, p1_x, p2_x]), block_X])
        full_Y = jnp.concatenate([jnp.array([y0, p1_y, p2_y]), block_Y])

        # 5. 时间映射 (平铺到采样点)
        dt_du_sampled = jnp.repeat(dt_du, self.T // self.N)

        # 6. 几何导数映射
        x_p, y_p = self.dB @ full_X, self.dB @ full_Y
        x_pp, y_pp = self.d2B @ full_X, self.d2B @ full_Y
        x_ppp, y_ppp = self.d3B @ full_X, self.d3B @ full_Y

        # 7. 物理量代数映射 (使用小 epsilon 防止除以 0)
        vx, vy = x_p / (dt_du_sampled + 1e-6), y_p / (dt_du_sampled + 1e-6)
        v_abs = jnp.sqrt(vx**2 + vy**2 + 1e-6)

        ax, ay = x_pp / (dt_du_sampled**2 + 1e-6), y_pp / (dt_du_sampled**2 + 1e-6)
        jx, jy = x_ppp / (dt_du_sampled**3 + 1e-6), y_ppp / (dt_du_sampled**3 + 1e-6)

        # 8. 动力学控制量 (转向角 delta 与滑移角 beta)
        kappa = (x_p * y_pp - y_p * x_pp) / (v_abs**3 + 1e-6)

        # 转向约束稳定性处理
        denom = 1.0 - (l_r * kappa)**2
        tan_delta = jnp.where(denom > 1e-3, (l * kappa) / jnp.sqrt(jnp.maximum(denom, 1e-6)), 0.0)
        delta = jnp.arctan(tan_delta)
        beta = jnp.arctan((l_r / l) * tan_delta)
        
        return {
            "pos": (self.B @ full_X, self.B @ full_Y),
            "vel": (vx, vy),
            "acc": (ax, ay),
            "jerk": (jx, jy),
            "delta": delta,
            "beta": beta,
            "kappa": kappa
        }