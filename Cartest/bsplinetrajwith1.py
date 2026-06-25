# bsplinetraj.py
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from pathlib import Path

class BSplineTrajectoryGenerator:
    """离线预计算 B-spline 基矩阵，运行时只用矩阵乘法"""
    
    def __init__(self, basis_path: Path):
        data = np.load(basis_path)
        self.B = jnp.array(data["B"])      # [T, n_ctrl]
        self.dB = jnp.array(data["dB"])    # [T, n_ctrl]
        self.d2B = jnp.array(data["d2B"])  # [T, n_ctrl]
        self.T = self.B.shape[0]
        self.n_ctrl = self.B.shape[1]
        self.dt = float(data["dt"])
        self.total_time = float(data["total_time"])
        
        # 从离线数据中读取 B-spline 的阶数 (degree)
        self.degree = int(data.get("degree", 3))
        
        # 计算首段节点间的跨度，用于精确还原速度与控制点的代数映射
        self.dt_knot = self.total_time / (self.n_ctrl - self.degree)

    def _compute_fixed_p0_p1(self, x0: jnp.ndarray, v0: jnp.ndarray):
        """基于当前位置和速度矢量，精确解析出 clamped 三次 B-spline 的前两个控制点"""
        p0 = x0
        # 严格遵循 B-spline 导数边界公式: v(0) = (degree / dt_knot) * (P1 - P0)
        p1 = p0 + (self.dt_knot / float(self.degree)) * v0
        return p0, p1

    def evaluate(self, control_points_free: jnp.ndarray, x0: jnp.ndarray, v0: jnp.ndarray):
        """
        计算单条轨迹，同时锚定起点位置 P0 和速度映射点 P1
        control_points_free: [n_ctrl - 2, 2] 外部优化器传入的自由控制点 (从 P2 开始)
        x0: [2,] 当前物体的实际物理位置 (x, y)
        v0: [2,] 当前物体的实际速度矢量 (vx, vy)
        """
        p0, p1 = self._compute_fixed_p0_p1(x0, v0)
        full_ctrl_pts = jnp.concatenate([p0[None, :], p1[None, :], control_points_free], axis=0)
        
        pos = jnp.dot(self.B, full_ctrl_pts)
        vel = jnp.dot(self.dB, full_ctrl_pts)
        acc = jnp.dot(self.d2B, full_ctrl_pts)
        return pos, vel, acc
    
    def evaluate_batch(self, control_points_free_batch: jnp.ndarray, x0: jnp.ndarray, v0: jnp.ndarray):
        """
        并行计算多条样本轨迹 (供 IGO 采样并行计算使用)
        control_points_free_batch: [batch, n_ctrl - 2, 2]
        """
        batch_size = control_points_free_batch.shape[0]
        p0, p1 = self._compute_fixed_p0_p1(x0, v0)
        
        p0_batch = jnp.tile(p0[None, None, :], (batch_size, 1, 1))
        p1_batch = jnp.tile(p1[None, None, :], (batch_size, 1, 1))
        full_ctrl_batch = jnp.concatenate([p0_batch, p1_batch, control_points_free_batch], axis=1)
        
        pos = jnp.einsum('ti,bid->btd', self.B, full_ctrl_batch)
        vel = jnp.einsum('ti,bid->btd', self.dB, full_ctrl_batch)
        acc = jnp.einsum('ti,bid->btd', self.d2B, full_ctrl_batch)
        return pos, vel, acc
    
    def to_advanced_vehicle_states(self, pos, vel, acc, wheel_base=2.8, lr=1.4):
        """
        🎯 方案A核心：根据 Transformation.py 的非线性运动学方程逆解出全部状态量
        
        Transformation 运动学:
           dx = v * cos(psi + beta)
           dy = v * sin(psi + beta)
           beta = arctan(lr * tan(steer) / wheel_base)
        
        返回: states 形状为 [..., T, 6]，每一帧对齐: [x, y, v, psi, acc, steer]
        """
        # 1. 计算实际线速度大小 v
        v = jnp.linalg.norm(vel, axis=-1)
        v_safe = v + 1e-6
        
        # 2. 得到质心的实际运动航向角 (psi + beta)
        psi_plus_beta = jnp.arctan2(vel[..., 1], vel[..., 0])
        
        # 3. 计算轨迹曲率 kappa = (vx*ay - vy*ax) / v^3
        curvature = (vel[..., 0] * acc[..., 1] - vel[..., 1] * acc[..., 0]) / (v_safe**3)
        
        # 4. 逆解前轮转向角 steer (delta)
        # 根据阿克曼几何及质心滑移逼近，曲率与前轮转角满足：tan(steer) = curvature * wheel_base
        steer = jnp.arctan(curvature * wheel_base)
        
        # 5. 逆解质心侧偏角 beta
        beta = jnp.arctan(lr * jnp.tan(steer) / wheel_base)
        
        # 6. 分离并解出真实的车辆车体航向角 psi
        psi = psi_plus_beta - beta
        
        # 7. 计算车体实际纵向加速度 (切向加速度：加速度在速度方向上的投影)
        v_dir = vel / v_safe[..., None]
        a_long = jnp.sum(acc * v_dir, axis=-1)
        
        # 组合成完全契合 Transformation.py 物理定义的 6 维状态序列
        return jnp.stack([pos[..., 0], pos[..., 1], v, psi, a_long, steer], axis=-1)

        