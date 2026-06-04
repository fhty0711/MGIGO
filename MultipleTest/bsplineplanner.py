# BSplinePlanner.py
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from pathlib import Path


# 请注意，黑箱优化将会大规模调用。所以外面调用的函数用jit包装。多个样本并行计算请用vmap包装。 内部矩阵算法等能用Jax 就用jax。（jax应该是有jax.nn的）
#总之，一定要利用好jax 的性质 请勿让gpu空转或者jax不停地重启预热！ 
class BSplineTrajectoryGenerator:
    """离线预计算 B-spline 基矩阵，运行时只用矩阵乘法"""
    
    def __init__(self, basis_path: Path):
        data = np.load(basis_path)
        self.B = jnp.array(data["B"])      # [T, n_ctrl]
        self.dB = jnp.array(data["dB"])
        self.d2B = jnp.array(data["d2B"])
        self.T = self.B.shape[0]
        self.n_ctrl = self.B.shape[1]
        self.dt = float(data["dt"])
        self.total_time = float(data["total_time"])
    
    def evaluate(self, control_points: jnp.ndarray):
        """
        control_points: [n_ctrl, 2]  (x, y)
        返回: (pos, vel, acc) 各为 [T, 2]
        """
        pos = jnp.dot(self.B, control_points)
        vel = jnp.dot(self.dB, control_points)
        acc = jnp.dot(self.d2B, control_points)
        return pos, vel, acc
    
    def evaluate_batch(self, control_points_batch: jnp.ndarray):
        """
        control_points_batch: [batch, n_ctrl, 2]
        返回: (pos, vel, acc) 各为 [batch, T, 2]
        """
        pos = jnp.einsum('ti,bid->btd', self.B, control_points_batch)
        vel = jnp.einsum('ti,bid->btd', self.dB, control_points_batch)
        acc = jnp.einsum('ti,bid->btd', self.d2B, control_points_batch)
        return pos, vel, acc
    
    def to_vehicle_states(self, pos, vel, acc, wheel_base=2.8):
        """
        从 B-spline 的运动学量计算车辆状态 [T, 6],
        状态: [x, y, v, psi, a_long, steer]
        """
        v = jnp.linalg.norm(vel, axis=-1)
        psi = jnp.arctan2(vel[..., 1], vel[..., 0])
        
        # 纵向加速度（速度方向投影）
        v_norm = v + 1e-6
        v_dir = vel / v_norm[..., None]
        a_long = jnp.sum(acc * v_dir, axis=-1)
        
        # 转向角（从曲率推导）
        curvature = (vel[..., 0] * acc[..., 1] - vel[..., 1] * acc[..., 0]) / (v**3 + 1e-6)
        steer = jnp.arctan(curvature * wheel_base)
        
        # 返回车辆状态
        return jnp.stack([pos[..., 0], pos[..., 1], v, psi, a_long, steer], axis=-1)