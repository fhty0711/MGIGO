# spline.py
"""Offline B-spline basis precomputation for JAX MPC.
无量纲纯几何基矩阵离线生成器。
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline

DEFAULT_N_SAMPLING = 150    # 全时域稠密采样点数 T
DEFAULT_N_CONTROL = 10     # 控制点个数 N
DEFAULT_DEGREE = 4         # 三次样条
DEFAULT_BASIS_FILE = Path(__file__).with_name("bspline_basis.npz")


def generate_dimensionless_basis(
    n_sampling: int,
    n_control: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成完全无量纲域 u 在 [0, 1] 之间的位置、一阶、二阶、三阶几何导数基矩阵。"""
    knot_count = n_control + degree + 1
    interior_count = knot_count - 2 * (degree + 1)
    
    if interior_count > 0:
        inner_knots = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    else:
        inner_knots = np.array([], dtype=np.float64)
        
    knots = np.concatenate([
        np.zeros(degree + 1),
        inner_knots,
        np.ones(degree + 1)
    ])
    
    u_eval = np.linspace(0.0, 1.0, n_sampling)
    
    B = np.zeros((n_sampling, n_control), dtype=np.float64)
    dB = np.zeros((n_sampling, n_control), dtype=np.float64)
    d2B = np.zeros((n_sampling, n_control), dtype=np.float64)
    d3B = np.zeros((n_sampling, n_control), dtype=np.float64)

    for i in range(n_control):
        c = np.zeros(n_control, dtype=np.float64)
        c[i] = 1.0
        spl = BSpline(knots, c, k=degree, extrapolate=False)
        B[:, i] = spl(u_eval, nu=0)   # 位置
        dB[:, i] = spl(u_eval, nu=1)  # 速度 (几何)
        d2B[:, i] = spl(u_eval, nu=2) # 加速度 (几何)
        d3B[:, i] = spl(u_eval, nu=3) # Jerk (几何)

    return B, dB, d2B, d3B, u_eval

def save_bspline_basis(
    output_path: Path = DEFAULT_BASIS_FILE,
    *,
    n_sampling: int = DEFAULT_N_SAMPLING,
    n_control: int = DEFAULT_N_CONTROL,
    degree: int = DEFAULT_DEGREE,
) -> Path:
    """生成无量纲基矩阵并固化为 npz 静态工件"""
    B, dB, d2B, d3B, u_eval = generate_dimensionless_basis(n_sampling, n_control, degree)
    np.savez(
        output_path,
        B=B,
        dB=dB,
        d2B=d2B,
        d3B=d3B,
        u_eval=u_eval,
        degree=degree
    )
    print(f"✅ Precomputed basis (including d3B) saved to: {output_path.resolve()}")
    return output_path

if __name__ == "__main__":
    save_bspline_basis()