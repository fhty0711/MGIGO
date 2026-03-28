
import jax.numpy as jnp
import numpy as onp
from scipy.interpolate import BSpline

# ====================== 1. 全局配置 ======================
DT = 0.1             # 时间步长
HORIZON = 100        # 预测时域点数
TOTAL_TIME = HORIZON * DT # 总预测时间
POLY_ORDER = 5       # B-spline 阶次
NUM_CONTROL_POINTS = 10 # 控制点数量
TOTAL_DIM = 2 * NUM_CONTROL_POINTS # 总优化变量维度 (Qx + Qy)
LANE_WIDTH = 3.7     # 车道宽度 (用于绘图/目标设置)
WHEELBASE = 2.8      # 轴距 L (用于微分平坦理论，尽管目前未显式使用)


# ====================== 1. 全局配置 ======================
DT = 0.1             
HORIZON = 100        
TOTAL_TIME = HORIZON * DT  # 10.0 seconds
POLY_ORDER = 5       

# --- 关键维度 ---
NUM_CONTROL_POINTS_FULL = 15 # 完整的 B-spline 控制点数量 (N)
NUM_CONTROL_POINTS_OPT = 13  # 优化变量 Q 的有效数量 (N - 2)
TOTAL_DIM = 2 * NUM_CONTROL_POINTS_OPT # 总优化变量维度 (2 * 13 = 26)

# --- 关键常数 ---
LANE_WIDTH = 3.7   


def create_5_tap_filter_matrix(N):
    F = onp.zeros((N, N))
    W = onp.array([1, 26, 66, 26, 1]) / 120.0
    for i in range(N):
        if i == 0 or i == N - 1:
            F[i, i] = 1.0; continue
        for r in range(-2, 3):
            q_idx_raw = i + r
            weight = W[r + 2]
            j = q_idx_raw
            j_final = abs(j) if j < 0 else (2 * (N - 1) - j if j >= N else j)
            F[i, j_final] += weight
    return jnp.array(F, dtype=jnp.float32)

F_MATRIX = create_5_tap_filter_matrix(NUM_CONTROL_POINTS_FULL)

INTERNAL_KNOTS_COUNT = NUM_CONTROL_POINTS_FULL - POLY_ORDER
KNOT_DELTA = TOTAL_TIME / INTERNAL_KNOTS_COUNT
internal_knots = onp.arange(1, INTERNAL_KNOTS_COUNT + 1) * KNOT_DELTA
knots = onp.concatenate([onp.zeros(POLY_ORDER + 1), internal_knots, onp.full(POLY_ORDER, TOTAL_TIME)])
t_eval = onp.arange(HORIZON) * DT

def compute_basis_matrix(knots, t_eval, k, nu):
    basis = onp.zeros((len(t_eval), NUM_CONTROL_POINTS_FULL))
    for i in range(NUM_CONTROL_POINTS_FULL):
        c = onp.zeros(NUM_CONTROL_POINTS_FULL); c[i] = 1.0
        spl = BSpline(knots, c, k=k, extrapolate=True)
        basis[:, i] = spl(t_eval, nu=nu)
    return jnp.array(basis, dtype=jnp.float32)

N5_BASIS, N5_PRIME, N5_DOUBLE_PRIME, N5_TRIPLE_PRIME = [
    compute_basis_matrix(knots, t_eval, POLY_ORDER, nu=i) for i in range(4)
]