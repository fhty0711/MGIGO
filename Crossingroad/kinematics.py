import jax.numpy as jnp
from jax import jit
import jax.nn as nn

@jit
def sigmoid_traj_mapping(z, s_in, v_0, t_steps):
    """
    z: [t_in, v_pass, k]
    t_steps: 时间序列 [0, ..., T]
    """
    t_in, v_pass, k = z
    # 速度曲线: v(t) = v0 + (v_pass - v0) * sigmoid(k * (t - t_in))
    v_t = v_0 + (v_pass - v_0) * nn.sigmoid(k * (t_steps - t_in))
    # 积分得到里程 s(t)
    dt = t_steps[1] - t_steps[0]
    s_t = s_in + jnp.cumsum(v_t) * dt
    return s_t