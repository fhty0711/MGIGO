import jax.numpy as jnp
from jax import vmap, jit

class LebesgueEvaluator:
    def __init__(self, luts, s_samples, K_matrix):
        self.luts = luts           # [M, S, Chunks]
        self.s_samples = s_samples # LUT 对应的里程点
        self.K = K_matrix         # 时间核 [T, T]

    @jit
    def compute_cost(self, all_masks, t_eval, weights, lambda_eff):
        """
        all_masks: [M, T, Chunks]
        """
        M, T, _ = all_masks.shape
        
        # 1. 计算两两冲突测度 [M, M, T]
        # 使用位运算交集
        def pair_overlap(m1, m2):
            return jnp.sum(jnp.bit_count(jnp.bitwise_and(m1, m2)), axis=-1)

        # 遍历所有对 (i, j)
        c_matrix = vmap(vmap(pair_overlap, in_axes=(None, 0)), in_axes=(0, None))(all_masks, all_masks)
        
        # 2. 交互能 (时空排斥)
        # J_int = ΣΣ w_ij * c_ij^T * K * c_ij
        def get_energy(c_ij):
            return jnp.dot(c_ij, jnp.dot(self.K, c_ij))
        
        j_int = jnp.sum(vmap(vmap(get_energy))(c_matrix) * weights)

        # 3. 效率能 (时间加权占用积分)
        occupancy = jnp.sum(jnp.bit_count(all_masks), axis=(-1, -2)) # [T]
        j_eff = jnp.sum(occupancy * t_eval)
        
        return j_int + lambda_eff * j_eff 