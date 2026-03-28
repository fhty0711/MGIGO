import jax.numpy as jnp
import jax
from jax import jit

class HybridIntersectionMap:
    def __init__(self, core_half_width=15.0, core_res=0.5, arm_res=2.0, arm_len=30.0):
        self.c_hw = core_half_width
        self.c_res = core_res
        # 核心区每边栅格数 (例如 30/0.5 = 60)
        self.n_side = int(2 * self.c_hw / self.c_res)
        self.core_total = self.n_side ** 2
        
        self.a_res = arm_res
        self.a_len = arm_len
        self.n_arm = int(self.a_len / self.a_res)
        
        # 索引偏移：Core(0), South(1), East(2), North(3), West(4)
        self.offsets = jnp.array([
            0, 
            self.core_total, 
            self.core_total + self.n_arm,
            self.core_total + 2 * self.n_arm,
            self.core_total + 3 * self.n_arm
        ])
        
        self.total_bits = self.core_total + 4 * self.n_arm
        self.num_chunks = (self.total_bits + 31) // 32

    @jit
    def pos_to_mask(self, x, y):
        mask = jnp.zeros(self.num_chunks, dtype=jnp.uint32)
        
        # 区域判定
        is_in_core = (jnp.abs(x) <= self.c_hw) & (jnp.abs(y) <= self.c_hw)
        
        # 1. 核心区索引计算 (映射到一片连续栅格)
        def get_core_idx():
            # 将 (-15, 15) 映射到 (0, 59)
            row = jnp.floor((y + self.c_hw) / self.c_res).astype(jnp.int32)
            col = jnp.floor((x + self.c_hw) / self.c_res).astype(jnp.int32)
            # 限制边界
            row = jnp.clip(row, 0, self.n_side - 1)
            col = jnp.clip(col, 0, self.n_side - 1)
            return row * self.n_side + col

        # 2. 直臂区逻辑 (简化为 1D 距离)
        def get_arm_idx():
            # 判定在哪条臂上并返回对应偏移
            is_s = (y < -self.c_hw); is_n = (y > self.c_hw)
            is_w = (x < -self.c_hw); is_e = (x > self.c_hw)
            
            dist = jnp.where(is_s | is_n, jnp.abs(y) - self.c_hw, jnp.abs(x) - self.c_hw)
            local_idx = jnp.floor(dist / self.a_res).astype(jnp.int32)
            local_idx = jnp.clip(local_idx, 0, self.n_arm - 1)
            
            offset = jax.lax.select(is_s, self.offsets[1], 
                     jax.lax.select(is_e, self.offsets[2],
                     jax.lax.select(is_n, self.offsets[3], self.offsets[4])))
            return offset + local_idx

        # 最终索引选择
        final_idx = jax.lax.cond(is_in_core, get_core_idx, get_arm_idx)
        
        # 设置 bitmask
        return mask.at[final_idx // 32].set(jnp.uint32(1 << (final_idx % 32)))