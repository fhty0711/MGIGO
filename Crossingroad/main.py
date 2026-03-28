from utils.geometry_reference import get_intersection_paths
from utils.map_utils import HybridIntersectionMap
from solver.MPCsolverM22 import mmog_igo_optimizer_mpc
from upperlayeroptimization1 import full_fitness_fn

# 1. 生成物理路径
reference_paths, s_samples = get_intersection_paths(s_steps=200)

# 2. 初始化混合地图
hybrid_map = HybridIntersectionMap(core_half_width=15.0, core_res=0.5)

# 3. 预计算 LUT (离线或初始化时执行一次)
# 使用 vmap 加速坐标转掩码的过程
import jax
from jax import vmap

@jax.jit
def build_all_luts(paths):
    # paths: [M, s_steps, 2]
    # 嵌套 vmap：先遍历 Agent (M)，再遍历路径点 (s_steps)
    v_pos_mask = vmap(vmap(hybrid_map.pos_to_mask, in_axes=(0, 0)), in_axes=0)
    return v_pos_mask(paths[..., 0], paths[..., 1])

luts = build_all_luts(reference_paths)

# 4. 存入 context
my_context = {
    'evaluator': LebesgueEvaluator(luts, s_samples, K_matrix),
    'luts': luts,
    's_samples': s_samples,

}


# --- 执行流程 ---
# 1. 准备 context 数据
# 2. 调用优化器
final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
    key=random.PRNGKey(42),
    T=100, dt=0.1, M=3, K=4, B=200, B0=80, # 参数对齐 Upperlayeroptimization1
    dims=[3, 3, 3], T_0=10, 
    fitness_fn_total=full_fitness_fn
)