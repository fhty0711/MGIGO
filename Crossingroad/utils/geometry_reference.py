import numpy as np
from shapely.geometry import LineString
import jax.numpy as jnp

def get_intersection_paths(s_steps=200, s_max=60.0):
    """
    使用 Shapely 生成三车的参考路径坐标
    - Agent 0: 南向北直行 (x=1.75, y从-30到30)
    - Agent 1: 东向西直行 (y=1.75, x从30到-30)
    - Agent 2: 南向西左转 (从南口进入，在路口中心转向西口)
    """
    # 定义标准车道宽度一半 (假设3.5m车道，中心线在1.75m)
    offset = 1.75
    
    # 1. 定义几何线段
    # 南北直行
    line_sn = LineString([(offset, -30), (offset, 30)])
    # 东西直行
    line_ew = LineString([(30, offset), (-30, offset)])
    # 南左转西 (控制点：南口 -> 稍微进入路口 -> 转向西口)
    line_sw = LineString([(offset, -30), (offset, -offset), (-offset, offset), (-30, offset)])

    lines = [line_sn, line_ew, line_sw]
    
    all_coords = []
    # 2. 等间距插值 (基于里程 s)
    s_vals = np.linspace(0, s_max, s_steps)
    
    for line in lines:
        # 使用 line.interpolate 确保点在路径上均匀分布
        # interpolate 参数是距离起点的长度
        pts = [line.interpolate(s) for s in s_vals]
        all_coords.append([(p.x, p.y) for p in pts])
        
    # 返回 JAX 数组 [M, s_steps, 2] 和 里程采样点 [s_steps]
    return jnp.array(all_coords), jnp.array(s_vals)

if __name__ == "__main__":
    # 测试代码
    paths, s_samples = get_intersection_paths()
    print(f"Path shape: {paths.shape}") # 应为 (3, 200, 2)
    print(f"Mileage samples: {s_samples[:5]}") # 显示前5个里程点