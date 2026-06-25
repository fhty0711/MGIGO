import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/braunschweig/gmm_igo/gmm_igo')
from TSP import plackett_luce_igo_optimizer_tsp

# ==================== 1. 准备数据 ====================

n_cities = 20  # 城市数量
key = jax.random.PRNGKey(42)

# 生成随机城市坐标
coords = jax.random.uniform(key, (n_cities, 2), minval=0, maxval=100)

# 定义适应度函数
def compute_tour_length(permutation, coords):
    city_coords = coords[permutation]
    diffs = city_coords - jnp.roll(city_coords, -1, axis=0)
    distances = jnp.sqrt(jnp.sum(diffs**2, axis=1))
    return jnp.sum(distances)

def fitness_fn(permutation, context):
    coords = context['coords']
    length = compute_tour_length(permutation, coords)
    return length  # 最大化 → 用负长度

# 打包上下文
context = {'coords': coords}

# ==================== 2. 设置超参数 ====================

T = 400           # 迭代代数
B = 100           # 每代采样数量（群体大小）
B0 = 20           # 精英数量（选前 B0 个最好的样本）
dt = 2.0         # 学习率

# ==================== 3. 运行优化 ====================

# 初始化随机数种子
key, subkey = jax.random.split(key)

# 运行求解器
final_eta, history = plackett_luce_igo_optimizer_tsp(
    key=subkey,
    T=T,
    n_cities=n_cities,
    B=B,
    B0=B0,
    dt=dt,
    fitness_fn_total=fitness_fn,
    context=context,
    initial_eta=None  # 用默认的全 0 初始化
)

# ==================== 4. 提取结果 ====================

# 从 eta 贪心解码最优路径：从 0 号城市出发，每步选转移偏好最高的未访问城市
def decode_best_tour(eta):
    n = eta.shape[0]
    visited = jnp.zeros(n, dtype=jnp.bool_)
    tour = jnp.zeros(n, dtype=jnp.int32)
    current = 0
    tour = tour.at[0].set(current)
    visited = visited.at[current].set(True)
    for step in range(1, n):
        logits = eta[current, :] - visited * 1e9  # 屏蔽已访问城市
        current = jnp.argmax(logits)
        tour = tour.at[step].set(current)
        visited = visited.at[current].set(True)
    return tour

best_permutation = decode_best_tour(final_eta)
best_length = compute_tour_length(best_permutation, coords)

print(f"最优路径: {best_permutation}")
print(f"最优路径长度: {best_length:.4f}")

# 提取历史记录（每次迭代的最佳和平均适应度）
best_history = history[0]   # 每次迭代的最佳适应度
mean_history = history[1]   # 每次迭代的平均适应度

print(f"\n初始最佳适应度 (代 0): {best_history[0]:.4f}")
print(f"最终最佳适应度 (代 {T-1}): {best_history[-1]:.4f}")
print(f"初始平均适应度 (代 0): {mean_history[0]:.4f}")
print(f"最终平均适应度 (代 {T-1}): {mean_history[-1]:.4f}")

# ==================== 5. Baseline 对比 ====================

# 5a. 随机路径
key, subkey = jax.random.split(key)
random_perm = jax.random.permutation(subkey, n_cities)
random_length = compute_tour_length(random_perm, coords)
print(f"\n--- Baseline 对比 ---")
print(f"随机路径长度: {random_length:.2f}")

# 5b. 最近邻贪心 (从每个城市出发，取最优)
def nearest_neighbor(start, coords):
    n = coords.shape[0]
    visited = jnp.zeros(n, dtype=jnp.bool_)
    tour = jnp.zeros(n, dtype=jnp.int32)
    current = start
    tour = tour.at[0].set(current)
    visited = visited.at[current].set(True)
    for step in range(1, n):
        dists = jnp.sqrt(jnp.sum((coords - coords[current])**2, axis=1))
        dists = dists + visited * 1e9  # 屏蔽已访问
        current = jnp.argmin(dists)
        tour = tour.at[step].set(current)
        visited = visited.at[current].set(True)
    return tour

best_nn_length = float('inf')
best_nn_tour = None
for start in range(n_cities):
    tour = nearest_neighbor(start, coords)
    length = compute_tour_length(tour, coords)
    if length < best_nn_length:
        best_nn_length = length
        best_nn_tour = tour
print(f"最近邻贪心 (最优起点): {best_nn_length:.2f}")

# 5c. 2-opt 局部优化 (numpy 版本，快速)
import numpy as np

def compute_tour_length_np(permutation, coords_np):
    city_coords = coords_np[permutation]
    diffs = city_coords - np.roll(city_coords, -1, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(distances)

def two_opt_np(tour, coords_np, max_iter=500):
    n = len(tour)
    best_tour = np.array(tour)
    best_len = compute_tour_length_np(best_tour, coords_np)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_tour = best_tour.copy()
                new_tour[i:j+1] = new_tour[i:j+1][::-1]
                new_len = compute_tour_length_np(new_tour, coords_np)
                if new_len < best_len:
                    best_tour = new_tour
                    best_len = new_len
                    improved = True
    return best_tour, best_len

coords_np = np.array(coords)
nn_tour_np = np.array(best_nn_tour)
igo_tour_np = np.array(best_permutation)

_, nn2opt_length = two_opt_np(nn_tour_np, coords_np, max_iter=800)
print(f"最近邻 + 2-opt: {nn2opt_length:.2f}")

_, igo2opt_length = two_opt_np(igo_tour_np, coords_np, max_iter=800)
print(f"IGO + 2-opt: {igo2opt_length:.2f}")

print(f"\nIGO 贪心解码: {best_length:.2f}")
print(f"vs 最近邻:     {best_nn_length:.2f}  (差距 {best_length - best_nn_length:+.2f})")
