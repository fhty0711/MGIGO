# Constran — 通用黑箱优化 Cost 构造引擎

**Constran** 将多目标 + 约束转化为单一标量 cost，供 IGO/GMM 等零阶求解器使用。

核心理念：**自相似 σ 嵌套**。obj/√2ⁿ → σₖ → [√2·σ₁ + Φ] × n。Φ=0 时完全透明——多层约束不衰减信号。优先级由 baseline（0/1/2）和插入位置共同决定。

---
## 1. 三步上手

```python
from Constraintdealer.Constran import *

# ① 写目标和违反函数 — 违反 >0 表示违规
def my_obj(x, ctx):
    return jnp.sum((x[:2] - ctx['target'])**2)

def collision_penalty(x, ctx):
    d = jnp.sqrt(jnp.sum((x[:,:2] - ctx['obs_pos'])**2, axis=-1))
    return jnp.maximum(0.0, ctx['safe_dist'] - d)

# ② 声明层级 — priority 大的在外层（高优先级）
layers = [
    Deterministic(collision_penalty, mode='hard',   priority=3, aggregate='max'),
    Deterministic(lambda x,c: jnp.sum(x[2:]**2),
                  mode='tunable', priority=2, tune_preset='standard'),
    Deterministic(lambda x,c: jnp.sum((x[:2]-c['target'])**2),
                  mode='soft',    priority=1),
]

# ③ 构建 → 给求解器
cost_fn = build(my_obj, layers)
```

**priority 大的在外层（高优先级），先满足。**

---
## 2. 核心机制

### 2.1 T_alpha — 多段对数变换

原始 g 从 1e-6 到 1e10（16 个数量级），T_alpha 压缩到 0~6：

```
T_alpha(g) = sign(g) × T_target(|g|)
```

三段式：**地板**（g < resolution，T 恒定）→ **log 增长**（resolution ≤ g ≤ ceiling）→ **缓坡天花板**（g > ceiling，T 缓慢增长）。

三档标定表：

| 表 | 分辨率 | 地板 T(0⁺) | 天花板 T(∞) | 用途 |
|----|--------|-----------|------------|------|
| `TRANSFORM_SOFT` | 1e-2 | 0.003 | 4.5 | 目标优化、舒适性 |
| `TRANSFORM_TUNABLE` | 1e-4 | 0.02 | 6.0 | 通用可调 |
| `TRANSFORM_HARD` | 1e-6 | 0.08 | 6.5 | 安全、硬约束 |

地板 = 跨过分辨率的"门槛跳变"。缓坡天花板 = 极端违规时 T 仍缓慢增长，防止求解器迷失。

### 2.2 自相似 σ 嵌套

$$\text{obj}/\sqrt{2}^{\,n+1} \;\to\; \sigma_k \;\to\; [\,\sqrt{2}\cdot\sigma_1 + \Phi\,] \times n \;\to\; \sqrt{2}\cdot\sigma_1$$

- k 只在最内层（目标函数），约束链全用 σ₁
- 每层约束：`Φ = baseline + max(0,T(g)) [+ δ·σ₁(β·max(0,T(g)))]`
- Φ=0 时层透明——√2·σ₁(σ₁(·/√2)·√2·...) = σ₁，任意层数无衰减
- 最终 √2·σ₁ 包裹，输出 ∈ (-√2, √2) ≈ (-1.41, 1.41)

### 2.3 baseline — 优先级基础

| mode | baseline | 语义 |
|------|----------|------|
| `soft` | 0 | 透明，无违规时 Φ=0 |
| `tunable` | 1.0 | 常值偏移，违规时叠加 penalty |
| `hard` | 2.0 | 大偏移，严格压过内层累积 |

**小 priority = 内层（被后续 σ·m 放大，影响大），大 priority = 外层（直接输出）。**
安全约束放内层自然被放大优先，舒适约束放外层直接输出。

### 2.4 精确罚 + Tunable boost

```
Φ = baseline + max(0, T(g)) + δ · σ₁(β · max(0, T(g)))
     ↑ 优先级基础   ↑ 精确罚基线      ↑ 可调饱和 boost
```

| 模式 | Φ 公式 | 
|------|--------|
| **Soft** | `0 + max(0, T(g))` |
| **Tunable** | `1.0 + max(0, T(g)) + δ·σ₁(β·max(0,T(g)))` |
| **Hard** | `2.0 + max(0, T(g)) + δ·σ₁(β·max(0,T(g)))` |
（P.S 这个hard 实在是太硬了 一般用不上 tunable 搞的硬一些一般都行）

---
## 3. 优先级嵌套

### 3.1 基本用法

```python
layers = [
    # P1 最内层：安全约束 — 被 σ·m 放大 4×，天然最优先
    Deterministic(collision, mode='hard', priority=1, aggregate='max'),
    # P2：动力学 — 放大 3×
    Deterministic(curvature, mode='tunable', priority=2, aggregate='mean',
                  tune_preset='standard'),
    # P5 最外层：跟踪目标 — 直接输出，不放大
    Deterministic(tracking, mode='soft', priority=5, aggregate='sum'),
]
cost_fn = build(my_obj, layers)
```

安全(内层 HARD)自然优先于舒适(外层 SOFT)——靠自相似结构保证，不用调参数。

### 3.2 层数选择

| 层数 | 效果 |
|------|------|
| 1-5 | 推荐范围，自相似无衰减，每层可辨 |
| 5-10 | 中等嵌套 |
| 10-20 | 深层嵌套，Φ=0 时完全透明，无额外衰减 |

### 3.3 不同约束不同语义

```python
layers = [
    # P1 (内): 避障 — baseline=2.0, 被放大, 最优先
    Deterministic(lambda x,c: 1.0/(x[0]+0.01)-100.0,
                  mode='hard', priority=1, transform='hard'),
    # P3: 速度 — baseline=1.0
    Deterministic(lambda x,c: jnp.abs(x[1])-1.0,
                  mode='tunable', priority=3, transform='tunable'),
    # P5 (外): 能耗 — baseline=0, 不放大
    Deterministic(lambda x,c: x[2]**2-1.0,
                  mode='soft', priority=5, transform='soft'),
]

---
## 4. 每层怎么设 — 语义配置指南

### 决策三步

```
① 违反能被接受吗？
   ├─ 绝不行 → mode='hard', priority=小 (内层, 被放大)
   ├─ 严重时可以 → mode='tunable', priority=中
   └─ 只是偏好 → mode='soft', priority=大 (外层, 不放大)

② 一个坏点就毁全部吗？
   ├─ 是 → aggregate='max'
   ├─ 否 → aggregate='mean'
   └─ 看总量 → aggregate='sum'

③ 怎么安全？
    看因果链条。一般来说，只有最下面的满足物理约束了，上面搜出来的才有意义。
    （P.S 即使你避障了，如果jerk 出来是100 那没有任何意义）
```

### 常见语义速查

| 语义 | mode | priority | aggregate | transform |
|------|------|----------|-----------|-----------|
| 避障/防撞 | `hard` | 1 (内) | `max` | `hard` |
| 车道偏离 | `tunable` | 2 | `sum` | `tunable` |
| 曲率 | `tunable` | 3 | `mean` | `tunable` |
| jerk/舒适 | `soft` | 4 | `mean` | `soft` |
| 跟踪 | `soft` | 5 (外) | `mean` | `soft` |

---
## 5. 约束类型

| 类型 | 类 | 数学形式 | 关键参数 |
|------|-----|---------|----------|
| 确定性 | `Deterministic` | $g(x) \le 0$ | — |
| 机会约束 | `Chance` | $\mathbb{P}_\xi(g(x,\xi) \le 0) \ge 1-\alpha$ | `alpha`, `n_samples`, `noise_fn` |
| 鲁棒约束 | `Robust` | $\max_{\xi\in\Xi} g(x,\xi) \le 0$ | `uncertainty_set`, `n_grid` |
| 分布鲁棒 | `DRO` | $\inf_{\mathbb{Q}\in\mathcal{A}} \mathbb{Q}(g\le 0) \ge 1-\alpha$ | `alpha`, `ambiguity_set`, `n_samples_per_dist` |

### 5.1 机会约束 (Chance)

```python
Chance(
    g_fn=lambda x, xi, ctx: jnp.abs(x) - xi,  # xi ~ noise
    noise_fn=lambda key, n: random.normal(key, (n,)),
    alpha=0.1,        # 90% 置信度
    n_samples=200,    # MC 采样数
    mode='tunable', priority=2,
)
```
MC 分位数估计：$\hat{Q}_{1-\alpha}(g) = \inf\{q \mid \frac{1}{M}\sum \mathbb{I}_{g\le q} \ge 1-\alpha\}$。

### 5.2 鲁棒约束 (Robust)

```python
Robust(
    g_fn=lambda x, xi, ctx: g(x, xi),
    uncertainty_set=jnp.linspace(-1, 1, 40),  # 离散网格
    mode='hard', priority=3,
)
```
通过 `lax.scan` 对网格逐点求最大值。适合时域鲁棒：对轨迹的 N 个时间采样点取 max。

### 5.3 分布鲁棒 (DRO)

```python
DRO(
    g_fn=lambda x, xi, ctx: g(x, xi),
    ambiguity_set=[dist1, dist2, dist3],  # 模糊集
    alpha=0.1,
    mode='hard', priority=3,
)
```
最坏分布下的机会约束。

### 5.4 B-spline 时域约束 & 分位数聚合

B-spline 轨迹约束只能通过上百个时域采样点评估，等价于时域鲁棒约束。一般不用 `max`（太敏感、异常点毁全部），也不用 `mean`（太宽松），**推荐用分位数**：

```python
Deterministic(
    g_fn=bspline_violation,  # 返回 [T] 向量，每个时间点一个违规值
    aggregate='q95',         # 95% 分位数：允许 5% 时间点违规
    mode='tunable', priority=2,
)
```

| aggregate | 语义 | 适用场景 |
|-----------|------|---------|
| `'q99'` | 99% 分位数 | 接近 max，但容错 1% 野点 |
| `'q95'` | 95% 分位数 | **推荐**，鲁棒且不过度敏感 |
| `'q90'` | 90% 分位数 | 更宽松，允许 10% 时间违规 |
| `'max'` | 最坏点 | 太敏感，B-spline 不推荐 |
| `'mean'` | 平均值 | 太宽松，违规可能被平均掉 |
| `'sum'` | 总和 | 累计违规量 |

---
## 6. 变换表与预设 （连续谱，从最软到最硬。但是考虑到精度分辨率一般也有1e-2 其实都挺硬）

### T 表 (α)

| 表 | knots_g | knots_T |
|----|---------|---------|
| SOFT | [1e-2, 5e-2, 1e-1, 0.5, 1, 10, 100, 1e4, 1e6, 1e8, 1e10] | [0.003, 0.015, 0.06, 0.25, 0.7, 2.2, 3.5, 4.0, 4.2, 4.4, 4.5] |
| TUNABLE | [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 100, 1e4, 1e6, 1e8, 1e10] | [0.02, 0.06, 0.15, 0.4, 0.8, 1.5, 3.0, 4.5, 5.0, 5.3, 5.7, 6.0] |
| HARD | [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 100, 1e4, 1e6, 1e8, 1e10] | [0.08, 0.15, 0.3, 0.6, 1.2, 2.0, 3.0, 4.5, 5.5, 5.8, 6.2, 6.5] |

### Tune 预设 (β, δ)

| preset | β | δ | 语义 |
|--------|------|------|------|
| mild | 0.15 | 0.5 | 极软 |
| standard | 0.5 | 0.7 | 标准（默认） |
| firm | 1.0 | 0.75 | 适中 |
| strong | 2.5 | 0.8 | 较硬 |
| nearhard | 8.0 | 1.0 | 近硬 |
| __hard__ | 20.0 | 1.2 | 纯硬（极少用） |

---
## 7. 数值特性

| 参数 | 值 | 说明 |
|------|-----|------|
| 结构 | obj/√2ⁿ⁺¹ → σₖ → [√2·σ₁ + Φ] × n → √2·σ₁ | 自相似，无衰减 |
| 输出范围 | (-√2, √2) ≈ (-1.41, 1.41) | 最终 σ·m 包裹 |
| Φ=0 透明 | 5×SOFT = obj_only | 任意层数 |
| baseline | 0 / 1.0 / 2.0 | SOFT / TUNABLE / HARD |
| k_inner | 0.1 | 按目标范围自选（0.1 为了防止饱和；一般1.0即可） |

---
## 8. 常见问题

**Q: priority 大的在外层？**
A: 对。大 priority = 外层 = 优先优化。

**Q: 为什么不用 k 参数？**
A: 自相似结构全链用 σ₁。k 只在最内层 σₖ 控制目标饱和速度。

**Q: 天花板会卡死求解器吗？**
A: 不会。天花板是缓坡，极端违规时 T 仍缓慢增长。

**Q: 地板会制造盲区吗？**
A: 故意的。分辨率以下的违规不算违规。

**Q: 不同物理量不同尺度能混用吗？**
A: 能。每个约束用自己的 T 表匹配物理尺度。
