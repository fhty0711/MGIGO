# Constran 用户手册

**Constran** 是约束转化引擎——你写精确罚函数，它自动组装成求解器能用的代价函数。

---

## 目录

1. [三步上手](#1-三步上手)
2. [四种约束类型](#2-四种约束类型)
3. [三种模式：Hard / Tunable / Soft](#3-三种模式)
4. [Tunable 参数怎么选](#4-tunable-参数怎么选)
5. [语义速查表](#5-语义速查表)
6. [MPC 用法](#6-mpc-用法)
7. [多智能体用法](#7-多智能体用法)
8. [常见问题](#8-常见问题)

---

## 1. 三步上手

```python
from Constraintdealer.Constran import *

# ① 写目标函数
def my_obj(x, ctx):
    return jnp.sum((x[:2] - ctx['target'])**2) + 0.1 * jnp.sum(x[2:]**2)

# ② 声明约束
constraints = [
    Deterministic(lambda x, ctx: x[0] + x[1] + 4,    # g(x) ≤ 0
                  mode='hard', priority=1, delta=1.5),
    Deterministic(lambda x, ctx: x[1] - x[0],
                  mode='soft', priority=2),
]

# ③ 构建 → 给求解器
cost_fn = build(my_obj, constraints)

# 求解器直接使用
result = mmog_igo_optimizer_mpc(
    ..., fitness_fn_total=cost_fn, context=ctx
)
```

**你只需要写 `g(x) ≤ 0` 形式的约束函数。** 正值 = 违反，负值/零 = 满足。Constran 自动做对数变换、饱和嵌套、优先级组装。

---

## 2. 四种约束类型

| 类型 | 你写 | Constran 自动计算 g_raw |
|------|------|----------------------|
| `Deterministic` | `g(x)` | `g(x)` 直接 |
| `Chance` | `g(x, ξ)` + 噪声分布 | `Q_{1-α}(g(x,ξ))` MC 分位数 |
| `Robust` | `g(x, ξ)` + 不确定性集 Ξ | `max_{ξ∈Ξ} g(x,ξ)` lax.scan |
| `DRO` | `g(x, ξ)` + 模糊集 {P_k} | `max_{P_k} Q_{1-α}^{P_k}(g)` |

### 2.1 Deterministic — 确定性约束

```python
# g(x) ≤ 0，不涉及噪声
Deterministic(lambda x, ctx: x[0]**2 + x[1]**2 - 25,   # 在圆内
              mode='hard', priority=1)

# 双边 band 约束: c_lo ≤ h(x) ≤ c_hi
# → 写成: max(c_lo - h(x), h(x) - c_hi) ≤ 0
Deterministic(lambda x, ctx: jnp.maximum(
                 0.3 - abs(x[0] - x[1] - 2),           # |x1-x2-2| ≥ 0.3
                 abs(x[0] - x[1] - 2) - 0.3),
              mode='hard', priority=1)
```

### 2.2 Chance — 概率约束

```python
# P(g(x,ξ) ≤ 0) ≥ 1-α，ξ 的分布已知
Chance(
    lambda x, xi, ctx: jnp.linalg.norm(x[:2] + xi) - 3.0,  # g(x,ξ)
    noise_fn=lambda key, shape: jax.random.normal(key, shape) * 0.2,
    alpha=0.1,        # 90% 概率满足
    n_samples=100,    # MC 样本数
    mode='hard', priority=1,
)
```

**选 n_samples：** `n_samples ≥ 10/α`。α=0.1 时 ≥ 100，α=0.01 时 ≥ 1000。

### 2.3 Robust — 鲁棒约束

```python
# g(x,ξ) ≤ 0 对所有 ξ∈Ξ 成立
xi_grid = jnp.concatenate([
    jnp.linspace(-3.0, -2.0, 20),    # 非凸集 [-3,-2] ∪ [1,2]
    jnp.linspace( 1.0,  2.0, 20),
])

Robust(
    lambda x, xi, ctx: (x[0] + xi)**2 + x[1]**2 - 10.0,
    uncertainty_set=xi_grid,
    n_grid=40,
    mode='hard', priority=1,
)
```

**支持非凸、不连通的不确定性集**——传统 LMI 方法做不到。

### 2.4 DRO — 分布鲁棒约束

```python
# inf_{P∈𝒫} P(g≤0) ≥ 1-α，分布本身不确定
DRO(
    lambda x, xi, ctx: g(x, xi),
    ambiguity_set=[
        lambda key, shape: jax.random.normal(key, shape) * 0.1,       # P1
        lambda key, shape: jax.random.normal(key, shape) * 0.3,       # P2 (更宽的噪声)
        lambda key, shape: jax.random.uniform(key, shape, -0.5, 0.5), # P3 (均匀噪声)
    ],
    alpha=0.1,
    mode='hard', priority=1,
)
```

---

## 3. 三种模式

| 模式 | 机制 | 一句话 | 何时用 |
|------|------|--------|--------|
| **Hard** | `jnp.where(g>0, T(g)+δ, inner)` | "任何违反都不可接受" | 安全、法规、物理极限 |
| **Tunable** | `δ·σ(β·T(g)) + inner` | "违反越差代价越大，但有上限" | 大部分场景（默认首选） |
| **Soft** | `T(g) + inner` | "违反越差代价越大，无上限" | 偏好、最简场景 |

### 关键区别

```
g = 0.001 (微小违反)  →  Hard: 代价≥0.832  |  Tunable β=5: 几乎满罚  |  Soft: 几乎无感
g = 100   (严重违反)  →  Hard: 代价≈0.987  |  Tunable β=1: 封顶≈0.82  |  Soft: 代价≈0.99
```

**Hard 的 $g=0.001$ 和 $g=100$ 都 > 所有内层。Tunable 的 $g>10$ 几乎封顶。Soft 永不封顶。**

### 模式选择速查

| 想表达的语义 | 用 |
|-------------|-----|
| "绝对不能碰" | Hard |
| "碰了很严重，但 10m 和 100m 一样糟" | Tunable β=1~5 |
| "碰得越深越糟，永不封顶" | Soft |
| "小擦边无所谓，别大撞就行" | Tunable β=5~50 |
| "多点擦边比单点深撞差" | Tunable(β=1)→Σ→σ (见 §5) |
| "单点深撞比多点擦边差" | Tunable(β=5)→Σ→σ 或加 1.5 floor |

---

## 4. Tunable 参数怎么选

```
贡献 = δ · σ(β · T(g))
         ↑       ↑
      最大幅度  过渡锐度
```

### β — "多快触发"

**公式：** `β ≈ 0.58 / log(1 + g_accept)`

其中 `g_accept` = 你认为"可以容忍"的最大违反量。

| 可接受违反 | β | 效果 |
|-----------|-----|------|
| ~10 | 0.2 | 非常软，大违反才感到 |
| ~1 | 0.8 | 标准过渡 |
| ~0.1 | 6 | 小违反即触发 |
| ~0.01 | 60 | 很锐 |
| ~0.001 | 500+ | 几乎即触即满 |

### δ — "最多扣几分"

内层内容经 σ 后输出约 [0, 0.7]。以此为参考：

| δ | 效果 |
|---|------|
| 0.1~0.5 | 轻偏好，几乎不改变排名 |
| 1.0~2.0 | 与目标同级竞争 |
| 3.0~5.0 | 强偏好，通常压倒目标 |

### 四档套餐

```python
# 轻微偏好
Chance(..., mode='tunable', priority=3, delta_soft=0.3, beta=0.2)

# 标准软约束
Chance(..., mode='tunable', priority=2, delta_soft=1.0, beta=1.0)

# 较强偏好
Chance(..., mode='tunable', priority=2, delta_soft=2.0, beta=5.0)

# 近似硬约束（光滑版）
Chance(..., mode='tunable', priority=1, delta_soft=3.0, beta=50)
```

---

## 5. 语义速查表

### 违反定义

| 语义 | 写法 |
|------|------|
| 违反量与穿透深度正比 | `pen_i`（有符号，直接用） |
| 满足就够了，不需要"更满足" | `max(0, pen_i)` |
| 每个违反点有最低代价 | `1.5 + pen_i`（带 floor） |

### 聚合方式

| 语义 | 聚合 | 含义 |
|------|------|------|
| 总违反量重要 | `Σ` | 全程擦边 > 单点碰撞 |
| 只看最坏一步 | `max` | 不允许任何一步出格 |
| 违反步数重要，深度不重要 | `count` | 广度 > 深度 |
| 允许 5% 步违规 | `Q₉₅` | 鲁棒性 |

### 完整语义 → 方案映射

```
"任何碰撞都不行"
  → Deterministic(pen>0), mode='hard'

"碰撞越深越差，没有上限"
  → 聚合 Σ, mode='soft'

"碰撞越深越差，但 10m 和 100m 一样糟"
  → 聚合 Σ, mode='tunable', β=1

"宁可单点深撞，也别全程擦边"
  → 每步 Tunable(β=1) → Σ → σ（大穿透被压扁）

"小擦边随便，别大穿透就行"
  → 每步 Tunable(β=5) → Σ → σ（放大单点大穿透）

"只是偏好，好跟踪可补偿小擦边"
  → mode='soft' 或 mode='tunable' with δ=1, β=1
    放在 Hard 层里面，不加 jnp.where

"安全就是安全，不需要奖励远离"
  → max(0, pen_i), 或 penalize_only_soft=True

"每步独立评估，不能有任何一步有大违反"
  → 聚合 max (不是 Σ), mode='tunable', β=5
```

---

## 6. MPC 用法

### 关键原则：build 一次，ctx 传动态信息

```python
# ✓ 正确
cost_fn = build(my_obj, constraints)    # ← 只在循环外调一次

for step in range(T_mpc):
    ctx = {
        'target': targets[step],        # 动态目标
        'obs_pos': obs_positions[step], # 动态障碍物
        'current_state': state,         # 当前状态
    }
    result = solver(..., fitness_fn_total=cost_fn, context=ctx)

# ✗ 错误 — 每次都 rebuild → 每次都 JIT 重编译
for step in range(T_mpc):
    cost_fn = build(my_obj, constraints)  # 不要！
    result = solver(...)
```

### 完整 MPC 示例

```python
from Constraintdealer.Constran import *

# 轨迹上每个点的碰撞约束
def obstacle_violation(z_flat, ctx):
    trajectory = rollout(z_flat, ctx['current_state'])
    penetration = safe_dist - compute_min_dists(trajectory, ctx['obs_pos'])
    return jnp.sum(jnp.where(penetration > 0, penetration, 0.0))

# 目标：跟踪 + 控制代价
def tracking_objective(z_flat, ctx):
    trajectory = rollout(z_flat, ctx['current_state'])
    return (jnp.sum(jnp.linalg.norm(trajectory - ctx['target'], axis=1)) * 2.0
            + jnp.linalg.norm(trajectory[-1] - ctx['target']) * 15.0
            + ...)  # 控制代价

# 两层嵌套
cost_fn = build(
    tracking_objective,
    [
        Deterministic(obstacle_violation,
                      mode='tunable', priority=1,
                      delta_soft=2.0, beta=1.0),
    ],
    k_inner=0.1,
)

# MPC 循环
cost_fn_jit = jax.jit(cost_fn)
for step in range(T_mpc):
    ctx = {'target': targets[step], 'obs_pos': obs[step],
           'current_state': state}
    mu, L, pi, v = mmog_igo_optimizer_mpc(
        key, T=200, dt=0.1, M=30, K=5, B=80, B0=40,
        dims=[H*2], T_0=50,
        fitness_fn_total=cost_fn_jit,
        initial_mu_k=mu, initial_L_inv_k=L, initial_v_k=v,
        context=ctx,
    )
```

### 如果求解器自己 JIT 了循环

MPCsolverM22 内部用 `lax.scan` 已经 JIT 了整个优化循环。此时 `build()` 返回的函数会被内联，外层 JIT 是冗余的。可以关掉：

```python
cost_fn = build(my_obj, constraints, jit_cost=False)
```

---

## 7. 多智能体用法

```python
from Constraintdealer.Constran import build_multi_agent

agent_fns = build_multi_agent({
    0: (  # Agent 0: 追踪目标，有避障约束
        lambda x, ctx: jnp.sum((x[:2] - ctx['t0'])**2),
        [Deterministic(lambda x, ctx: x[0] - 1.0, mode='hard', priority=1)]
    ),
    1: (  # Agent 1: 追踪目标，无约束
        lambda x, ctx: jnp.sum((x[2:] - ctx['t1'])**2),
        []
    ),
})

# 每个 agent 的 cost_fn 签名: (agent_idx, joint_x, ctx) -> scalar
result = mmog_igo_rne_blocks_solver(
    ..., fitness_fn_j=agent_fns[0], ...
)
```

---

## 8. 常见问题

### Q: Hard / Tunable / Soft 怎么选？

默认 Tunable 起手，调 β 和 δ。需要绝对优先级 → Hard。最简单场景 → Soft。

### Q: δ 不设会怎样？

`autodelta()` 自动赋值：最外层 Hard → 1.5，内层 Hard → 3.0。Tunable 默认 δ_soft=2.0, β=5.0。

### Q: 约束满足时会奖励吗？

默认**会**——`T(g)<0` 降低代价。这是设计意图：深度在可行域内部的解排名更高。如果不需要，设 `penalize_only_soft=True`。

### Q: 会被"深度满足"扰乱吗？

实际工程中不会——大多数约束有物理下界（`max(0, pen)` 或 `|h(x)|` 天然 ≥ 0）。如果约束可以无限负（罕见），用 Tunable 封顶。

### Q: 多点擦边 vs 单点深撞怎么控制？

三个旋钮：① floor（如 1.5+pen）② 聚合方式（Σ vs max）③ 模式参数（β, δ）。见 §5 速查表。

### Q: 能不能和原来的 exact penalty 混用？

可以。`build()` 返回的就是 `(x, ctx) -> scalar`，和其他 cost function 签名一致。可以在外层再加自己的处理。

### Q: 约束函数里能做复杂计算吗？

可以。`g_fn` 可以是任意 JAX 计算——rollout 轨迹、查表、神经网络推理。只要返回标量。

### Q: random.PRNGKey(0) 固定种子对吗？

Chance/DRO 约束用固定种子 `PRNGKey(0)`，保证同一次 `build()` 内 cost function 是确定的（否则 JAX 会重编译）。如需随机性，在 `ctx` 中传入 key 并在 `g_fn` 内显式使用。
