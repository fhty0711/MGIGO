# 约束转化方法论：从约束优化到黑箱标量代价

## 目录

1. [问题与设计目标](#1-问题与设计目标)
2. [数学基础](#2-数学基础)
3. [五种约束类型](#3-五种约束类型)
4. [优先级嵌套系统](#4-优先级嵌套系统)
5. [硬约束 vs 软约束 vs 可调约束](#5-硬约束-vs-软约束-vs-可调约束)
6. [通用 M 层框架](#6-通用-m-层框架)
7. [对数变换：消灭数值爆炸](#7-对数变换消灭数值爆炸)
8. [深层嵌套与 δ 的标度律](#8-深层嵌套与-δ-的标度律)
9. [完整操作手册](#9-完整操作手册)
10. [代码索引](#10-代码索引)

---

## 1. 问题与设计目标

### 1.1 我们要解决什么问题

考虑最一般的约束优化问题：

$$
\begin{aligned}
\min_{x \in \mathbb{R}^d} &\quad f(x) \\
\text{s.t.} &\quad g_i(x, \xi) \le 0, \quad i = 1,\dots,M
\end{aligned}
$$

其中约束 $g_i$ 可能涉及不确定性 $\xi$，并且约束之间有**字典序优先级**
（某些约束比另一些更重要，必须优先满足）。

MMOG-IGO 求解器只需要一个**标量代价** $\mathcal{L}(x)$ 来对候选解做
`argsort` 排序。我们需要一个转化方案，满足：

1. **保序**：$x$ 越好 → $\mathcal{L}(x)$ 越小
2. **严格优先级**：任何对高层约束的违反，必须比所有低层代价更大
3. **float32 数值稳定**：全动态范围内可区分（~7 位有效数字）
4. **黑箱友好**：不需要知道 $f_{\max}$，$f$ 可达 $10^8$ 甚至为负

### 1.2 两个核心原则

**原则一：多段 α 变换（T_alpha）。**

$$
\boxed{\mathcal{T}_\alpha(x) = \text{sign}(x) \cdot T_{\text{target}}(|x|)}
$$

$T_{\text{target}}$ 通过预设的分段表在对数空间插值。小 $|x|$ 时 $T_{\text{target}}$ 为常数
（地板），大 $|x|$ 时恢复对数行为。地板消灭了小违反的数值盲区——
$g=10^{-10}$ 的微小违反立刻被拉升到 $\mathcal{T} \approx 0.7$，求解器立即可感知。
详见 [§7](#7-多段-α-变换与预设表)。

纯 `log_transform`（$T(x)=\text{sign}(x)\log(1+|x|)$）仍可作为 `transform='log'` 使用。

**原则二：Tunable 连续谱。** 所有约束都是加性的，只有两种模式：

| 模式 | 机制 | β 控制什么 |
|------|------|-----------|
| **Tunable** | $\delta \cdot \sigma(\beta \cdot \mathcal{T}(g)) + \text{inner}$ | β: 0.1→软偏好, 1→标准, 100→硬, 1e7→纯硬 |
| **Soft** | $\mathcal{T}(g) + \text{inner}$ | 无参数, 最简 |

没有 `jnp.where`，没有分枝。`mode='hard'` 自动映射为 `Tunable + β=1e7`。

## 2. 数学基础

### 2.1 饱和函数 $\sigma_k$

$$
\boxed{\sigma_k(x) = \frac{kx}{\sqrt{1 + (kx)^2}}}
$$

**基本性质：**

1. **奇函数**：$\sigma_k(-x) = -\sigma_k(x)$。
2. **值域**：严格在 $(-1, 1)$ 内，与 $k$ 无关。
3. **$k$ 控制"拐点"**：$\sigma_k(1/k) = 1/\sqrt{2} \approx 0.707$。
4. **嵌套有精确闭式**：$\sigma_1^{(n)}(1) = 1/\sqrt{n+1}$（归纳法可证）。

### 2.2 多段 α 变换 $\mathcal{T}_\alpha$

$$
\boxed{\mathcal{T}_\alpha(x) = \text{sign}(x) \cdot T_{\text{target}}(|x|)}
$$

$T_{\text{target}}$ 通过预设的 $(g_i, T_i)$ 结点在对数空间线性插值：

- **小 $|x|$**：$T_{\text{target}}$ 为常数地板（如 0.7）——微小违反也能被感知
- **大 $|x|$**：$T_{\text{target}} \approx \log(|x|)$——恢复标准对数压缩

**预设表（约束）：**

| 预设 | 地板 $T(0^+)$ | 过渡 | 适用 |
|------|-------------|------|------|
| `'tight'` | 0.3 | 缓 | 近 log，但无盲区 |
| `'standard'` | 0.7 | 标准 | 默认 |
| `'sharp'` | 1.0 | 急 | 小违反立即可感 |
| `'wide'` | 0.3 | 宽 | 大范围线性，适合偏好 |
| `'log'` | 0 | — | 原版 log_transform，无地板 |

**预设表（目标）：** `'standard'`（默认），`'flat'`（更平），`'log'`。

**为什么需要地板？** 原版 $\log(1+|x|)$ 在小 $x$ 区 $\mathcal{T}(x) \approx x$。
$g=10^{-8}$ 经过 $\mathcal{T}$ 后仍是 $10^{-8}$，再经 $\sigma$ 嵌套后和 $g=0$ 无差别——
求解器感知不到微小违反。地板消灭了这个盲区。

### 2.3 嵌套级联的数学结构

单层 $\sigma$ 将 $(-\infty, \infty)$ 映射到 $(-1, 1)$。多层嵌套时：

$$
\sigma_1^{(n)}(1) = 1/\sqrt{n+1} \quad \text{（精确）}
$$

**证明（归纳法）：** $\sigma_1^{(n+1)}(1) = \sigma_1(1/\sqrt{n+1}) = 1/\sqrt{n+2}$。

这是整套 $\delta$ 选值理论的基石。

---

## 3. 五种约束类型

| # | 类型 | 数学形式 | g_raw 计算 |
|---|------|---------|-----------|
| 0 | 无约束 | $\min f(x)$ | — |
| 1 | **确定性** | $g(x) \le 0$ | $g(x)$ 直接 |
| 2 | **机会约束** | $P(g(x,\xi) \le 0) \ge 1-\alpha$ | $Q_{1-\alpha}(g(x,\xi))$ MC 分位数 |
| 3 | **鲁棒** | $g(x,\xi) \le 0\;\forall\xi\in\Xi$ | $\max_{\xi\in\Xi} g(x,\xi)$ lax.scan |
| 4 | **分布鲁棒** | $\inf_{P\in\mathcal{P}} P(g \le 0) \ge 1-\alpha$ | $\max_{P\in\mathcal{P}} Q_{1-\alpha}$ |

（各类型详细说明和代码模板见 [ConstranUser_README.md](ConstranUser_README.md)）

---

## 4. 优先级嵌套系统

### 4.1 嵌套公式

对于 $M$ 个约束，嵌套从内到外构建。每层都是**加性的**——没有 `jnp.where`：

$$
\boxed{\mathcal{L}_0 = \sigma_{k_{\text{inner}}}\!\big(\mathcal{T}_\alpha^{\text{obj}}(f(x))\big)}
$$

$$
\boxed{\mathcal{L}_i = \begin{cases}
\sigma_1\!\big(\delta_i \cdot \sigma(\beta_i \cdot \mathcal{T}_\alpha^i(g_i)) + \mathcal{L}_{i-1}\big) & \text{Tunable} \\[8pt]
\sigma_1\!\big(\mathcal{T}_\alpha^i(g_i) + \mathcal{L}_{i-1}\big) & \text{Soft}
\end{cases}}
$$

$$
\boxed{\mathcal{L}(x) = \mathcal{L}_M}
$$

其中 $i=1$ 是最内层约束，$i=M$ 是最外层。每层独立选择：
- **变换表** `transform`：$\mathcal{T}_\alpha^i$（tight/standard/sharp/wide/log）
- **Tunable 参数**：$(\beta_i, \delta_i)$，从软到硬的连续谱

### 4.2 Tunable 连续谱：从软到硬

**$\beta$ 是唯一的关键参数。** 所有约束都是 `contrib + inner` 的加性形式：

| β | 行为 | 违反 $g=0.001$ | 违反 $g=100$ | 预设名 |
|---|------|---------------|-------------|--------|
| 0.1 | 极软，大违反才触发 | 几乎无感 | 温和 | `'mild'` |
| 0.5 | 标准软 | 轻微 | 明显 | `'standard'` |
| 1.0 | 适中 | 可感 | 强 | `'firm'` |
| 5.0 | 较硬 | 立刻触发 | 满罚 | `'nearhard'` |
| 100 | 硬 | 即触即满 | 封顶 | — |
| $10^7$ | 纯硬（≈ 旧 Hard） | 和 $g=100$ 几乎一样 | 封顶 | `mode='hard'` |

**β ≥ 100 时过渡宽度 < $10^{-7}$，float32 不可分辨——等价于旧版的 Hard 模式。**
`mode='hard'` 自动映射为 `Tunable + β=1e7`，无需手写 β。

**$\delta$ 控制最大贡献幅度：** 内层内容 σ 后约 $[0, 0.7]$。$\delta=1\sim3$ 可与目标同级竞争或压倒。

**Tunable 预设套餐：**

| 预设 | β | δ | 适用 |
|------|---|---|------|
| `'mild'` | 0.1 | 1.0 | 舒适/效率偏好 |
| `'standard'` | 0.5 | 1.0 | 标准软约束 |
| `'firm'` | 1.0 | 2.0 | 重要偏好 |
| `'strong'` | 2.0 | 2.0 | 较强约束 |
| `'nearhard'` | 5.0 | 3.0 | 近似硬约束 |

### 4.3 位置 = 天然权重

即使 δ 和 β 相同，外层也比内层更有影响力——因为外层贡献不经内层 σ 压缩：

```
同样 g=100 的违反：
  L1 (最外层, 0层σ压缩): 贡献 ≈ 4.6 (原封不动)
  L2 (中层,   1层σ压缩): 贡献 ≈ 0.98
  L3 (最内层, 2层σ压缩): 贡献 ≈ 0.70
```

嵌套本身就提供了分层。Tunable 的 δ 和 β 在这个基础上做微调。

### 4.4 三层示例

```python
cost_fn = build(
    objective_fn,
    [
        Deterministic(static_viol, mode='hard', priority=1,    # → Tunable β=1e7
                      delta=1.5, transform='sharp'),
        Chance(ped_viol, mode='tunable', priority=2,
               tune_preset='firm', transform='standard'),
        Deterministic(comfort, mode='soft', priority=3,
                      transform='wide'),
    ],
    k_inner=0.1, obj_transform='standard',
)
```

没有 `jnp.where`，所有层都是连续可微的。

---

## 5. 通用 M 层框架

### 5.1 每层独立配置四张表

| 配置项 | 控制什么 | 预设 |
|--------|---------|------|
| `transform` | 违反感知基线 | tight/standard/sharp/wide/log |
| `tune_preset` 或 `(beta, delta_soft)` | 软硬程度 + 影响力 | mild/standard/firm/strong/nearhard |
| `mode` | Tunable 或 Soft | 'tunable'（默认）, 'soft', 或 'hard'（→ Tunable β=1e7） |
| `priority` | 嵌套顺序 | 1=最高（最外层） |

### 5.2 设计规则

```
OUTERMOST (最高优先级)
    ↑
    ├── 大 β (≥100) — 硬约束, 安全关键
    ├── 中 β (1~10)  — 重要但可调
    ├── 小 β (0.1~1) — 软偏好, 可妥协
    ├── Soft         — 最简, 无参数
    ↓
INNERMOST (目标函数, k=0.1)
```

---

## 6. 深层嵌套与 $\delta$

$\sigma_1^{(n)}(1) = 1/\sqrt{n+1}$ 的闭式保证了：**越深越容易分离，$\delta$ 可以越小。**

T_alpha 的地板进一步降低了 $\delta$ 需求：

| transform | T(0⁺) | 建议 δ (外层) | 建议 δ (内层) |
|-----------|-------|-------------|-------------|
| sharp | 1.0 | 0.1~0.3 | 0.3~0.5 |
| standard | 0.7 | 0.3~0.5 | 0.5~0.7 |
| tight | 0.3 | 0.6~0.8 | 0.8~1.0 |

**$\delta$ 太大** → 求解器不敢靠近约束边界，解太保守。
**$\delta$ 太小** → 层级分离脆弱。

---

## 7. 完整操作手册

**Step 1** — 列出约束，分配优先级，选择变换表和 Tunable 参数。

**Step 2** — 写 g_raw 计算函数（正值=违反，负值=满足）。

**Step 3** — 选 $k_{\text{inner}}$（默认 0.1）和 `obj_transform`（默认 'standard'）。

**Step 4** — 从内到外构建：

```python
from Constraintdealer.Constran import *

cost_fn = build(
    objective_fn,
    [
        Deterministic(g1, mode='hard', priority=1,
                      delta=1.5, transform='sharp'),
        Deterministic(g2, mode='tunable', priority=2,
                      tune_preset='firm', transform='standard'),
        Deterministic(g3, mode='soft', priority=3,
                      transform='wide'),
    ],
    k_inner=0.1,
)
```

**Step 5** — 验证区分度。

---

## 8. 代码索引

| 约束类型 | 函数 | 文件 |
|---------|------|------|
| 翻译器 (T_alpha) | `build()`, `Deterministic`, `Chance`, `Robust`, `DRO` | `Constran.py` |
| 翻译器 (log_transform) | `build()`, `Deterministic`, ... | `Constran.py` |
| 确定性测试 | `cost_sat_hierarchical*` | `Constraints.py` |
| 机会约束测试 | `cost_sat_hierarchical4-10` | `Constraints.py` |
| 鲁棒测试 | `cost_robust1-2` | `RobustConstraints.py` |
| 生产级 MPC | — | `Hybridsystemtest.py` |
| 用户手册 | — | `ConstranUser_README.md` |

---

## 附录：快速参考卡片

```python
# ─── 核心函数 ───
sigma_k(x, k=1.0)      # 饱和: kx/√(1+(kx)²), 输出 ∈ (-1,1)
T_alpha(x, knots_g, knots_T)  # 多段 α 变换

# ─── 约束声明 ───
Deterministic(g_fn, mode='tunable', priority=1,
              transform='standard', tune_preset='firm')
Deterministic(g_fn, mode='soft', priority=2,
              transform='wide')
Deterministic(g_fn, mode='hard', priority=1,    # → Tunable β=1e7
              delta=1.5, transform='sharp')

# ─── 预设速查 ───
# transform: tight, standard, sharp, wide, log
# tune_preset: mild(0.1,1.0), standard(0.5,1.0), firm(1.0,2.0),
#              strong(2.0,2.0), nearhard(5.0,3.0)
# obj_transform: standard, flat, log

# ─── 构建 ───
cost_fn = build(obj_fn, constraints, k_inner=0.1,
                obj_transform='standard')
```

β —— 控制"多快触发"
$$
\beta \approx \frac{0.58}{\log(1 + g_{\text{accept}})}
$$

$g_{\text{accept}}$ = 你认为"可以容忍"的最大违反量。

可接受违反	β	含义
~10	0.2	宽过渡，大违反才感到
~1	0.8	标准
~0.1	6	小违反即触发
~0.01	60	很锐
~0.001	500+	几乎即触即满
δ —— 控制"最多影响多少"
内层内容（目标 + 内层约束）经 σ 后输出量级约 $[0, 0.7]$。

δ	效果
0.1–0.5	轻偏好，几乎不改变排名
1.0–2.0	与目标同级竞争
3.0–5.0	强偏好，通常压倒目标
>5	近似硬约束
四档套餐
场景	δ	β	g≈0.001	g≈1	g≈10
轻偏好	0.3	0.2	~0	+0.04	+0.13
标准软	1.0	1.0	+0.001	+0.57	+0.92
较强	2.0	5.0	+0.01	+1.92	+1.99
近似硬	3.0	50	+0.15	+3.00	+3.00
Soft 等于 Tunable 取 β→0, δ→∞ 的极限——对数增长永不饱和。选 Tunable 就是给这个增长加了个上限。

总结一下就是：

数学层： 三种模式的单调性都正确，不会破坏排序。

工程层： 选哪个、参数怎么设，取决于你对"违反"的语义定义：

你想说	用
"任何违反都不可接受，不管多小"	Hard, jnp.where
"违反越大越差，没有上限"	Soft, T(g)
"违反有上限——超过某个程度后都一样糟糕"	Tunable, 调 δ 定上限
"小违反可以忍，但要平滑过渡"	Tunable, 调 β 定容忍区宽度
"多点小违反 vs 单点大违反，我要区分"	Soft 或 Tunable β<1
"违反只看最坏的那个点"	用 lax.scan(max) 而不是 sum

---

## 11. 动态 Gamma — 硬约束的敏感度保持

### 11.1 问题：硬约束在违反区内 cost 平坦

硬约束公式 `σ(T(g) + δ)` 中，δ=1.5 是固定的。当违反 g 很小时，
T(g) ≪ δ，δ 压倒了违反信号：

```
g=100: T(g)=4.6 → σ(6.1)=0.987
g=1:   T(g)=0.69 → σ(2.19)=0.910
g=0.1: T(g)=0.095 → σ(1.60)=0.847
g=0.01: T(g)=0.010 → σ(1.51)=0.834  ← δ 完全压制 T(g)
g=0.001:T(g)=0.001 → σ(1.50)=0.832  ← 和 g=0.01 几乎没区别
```

在 g∈[0.001, 0.1] 区间，cost 变化量仅 ~0.015。求解器在这段违反区内"瞎了"——分不清
微小违反和较大违反，找不到回可行域的方向。

### 11.2 解法：动态 γ = δ / T(g_best)

在违反项前加一个来自 ctx 的缩放因子：

```
σ( T(g) · γ + δ + inner )
```

每次求解器收敛后，从精英解（当前最好的解）上评估违反值 g_best，计算：

```
γ = δ / T(g_best)
```

这样 T(g_best)·γ = δ —— 最优违反的贡献恰好等于偏移量。无论违反多小。

```
g_best=100: γ=0.32 → T(0.1)*γ=0.03  (大违反时 γ 小，不放大)
g_best=0.1: γ=15.7 → T(0.1)*γ=1.50  (小违反时 γ 大，放大到和 δ 同级)
g_best=0.01: γ=150 → T(0.01)*γ=1.50  (始终把贡献拉到 δ 量级)
```

灵敏度对比（g_best=0.05）：

| g | 固定 γ=1 | 动态 γ=30.7 |
|---|----------|------------|
| 0.2 | σ=0.860（平坦）| σ=0.990（敏感）|
| 0.1 | σ=0.847（平坦）| σ=0.975（敏感）|
| 0.05 | σ=0.840（平坦）| σ=0.949（knee区）|
| 0.03 | σ=0.837（平坦）| σ=0.924（knee区）|
| Δσ | 0.023 | 0.066（3x 更敏感）|

### 11.3 API：build_dynamic + update_gamma_from_elite

```python
from Constraintdealer.Constran import build_dynamic, Deterministic, update_gamma_from_elite

# 构建 — 和 build() 完全相同的接口
constraints = [
    Deterministic(viol_obstacle, mode='hard', priority=1),
    Deterministic(viol_pedestrian, mode='hard', priority=2),
]
cost_fn = build_dynamic(objective_fn, constraints)

# MPC 循环
for step in range(T_mpc):
    # gamma 默认 1.0 — 安全，等同于 build()
    result = solver(..., fitness_fn_total=cost_fn, context=ctx)

    # 更新 gamma 为下一步准备
    elite_x = extract_best(result)
    ctx = update_gamma_from_elite(
        ctx,
        [(viol_obstacle, 1), (viol_pedestrian, 2)],  # (viol_fn, priority)
        elite_x,
    )
```

**关键属性**：
- gamma 默认 = 1.0（ctx 里没有 key 也正常工作，无需 calibrate）
- `update_gamma_from_elite` 每次从 scratch 重算（不依赖旧值，无需 reset）
- gamma 在 ctx 里 → 不触发 JAX 重编译
- Sigma 限幅：无论 γ 多大，cost ≤ 1.0

### 11.4 为什么单靠 log_transform 不够

log_transform 压缩了量级但保序——T(0.1)=0.095 > T(0.01)=0.010，是对的。
问题是 **δ=1.5 这个固定偏移量**。T(g)+δ 中 δ 的占比随着 g→0 趋于 100%，
"违反信号 / 総输入" 趋近于零。不是 log_transform 的问题，是加法结构的问题。
γ 修复了这个结构性问题。

### 11.5 与动态 k 的互补

| 参数 | 控制什么 | 防止什么 |
|------|---------|---------|
| 动态 k (objective) | objective 在 sigma 中的 knee 位置 | objective 量级漂移导致平坦 |
| 动态 γ (hard constraint) | 违反贡献相对于 δ 的比例 | δ 固定导致违反区平坦 |

两者都不触发 JAX 重编译。见 ObjectiveComposer_README.md §8。

### 11.6 软约束不需要动态 γ

软约束 `σ(T(g) + σ(T(f), k_obj))` 中没有 δ 偏移——T(g) 直接参与加性竞争。
只要 objective 的动态 k 保证 σ(T(f)) 不饱和，余量就足以接收 T(g)。
框架给了你全套积木，语义你自己定。

---

## 12. Constran — 多段 α 变换（无动态追踪）

### 12.1 动机：消除对求解器输出的依赖

动态 gamma 和动态 k 虽然有效，但依赖精英解追踪——存在三个问题：
1. **自循环**：精英解质量影响 k/gamma，k/gamma 反过来影响精英解
2. **噪音大**：精英解的 term 值在优化初期极不稳定
3. **评估成本高**：每次更新都要用校准函数重算轨迹

**根本解法：改造变换函数本身。** 如果 `T(x)` 的输出本身就落在 sigma 的敏感区，
就不需要事后修正。

### 12.2 分段 T 变换

标准对数变换 `T(x) = sign(x)·log(1+|x|)` 的问题是：小输入 T≈x（趋向 0，弱信号），
大输入 T≈log|x|（过压缩）。分段 T 直接定义目标输出值：

```
T(x) = sign(x) · T_target(|x|)
```

`T_target` 是在 log|x| 空间分段线性插值的函数，通过 knot 表定义。
小 x 时 T_target 保持常数 → **真正的地板**，不随 x→0 而衰减。
大 x 时 T_target ≈ log|x| → 退回标准对数行为。

| \|x\| 范围 | T_target | 含义 |
|---------|----------|------|
| ~1e-6 | 0.7 | 地板——极微小值保持 T≈0.7 |
| ~1e-4 | 0.8 | 地板区 |
| ~1e-2 | 0.9 | 过渡 |
| ~1e-1 | 1.0 | 接近标准 log |
| ~1e0 | 1.5 | 标准 log 区域 |
| ~1e1 | 2.5 | 轻度放大 |
| ~1e2 | 4.0 | 标准压缩 |
| ~1e4 | 7.0 | 大值压缩 |
| ~1e6 | 10.0 | 极大值压缩 |

knot 之间通过 log(|x|) 线性插值 T_target 实现平滑过渡。

### 12.3 效果

δ=1.0 配合多段 α，违反值从 1e-6 到 1e6 跨越 12 个数量级，
σ 始终稳定在 [0.71, 0.97]：

```
g=1e-06: T=0.010  σ=0.711  ✓ knee区
g=1e-04: T=0.049  σ=0.724  ✓ knee区
g=1e-02: T=0.405  σ=0.815  ✓ knee区
g=1e-01: T=0.693  σ=0.861  ✓ knee区
g=1e+00: T=0.693  σ=0.861  ✓ knee区
g=1e+01: T=1.792  σ=0.941  ✓ knee区
g=1e+02: T=3.045  σ=0.971  ✓ knee区
g=1e+04: T=6.215  σ=0.990  knee区
g=1e+06: T=9.904  σ=0.996  knee区
```

**零精英追踪。零动态更新。零 JAX 重编译风险。**

### 12.4 API（与 Constran 完全兼容）

```python
from Constraintdealer.Constran import build, Deterministic

constraints = [
    Deterministic(viol_obstacle, mode='hard', priority=1),
    Deterministic(viol_pedestrian, mode='hard', priority=2),
]
cost_fn = build(objective_fn, constraints)

# 直接给求解器——和 Constran.build() 完全相同的使用方式
result = mmog_igo_optimizer_mpc(..., fitness_fn_total=cost_fn, ...)
```

与 Constran.py 的区别：
- `T_alpha` 替换 `log_transform`
- δ 默认 1.0（替换 1.5）
- 无 `build_dynamic`、`update_gamma_from_elite` 等动态追踪 API
- `Deterministic`、`Chance`、`Robust`、`DRO` 接口完全相同

### 12.5 自定义 knot 表

```python
from Constraintdealer.Constran import T_alpha, sigma_k
import numpy as np

my_knots_g = np.array([1e-4, 1e0, 1e4])
my_knots_a = np.array([100,   1.0,  0.01])

# 在自定义 g_fn 中使用
def my_violation(x, ctx):
    g_raw = compute_violation(x)
    return T_alpha(g_raw, my_knots_g, my_knots_a)
```
