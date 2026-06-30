# 约束转化方法论：从约束优化到黑箱标量代价

## 1. 问题与设计目标

考虑约束优化问题 $\min_x f(x)$ s.t. $g_i(x) \le 0$，约束有字典序优先级。MGIGO 求解器只需一个标量代价做 `argsort` 排序。

设计目标：
1. **保序**：$x$ 越好 → $\mathcal{L}(x)$ 越小
2. **严格优先级**：外层约束违规必须比所有内层代价更大
3. **float32 可辨**：~7 位有效数字，每层 ≥ 1000 可辨识等级
4. **黑箱友好**：$f$ 可达 $10^{13}$ 甚至为负

---
## 2. 数学基础

### 2.1 饱和函数 $\sigma_k$

$$\sigma_k(x) = \frac{kx}{\sqrt{1 + (kx)^2}}, \quad \text{输出} \in (-1, 1)$$

$k$ 控制拐点：$\sigma_k(1/k) = 1/\sqrt{2} \approx 0.707$。所有约束层使用统一 $k = 0.7$。

### 2.2 多段 α 变换 $T_\alpha$

$$T_\alpha(g) = \text{sign}(g) \cdot T_{\text{target}}(|g|)$$

$T_{\text{target}}$ 通过分段的 log-linear 插值实现，三段式：
- **地板**（$|g| < g_0$）：$T = T_0$ 恒定，$g=0$ 时恰好 $T=0$
- **log 增长**（$g_0 \le |g| \le g_{\text{ceil}}$）：$T \propto \log|g|$
- **缓坡天花板**（$|g| > g_{\text{ceil}}$）：$T$ 缓慢增长，永不绝对平坦

三档标定：

| 模式 | $g_0$ (分辨率) | $T_0$ (地板) | $T_{\text{ceil}}$ (天花板) | $\alpha \approx 1/g_0$ |
|------|---------------|-------------|-------------------------|----------------------|
| SOFT | $10^{-2}$ | 0.003 | 4.5 | $10^2$ |
| TUNABLE | $10^{-4}$ | 0.02 | 6.0 | $10^4$ |
| HARD | $10^{-6}$ | 0.08 | 6.5 | $10^6$ |

$\alpha$ 隐含在 knot table 的分辨率中。地板 = 违规门槛，天花板 = 防 σ 饱和 + 缓坡防迷失。

### 2.3 自相似 σ 嵌套

$$\mathcal{L}(x) = \sqrt{2}\cdot\sigma_1\!\left(\cdots \sqrt{2}\cdot\sigma_1\!\left(\sigma_{k_{\text{in}}}\!\left(\frac{T_{\text{obj}}(f(x))}{\sqrt{2}^{\,n+1}}\right) + \Phi_1\right) + \cdots + \Phi_n\right)$$

- $\Phi_i = \text{baseline}_i + \max(0, T(g_i)) + \delta_i \cdot \sigma_1(\beta_i \cdot \max(0, T(g_i)))$
- $\text{baseline} \in \{0, 1.0, 2.0\}$：SOFT/TUNABLE/HARD
- $\Phi=0$ 时层透明：$\sqrt{2}\cdot\sigma_1(\sigma_1(\cdot/\sqrt{2})\cdot\sqrt{2}) = \sigma_1$，任意层数无衰减
- 输出 $\in (-\sqrt{2}, \sqrt{2}) \approx (-1.41, 1.41)$

### 2.4 优先级

- **小 priority** = 内层，被后续 $\sigma\cdot m$ 放大 → 影响大 → 安全约束
- **大 priority** = 外层，直接输出 → 影响小 → 舒适约束

---
## 3. 连续谱：SOFT → TUNABLE → HARD

$\alpha$（T 表）定分辨率和地板，$\beta, \delta$ 定硬度。三者搭配产生连续谱。

**ratio** = 外层约束门槛跳变 / 内层 $\sigma$ 收敛值：

$$\text{ratio} = \frac{\sigma_k(\Phi_{\text{threshold}} + \text{inner}_{\max}) - \sigma_k(\text{inner}_{\max})}{\sigma_k(\text{inner}_{\max})}$$

| preset | $\beta$ | $\delta$ | SOFT ratio | TUNABLE ratio | HARD ratio |
|--------|---------|----------|-----------|-------------|----------|
| mild | 0.15 | 0.5 | 0.00 | 0.03 | 0.13 |
| standard | 0.5 | 0.7 | 0.01 | 0.04 | 0.16 |
| firm | 1.0 | 0.75 | 0.01 | 0.05 | 0.20 |
| strong | 2.5 | 0.8 | 0.01 | 0.09 | 0.33 |
| nearhard | 8.0 | 1.0 | 0.04 | 0.25 | 0.73 |
| __hard__ | 20.0 | 1.2 | 0.11 | 0.58 | 1.05 |

- **SOFT**：ratio 0.00→0.11，永远竞争不过内层。约束是偏好，内层目标更优就让内层胜出。
- **TUNABLE**：ratio 0.03→0.58，从软到硬的连续过渡。
- **HARD**：ratio 0.13→1.05，从竞争到严格优先。__hard__ (ratio=1.05) 门槛刚好压过内层最大，现实中几乎不用。

---
## 4. 数值验证

### 4.1 N=20 层嵌套

$k=0.7$, TUNABLE standard, $g=0.1$：全部 20 层 $\ge 1\times$ float32 可辨，前 11 层 $\ge 1000\times$。

### 4.2 全违规范围

$g$ 从分辨率到天花板，cost 从 0.02 到 0.98，所有台阶 $\ge 10,000\times$ f32 可辨。

### 4.3 异质约束

碰撞（HARD，$1/(x+\varepsilon)$ 非线性）、速度（TUNABLE，线性）、能耗（SOFT，二次）三种不同物理量混合嵌套——优先级正确，非线性被 T 表天花板兜底。

---
## 5. 代码索引

| 函数/变量 | 位置 | 说明 |
|----------|------|------|
| `T_alpha()` | Constran.py:122 | 多段 log-like 变换 |
| `sigma_k()` | Constran.py:273 | 饱和函数 |
| `_assemble_nest()` | Constran.py:477 | 嵌套组装 |
| `build()` | Constran.py:516 | 公共 API |
| `TRANSFORM_SOFT` | Constran.py:45 | SOFT T 表 |
| `TRANSFORM_TUNABLE` | Constran.py:50 | TUNABLE T 表 |
| `TRANSFORM_HARD` | Constran.py:55 | HARD T 表 |
| `TUNE_PRESETS` | Constran.py:199 | β,δ 预设 |
| `NEAR_HARD_BETA` | Constran.py:197 | 硬约束默认 β=8.0 |
| `ConstraintSpec.baseline` | Constran.py:286 | 0=SOFT, 1=TUNABLE, 2=HARD |
| `Deterministic` | Constran.py:348 | 确定性约束 |
| `Chance` | Constran.py:353 | 机会约束 $\mathbb{P}(g\le 0)\ge 1-\alpha$ |
| `Robust` | Constran.py:366 | 鲁棒约束 $\max_{\xi\in\Xi} g(x,\xi)\le 0$ |
| `DRO` | Constran.py:373 | 分布鲁棒约束 |

## 6. B-spline 时域约束

B-spline 轨迹通过上百个时域采样点评估约束，等价于时域鲁棒。推荐 `aggregate='q95'`（95% 分位数），比 `max` 更鲁棒（容忍 5% 野点），比 `mean` 更严格（不会平均掉违规）。
