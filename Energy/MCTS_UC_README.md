# MCTS + IGO for Unit Commitment

## 优化问题（数学描述）

### 集合

| 符号 | 含义 | 来源 |
|------|------|------|
| $\mathcal{G}$ | 火电机组，$g \in \mathcal{G}$ | `Generator.py:GeneratorSet` |
| $\mathcal{B}$ | 电池储能，$b \in \mathcal{B}$ | `Energy_storage.py:Battery` |
| $\mathcal{W}$ | 风电机组，$w \in \mathcal{W}$ | `Energy_storage.py:WindTurbine` |
| $\mathcal{S}$ | 光伏机组，$s \in \mathcal{S}$ | `Energy_storage.py:SolarPV` |
| $\mathcal{H}$ | 氢储能，$h \in \mathcal{H}$ | `Energy_storage.py:HydrogenStorage` |
| $\mathcal{T}$ | 时间步，$t \in \{1, \dots, 24\}$ | |

### 决策变量（连续，无 01 变量）

对每种资产，求解器控制一个连续变量 $x_{*,t} \in \mathbb{R}$，物理出力通过模型解码：

**火电** — 增量式编码，爬坡由构造保证：
$$p_{g,t} = \text{clip}\!\left(p_{g,t-1} + r_g \cdot \tanh(x_{g,t}) + \underbrace{\mathbb{1}_{\text{off}\to\text{on}} \cdot \max(0,\; p_g^{\min}\!\cdot\!\sigma(\alpha x_{g,t}) - p_{g,t-1})}_{\text{启动跳跃}},\; 0,\; p_g^{\max}\right)$$

其中 $r_g$ = `GeneratorSet.ramp[g]`, $\alpha$ = `GeneratorSet.alpha`。

**电池** — 连续充放，无 01 变量，$\sigma$ 平滑混合充放效率：
$$p_{b,t} = P_b^{\max} \cdot \tanh(x_{b,t})$$
$$\eta_{b,t} = \phi \cdot \eta_b^d(|p|) + (1-\phi) \cdot \eta_b^c(|p|), \quad \phi = \sigma\!\left(\beta \cdot p_{b,t}\right)$$
$$E_{b,t+1} = E_{b,t} - \phi \cdot p_{b,t} \cdot \eta_b^d \cdot \Delta t - (1-\phi) \cdot p_{b,t} / (\eta_b^c + \epsilon) \cdot \Delta t$$

其中 $\eta_b^d(|p|) = \eta_b^{d0} - \eta_b^{\text{loss},d}\!\cdot\!(|p|/P_b^{\max})^2$ 是 C-rate 依赖效率，$\eta_b^c$ 类似。

**风电 / 光伏** — 物理功率曲线 + 弃电变量：
$$p_{w,t} = (1 - x_{w,t}^{\text{curt}}) \cdot f_w(v_t), \quad 0 \leq x_{w,t}^{\text{curt}} \leq 1$$
$$p_{s,t} = (1 - x_{s,t}^{\text{curt}}) \cdot f_s(G_t, T_t^{\text{amb}}), \quad 0 \leq x_{s,t}^{\text{curt}} \leq 1$$

$f_w(v)$ 是光滑分段功率曲线（切入→三次→额定→切出，$\sigma$ 过渡），$f_s(G, T)$ 是单二极管光伏模型。

**氢储能** — 电解槽 + 储罐 + 燃料电池，$\sigma$ 平滑模式切换：
$$p_{h,t}^{\text{grid}} = (1-\phi_h) \cdot p_{h,t}^{\text{fc}} - \phi_h \cdot p_{h,t}^{\text{ely}}, \quad \phi_h = \sigma(\beta_h x_{h,t})$$
$$M_{h,t+1}^{\text{H}_2} = M_{h,t}^{\text{H}_2} + \eta_h^e(p^{\text{ely}}) \cdot p_{h,t}^{\text{ely}} \cdot \Delta t - \frac{p_{h,t}^{\text{fc}}}{\eta_h^f(p^{\text{fc}})} \cdot \Delta t$$

### 目标函数

$$\min \sum_{t \in \mathcal{T}} \Bigg[ \underbrace{\sum_{g \in \mathcal{G}} \big(a_g p_{g,t}^2 + b_g p_{g,t} + c_g u_{g,t} + s_g \cdot \max(0, u_{g,t} - u_{g,t-1})\big)}_{\text{火电：燃料 + 启动}}$$

$$+ \underbrace{\sum_{b \in \mathcal{B}} \big(\text{charge\_cost}_b(p_{b,t}^{\text{chg}}) + \text{discharge\_cost}_b(p_{b,t}^{\text{dis}}) + \text{degradation}_b(|p_{b,t}|)\big)}_{\text{电池：充放损耗 + 老化}}$$

$$+ \underbrace{\sum_{w \in \mathcal{W}} \text{curtail\_cost}_w(x_{w,t}^{\text{curt}} \cdot p_{w,t}^{\text{avail}})}_{\text{弃风惩罚}} + \underbrace{\sum_{s \in \mathcal{S}} \text{curtail\_cost}_s(x_{s,t}^{\text{curt}} \cdot p_{s,t}^{\text{avail}})}_{\text{弃光惩罚}} \Bigg]$$

其中 $u_{g,t} = \mathbb{1}[p_{g,t} > p_g^{\min}/2]$（开关状态从出力自然涌现）。

### 约束层级（Constran 组装）

**P1 (hard)** — 物理极限：
$$p_{g,t} \leq p_g^{\max}, \quad E_{b,t} \in [E_b^{\min}, E_b^{\max}], \quad M_{h,t}^{\text{H}_2} \in [M_h^{\min}, M_h^{\max}]$$

**P2 (hard / Chance)** — 电力平衡：
$$P\!\left(\sum_{g} p_{g,t} u_{g,t} + \sum_{b} p_{b,t} + \sum_{w} p_{w,t} + \sum_{s} p_{s,t} + \sum_{h} p_{h,t}^{\text{grid}} \;\geq\; D_t + \xi_t^{\text{load}}\right) \geq 1-\alpha$$
其中 $\xi_t^{\text{load}} \sim \mathcal{N}(0, \sigma_D^2)$, $\xi_{t}^{\text{wind}}, \xi_{t}^{\text{solar}}$ 影响 $p_{w,t}, p_{s,t}$。

**P3 (tunable)** — 最小出力：
$$u_{g,t} \cdot p_g^{\min} \leq p_{g,t} \quad \text{（运行时不低于技术最小出力）}$$

**P4 (tunable)** — 火电过发：
$$\max\!\left(0,\; \sum_g p_{g,t} u_{g,t} + \text{batt\_dis}_t - \big(D_t - \text{renewable\_consumed}_t\big)\right)$$
（风光过剩可充电池或弃电，不计入过发）

**P5 (DRO 可选)** — 分布鲁棒：将 P2 替换为
$$\inf_{Q \in \mathcal{Q}} \; Q\!\left(\text{balance}_t \geq 0\right) \geq 1-\alpha$$
其中 $\mathcal{Q}$ 是模糊集 = {名义高斯, 高波动, 偏移均值, 厚尾 Student-t}。

### MCTS + IGO：双向反馈，非单向传递

单向不行——MCTS 如果错了，IGO 不能推翻。必须是闭环。

```
┌──────────────────────────────────────────────┐
│              MCTS（粗搜，离散）                 │
│  树搜索启停区间 | 无参数 Score                 │
│  输出: top-K 候选启停表 + "争议机组" 列表       │
└──────────────────┬───────────────────────────┘
                   │ K 个候选
                   ▼
┌──────────────────────────────────────────────┐
│              IGO（精搜 + 诊断，连续）            │
│  对每个候选，48~72 blocks 联合优化全 24h         │
│                                              │
│  诊断信号（非仅 total cost）:                   │
│   卡 p_max → 这台开少了                        │
│   趴 p_min → 这台开多了                        │
│   SoC 触底/顶 → 电池容量分配不对                │
│   过度弃电 → 风光没消纳好                       │
│                                              │
│  输出: cost + 诊断向量                         │
└──────────────────┬───────────────────────────┘
                   │ cost + 诊断
                   ▼
┌──────────────────────────────────────────────┐
│              MCTS 修正（回传）                  │
│  诊断 → 树更新:                                │
│   "卡 p_max" → ON 节点败率 ↑（应多开）          │
│   "趴 p_min" → ON 节点败率 ↑（应关）            │
│   cost 显著优于预期 → 沿路径胜率 ↑               │
│                                              │
│  MCTS 重新选 top-K → 下一轮                    │
└──────────────────────────────────────────────┘
```

三个关键：

1. **MCTS 输出争议列表，不只一个解**。Score 方差大的节点说明 UCB1 还没信心——这些机组需要在 IGO 里重点诊断。

2. **IGO 诊断是每台机独立的**。"Gen 5 出力卡在 p_max" 只影响 Gen 5 在树里对应节点的胜负统计，不污染其他机组的决策。诊断信号让 MCTS 下次优先重新探索这些争议机组。

3. **IGO 的 GMM 保持 K=3 分量**。一个接近 MCTS 建议（利用），两个自由探索不同出力分配。如果自由分量发现更好的 dispatch 模式（比如 Storage 充放策略不同），不影响启停——但影响 cost → 影响 MCTS 的胜负回传。

### 与传统 MILP 对比

| | Gurobi (MILP) | MCTS + IGO |
|------|-------------|-----------|
| 离散变量 | $24 \times |\mathcal{G}|$ 个 01 + Big-M 约束 | MCTS 树搜索启停区间 |
| 连续变量 | $24 \times (|\mathcal{G}| + |\mathcal{B}| + \cdots)$ | IGO 多块联合优化 |
| 成本函数 | 分段线性化 (SOS2) | 精确二次 + 非线性效率 |
| 储能 | 01 充/放变量 + Big-M | $\sigma$ 平滑混合，连续变量 |
| 不确定性 | 场景树 (5-10 个场景) | 100 MC 样本, Chance/DRO 天然 |
| 全局性 | B&B 证明最优 | MCTS 概率近似最优 |

## 动机

当前 IGO 将所有决策编码为连续变量，5-10 台机能找到好的可行解。但随着规模增大：

- 30 台机组 → 2^30 种启停组合 → 连续编码的隐式探索不足
- Gurobi 用 Branch & Bound 显式搜索离散空间
- IGO 缺的就是这个"分支"能力

MCTS 补上离散搜索，IGO 负责连续调度。

## 24h 联合优化（非马尔可夫场景）

MPC 递推在 5 台机上勉强凑效，但有根本局限：**启动成本、最小启停时间、储能跨日调度，本质上是非马尔可夫的**。前一小时的决策影响后一小时的成本，不能贪心递推。

Gurobi 的做法：一次性建 24×N 个 01 变量 + 24×N 个连续变量，MILP 全局解。MCTS 的对应做法：

```
树结构：每个叶节点 = 一台机组在 24h 内的完整启停表

  例：Gen 0 的 24h 启停 = [ON, ON, ON, ON, OFF, OFF, ..., ON, ON]
       ↑ 这是一个 "宏决策"：什么时候开，什么时候关

  树的每层展开一台机组 → 深度 = N_gen（不是 24×N）
  每个叶节点包含该机组完整的 24h 开关序列
```

**为什么深度是 N 而不是 24×N**？因为机组的启停有强时序约束——一旦决定了"开机区间"和"关机区间"，24 个时步的启停就确定了。树不需要逐小时展开，只需要决策每台机组的启停区间。

### 直接求解

**逐时递推不够**——启动成本、储能跨日套利、提前爬坡接需求跳升，本质上是非马尔可夫的。必须把所有决策量一次性暴露给求解器。

```
MCTS 提出启停表 → IGO 联合优化全 24h 所有变量

决策量 = 24 × (N_gen + N_batt + N_wind_curt + N_solar_curt + N_h2 + ...)
       = 24 × (30 + 5 + 1 + 1 + 1) = 912 维

多块求解器: M≈48 blocks, 每块 D=8~10

  Block 0:  hour 0, baseload gens (D=7)
  Block 1:  hour 0, mid+peak gens (D=7)
  Block 2:  hour 0, 储能+风光弃电 (D=6)
  Block 3:  hour 1, baseload gens (D=7)
  ...
  Block 47: hour 23, 储能+风光弃电 (D=6)
```

**所有变量在一轮 IGO 里一次性联合优化**。爬坡由增量式编码满足，启停由 MCTS 给出，但出力值全开放。48 blocks × 300 iter × 300 samples ≈ 几十秒——只在层2/3调用几十次。

**三层评估在 24h 场景下**：

| 层 | 24h 场景 | 计算 | 调用 |
|----|---------|------|------|
| 层1 启发式 | 按 merit order 估算 24h 成本 | O(N×H)，μs | 数千 |
| 层2 轻量 IGO | 全 24h, M=48 blocks, T=100, B=30, MC=30 | ~3-5s | 30-50 次 |
| 层3 精评 IGO | 全 24h, M=48 blocks, T=300, B=200, MC=100, +Chance/DRO | ~15-30s | 3-5 次 |

**与 Gurobi 24h MILP 的对应**：

| | Gurobi | MCTS + IGO |
|------|--------|-----------|
| 离散变量 | 24×N 个 01 变量 | MCTS 树搜索启停区间 |
| 连续变量 | 24×N 个 | 48 blocks × 912 维, IGO 联合优化 |
| 时序约束 | Big-M + 线性不等式 | 增量式编码（爬坡）+ 启停区间（最小启停）|
| 非凸成本 | 必须线性化 (SOS2) | 直接塞进去 |
| 全局性 | B&B 证明最优 | MCTS 概率近似最优 |

## 求解流程（双向版）

### Step 1 — 分块

50 台机按类型分成 3 块，每块 MCTS 独立建树：

```python
baseload  = gens[0:15]    # 大容量、便宜
mid_merit = gens[15:35]   # 中等
peakers   = gens[35:50]   # 小、贵、只在高峰开
```

### Step 2 — MCTS 粗搜

每块内部建树。每个节点 = 一台机组的 `ON` / `OFF` 二选一。

**节点评分（无参数）**

```math
\text{Score}(i) = 
\begin{cases}
+\infty, & N_i = 0 \\[6pt]
\dfrac{W_i}{N_i} + \sqrt{\dfrac{2 \ln N_{\text{parent}}}{N_i}}, & N_i > 0
\end{cases}
```

Selection 按 Score 选。Expansion 产生 ON/OFF 分支。

**层1 启发式**：MCTS 遍历数千次，每次用 `fuel_cost(merit_order)` 微秒级评估。不碰 IGO。

**MCTS 输出**：每块 top-K 候选启停表 + 争议机组列表（Score 方差大的节点 → 这些机组需要在 IGO 里重点检查）。

### Step 3 — IGO 精搜 + 诊断

把三块的 top-K 候选组装成全 24h 启停表。IGO 联合优化全 24h 连续变量（48~72 blocks, D≤10）。

IGO 不只返回 cost，还返回**每台机的诊断信号**：

| 信号 | 含义 | 传给 MCTS |
|------|------|----------|
| 出力卡 p_max | 开少了，出力到顶 | 该 ON 节点败率 ↑ |
| 出力趴 p_min | 开多了，长期低负荷 | 该 ON 节点败率 ↑ |
| SoC 反复触底/顶 | 电池容量配错 | 下一轮调整电池块 |
| 过度弃电 | 风光消纳不足 | 更新风电/光伏的 curtail 偏好 |

### Step 4 — MCTS 修正

诊断信号回传 MCTS，更新树统计量。MCTS 重新选 top-K → 进入下一轮。

**诊断如何影响胜负**：不只是 ON vs OFF 比 cost。一台机 ON 但出力卡 p_max——说明 IGO 在抱怨"你只开一台不够，我需要更多"。MCTS 把这个信号转成：*在该节点的父节点上，考虑再多开一台*。本质上是在树上向上回传"容量不足"的信号。

### Step 5 — 收敛

当 MCTS 的 top-K 候选连续两轮 cost 变化 < 1%，且争议机组列表为空（所有机组 Score 方差都小），停止。

**各层级调用统计**：

| | 层1 启发式 | 层2 轻量 IGO | 层3 精评 IGO |
|------|----------|------------|------------|
| 触发条件 | 每次 MCTS 遍历 | 候选进入 top-K | 最终精评 |
| 计算 | O(N)，μs | 48 blocks × T=100, B=30 | 48 blocks × T=300, B=200 |
| 耗时 | μs | 3-5s | 15-30s |
| 调用次数 | 数千 | 30-50 次 | 3-5 次 |
| 输出 | cost 估计 | cost + 初步诊断 | cost + 完整诊断向量 |

### 第三步：跨块协调

Master 收集三块的最优启停方案，组装成 50 台机的完整启停表。校验总出力 ≥ demand（含 Chance/DRO）。如果不足，Master 通知相关块增加开机数，重新精评。

### 第四步：IGO 精细调度

固定启停方案，IGO 优化所有连续变量（出力、储能充放）同时满足 Chance/DRO 约束。复用已有的 `stochastic_uc.py` + Constran。

## 参数

| 参数 | 10 台 (1 block) | 30 台 (2 blocks) | 50 台 (3 blocks) |
|------|----------------|-----------------|-----------------|
| 块数 × 块深 | 1×10 | 2×8 | 3×6 |
| 层1 启发式 | ~5000 次 | ~8000 次 | ~12000 次 |
| 层2 轻量 IGO | 20 次 | 40 次 | 60 次 |
| 层3 精评 IGO | 5 次 | 8 次 | 10 次 |
| 层2 IGO 参数 | T=100 B=30 K=1 MC=30 | 同左 | 同左 |
| 层3 IGO 参数 | T=300 B=200 K=3 MC=100 | 同左 | 同左 |
| 预计总时间 | ~3s | ~10s | ~20s |
| 预计 VRAM | 2-3 GB | 3-4 GB | 4-5 GB |
| UCB1 C | 0.3 | 0.3 | 0.3 |

## 与纯 IGO / Gurobi 对比

| 维度 | 纯 IGO | Gurobi (MILP) | MCTS + IGO |
|------|--------|--------------|-----------|
| 离散搜索 | 连续编码隐式 | B&B 枚举 | UCB1 + 分块 |
| 全局性 | 局部最优 | 全局（简化模型） | 概率近似全局（真实模型） |
| 建模 | 真实物理 | 必须线性化 | 真实物理 |
| 不确定性 | Chance/DRO 天然 | 场景树削减 | Chance/DRO（层3） |
| 50 gens | 2.5s | 30-120s | ~20s |
| IGO 调用 | 5 rounds | — | ~70 次 (85% 轻量) |

## 文件位置

```
Energy/
  mcts_uc.py           ← MCTS + IGO 实现（待写）
  stochastic_uc.py     ← IGO + Constran（复用为 Rollout）
  UC_README.md
  MCTS_UC_README.md    ← 本文档
```

## 替代/补充方案

- **Beam Search**：每层保留 K 个最优，比 MCTS 确定，需要好 heuristic
- **Simulated Annealing**：扰动当前方案 + IGO 评估邻域，更简单
- **Genetic Algorithm**：启停方案做交叉/变异，IGO 评估 fitness

## RTX 5070 显存

轻量架构下 99% 的 MCTS 节点是 O(N) 启发式，不碰 GPU。显存只在层2/3 IGO 时使用。

| 组件 | 占用 | 备注 |
|------|------|------|
| JAX 预分配 | 可控 | `XLA_PYTHON_CLIENT_ALLOCATOR=platform` 按需取 |
| JIT 编译缓存 | 1-2 GB | 单进程共用，不复制 |
| 层3 MC vmap | 0.5-1 GB | 100 MC 样本的 trace |
| 层2 轻量 IGO | ~500 MB/次 | 串行跑，用后释放 |

**推荐配置**：单进程 + `XLA_PYTHON_CLIENT_ALLOCATOR=platform`，VRAM 3-5 GB，RTX 5070 充裕。
