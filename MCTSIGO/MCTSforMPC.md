# 引导树 + IGO：滚动时域 MPC 中的离散-连续联合求解

> **这不是真正的 MCTS。** 我们借用 MCTS 的**形式**（树结构、UCB 评分、Selection/Backprop），但底层模型根本不同：树是静态预建的索引结构，所有离散组合路径从一开始就存在，Selection 是在有限路径集中选择下一条要评估的路径，Backprop 是把评估结果登记到路径经过的节点上。不存在"节点价值传播"——所有 cost（包括离散变量的）都是非马尔可夫的，无法在树结构上做加法分解。详细论证见 §10。

> MCTS.md §6.3 对实时 MPC 的处理有根本缺陷：将"拓扑候选少 → 串行评估也不会太慢"视为优势，忽略了**时序耦合和非马尔可夫代价**导致的串行评估不可行。本文档专门阐述引导树 + IGO 在 MPC 场景下的正确求解形式。

---

## 1. 为什么 MCTS.md 的 MPC 方案不对

### 1.1 回顾：MCTS.md §6.3 的做法

```
For 每个 MPC 时步 t:
  ① MCTS 搜索当前时步的拓扑 s_t（离散节点少，几百次层1遍历，μs）
  ② 选出 top-2 拓扑候选: s_t^A, s_t^B
  ③ 对每个候选串行跑 IGO: IGO(s_t^A) → cost_A, IGO(s_t^B) → cost_B
  ④ 选 cost 较低的，执行对应的 x_t
  ⑤ Warm Start: GMM + 树统计量继承到 t+1
```

### 1.2 问题一：时序耦合让串行评估失去意义

非马尔可夫代价——启动费、最小启停时间、储能 SoC 跨时步继承——意味着 **$s_t$ 的好坏不是 $s_t$ 自己决定的，而是由 $s_t$ 与 $s_{t-1}$ 的过渡、以及 $s_t$ 对 $t+1, t+2, \dots$ 的影响共同决定的**。

串行评估 $s_t^A$ 和 $s_t^B$ 时，IGO 内部必须对未来时步 $t+1 \dots t+H$ 的拓扑做假设。如果假设不当，cost 评估就不准——这不是 IGO 迭代数的问题，是**评估本身就带着错误前提**。

具体来说，以两个典型场景为例：

**能源 UC**：评估 $s_t^A$（Gen 3 开机）时，IGO 优化 $t \dots t+H$ 的连续参数。但 $t+1$ 时 Gen 3 能不能关？如果最小开机时间是 4h，那 $t+1, t+2, t+3$ 都不能关——这个约束必须在 IGO 里编码。但 IGO 不知道你下一时步的 MCTS 会选什么拓扑——**当前时步的候选评估，依赖一个尚未做出的未来离散决策**。

**自动驾驶**：评估 $s_t^A$（路口左转）vs $s_t^B$（路口直行）。左转后的车道不同、允许的速度范围不同、与他车的交互模式不同。$t+2$ 时是否能变道、$t+3$ 时是否能加速到目标速度——这些完全取决于 $s_t$ 的选择。如果 IGO 在评估左转候选时假设未来拓扑是"保持左转后车道"，而实际下一时步 MCTS 可能选择"立刻变道"，则 cost 评估失准。**更致命的是**：左转的轨迹曲率大、通过速度低，而直行轨迹平滑、速度高——两个拓扑候选对应的连续参数（速度曲线、转向序列）完全不同，串行 IGO 各自优化时没有任何信息复用。

**串行评估的根本矛盾**：评估 $s_t$ 需要知道 $s_{t+1}$，但 $s_{t+1}$ 是下一时步 MCTS 才决定的。

### 1.3 问题二：即使候选少，串行 IGO 也是浪费

MCTS.md 说"离散节点少 → MCTS 快 → 省下的时间给 IGO"。但忽略了：

- 两个候选各跑一次 IGO（$T=150$）→ 两个独立的自然梯度流从几乎相同的起点收敛到两个不同的局部最优。其中大量的中间迭代在重复探索相似的区域
- 如果两个拓扑候选的差异只是"Gen 3 开 vs 关"，那么连续参数中 Gen 1, 2, 4, 5 的出力分配、储能充放策略在两次 IGO 中大概率相似——**这些计算被重复做了两遍**
- **自动驾驶更严重**：左转 vs 直行的速度曲线完全不同（左转需要减速、曲率大；直行匀速、曲率小），但两者仍共享大量底层计算——道路几何、他车预测、安全约束评估。串行跑两次 IGO 意味着两次独立采样、两次独立 roll-out、两次独立约束评估——这些本该在单次多模态优化中分摊

正确做法不是"串行跑两次 IGO"，而是让 **IGO 在多模态分布下一次同时探索多个拓扑对应的连续参数区域**——$K>1$ 的 GMM 分量本身就具备这个能力。

---

## 2. 正确形式：时域引导树 + 多模态 IGO

### 2.1 核心思路

不把引导树和 IGO 的界面画在"引导树决定 $s_t$，IGO 优化 $x_t$"上——这个界面在非马尔可夫场景下不存在。

正确的界面是：

> **引导树在时域 $[t, t+H]$ 内的预建拓扑路径集中选择下一条要评估的路径 $\mathbf{s}_{t:t+H}$，IGO 联合优化该路径对应的全时域连续参数 $\mathbf{x}_{t:t+H}$。只执行 $s_t, x_t$；剩余的 $s_{t+1:t+H}, x_{t+1:t+H}$ 丢弃。**

这就是标准 MPC 的"预测整个时域，只执行第一步"——但应用到离散+连续二分结构上。引导树不是逐步构建的：时域内所有可行的拓扑序列 $\mathbf{s}_{t:t+H}$ 通过宏决策预先枚举（见 §4.1），树只是组织这些路径的索引结构。Selection 走完一条路径 = 选中一个完整的时域拓扑序列，送入 IGO 评估。

### 2.2 索引森林的结构

**关键设计**：索引森林是静态预建的——不是在搜索中动态扩展的。所有路径在建模时已存在。它不是一棵树，而是一个**森林**：第一层就有多个独立 root，每个 root 对应一类离散决策维度。

```
索引森林（2 机组 + 1 储能, 3 个 root, 无需时序排序）

  Root A: 机组启停          Root B: 储能模式         Root C: 联络线
  Gen0: ON/OFF             Storage: CHARGE/DIS/IDLE   Tie: IMPORT/EXPORT
  Gen1: ON/OFF

  完整路径 = Root A 的一条叶路径 + Root B 的一条叶路径 + Root C 的一条叶路径
          = (Gen0=ON, Gen1=OFF) + (Storage=CHARGE) + (Tie=IMPORT)
```

**每个 root 内部是一棵独立的索引树**，叶节点包含该 root 对应的部分策略。Forest 的一条完整路径 = 跨所有 root 的叶节点组合，得到一个完整离散策略 $\mathbf{s}$。

**为什么是森林而非单棵树**：
- 不同类别的决策之间没有天然的顺序关系（机组启停 vs 储能模式，谁先谁后？）
- 单棵树需要人为指定 root → depth 顺序，这个顺序影响 UCT 的搜索行为但不反映问题结构
- 森林让每类决策独立被 UCT 搜索，组合 cost 作为联合信号回传给所有 root
- 每个 root 可以有不同的分支因子、不同的 K 值、不同的 UCT 参数

**Root 的粒度：从粗到细**。每个 root 可以是：

```
粗粒度 root（agent 级）:
  Agent A: [保守策略, 激进策略, 跟随策略]    ← 3 个选项
  Agent B: [全开, 半开, 全关]

细粒度 root（参数级）:
  Gen0: [ON, OFF]
  Gen1: [ON, PARTIAL, OFF]                  ← 粒度细化
  Storage: [CHARGE, DISCHARGE, IDLE]

超细粒度 root（微调级）:
  Gen1_partial_level: [30%, 50%, 70%]       ← 在 PARTIAL 下进一步细化
```

粗 root 先做"方向性"决策（哪个 agent 用什么策略），细 root 再做"参数性"决策（具体哪台机怎么开）。粗 root 的分支少、收敛快；细 root 的分支多、依赖粗 root 的上下文。**上下层之间不通过树结构连接**——它们只是恰好同时被评估，通过联合 cost 隐式协调。

**关键：层宽而总深较浅**。森林组织为**分层结构**——每层代表一个粒度级别，层内可以有多个 root（宽），总层数少（浅）。

```
Layer 0 (粗粒度):  [Root: 竞价策略]                         ← 1 root, 方向性
Layer 1 (中粒度):  [Root: 机组A, Root: 机组B, Root: 储能]    ← 3 roots, 分类
Layer 2 (细粒度):  [Root: 联络线, Root: 备用容量]            ← 2 roots, 辅助

总深度 = 3 层, 总宽度 = 6 roots
完整路径 = Layer0 的选择 × Layer1 的选择 × Layer2 的选择
```

每层是一组独立演化的 root。同层 root 之间无顺序依赖——它们代表同一粒度级别上的不同决策类别。层与层之间是粗→细的渐进关系。

这不是"把一棵树拆成几棵"——而是**离散组合搜索根本就不是树状的**。森林从粗到细、从少到多逐层生长：
- 加层（纵向）：在已有层下方加入更细粒度的新层 → 决策变得更精细
- 加 root（横向）：在已有层内加入新的决策类别 → 决策覆盖面更广
- 旧层/旧 root 可裁剪（决策维度过期）
- 每 root 内部可加深（分支变多：ON/OFF → ON/PARTIAL/OFF）
- 所有 root 的 UCT 状态（Q, n）对其他 root 透明——它们只通过联合 cost 互相感知

这就是"一层一层加 agent 或者加策略的精细度"：森林的层数 = 当前活跃的决策维度/agent 数，而不是时序步数。

**森林的 UCT Selection**：每个 root 独立运行 UCT，从各自的根走到各自的叶。所有 root 的叶节点策略合并为完整策略 $\mathbf{s}$，一次性送入评估器。评估结果（cost）**同时回传给所有 root**，更新各自路径上节点的 Q。

这意味着：如果一个 root 选了烂分支，所有人都收到差评——所有 root 共同承担联合决策的后果。这正是非马尔可夫 cost 的要求：cost 不可分解到单个 root，只能整体评估、整体回传。

**JAX 编译注意事项**：Forest 是纯 Python 层——只改变 `for root in roots` 的循环次数，不改变 JAX 计算图。`igo_evaluate` 的输入维度 $D$ 不变则无需重编译。建议**预注册所有可能的 agent 参数配置**：

```python
AGENT_CONFIGS = {
    "GenCo_small":  {"D": 5,  "K": 3, "B": 80},
    "GenCo_large":  {"D": 15, "K": 5, "B": 200},
    "Storage_op":   {"D": 3,  "K": 3, "B": 60},
}
```

Forest 动态加 root 时只需指定 `config_key`——只要 key 在预注册列表中，IGO 的 JIT 编译就是固定的。新 agent 类型的加入 = 新 config 注册 + 一次提前编译，不影响运行时。

**自动驾驶的树结构**：

```
时域索引树（H=8, 100ms/步, 总时域 0.8s）

拓扑单元: 路口选择 + 变道决策

        t       t+1     t+2     t+3     t+4     t+5     t+6     t+7
路口:  ─── 直行 ─── 直行 ─── 直行 ─── 左转?─── 左转道── 左转道── 左转道
变道:  ─── 保持 ─── 保持 ─── 右变?─── 右道 ─── 保持 ─── 保持 ─── 保持

变更事件（≤3个）:
  Event 1: 在 t+2 右变道 → 分支: 变 / 不变
  Event 2: 在 t+3 左转   → 分支: 左转 / 直行
  Event 3: 在 t+5 加速超越 → 分支: 超车 / 跟车
```

**驾驶拓扑的宏决策形式**：

每个离散决策不是"在 $\tau$ 时刻做什么"，而是**一个有时域跨度的 maneuver**：

| 宏决策 | 编码 | 自由参数（留给 IGO） |
|--------|------|---------------------|
| 变道 | $\text{LANE\_CHANGE}(\text{target}, \tau_{\text{start}}, \tau_{\text{end}})$ | 具体变道轨迹的横向加速度剖面 |
| 路口转向 | $\text{TURN}(\text{direction})$ at 路口位置 | 转向速度、曲率剖面 |
| 超车 | $\text{OVERTAKE}(\tau_{\text{start}})$ | 超车时机、加速幅度 |
| 让行 | $\text{YIELD}(\text{to\_agent\_id})$ | 减速时机、让行距离 |

索引树深度 = 时域内 maneuver 事件数。在 0.8s 时域内，一辆车最多执行 1-2 个完整 maneuver。树极浅。

### 2.3 引导树 + 多模态 IGO 的闭环

```
For 每个 MPC 时步 t:
  
  ① 引导树路径采样（层1）
     索引树覆盖时域 [t, t+H] 内所有可行的拓扑序列
     每个叶节点（路径）= 一个完整的拓扑序列候选 s_{t:t+H}
     层1 用 Constran 成本快评（确定性，μs，全时域），遍历数百次
     Selection 通过 UCT 评分（§8）选择下一条要评估的路径
     输出: top-K 个时域拓扑序列候选（累计评估次数最多的 K 条路径）
  
  ② 多模态 IGO 联合评估（层3）
     不是对 K 个候选各跑一次 IGO
     而是跑一次 IGO，但 GMM 保持 K>1 分量，每分量探索不同拓扑候选对应的连续参数区域
     
     关键：IGO 优化的是全时域连续参数 x_{t:t+H}
     成本 = Σ_{τ=t}^{t+H} cost(s_τ, x_τ, s_{τ-1}, x_{τ-1})
     非马尔可夫项（启动费、SoC 继承、爬坡）在时域求和内部自然处理
     
     诊断信号（每分量独立）：
       - 某分量 cost 显著低 → 对应拓扑序列更优
       - 某分量出力频繁触边界 → 对应拓扑序列有容量问题
       - SoC/储能边界 → 时域内的容量配置不当
       
       驾驶场景的诊断信号：
       - 加速度频繁触上下限 → 拓扑太激进（如频繁超车）或太保守（如不敢加速）
       - 安全距离频繁触边界 → 拓扑风险过高，应考虑更保守的 maneuver
       - 横向加速度超舒适阈值 → 变道/转向的 maneuver 安排太紧，需放宽时域窗口
       - 进度显著落后 → 拓扑过于保守（一直跟车不超），应尝试更主动的 maneuver
       - 曲率不连续（jerk 频繁触限）→ 连续参数优化不充分，需增加 IGO 迭代或调整约束 β
  
  ③ 选择 + 执行
     选 cost 最低的 GMM 分量对应的拓扑 s_t 和连续参数 x_t
     执行 s_t, x_t
     诊断信号存入跨时步记忆
  
  ④ 时域滑窗 Warm Start
     时域从 [t, t+H] 滑到 [t+1, t+H+1]
     GMM: K 个分量的 (μ, S, π) 全部继承（滑窗后最优解的大部分仍然有效）
     索引树: 裁剪掉 t 对应的那层，t+H+1 对应的新层从先验初始化
     样本复用: 上一时步的精英样本如果与新拓扑序列兼容，加权复用
```

### 2.4 为什么多模态 IGO 一次运行优于串行 K 次

| | 串行 K 次 IGO | 多模态单次 IGO |
|---|---|---|
| **计算量** | $K \times T \times B$ 次 cost 评估 | $1 \times T \times B$ 次 cost 评估（B 个样本分散到 K 个分量） |
| **信息共享** | 无（两次独立优化） | 分量间通过 Mixture 分母共享梯度信息 |
| **探索覆盖** | 各候选独立收敛，可能遗漏中间模式 | K 个分量同时覆盖 K 个模态 + 交叉区域 |
| **非马尔可夫处理** | 每个候选单独做时域求和 | 同一次 IGO 内所有样本做全时域求和 |
| **适合场景** | K 个拓扑差异巨大，连续参数完全不可比 | K 个拓扑差异局部（几台机不同），连续参数大量重叠 |

对于 MPC，拓扑候选之间的差异通常是局部的（"这台机开还是关"、"走左边还是右边"），连续参数大量重叠——多模态 IGO 天然更高效。

### 2.5 求解器接口：Forest 如何调用 IGO 和博弈求解器

Forest 不直接调用 IGO——它调用用户注入的 **`igo_evaluate(strategy, mode) → float`**。这个函数内部可以是任何东西。

**三种保真度对应三种调用方式**：

```
igo_evaluate(strategy, mode):

  mode='none':
    → Constran 快评，x=x₀（固定），μs 级
    → 只问"这个拓扑结构上可行吗？"

  mode='light':
    → IGO 轻量优化，T=100, B=30
    → 问"给定这个拓扑，优化后的 cost 大概多少？"
    → 附带诊断信号：哪些机组频繁触边界？储能 SoC 是否到极限？

  mode='full':
    → IGO 精评，T=300, B=200, MC=100
    → 最终裁决——这个拓扑到底好不好？
```

**单智能体 IGO 注入方式**：

```python
# Constran 构建 cost
cost_fn = build(objective, constraints)

# 包装为 forest 需要的评估器
def my_evaluate(strategy, mode):
    ctx = {'strategy': strategy, ...}
    if mode == 'none':
        return float(cost_fn(x0, ctx))
    elif mode == 'light':
        result = mmog_igo_optimizer_mpc(
            key, T=100, B=30, ...,
            fitness_fn_total=cost_fn, context=ctx)
        return float(result.best_cost)
    elif mode == 'full':
        result = mmog_igo_optimizer_mpc(
            key, T=300, B=200, MC=100, ...,
            fitness_fn_total=cost_fn, context=ctx)
        return float(result.best_cost)

# 注入森林
solver = build_guide_tree_solver(problem, igo_solver=my_evaluate)
solver.solve()
```

**多智能体 RNE 注入方式**：每个 agent 有自己的 forest，联合 evaluate 调用 RNE 求解器。

```python
forest_A = build_guide_tree_solver(problem_A, igo_solver=joint_eval_A)
forest_B = build_guide_tree_solver(problem_B, igo_solver=joint_eval_B)

def joint_evaluate(strategy_A, strategy_B, mode):
    if mode == 'full':
        result = mmog_igo_rne_blocks_solver(
            M_agent=2,
            fitness_fn_j=agent_cost,
            context={'topology': (strategy_A, strategy_B)},
            ...)
        return result.nash_costs  # (cost_A, cost_B)
```

见 §9 多树异构的完整讨论。

### 2.6 高频 MPC 场景：Warm Start + 反驳/接受

高频场景下，森林不是每时步从零开始——上一轮的 Q 统计量和 GMM 参数作为下一轮的初始值。

```
时步 t ──────────────────────────→ 时步 t+1

Forest(t):                          Forest(t+1):
  Q_none, Q_light, Q_full           Q 继承（衰减因子 γ≈0.85）
  n 计数                            n *= γ（旧计数逐渐让位）
  GMM (μ, Σ, π)                     GMM 继承（裁剪+追加）
  精英列表                          精英列表清空（新时步重新投票）
  跨时步诊断记忆                    更新：哪些机组/模式在 t 时步表现好

评估 → 反驳或接受:
  t 时步: path (Gen0=ON, Storage=CHARGE) → cost=0.37 → 精英 ✅
  t+1 时步: 需求变了, 同样 path → cost=0.68 → 没进精英
    → Gen0=ON 的 Q 收到反驳信号（投票减少）
    → Forest 学到：这个需求水平下 Gen0=ON 不够用
```

**核心**：不是每时步"重新搜索"，而是**持续性认知**。forest 跨时步积累对离散结构的知识，通过反复评估来接受好结构、反驳坏结构。

---

## 3. 非马尔可夫代价的时域内处理

### 3.1 代价函数的时域形式

MCTS.md 的 cost 定义是每时步独立的。MPC 的 cost 必须写成时域求和形式：

$$\mathcal{L}(\mathbf{s}_{t:t+H}, \mathbf{x}_{t:t+H}) = \sum_{\tau=t}^{t+H} \Big[ f_\tau(s_\tau, x_\tau) + c_{\text{trans}}(s_\tau, x_\tau, s_{\tau-1}, x_{\tau-1}) \Big]$$

其中 $c_{\text{trans}}$ 捕获所有非马尔可夫项：

| 非马尔可夫项 | $c_{\text{trans}}$ 形式 | 能否在单时步内确定 |
|-------------|------------------------|-------------------|
| 启动费 | $s_g \cdot \max(0, u_\tau - u_{\tau-1})$ | ✗ 依赖 $s_{\tau-1}$ |
| 最小启停时间 | $\infty$ 若违反（Constran Hard 层） | ✗ 依赖历史 |
| 爬坡约束 | $\|p_\tau - p_{\tau-1}\|$ 超限惩罚 | ✗ 依赖 $x_{\tau-1}$ |
| 储能 SoC | $E_{\tau+1} = E_\tau + \dots$ | ✗ 时域递推 |
| 氢储罐质量 | $M_{\tau+1} = M_\tau + \dots$ | ✗ 时域递推 |

**驾驶场景的非马尔可夫项**：

| 非马尔可夫项 | $c_{\text{trans}}$ 形式 | 说明 |
|-------------|------------------------|------|
| Jerk 约束 | $\|a_\tau - a_{\tau-1}\| / \Delta t$ 超限惩罚 | 舒适性，依赖上一时步加速度 |
| 横向加速度累积 | $\sum_{k=\tau_0}^{\tau} \|a_{\text{lat},k}\|$ 超舒适阈值 | 变道 maneuver 跨多时步，累积效应 |
| 安全距离递推 | $d_{\tau+1} = d_\tau + (v_{\text{front},\tau} - v_\tau)\Delta t$ | 跟车距离是递推状态，单时步评估不完整 |
| 进度累积 | $\text{progress}_\tau = \sum_{k=t}^{\tau} v_k \Delta t$ | 必须到达目标，纯时域求和 |
| 交通规则 | 闯红灯、逆行在单时步触发即 Hard 违约 | Hard 模式，但检测需要时序上下文（如黄灯到红灯的过渡） |
| 他车交互 | $c_{\text{interact}}(s_\tau, s_{\text{other},\tau})$ | 让行/抢行决策影响他车行为，他车响应有延迟（非瞬时） |

**这些项在 IGO 的 cost 评估中，通过对全时域做一次前向 rollout 自然处理**——不需要索引树感知它们。索引树只需要知道"给定拓扑序列 $\mathbf{s}_{t:t+H}$，IGO 能算出对应的最优连续参数的总 cost"，用于更新该路径的评估统计量。

驾驶场景的关键区别：非马尔可夫项中 **Jerk 和安全距离递推是最频繁触发的两项**，远多于 UC 中的启动费和启停约束。这意味着驾驶的 IGO 在时域内需要更密集的约束评估——但因为时域较短（$H=8\sim10$ vs UC 的 $H=24$），总计算量仍可控。

### 3.2 Constran 在时域 cost 中的角色

Constran 构建的 cost_fn 接受 $(x, ctx)$，其中 $x$ 是**全时域拼接的连续参数向量**，$ctx$ 包含拓扑序列 $\mathbf{s}_{t:t+H}$ 和初始状态 $(s_{t-1}, x_{t-1})$。

```
x = [x_t, x_{t+1}, ..., x_{t+H}]  ← 全时域连续参数，一次性输入

cost_fn(x, ctx):
    total = 0
    carry = (s_{t-1}, x_{t-1}, E_{t-1}, M_{t-1})  ← 跨时步状态
    for τ in t .. t+H:
        cost_τ, carry = step_cost(τ, s_τ, x_τ, carry)
        total += cost_τ
    return total
```

Constran 的三种模式在时域内照常工作——Hard/Tunable/Soft 约束在每个时步独立施加，但违反量在时域求和前通过 $\sigma$-饱和嵌套压缩。

**关键保证**：因为 cost_fn 一次性看到全时域，$T$（log-transform）和 $\sigma$ 在整个时域 cost 上施加——不是每时步独立饱和。这意味着"一个时步严重违反"和"多个时步轻微违反"可以通过 Tunable 的 $\beta$ 区分（见 ConstranUser_README.md §5 的聚合语义）。

---

## 4. 工程实现

### 4.1 宏决策压缩：从时域枚举到静态索引树

所有路径在建模阶段就已存在——宏决策压缩决定了索引树的结构。对于 MPC 时域 $H=8\sim12$，拓扑变更事件数通常 $\leq 3$：

```python
# 索引树的结构：所有路径预建，树只管索引和统计
class HorizonIndexTree:
    """
    树深度 = 最大变更事件数 (≤ 3-4)
    每个节点 = (机组, 变更类型, 变更时步)
    每条根→叶路径 = 一个完整的时域拓扑序列 s_{t:t+H}
    
    变更类型: ON_at(t) / OFF_at(t) / 不变
    树不是动态扩展的——所有合法变更事件在初始化时枚举完毕
    """
    
    def build_tree(self, gens, horizon_H, min_up_down):
        events = []
        for g in gens:
            # 根据最小启停时间约束，枚举可能的变更时步
            possible_on_times = self._feasible_startup_times(g, horizon_H)
            possible_off_times = self._feasible_shutdown_times(g, horizon_H)
            events.extend([(g, 'ON', t) for t in possible_on_times])
            events.extend([(g, 'OFF', t) for t in possible_off_times])
        
        # 树深度 = len(events)，但通过剪枝控制
        # 变更时步必须单调非减（时间不能倒流）
        # 同一机组不能连续两个 ON 或两个 OFF
        return build_tree_from_events(events, pruning_rules)
```

对于自动驾驶场景（$\leq 5$ 个离散决策，$H=5\sim8$），变更事件更少——一台车在 1 秒时域内最多变 1-2 次道或过 1-2 个路口。

### 4.2 多模态 IGO 的分量-拓扑绑定

IGO 的 $K=3$ 个 GMM 分量如何与拓扑候选对应：

```
方案 A — 显式绑定（拓扑候选少且明确时）
  分量 0: 绑定到拓扑候选 s_{t:t+H}^A
  分量 1: 绑定到拓扑候选 s_{t:t+H}^B  
  分量 2: 自由探索（可能发现更好的拓扑-参数组合）

  采样时：
    分量 0,1 的 mean 初始化为各自拓扑候选对应的连续参数 warm start
    分量 2 的 mean 初始化在 A 和 B 之间（插值或随机）
  
  更新时：
    所有分量共享同一个 cost_fn，但 cost_fn 内部根据分量 ID 使用不同的 ctx['topology']

方案 B — 隐式探索（拓扑候选多或不确定时）
  所有分量的 mean 初始化为同一个 warm start
  cost_fn 内部使用连续松弛（sigmoid 编码拓扑）
  IGO 自然收敛到不同模态 → 不同分量对应不同 emergent 拓扑
```

**推荐**：MPC 场景用方案 A（显式绑定），因为引导树已经给出了明确的 top-K 拓扑候选路径，浪费 warm start 信息不合理。方案 B 适用于一次性大规模优化。

### 4.3 时域滑窗 Warm Start

```
时步 t 的最终 GMM:
  (μ_k^t, S_k^t, π_k^t)  for k = 0, 1, 2

时域滑窗: [t, t+H] → [t+1, t+H+1]
  → 裁剪: 丢弃 μ_k^t 中对应时步 t 的参数段
  → 追加: μ_k^t 末尾追加时步 t+H+1 的参数段（从 μ_k^t 的最后一步外推或随机初始化）
  → S_k, π_k 保持不变（协方差结构在滑窗后基本不变）

索引树:
  → 裁剪: 移除时步 t 对应的树层（该层路径的前缀已过期）
  → 继承: 保留时步 t+1..t+H 的节点统计量 (Q_j^mode, n_j^mode)
  → 衰减: Q_j^mode 不变（是经验均值，不需要衰减）
           n_j^mode *= γ  (γ ≈ 0.85-0.9, 旧计数逐渐让位给新时步的评估)
  → 扩展: 为时步 t+H+1 新增节点，从先验初始化
```

### 4.4 跨时步诊断记忆

诊断信号不只回传更新索引树的路径统计量——更重要的是**积累跨时步的模式记忆**：

```python
# 跨时步诊断记忆（伪代码）
class TemporalDiagnostics:
    """
    记录每个 (机组, 状态) 在多个时步中的表现模式
    """
    def __init__(self, n_gens, memory_decay=0.95):
        self.gen_stats = {}  # g -> {pattern: running_average}
        self.decay = memory_decay
    
    def update(self, t, gen_id, signal):
        # signal = 'at_pmax' | 'at_pmin' | 'frequent_switch' | ...
        # 指数衰减更新，越近的时步权重越大
        for g, sig in zip(gen_id, signal):
            if g not in self.gen_stats:
                self.gen_stats[g] = {}
            for s in sig:
                old = self.gen_stats[g].get(s, 0.0)
                self.gen_stats[g][s] = old * self.decay + (1 - self.decay)
    
    def get_prior_bias(self, gen_id):
        # 返回索引树节点先验偏置
        # 'at_pmax' 频繁 → ON 节点败率上调（应多开）
        # 'at_pmin' 频繁 → ON 节点败率上调（应关）
        # 'frequent_switch' → 该机组的变更事件惩罚加大
        return bias_for_mcts
```

### 4.5 单次 IGO 内的候选并行评估

与 MCTS.md §7.1 的三层并行粒度不同，MPC 场景的并行发生在 GMM 分量级别：

```python
# 单次 IGO，K 个分量的自然梯度更新
# 采样: 从 K 个分量独立采样 B_k 个样本
# 评估: vmap 对所有样本计算 cost_fn(samples, ctx_k)
#       其中 ctx_k 包含对应拓扑序列
# 更新: 每个分量用自己的精英样本独立更新

# JAX 代码路径与 blockwise_mgigo 完全一致
# 唯一的区别: ctx 是 per-component 的（包含不同拓扑）
```

**计算量对比**：
- 串行 K 次 IGO：$K \times T \times B$ 次 cost_fn 评估
- 多模态单次 IGO：$T \times (K \times B_k)$ 次 cost_fn 评估，其中 $\sum B_k = B$
- 如果 $B_k = B/K$，总评估次数相同。但单次 IGO 的 $T$ 可以更小（因为分量间共享梯度信息加速收敛），且 JIT 编译只发生一次

---

## 5. 与 MCTS.md §6.3 的关键区别总结

| 维度 | MCTS.md §6.3（旧） | 本文档（新） |
|------|-------------------|------------|
| **离散搜索方式** | 当前时步 $s_t$，串行评估候选 | 时域索引树：所有路径预建，UCT 引导采样 $\mathbf{s}_{t:t+H}$ |
| **树模型** | 动态扩展的标准 MCTS 树 | 静态索引树：路径集预建，树只管索引和统计（§10） |
| **IGO 评估对象** | 单个 $s_t$ + 连续参数 | 全时域 $(\mathbf{s}_{t:t+H}, \mathbf{x}_{t:t+H})$ |
| **非马尔可夫代价** | IGO 隐式处理（不透明） | 时域前向 rollout 显式求和（树不做加法分解） |
| **候选评估方式** | 串行 K 次 IGO | 单次多模态 IGO（K 分量） |
| **层结构** | 层2 消失，层3 = 最终 | 层2 消失，层3 = 多模态 IGO |
| **树深度** | $N$（单时步） | 时域内变更事件数 $\ll H \times N$（路径预枚举） |
| **Warm Start** | GMM + 树统计量 | GMM + 索引树统计量 + 时域滑窗裁剪 + 跨时步记忆 |

---

## 6. 适用范围与退化

### 6.1 当 $H=1$ 时（无预测时域）

如果 MPC 的预测时域退化为 1（只看当前步），本文档退化回 MCTS.md §6.3 的方案——因为不存在时序耦合。但实际 MPC 几乎总是 $H \geq 3$。

### 6.2 当离散节点数 = 0 时（纯连续 MPC）

如果问题没有离散拓扑（纯连续 MPC），整个引导树层消失。IGO 单独求解连续 MPC。这就是 `Energy/stochastic_uc.py` 的当前做法——纯连续 IGO 滚动 MPC。

### 6.3 当非马尔可夫性很弱时

如果跨时步耦合很弱（如启动费很小、无最小启停约束），可以近似退化为 MCTS.md §6.3 的串行方案。此时串行评估的误差在可接受范围内，换取更简单的实现。

---

## 7. 自动驾驶特化

### 7.1 安全关键约束的 Constran 编码

驾驶与 UC 在约束上的根本区别：**安全约束（碰撞、闯红灯、驶出路面）是 Hard 且不可妥协的，而舒适/效率是 Tunable/Soft**。

Constran 的优先级分配：

```
P1 (Hard):  碰撞避免       g_coll(x, ctx) = safe_dist - min_distance(traj, obs) ≤ 0
            交通规则       g_rule(x, ctx) = violation_flag(traj, map) ≤ 0
            道路边界       g_bound(x, ctx) = |lateral_offset| - lane_width/2 ≤ 0

P2 (Tunable): 舒适性      g_comfort(x, ctx) = |jerk| - jerk_max ≤ 0,  δ=2.0, β=1.0
              横向舒适     g_lat(x, ctx) = |a_lat| - a_lat_max ≤ 0,  δ=2.0, β=5.0

P3 (Soft):    效率        g_progress(x, ctx) = target_speed - v_τ ≤ 0 (速度不足)
              能耗        g_energy(x, ctx) = |a_τ · v_τ| - eco_power ≤ 0
```

**关键**：P1 的碰撞约束不能像 UC 的功率平衡那样用 Tunable 慢慢调。碰撞 = Hard，违反时 $\sigma$-饱和嵌套的 `jnp.where(g>0, T(g)+δ, inner)` 确保碰撞候选的 cost 一定压倒所有无碰撞候选（$\delta=3\sim5$ 把碰撞输出推到 $\sigma$ 饱和区顶部）。

### 7.2 感知不确定性的在线处理

驾驶的 Chance 约束来源与 UC 根本不同：

| | UC | 自动驾驶 |
|--|-----|---------|
| 不确定性来源 | 需求/风光（分布已知） | 感知噪声 + 他车意图（分布未知/多模态） |
| 样本生成 | `noise_fn(key, shape)` 从已知分布采样 | 需要从感知模块输出（检测框 + 不确定性估计） |
| 时变性 | 静态（场景固定后分布不变） | 高度动态（每帧感知结果不同） |
| 处理方式 | 离线 MC + 多批次分位数 | 在线轻量 MC + 预测采样 |

**驾驶的混合策略**：

```python
def driving_chance_cost(x, ctx):
    # ctx['perception'] 包含当前帧的检测结果 + 不确定性
    # 每个检测目标: (mean_pose, cov_pose, intent_probs, class)
    
    total_risk = 0.0
    for obj in ctx['perception']['objects']:
        # 方案 A: 如果 intent_probs 集中（如 0.95 概率直行），
        #         直接用最可能意图做确定性评估（快）
        if max(obj.intent_probs) > 0.9:
            pred_traj = rollout_deterministic(obj, obj.intent_probs.argmax())
            total_risk += collision_risk(x, pred_traj)
        
        # 方案 B: 如果意图不确定（如 0.4/0.3/0.3 分给三种可能），
        #         在线 MC 采样，每种意图按概率加权
        else:
            for intent, prob in enumerate(obj.intent_probs):
                if prob > 0.1:  # 忽略极小概率意图
                    # 少量样本（n=5~10），按意图分布采样轨迹
                    mc_trajs = sample_trajectories(obj, intent, n=10)
                    risk_samples = vmap(lambda traj: collision_risk(x, traj))(mc_trajs)
                    total_risk += prob * jnp.quantile(risk_samples, 0.9)
    
    return total_risk
```

**为什么不在线跑 100 个 MC 样本**：驾驶的碰撞检测比 UC 的功率求和昂贵得多（需要几何计算），100 个样本 × 8 个时步 × 5 个目标 ≈ 4000 次碰撞检测，在 30ms 预算内不可行。意图引导的样本分配把有效样本集中在不确定目标上。

### 7.3 多车交互：他车不是噪声，是反馈

UC 的不确定性（需求、风光）是单向的——它们影响约束，但不受我们决策影响。驾驶的不确定性是**双向的**——我们抢行，他车就让；我们让行，他车就抢。

这意味着索引树的某些分支不仅影响自车的拓扑，还**改变他车的响应模式**：

```
Event: 路口，自车 vs 他车
  分支 A: 自车抢行 → 他车减速让行 → 自车快速通过
          IGO cost: 时间短，但安全距离小 (Hard 约束可能触边)
  
  分支 B: 自车让行 → 他车先过 → 自车等待后通过
          IGO cost: 时间长，但安全距离充裕
  
  分支 C: 自车微加速试探 → 他车反应不确定 →
          需要在线 MC 采样他车响应（激进/保守司机模型）
```

**对索引树的影响**：每个涉及他车交互的节点，他车的响应不是确定的——需要在 IGO 的 cost 评估中对他车行为做 MC。这使得驾驶场景的索引树需要覆盖更多不确定性分支。

**实用简化**：对每个他车维护一个"交互模式"（aggressive / normal / cautious），作为索引树覆盖的路径空间的一部分。不是对每帧所有他车都枚举三种模式——只对**与本车拓扑决策直接交互的他车**枚举。

### 7.4 驾驶连续参数的块结构

驾驶的连续参数天然按语义分块，适合 Blockwise MGIGO：

```
Block 0: 纵向速度剖面  v_τ     for τ = t..t+H  (D = H ≤ 10)
Block 1: 横向偏移剖面  d_τ     for τ = t..t+H  (D = H ≤ 10)
Block 2: 跟车距离目标  d_safe  (标量, D = 1)

或按 maneuver 阶段分块：
Block 0: 变道前减速阶段   v_τ, d_τ for τ ∈ [t_0, t_1]
Block 1: 变道横向移动阶段 d_τ    for τ ∈ [t_1, t_2]
Block 2: 变道后加速阶段   v_τ     for τ ∈ [t_2, t_end]
```

分块数少（2-3 块），每块维度低（D ≤ 10），单 GPU 轻松跑。

### 7.5 驾驶 vs UC 的参数速查

| 参数 | 驾驶 (H=8, 5 maneuvers) | 能源 UC (H=12, 30 gens) |
|------|--------------------------|--------------------------|
| 索引树深度 | 1-3（时域内 maneuver 事件） | 3-6（时域内启停变更事件） |
| 索引树遍历次数 | ~200-500 | ~8000-12000 |
| 分块数 | 2-3（纵向/横向/跟车） | 15-30（时步 × 机组类型） |
| 层3 IGO | T=150, B=80, MC=15 | T=100, B=30, MC=30 |
| Constran 优先级 | P1 安全(Hard) > P2 舒适(Tunable) > P3 效率(Soft) | P1 物理(Hard) > P2 平衡(Chance) > P3 最小出力(Tunable) |
| 主要非马尔可夫项 | Jerk, 安全距离递推, 横向累积 | 启动费, 爬坡, SoC 递推 |
| 不确定性焦点 | 他车意图（在线 MC + 意图引导） | 需求/风光（离线 MC + DRO） |
| 总耗时 | 15-30 ms | 10-20 s |
| 硬件 | Orin/Thor, VRAM ~1 GB | RTX 5070, VRAM 3-5 GB |

### 7.6 多智能体博弈：联合引导树 + RNE 求解器

§7.3 讨论了他车交互的双向性——自车决策改变他车响应。但更根本的问题在于：**当交互双方（或多方）各自独立优化自己的目标时，单边优化的解不是博弈均衡——双方都有动机偏离**。这需要显式求解 Nash 均衡。

`MPC_G_MS.py`（blockwise RNE 求解器）可以像 `MPCsolverM22.py` 融入单智能体引导树一样，融入联合引导树。

这个框架不仅适用于自动驾驶的车车交互，同样适用于**竞争性电力市场**——各发电商/储能商独立 minimize 自身成本，市场出清价格由所有人的报价共同决定。

**从单边优化到博弈均衡**：

```
单边引导树 + IGO（单主体控制所有决策变量）:
  索引树覆盖单主体的拓扑路径集
  IGO 优化连续参数 x

联合引导树 + RNE（多主体各自优化自己的目标）:
  每个 agent 有自己的索引树，覆盖各自的拓扑路径集
  联合路径 = 笛卡尔积 (s_0, ..., s_{M-1})
  RNE 求解联合连续策略的 Nash 均衡 (x_0*, ..., x_{M-1}*)
  每个 agent i 的均衡 cost_i* 回传给自己的索引树做 Backprop
```

**为什么需要联合引导树**：

单独优化一方拓扑（假设另一方被动）在两个领域都失效：

| 场景 | 失效模式 | 正确做法 |
|------|---------|---------|
| 驾驶：抢行博弈 | 自车单边优化选"抢行"，但他车也选"抢行"→ 碰撞 | 联合引导树发现混合均衡或折中拓扑 |
| 驾驶：路口协商 | (左转,让行)和(直行,抢行)是两个不同均衡，单边看不到全貌 | 联合引导树探索两个均衡区域 |
| **UC：储能竞争出价** | 储能 A 单边优化选"高峰放电"，但储能 B 也选同样策略 → 高峰电价被压低，双方利润缩水 | 联合引导树发现储能 A 高峰放电 + 储能 B 错峰放电的均衡 |
| **UC：发电商竞争** | GenCo A 单边优化多开便宜机组 → 电价低 → 但 GenCo B 也开便宜机组 → 电价更低 → 双方都亏 | 联合引导树发现容量博弈的 Cournot 均衡 |

**联合索引树结构 — 驾驶 vs UC**：

```
驾驶（2 车路口）:                      UC（2 发电商竞争）:

自车: 左转? / 直行                     GenCo A: Gen1+2 / Gen1 only?
他车: 抢行? / 让行                     GenCo B: Gen3+4 / Gen3 only? / 全停?

联合:                                 联合:
  路径1: (左转, 让行) cost=(低,中)        路径1: (A:1+2, B:3+4) → 供过于求, 电价低, 双方利润低
  路径2: (左转, 抢行) cost=(极高,高)      路径2: (A:1+2, B:3)   → 供需平衡, 电价适中, A利润高
  路径3: (直行, 让行) cost=(中,低)        路径3: (A:1,   B:3+4) → 供需平衡, 电价适中, B利润高
  路径4: (直行, 抢行) cost=(中,中)        路径4: (A:1,   B:3)   → 供不应求, 电价高, 双方利润中等
```

**UC 竞争博弈的 cost 函数结构**：

与单主体 UC（MCTS_UC_README.md）不同，竞争 UC 的每个 agent 有自己的 cost 函数，且通过市场出清耦合：

```python
def agent_cost(agent_idx, joint_x, ctx):
    """
    agent_idx: 发电商编号
    joint_x:   所有发电商的连续参数拼接
    ctx:       包含需求、燃料价、各agent的机组归属
    """
    # 1. 从 joint_x 提取所有发电商的出力
    all_outputs = decode_all_agents_output(joint_x, ctx)
    
    # 2. 市场出清：汇总出力 → 确定出清价格
    total_supply = sum(all_outputs)
    clearing_price = inverse_demand_curve(total_supply, ctx['demand'])
    # 如果 total_supply < demand: 价格 = VOLL (失负荷价值), 极高
    # 如果 total_supply ≥ demand: 价格 = P(total_supply)  (递减)
    
    # 3. Agent 自己的成本
    my_gens = ctx['agent_generators'][agent_idx]
    my_output = sum(all_outputs[g] for g in my_gens)
    fuel_cost = sum(fuel_cost_fn(g, all_outputs[g]) for g in my_gens)
    startup_cost = sum(startup_fn(g, ctx['prev_state'][g]) for g in my_gens)
    
    # 4. 利润 = 收入 - 成本（minimize -profit）
    revenue = my_output * clearing_price
    return -(revenue - fuel_cost - startup_cost)
```

**关键差异**：单主体 UC 的 cost 是 `Σ 燃料 + 启动 + 约束惩罚`。竞争 UC 的 cost 是 `-(收入 - 成本)`，其中收入通过市场出清依赖所有 agent 的联合出力。RNE 求解器通过 $M_{\text{inner}}$ 次背景采样，对每个 agent 估计"给定其他 agent 的当前策略分布，我的期望收益是多少"——这正是 Nash 均衡的样本估计。

**储能竞争的特殊性**：

储能不直接发电，而是做时域套利。这使得拓扑决策更微妙：

- 储能 A 的拓扑 = 时域内充放电区间（何时充、何时放）
- 储能 B 的拓扑同理
- 如果 A 和 B 都在同一时段放电 → 该时段电价被压低 → 双方套利空间缩小
- Nash 均衡：A 和 B 错开放电时段（或 A 充电时 B 放电，反之亦然）
- 索引树需要同时覆盖 A 和 B 的充放区间组合

这就回到了 MCTS_UC_README 的宏决策设计——每台储能的"充放区间"是一个宏决策，联合 MCTS 一次性搜索所有储能的充放区间组合。

**RNE 求解器在引导树中的角色**：

对应 MCTS.md §3.2 的三层金字塔，RNE 求解器充当层2/3 的评估器：

```
联合引导树遍历（层1，μs）:
  遍历数百到数千次，用简化博弈矩阵快评联合拓扑
  驾驶: 纯策略枚举 + 启发式 cost（碰撞/通过/时间）
  UC:   纯策略枚举 + merit-order 近似出清价
  输出: top-K 个联合拓扑候选

RNE 求解器（层3）:
  对 top-K 个联合拓扑候选，跑一次多模态 RNE
  K 个 GMM 分量各绑定一个联合拓扑候选
  求解对应拓扑下的 Nash 均衡连续策略
  
  RNE 自洽性保证：
    - 对 agent i，精英样本 = 在其他 agent 的均衡策略背景下，agent i 的最优响应
    - 所有 agent 同时满足 → Nash 均衡
    - 混合策略权重 π 反映不同均衡的相对优劣
```

**代码层面的融合**：

```python
from gmm_igo.MPC_G_MS import mmog_igo_rne_blocks_solver

# ============================================================
# 驾驶场景
# ============================================================
def joint_mcts_rne_driving(t, state, horizon_H):
    joint_topologies = mcts_search_joint(
        ego_maneuvers=[LANE_CHANGE, TURN, OVERTAKE],
        other_maneuvers=[YIELD, RUSH, LANE_CHANGE],
        horizon_H=8
    )
    result = mmog_igo_rne_blocks_solver(
        M_agent=2,  # 自车 + 关键他车
        block_to_agent_idx=[0]*ego_blocks + [1]*other_blocks,
        fitness_fn_j=lambda aid, x, ctx: driving_cost(aid, x, ctx),
        context={'topology': joint_topologies},
        ...
    )
    return extract_ego_action(result)

# ============================================================
# UC 竞争市场场景
# ============================================================
def joint_mcts_rne_uc_market(t, state):
    """
    GenCo A 控制 gens [0..4], GenCo B 控制 gens [5..9]
    各自 minimize 自身成本，市场出清决定电价
    """
    joint_commitments = mcts_search_joint(
        agent_A_choices=[commit_pattern(g) for g in [0,1,2,3,4]],
        agent_B_choices=[commit_pattern(g) for g in [5,6,7,8,9]],
        horizon_H=24
    )
    result = mmog_igo_rne_blocks_solver(
        M_agent=2,  # GenCo A, GenCo B
        N_blocks=10,  # 10 台机组各一块
        block_to_agent_idx=[0]*5 + [1]*5,
        fitness_fn_j=lambda aid, x, ctx: genco_profit(aid, x, ctx),
        # genco_profit 内部做市场出清:
        #   total_supply = sum(all_gens_output(x))
        #   price = inverse_demand(total_supply)
        #   return -(agent_revenue - agent_cost)
        context={
            'demand_curve': demand_forecast,
            'topology': joint_commitments,
        },
        ...
    )
    # 提取均衡: 各 GenCo 的 Nash 均衡出力和利润
    return extract_nash_commitments_and_dispatch(result)
```

**能源 UC 竞争博弈的附加场景**：

| 场景 | Agent 划分 | 博弈焦点 | 索引树覆盖的路径 |
|------|-----------|---------|---------------|
| **发电商容量博弈** | 每个 GenCo 一个 agent | 开多少机组 → 影响电价 → 影响对手利润 | 各 GenCo 的启停组合 |
| **储能套利竞争** | 每个储能商一个 agent | 充放时段分配 → 影响峰谷价差 → 影响对手套利空间 | 各储能的充放区间宏决策 |
| **风光+储能联合报价** | 可再生+储能捆绑为一个 agent | 弃电 vs 储存 vs 即发 → 影响实时电价 | 弃电策略 + 储能充放区间 |
| **跨区输电阻塞** | 各区 ISO 一个 agent | 区际联络线容量分配 → 影响本区电价 | 联络线容量申请 + 区内机组启停 |
| **辅助服务市场** | 调频/备用提供商各一个 agent | 预留容量 vs 电量市场 → 机会成本博弈 | 预留容量 + 电量市场出力 |

**RNE vs 单边 IGO 的适用分界**：

| 场景 | 用什么 | 原因 |
|------|--------|------|
| 单主体控制所有资源（传统 UC） | 单边 IGO (MPCsolverM22) | 无博弈，纯优化 |
| 驾驶：他车不交互 | 单边 IGO + 意图采样 (§7.2) | 博弈效应可忽略 |
| 驾驶：强交互（路口、抢行） | 联合引导树 + RNE | 双方决策互相影响 |
| UC：垄断 / 单一发电商 | 单边引导树 + IGO (MCTS_UC_README) | 无竞争者 |
| UC：寡头竞争（2-5 个 GenCo） | 联合引导树 + RNE | Cournot/Nash 均衡 |
| UC：完全竞争（很多小 GenCo） | 退化：单边 IGO + 价格接受者假设 | 单个 agent 无法影响电价 |
| 储能竞争出价 | 联合引导树 + RNE | 储能套利高度依赖竞争对手的充放时段 |

**计算量控制**：

RNE 求解器比单边 IGO 贵（$M_{\text{inner}} \times M_{\text{agent}}$ 倍的联合评估）：

| 场景 | $M_{\text{agent}}$ | $M_{\text{inner}}$ | 单次 RNE 耗时 | 备注 |
|------|-------------------|-------------------|-------------|------|
| 驾驶 2 车交互 | 2 | 10-15 | 15-25 ms | 时域短 (H=8), 块数少 |
| UC 2 GenCo | 2 | 15-20 | 3-8 s | 时域长 (H=24), 但块并行 |
| UC 3-5 GenCo | 3-5 | 10-15 | 8-20 s | 博弈树更深但仍在 GPU 预算内 |
| 储能 2-3 竞争者 | 2-3 | 10-15 | 2-5 s | 储能维度低 (SoC + 充放) |

**关键控制原则**：
- 只对**直接竞争**的 agent 建博弈（UC 中 3-5 个大发电商而非全部 50 台机组各自成 agent）
- 其余小参与者合并为"背景市场"或降级为价格接受者
- $M_{\text{agent}} \leq 5$ 时博弈树和 RNE 均可控

---

## 8. 多保真度 UCT 评分与 MGIGO 单优化器多模态探索

> **这不是标准 MCTS 的 UCT。** 这是在预建路径集上的多保真度 bandit 评分（见 §10）。树是静态索引——所有路径已经存在，UCT 只决定"下一条评估哪条路径"。$Q_j^{\text{mode}}$ 不是节点价值，而是"经过分支 $j$ 的路径集合"在 mode 保真度下的经验平均 cost。树结构上没有加法分解（非马尔可夫），Selection 走到底选中的是一条完整路径 $p$。

### 8.1 公式回顾

当 Selection 沿索引树走到节点，面临 $K$ 个分支（每个分支对应一个离散前缀选择）时，第 $j$ 个分支的 UCT 得分：

$$\text{UCT}_j = \widetilde{Q}_j + c \sqrt{\frac{\ln N_{\text{total}}}{n_j}}$$

其中 $n_j = n_j^{\text{none}} + n_j^{\text{light}} + n_j^{\text{full}}$（分支 $j$ 被评估的总次数），$N_{\text{total}} = \sum_{k=1}^K n_k$。

### 8.2 方案评估

**总体评价：这是在预建路径集上做多保真度 bandit 引导的最精确评分方案。** 三层独立计数的设计解决了一个隐含问题——不同保真度的评估结果如果在同一个节点统计量中混在一起做算术平均，高精度评估的信号会被低精度噪声稀释。

**三层独立 $Q$ 的递推**：

$$Q_j^{\text{mode}} \leftarrow Q_j^{\text{mode}} + \frac{1}{n_j^{\text{mode}}} \left( R_q - Q_j^{\text{mode}} \right), \quad \text{mode} \in \{\text{none}, \text{light}, \text{full}\}$$

每个 $Q_j^{\text{mode}}$ 是经验均值。关键性质：**$R_q$ 来自 Constran cost 函数，其输出已经经过最外层 $\sigma$-饱和**，天然有界在 $(-1, 1)$。因此 $Q_j^{\text{mode}} \in (-1, 1)$ 作为有界随机变量的经验均值自动继承有界性。

**饱和嵌套 $\widetilde{Q}_j$ —— 严格对齐 Constran 的递归 $\sigma$-包裹**：

$$\widetilde{Q}_j = \sigma\!\Big( Q_j^{\text{none}} + \sigma\!\big( Q_j^{\text{light}} + \sigma(Q_j^{\text{full}}) \big) \Big)$$

从内向外读，与 Constran `_assemble_nest` 的构建顺序（低优先级→高优先级，从内到外）严格同构：

```
Constran _assemble_nest              本方案 Q̃_j
─────────────────────────           ─────────────────────
inner = σ(T(objective))             σ(Q_full)    ← 最内层：重型
  ↓                                    ↓          离散+连续联合优化
for each constraint:                   ↓          IGO 精搜，最细约束
  inner = σ(T(g) + inner)            σ(Q_light + σ(Q_full))
  (Soft / Tunable / Hard)              ↓         ← 中间层：中型
                                       ↓          IGO 轻量探测
                                     σ(Q_none + σ(Q_light + σ(Q_full)))
                                                       ← 最外层：轻型
                                                       纯离散初筛，粗约束
                                                       被最外 σ 饱和
```

**为什么不需要额外的 $\delta$ 权重**：

1. **每个 $Q_j^{\text{mode}}$ 天然有界**：$R_q$ 是 Constran cost 的输出，已经通过最外层 $\sigma$ 饱和在 $(-1, 1)$ 内。$Q_j^{\text{mode}}$ 作为这些有界值的经验均值，自动 $\in (-1, 1)$。不存在"$Q_j^{\text{light}}$ 可以任意大而覆盖内层信号"的问题——它天然有界。

2. **递归 $\sigma$-包裹本身就是"层层封顶"**：$\sigma(Q_j^{\text{full}}) \in (-1, 1)$ → $Q_j^{\text{light}} + \sigma(Q_j^{\text{full}}) \in (-2, 2)$ → 外层 $\sigma$ 将其压回 $(-1, 1)$ → $Q_j^{\text{none}} + \text{inner} \in (-2, 2)$ → 最外层 $\sigma$ 再压回。每一层都只能在上层留下的 $(-1, 1)$ 区间内做有界修正。这正是 Constran 的做法——`inner = σ(T(g) + inner)`，$T(g)$ 是对数压缩后的有界量，加上 inner 后整体再 $\sigma$。

3. **重型信号的内在主导权来自嵌套顺序，不来自显式权重**：full 在最内层 → 它的信号穿过最多层 $\sigma$ → 每层 $\sigma$ 是 1-Lipschitz 的 → 最内层信号以几乎未衰减的形式传递到最外层。none 在最外层 → 它的信号只穿过一层 $\sigma$ → 对最终输出的影响被一层 $\sigma$ 压缩。注意：在 $\sigma(x) = x/\sqrt{1+x^2}$ 下，输入的 $x$ 越大越被压缩。当 $\sigma(Q^{\text{full}})$ 给出强信号（接近 ±1），$Q^{\text{light}} + \sigma(Q^{\text{full}})$ 被外层 $\sigma$ 处理后，light 的修正被压缩——这正是我们想要的："重型信号不可被推翻"。

4. **无评估时自然归零**：$Q_j^{\text{full}} = 0 \Rightarrow \sigma(0) = 0$，同理 light 和 none。未评估分支 $\widetilde{Q}_j = 0$，完全由探索项驱动。

**与 MCTS.md §2.3 标准 UCB1 的对比**：

| | MCTS.md UCB1 | 本方案 |
|--|-------------|--------|
| 利用项 | $\bar{X}_j$（所有评估混在一起平均） | $\widetilde{Q}_j$（三层独立、递归 $\sigma$-包裹） |
| 探索项 | $\sqrt{2\ln N_{\text{parent}} / n_j}$ | $c\sqrt{\ln N_{\text{total}} / n_j}$ |
| 计数 | 单一 $n_j$ | $n_j = n_j^{\text{none}} + n_j^{\text{light}} + n_j^{\text{full}}$ |
| 参数 | 无 | $c = 1/\sqrt{2}$ |

本方案的利用项更强：承认不同保真度的信息质量差异，通过递归 $\sigma$-包裹（非加权组合）让重型信号结构性主导。

### 8.3 Hoeffding 条件的验证

**问题**：标准 UCB1 的 $\sqrt{2\ln N / n}$ 中的 $\sqrt{2}$ 来自 Hoeffding 不等式 $P(|\bar{X} - \mu| \geq \epsilon) \leq 2\exp(-2n\epsilon^2/(b-a)^2)$，要求奖励 $X \in [a, b]$ 且 $\epsilon$ 覆盖真实均值 $\mu$ 的概率 $\geq 1 - 1/N$。本方案 $c = 1/\sqrt{2}$ 是否满足？

**推导**：

$\widetilde{Q}_j$ 是三个 $\sigma$ 的组合，输出 $\in (-1, 1)$。范围 $(b-a) = 2$。

Hoeffding 不等式（$n = n_j$ 个样本的均值 $\bar{X}_n$）：
$$P(|\bar{X}_n - \mu| \geq \epsilon) \leq 2\exp\!\left(-\frac{2n\epsilon^2}{(b-a)^2}\right)$$

设定置信水平 $1/N_{\text{total}}$：
$$2\exp\!\left(-\frac{2n_j\epsilon^2}{4}\right) = \frac{1}{N_{\text{total}}}$$

$$\epsilon = \sqrt{\frac{2(\ln N_{\text{total}} + \ln 2)}{n_j}} \approx \sqrt{\frac{2\ln N_{\text{total}}}{n_j}} \quad \text{for large } N_{\text{total}}$$

所以 $c_{\text{Hoeffding}} = \sqrt{2} \approx 1.414$ 是理论最优值。

**用户建议 $c = 1/\sqrt{2} \approx 0.707$ 或 $c = 1/2$，是否破坏 Hoeffding 条件？**

**直接回答**：$c = 1/\sqrt{2}$ 在 Hoeffding 意义下是**紧界不足**，置信度从 $1 - 1/N_{\text{total}}$ 降到约 $1 - 1/N_{\text{total}}^{0.25}$。但这在工程上不一定有问题，原因有三：

1. **Regret 阶不变**：UCB 的累积 regret 是 $O(\sqrt{n \log n})$，常数 $c$ 只影响 leading coefficient，不改变渐近阶。$c = 1/\sqrt{2}$ 的 regret 是 $c = \sqrt{2}$ 的约 $1/2$ —— 实际上可能更好，因为标准 Hoeffding 是 worst-case bound。

2. **$\widetilde{Q}$ 的有效范围可能小于 $(-1, 1)$**：三层 $\sigma$ 嵌套压缩后，通常情况下的 $\widetilde{Q}$ 集中在 $(-0.7, 0.7)$ 左右（$\sigma$ 的线性区），等效 $(b-a) < 2$，从而有效 $c$ 更接近理论界。

3. **多保真度的 counting 结构本身提供了额外的"隐式探索"**：当一个分支只有 $n_j^{\text{none}}$（无 light/full）时，$n_j$ 虽大但 $\widetilde{Q}_j$ 本质上只是 $\sigma(Q_j^{\text{none}})$ —— 低保真度的不确定性隐含在利用项的低置信度中，相当于在利用项内部就已经打了折扣，不再需要满额的探索 bonus 来对冲。

**结论**：$c = 1/\sqrt{2}$ 可以接受，$c = 1/2$ 偏保守但也可以工作。如果要严格满足 Hoeffding（例如用于安全关键场景的 theoretical guarantee），有两种修正方案：

**修正 A — 重新归一化 $\widetilde{Q}$ 到 $[0,1]$**：

令 $\widetilde{Q}'_j = (\widetilde{Q}_j + 1) / 2 \in [0, 1]$，范围 $(b-a) = 1$。此时：
$$\epsilon = 1 \cdot \sqrt{\frac{\ln N_{\text{total}} + \ln 2}{2n_j}} \approx \frac{1}{\sqrt{2}}\sqrt{\frac{\ln N_{\text{total}}}{n_j}}$$

**$c = 1/\sqrt{2}$ 精确满足 Hoeffding！** 这是最干净的修正——不改公式结构，只加一个仿射变换。

**修正 B — 分层 Hoeffding（利用三层独立计数）**：

每层独立满足 Hoeffding（每层的 $Q_j^{\text{mode}}$ 独立递推，各自有 $n_j^{\text{mode}}$ 个样本），然后通过 $\sigma$ 的 1-Lipschitz 性质保证 $\widetilde{Q}_j$ 继承浓度：
$$|\widetilde{Q}_j - \mu| \leq |Q_j^{\text{none}} - \mu^{\text{none}}| + |Q_j^{\text{light}} - \mu^{\text{light}}| + |Q_j^{\text{full}} - \mu^{\text{full}}|$$

每层用 union bound + 各自层级的样本数设定置信区间。这样总体置信度由三层共同保证，允许每层的探索 bonus 更小。

**推荐**：工程上用修正 A（$\widetilde{Q}'_j = (\widetilde{Q}_j + 1)/2$，$c = 1/\sqrt{2}$）。简单、满足理论界、公式结构不变。

### 8.4 MGIGO 单优化器实现 $K$ 个分支的同时探索

这是方案最巧妙的部分：当节点有 $K$ 个离散分支时，**不是对每个分支独立跑 $K$ 次 IGO**，而是用**一次 MGIGO 的 $K$ 个 GMM 分量同时探索 $K$ 个分支**。

**与 MCTSforMPC.md §2.3 多模态 IGO 的精确对应**：

```
索引树节点: ○ (parent node)
          / | \
         ○  ○  ○  ← K=3 个离散分支 (如 ON/OFF/SHUTDOWN)

MGIGO 单次运行:
  GMM K=3 分量
  分量 0 → 探索分支 0 的连续参数空间，ctx = {branch: 0, topology: s_0}
  分量 1 → 探索分支 1 的连续参数空间，ctx = {branch: 1, topology: s_1}
  分量 2 → 探索分支 2 的连续参数空间，ctx = {branch: 2, topology: s_2}
```

**采样阶段**：从 $K$ 个分量各采 $B/K$ 个样本。分量 $j$ 的样本 $x \sim \mathcal{N}(\mu_j, \Sigma_j)$，在 cost_fn 中与分支 $j$ 的拓扑 $s_j$ 绑定评估。这 $B$ 个样本覆盖了 $K$ 个不同拓扑对应的连续参数区域。

**精英筛选**：$B$ 个样本按 cost 排序，取前 $B_0$ 个为精英。关键是——精英筛选是**跨分量竞争**的：如果分支 0 的拓扑确实优于分支 1，则分量 0 的样本会有更多进入精英集，分量 0 的权重 $v_0$ 会增大。反之，质量差的分支对应的分量权重自然下降。

**更新阶段**：每个分量用自己的精英样本独立更新 $(\mu_j, \Sigma_j, \pi_j)$：
- 如果分量 $j$ 的样本经常进入精英 → $\pi_j \uparrow$，分量扩张
- 如果分量 $j$ 的样本很少进入精英 → $\pi_j \downarrow$，分量收缩
- 分量权重 $\pi_j$ 直接反映分支 $j$ 在当前探索下的竞争力

**Backprop**：一次 MGIGO 运行结束后，每个分量得到一个 cost 评估结果。这些结果作为 $R_q^{\text{light}}$ 或 $R_q^{\text{full}}$（取决于 IGO 参数）回传给索引树对应分支的节点，独立更新 $Q_j^{\text{mode}}$ 和 $n_j^{\text{mode}}$。

**与 $n_j^{\text{none}}, n_j^{\text{light}}, n_j^{\text{full}}$ 的联动**：

```
Selection 决策:
  UCT_j = σ(Q_j^n + σ(Q_j^l + σ(Q_j^f))) + c√(ln N_total / n_j)
  
  如果 n_j 小:  探索项主导 → 选冷门分支
  如果 n_j 大但只有 n_j^n:  Q_j^f=0, Q_j^l=0 → Q̃_j ≈ σ(Q_j^n)
                           信息质量低, 但 n_j 大抑制了探索项
                           → 该分支需要 light/full 评估
  → Selection 自动偏向"已有足够低保真度证据但仍需高保真度验证"的分支
```

**一个 MGIGO 优化器就够了**：因为 MGIGO 的 cost_fn 接受 $(x, ctx)$，ctx 中可以传 `branch_id`。同一个自然梯度流在 $K$ 个分量上并行更新，分量间通过 Mixture 分母共享梯度结构——这就是 `MPCsolverM22.py` 当前的代码路径，唯一的变化是 cost_fn 内部用 `lax.switch(branch_id, ...)` 选择对应的拓扑。

**与逐分支独立评估的对比**：

| | 逐分支独立 IGO | 单次 K 分量 MGIGO |
|--|--------------|-----------------|
| IGO 运行次数 | $K$ 次 | $1$ 次 |
| 分量间信息共享 | 无 | 通过 Mixture 分母共享梯度 |
| 跨分支竞争 | 无（各自优化） | 精英筛选天然跨分支竞争 |
| 分量权重 | N/A | $\pi_j$ → 混合策略 → 对应分支胜率 |
| JIT 编译 | $K$ 次 | $1$ 次 |
| 适合保真度 | full（每个分支单独全跑） | light / full（一次运行评估所有分支） |
| 不适合场景 | — | 分支数 $K$ > GMM 分量数 → 需分批 |

**分层评估策略**：

| 保真度 | 用什么评估 | $R_q$ 来源 | 更新哪些计数 |
|--------|----------|-----------|------------|
| none | Constran 快评（μs，无 IGO） | 单点 cost 评估 | $Q_j^{\text{none}}, n_j^{\text{none}}$ |
| light | 单次 $K$ 分量 MGIGO, T=100, B=30 | MGIGO 分量 $j$ 的 cost | $Q_j^{\text{light}}, n_j^{\text{light}}$ |
| full | 单次 $K$ 分量 MGIGO, T=300, B=200 | MGIGO 分量 $j$ 的 cost | $Q_j^{\text{full}}, n_j^{\text{full}}$ |

### 8.5 完整 Selection 流程

```python
def select_child(node, c=1.0/np.sqrt(2)):
    """
    node: 索引树节点 with K children
    每个 child 维护:
      Q_none, Q_light, Q_full  # 三层独立 Q 值 (均 ∈ (-1,1), R_q 来自 Constran σ-饱和)
      n_none, n_light, n_full  # 三层独立计数
    
    饱和嵌套公式 (对齐 Constran _assemble_nest):
      Q̃_j = σ( Q_none + σ( Q_light + σ( Q_full ) ) )
    
    关键: 每个 Q 天然有界 (继承自 Constran cost 的 σ-饱和输出)。
         不需要额外 δ 权重——递归 σ-包裹本身就是"层层封顶"。
         重型(内层)信号穿过最多层 σ, 结构性主导。
    """
    N_total = sum(child.n_none + child.n_light + child.n_full 
                  for child in node.children)
    
    best_uct = -float('inf')
    best_child = None
    
    for child in node.children:
        n_j = child.n_none + child.n_light + child.n_full
        
        if n_j == 0:
            # 未访问 → 优先探索
            return child
        
        # ① 递归 σ-包裹: 重型在内, 轻型在外
        Q_tilde = sigma(child.Q_none + 
                        sigma(child.Q_light + 
                              sigma(child.Q_full)))
        # Q_tilde ∈ (-1, 1) — 每层 σ 保证有界
        
        # ② 归一化到 [0, 1] 以匹配 Hoeffding 界
        Q_tilde_norm = (Q_tilde + 1.0) / 2.0
        
        # ③ 探索项 + UCT
        exploration = c * np.sqrt(np.log(N_total) / n_j)
        uct = Q_tilde_norm + exploration
        
        if uct > best_uct:
            best_uct = uct
            best_child = child
    
    return best_child
```

### 8.6 与 Constran $\sigma$-饱和嵌套的同构性

注意到 $\widetilde{Q}_j$ 的嵌套结构和 Constran 的递归 cost 组装（`_assemble_nest`）是**严格同构**的：

| Constran `_assemble_nest` | 本方案 $\widetilde{Q}_j$ |
|--------------------------|------------------------|
| `inner = σ(T(objective))` | $\sigma(Q_j^{\text{full}})$ — 重型核心信号，最内层先行饱和 |
| `inner = σ(T(g_P3) + inner)` (Soft/Tunable) | $\sigma(Q_j^{\text{light}} + \sigma(Q_j^{\text{full}}))$ — 中型修正包裹重型 |
| `inner = σ(T(g_P2) + inner)` (Soft/Tunable) | $\sigma(Q_j^{\text{none}} + \sigma(Q_j^{\text{light}} + \sigma(Q_j^{\text{full}})))$ — 轻型修正包裹全部 |
| `inner = σ(𝟙[g>0](T(g)+δ) + 𝟙[g≤0]inner)` (Hard) | —（Q 值嵌套不需要 binary 分支；重型信号通过嵌套最深层获得等效主导权） |

两者都使用 $\sigma(x) = x / \sqrt{1+x^2}$ 作为饱和函数，都使用递归向外包裹。核心设计原则完全一致：**外层 $\sigma$-包裹接受"本层的贡献 + 内层的饱和输出"，再整体饱和。** 内层（重型/高优先级）信号穿过最多层 $\sigma$（1-Lipschitz）以几乎未衰减的形式传递到底，外层（轻型/低优先级）信号只穿过少量 $\sigma$，对最终输出的修正被压缩。不需要显式权重——嵌套顺序本身就是优先级。这和 Constran 中 $T(g)$（对数压缩）使得每个约束的贡献有界化、然后通过递归 $\sigma$ 层层封顶的设计完全同构。

这意味着一件事：**如果 Constran 已经 JIT 编译在 GPU 上，$\widetilde{Q}_j$ 的计算可以复用同一个 $\sigma$ 内核，零额外编译开销。**

---

## 9. 多树异构：每棵树结构独立，联合求解器统一评估

§8 的 UCT 评分和 MGIGO 多模态探索都是在**单棵树**内讨论的。但实际场景中——多区域 UC、多车型自动驾驶、多 agent 混合策略博弈——天然需要**多棵结构不同的树**并行搜索，再汇聚到同一个 RNE/IGO 求解器做联合评估。

### 9.1 为什么需要多棵树

单棵树的隐含假设：所有离散决策单元是同质的、可以用同一套分块策略和 UCT 参数管理。这在以下场景不成立：

| 场景 | 树结构差异 | 为什么不能合并成一棵树 |
|------|----------|---------------------|
| **多区域 UC** | 区域 A: 30 台火电 + 5 储能，树深 35<br>区域 B: 10 台火电 + 2 风电，树深 12 | 树深不同、决策单元类型不同、macro-action 编码不同 |
| **混合机组类型** | Baseload: 启停区间宏决策，二元分支<br>Storage: 充放区间宏决策，三元分支（充/放/闲） | 分支因子不同、UCT 公式不同（三元分支需要不同探索系数） |
| **多 agent 博弈** | Agent 0: 自车 maneuver 树，深度 2-3，分支 3-5<br>Agent 1: 他车 maneuver 树，深度 2，分支 2-3 | 各 agent 的离散决策空间独立，不应强制共用一棵树 |
| **异构硬件集群** | 树 0: GPU 0 上的大规模索引树<br>树 1: GPU 1 上的小规模索引树 | 物理隔离，需跨设备协调 |

**核心原则**：**树的结构反映决策单元的物理结构**。不同物理结构 → 不同树结构 → 不能硬拼成一棵。但它们的评估在连续参数空间（IGO/RNE）中耦合 → 需要联合求解器。

### 9.2 架构：多树 → 联合 RNE 求解器

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Tree 0          │   │ Tree 1          │   │ Tree 2          │
│ Agent: GenCo A  │   │ Agent: GenCo B  │   │ Agent: Storage  │
│ Depth: 15       │   │ Depth: 10       │   │ Depth: 5        │
│ Branch: ON/OFF  │   │ Branch: ON/OFF  │   │ Branch: 充/放/闲│
│ UCT: c=1/√2     │   │ UCT: c=1/√2     │   │ UCT: c=1/2      │
│ Blocks: [0..14] │   │ Blocks: [15..24]│   │ Blocks: [25..29]│
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         │  top-K 拓扑候选      │  top-K 拓扑候选      │  top-K 拓扑候选
         ▼                     ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│              联合 RNE 求解器 (MPC_G_MS)                        │
│                                                              │
│  N_blocks = 30 (Tree 0: 15 + Tree 1: 10 + Tree 2: 5)        │
│  block_to_agent_idx = [0]*15 + [1]*10 + [2]*5                │
│                                                              │
│  每个 Tree 的 top-K 候选 → K 个 GMM 分量                       │
│  Tree 0 分量 0..K0-1 → agent 0 的 block GMMs                 │
│  Tree 1 分量 0..K1-1 → agent 1 的 block GMMs                 │
│  Tree 2 分量 0..K2-1 → agent 2 的 block GMMs                 │
│                                                              │
│  联合评估: agent i 的 cost = f_i(joint_x, ctx)               │
│  → 各 agent 各自 minimize 自己的 cost                         │
│  → 迭代至 Nash 均衡                                          │
└──────────────────────────────────────────────────────────────┘
```

**每棵树内部**：独立的索引树，独立的三层保真度计数（$n_j^{\text{none}}, n_j^{\text{light}}, n_j^{\text{full}}$），独立的 $\widetilde{Q}_j$ 饱和嵌套。各树之间不共享路径统计量——它们面对的是不同 agent 的不同离散决策空间，共享没有意义。

**联合求解器**：只关心连续参数空间。`block_to_agent_idx` 建立了"树的 block → agent"的映射。每个 agent 的 GMM 分量数可以不同（$K_0 \neq K_1 \neq K_2$），只要每块内部的 $K$ 一致（因为 vmap 需要静态维度）。

### 9.3 多 K 值方案：每 agent 独立分量数

标准 MPC_G_MS 要求所有 $N_{\text{blocks}}$ 使用相同的 $K$（因为 `vmap` 需要静态 shape）。多树场景中不同 agent 的候选数可能不同（如 GenCo A 有 5 个启停候选，Storage 只有 2 个充放候选）。

**方案：统一的 $K_{\text{max}}$ + per-agent mask**：

```python
# 所有 block 统一用 K_max = max(K_0, K_1, K_2)
K_max = 5

# per-agent 的有效分量数
K_active = jnp.array([5, 3, 2])  # agent 0: 5, agent 1: 3, agent 2: 2

# 在采样和更新时，block_to_agent_idx 查找对应的 K_active
# 多余分量用 mask 置零权重：
def sample_with_active_mask(block_idx, key):
    agent = block_to_agent_idx[block_idx]
    k_eff = K_active[agent]
    # 分量 0..k_eff-1: 正常采样
    # 分量 k_eff..K_max-1: 权重为 0, 不会被选中
    pi_masked = pi_all[block_idx].at[k_eff:].set(0.0)
    pi_masked = pi_masked / pi_masked.sum()  # re-normalize
    ...
```

对于 $K$ 值差别不大的场景（$\leq 2$ 倍），这种 mask 方案零开销（纯 JAX 向量化）。如果差别极大（某 agent 只有 1 个候选，另一个有 10 个），建议把 $K$ 小的 agent 的多余分量用作"自由探索"（不绑定拓扑，完全自由采样），相当于给该 agent 增加了探索自由度。

### 9.4 树间协调：Master 层

多树之间有两种耦合模式，决定 Master 的协调策略：

**模式 A — 仅通过 cost 耦合（最常见）**：

各树的拓扑决策互不约束，但通过联合 cost 函数间接耦合。典型场景：
- 竞争性电力市场：GenCo A 的启停不影响 GenCo B 是否能启停，但影响电价 → 影响 B 的利润
- 多车博弈：自车左转不禁止他车直行，但双方轨迹在路口交互

协调方式：**不需要显式 Master**。各树独立搜索自己的拓扑空间，top-K 候选送入 RNE 求解器。RNE 的均衡 cost 自动反映跨树拓扑组合的质量。每棵树的 Backprop 使用各自 agent 在均衡中的 cost。

```
Tree 0 搜索 s_0  →  top-K_0 →  ┐
Tree 1 搜索 s_1  →  top-K_1 →  ├─ RNE(s_0, s_1, s_2) → (cost_0, cost_1, cost_2)
Tree 2 搜索 s_2  →  top-K_2 →  ┘     │
       ▲                              │
       └──────── Backprop ────────────┘
         Tree 0 回传 cost_0
         Tree 1 回传 cost_1
         Tree 2 回传 cost_2
```

**模式 B — 全局约束耦合**：

各树的拓扑决策受共享资源约束。典型场景：
- 多区域 UC 统一出清：总发电 = 总需求，各区域出力受联络线容量限制
- 联合调度：所有储能的 SoC 受总容量限制

协调方式：**需要 Master 在 RNE 评估后校验全局约束**。如果联合拓扑违反全局约束（如总出力 < 需求），Master 通知相关树"增加开机"或"增加放电"，对应的树在下一次 Selection 时提高相关分支的先验胜率。

```python
class MultiTreeMaster:
    def coordinate(self, trees, rne_result):
        # 校验全局约束
        total_output = rne_result.total_supply
        if total_output < self.demand - tolerance:
            # 出力不足 → 通知所有 GenCo 树：增加开机
            shortage = self.demand - total_output
            for tree in trees:
                if tree.agent_type == 'GenCo':
                    tree.bias_on_nodes(shortage / len(trees))
        
        if self.congestion_detected(rne_result):
            # 联络线阻塞 → 通知相关区域树：调整出力分布
            self.adjust_regional_trees(trees, rne_result)
```

**模式 C — 分层树（Tree of Trees）**：

子区域的局部索引树结果汇总到上层协调树。上层树不覆盖具体机组的启停路径，而是覆盖"子区域出力分配"——这是更高层的离散决策。

```
                   ┌──────────────────┐
                   │   Master 索引树   │  ← 覆盖: 各区域出力配额
                   │   Depth = 3-5     │     分支: [100, 200, 300] MW
                   └──────┬───────┬────┘
                          │       │
              ┌───────────┘       └───────────┐
              ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│ Region 0 索引树      │        │ Region 1 索引树      │
│ 给定配额: 200MW      │        │ 给定配额: 150MW      │
│ 覆盖: 本区机组启停路径 │        │ 覆盖: 本区机组启停路径 │
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           └───────────┬───────────────────┘
                       ▼
           ┌──────────────────────┐
           │  联合 IGO/RNE 评估    │
           │  cost = Σ region_cost│
           │  + 联络线越限惩罚     │
           └──────────────────────┘
```

分层树的 Backprop 需要两次：子区域索引树用局部 cost 更新，Master 索引树用总 cost 更新。两层之间通过"配额约束"耦合——子区域在给定配额下搜索，Master 调整配额。

### 9.5 与 MPC_G_MS 的代码对应

```python
from gmm_igo.MPC_G_MS import mmog_igo_rne_blocks_solver

class MultiTreeRNE:
    """
    多树异构引导树 + RNE 求解器
    
    trees: List[IndexTree] — 每棵树独立结构，覆盖各自 agent 的路径空间
    block_to_agent_idx: Array[N_blocks] — 每块归属的 agent
    tree_to_blocks: List[List[int]] — 每棵树包含哪些 block
    """
    
    def __init__(self, trees, block_to_agent_idx):
        self.trees = trees
        self.block_to_agent = jnp.array(block_to_agent_idx)
        self.N_blocks = len(block_to_agent_idx)
        self.M_agent = max(block_to_agent_idx) + 1
        
        # 每棵树 → blocks 映射
        self.tree_to_blocks = []
        for tree in trees:
            self.tree_to_blocks.append(
                [i for i, a in enumerate(block_to_agent_idx) 
                 if a == tree.agent_id]
            )
    
    def step(self, t, state, horizon_H):
        # ① 各树独立路径采样（可并行）
        joint_topologies = []
        for tree in self.trees:
            topk = tree.search(
                horizon_H, 
                n_iters=tree.config.mcts_iters,
                c=tree.config.c_uct
            )
            joint_topologies.append(topk)  # List[top-K_i candidates]
        
        # ② 组装 RNE 的 per-block 初始化
        # 每棵树将其 top-K 候选映射到对应 block 的 GMM 分量
        mu_init, S_init, pi_init = self._assemble_rne_init(joint_topologies)
        
        # ③ RNE 联合求解
        result = mmog_igo_rne_blocks_solver(
            key, T=150, dt=0.15,
            N_blocks=self.N_blocks,
            M_agent=self.M_agent,
            K=self.K_max,
            B=80, B0=40,
            dims=self._all_block_dims(),
            T_0=50,
            fitness_fn_j=self._joint_fitness_fn,
            initial_mu_k=mu_init,
            initial_L_inv_k=S_init,
            initial_v_k=pi_init,
            context={'topologies': joint_topologies},
            block_to_agent_idx=self.block_to_agent,
            M_inner=15,
        )
        
        # ④ 提取各 agent 的均衡策略
        nash_actions = {}
        for agent_id in range(self.M_agent):
            # 取 agent 的 blocks 中 pi 最高的分量
            agent_blocks = [i for i, a in enumerate(self.block_to_agent) 
                           if a == agent_id]
            best_comp = result.final_pi[agent_blocks].mean(axis=0).argmax()
            nash_actions[agent_id] = {
                'topology': joint_topologies[agent_id][best_comp],
                'continuous': result.final_mu[agent_blocks, best_comp],
            }
        
        # ⑤ Backprop: 每棵树用各自 agent 的均衡 cost 回传
        # （从 RNE 的 metrics_history 中提取 per-agent cost 轨迹）
        for i, tree in enumerate(self.trees):
            agent_cost = result.metrics['mean_fitness'][i]
            tree.backprop(joint_topologies[i], agent_cost[-1])
        
        return nash_actions
```

### 9.6 多树 vs 单树分块的决策边界

| 条件 | 用单树分块 | 用多树异构 |
|------|----------|----------|
| 所有决策单元类型相同 | ✓ | 过度设计 |
| 分支因子统一（如全是 ON/OFF） | ✓ | 不需要 |
| 决策单元属于不同 agent | ✗ | ✓ 各 agent 独立树 |
| 分支因子不同（二元 vs 三元 vs 多元） | ✗ | ✓ 每树独立 UCT 参数 |
| 树深差异 > 3× | ✗ | ✓ 深树不应拖慢浅树 |
| 跨硬件分布（多 GPU） | ✗ | ✓ 每 GPU 一棵树 |
| 分层决策（配额 → 机组） | ✗ | ✓ Tree of Trees |

**关键原则**：树的结构反映决策者的物理边界。不同的 agent → 不同的树。不同的决策单元类型（机组 vs 储能 vs 联络线）→ 可以在同一棵树的不同 block，但如果分支因子或 UCT 参数不同 → 应该分树。

### 9.7 非博弈场景的多树：单智能体异构决策单元

多树不是博弈的专利。即使在**单智能体**场景中，决策单元类型不同也天然需要多棵树：

| 场景 | 树 0 | 树 1 | 树 2 | 为什么分树 |
|------|------|------|------|----------|
| UC 混合资产 | 火电启停树（二元分支） | 储能充放树（三元分支） | 风电弃电树（连续+离散） | 分支因子不同，macro-action 编码不同 |
| 多区域单调度中心 | 区域 A 机组树（深 30） | 区域 B 机组树（深 10） | — | 树深差异大，区域间仅通过联络线容量耦合 |
| 多时间尺度 | 日前启停树（24h horizon） | 实时调频树（1h horizon） | — | horizon 不同，索引树结构不同 |
| 自动驾驶 | 路径拓扑树（路口选择） | 速度模式树（激进/保守） | 交互模式树（让行/抢行） | 决策语义不同，UCT 参数 $c$ 不同 |

**这些场景的共同点**：所有树的服务对象是同一个 agent（同一个调度中心 / 同一辆自车）。树的离散搜索各自独立，但最终汇聚到**同一个单智能体 IGO 求解器**做联合连续评估——不存在 Nash 均衡，只有一个全局 cost。

**联合评估机制——完全对齐 blockwise MGIGO**：

```
Tree 0 (火电)              Tree 1 (储能)              Tree 2 (风电)
深度 15, 二元分支          深度 5, 三元分支           深度 3, 连续+离散混合
      │                        │                        │
      │ top-K_0                 │ top-K_1                 │ top-K_2
      ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│          单智能体 blockwise MGIGO (MPCsolverM22)              │
│                                                             │
│  Block 0..14: 火电 tree 的 15 个 block                       │
│  Block 15..19: 储能 tree 的 5 个 block                       │
│  Block 20..22: 风电 tree 的 3 个 block                       │
│                                                             │
│  各 block 独立采样（GMM K_i 分量）                             │
│  拼接成全维向量 x = [x_block0, ..., x_block22]               │
│  cost_fn(x, ctx) 一次性评估全维 x                             │
│  → 单一 scalar cost                                          │
│                                                             │
│  精英筛选: B 个全维样本按 cost 排序 → top B0                  │
│  各 block 用精英样本中"属于自己维度"的那段独立更新 (μ, S, π)    │
└─────────────────────────────────────────────────────────────┘
```

**这就是 blockwise MGIGO 的当前工作方式**——各 block 策略独立采样，但更新信号来自同一个联合 cost。多树只是把这个逻辑向上推了一层：**树只管离散拓扑候选，block 只管连续参数分布，两者的界面是"树的叶节点 → block 的 GMM 分量初始化"**。

**单智能体多树的 Backprop**：

与博弈场景不同，单智能体只有一个全局 cost。Backprop 时所有树共享同一个 $R_q$：

$$R_q = \text{joint\_IGO\_cost}(s^{\text{tree0}}, s^{\text{tree1}}, s^{\text{tree2}}, x^*)$$

每棵树用同一个 $R_q$ 更新自己的 $Q_j^{\text{mode}}$：
$$Q_j^{\text{mode}} \leftarrow Q_j^{\text{mode}} + \frac{1}{n_j^{\text{mode}}} (R_q - Q_j^{\text{mode}})$$

这意味着：如果 Tree 0 选了一个坏的拓扑（火电全关），即使 Tree 1 和 Tree 2 的拓扑完美，$R_q$ 也会很高（因为供电不足），三棵树都会收到一个差的 $R_q$。这看起来不公平——Tree 1 的贡献被 Tree 0 的坏选择污染了——但这是正确的：**在单智能体场景中，不存在"公平"问题，只有全局最优。差的 $R_q$ 会驱动 Tree 0 在下一次 Selection 中避开坏分支，同时 Tree 1 和 Tree 2 也会更积极地探索（因为它们收到差评后，各分支的相对 UCT 排序变化，exploration 项可能推动尝试不同拓扑）。**

如果需要对各树的贡献做归因（例如用于诊断），可以在 IGO 后做 ablation：固定其他树的拓扑，单独评估某一棵树的变化带来的 marginal cost 改善。但这只是诊断用途，不影响 Backprop。

### 9.8 与 MPC_G_MS 的关系

多树框架与两种求解器的对应：

| 场景 | 求解器 | block_to_agent_idx | 联合 cost |
|------|--------|-------------------|-----------|
| 单智能体多树（§9.7） | MPCsolverM22 | 全 0（所有 block 归 agent 0） | 单一 cost_fn |
| 多智能体多树（§7.6, §9.2-9.5） | MPC_G_MS | 按 agent 分配 | per-agent cost_fn |

两者在代码层面只有 `block_to_agent_idx` 和 cost_fn 的签名不同——求解器内核完全复用。

### 9.9 单 GPU 可行性：为什么多树不意味着多倍算力

多棵树结构上独立，不等于算力上独立。实际上建模良好的多树系统，**总计算量远小于各树独立求解之和**，完全可以在单 GPU 上跑。某公司内部实践：**1/5 个 3090 足够处理 16 车博弈 + 3 个连续任务**。

**为什么不是 N 倍算力**：

**1. 连续参数维度的线性增长，而非指数增长**

每增加一棵树（一个 agent），增加的只是该 agent 的连续参数维度，而非整个搜索空间：

```
16 车博弈:
  每车: 3 个连续任务 (速度剖面 + 转向 + 车道偏移)
  每车: 时域 H=8, D=8 → 每车连续维度 = 3×8 = 24
  总连续维度 = 16 × 24 = 384

  RNE 的 blockwise 分解:
    总 blocks ≈ 48 (16车 × 3任务)
    每 block D_max ≤ 10
    vmap 在 block 维和样本维双层并行
    → 单次 cost 评估 ≈ 和单 agent 48-block 问题相当 (如 50-UC 的规模)
```

这和单智能体 50 台机组的 UC 问题的连续参数规模（$24 \times 50 = 1200$ 维，分 48 blocks）在同一个量级。**多 agent 博弈增加的只是 agent 间联合评估的 $M_{\text{inner}}$ 次背景采样，而非维度爆炸。**

**2. 离散搜索在各树内独立但极度稀疏**

16 辆车，每辆车在 $H=8$ 的时域内最多 1-2 个 maneuver 事件。每棵树的深度 $\leq 3$，遍历次数 $\leq 300$。16 棵树 × 300 = 4800 次层1 评估——全是 Constran 快评（μs 级），总计 $\sim 5$ms。

离散搜索的瓶颈不在树的数量，而在单棵树的深度。浅树（深度 $\leq 3$）× 多棵 = 便宜。深树（深度 30）× 1 棵 = 贵。多树恰好避免了深树。

**3. JAX JIT 编译：一次编译，所有树复用**

所有树的 cost_fn 共享同一个 JIT 编译缓存。树之间的差异只在 ctx 参数（`branch_id`、`topology`），不需要重新编译。16 个 agent 的 per-agent cost_fn 如果有相同的结构（只是参数不同），JAX 的 `vmap` + `lax.switch` 在编译期把它们融合成单次 GPU kernel launch。

**4. RNE 的 $M_{\text{inner}}$ 是精度参数，不是 agent 数乘子**

$M_{\text{inner}} = 10\sim20$（背景样本数）是固定的，不随 agent 数增长。16 车的 $M_{\text{inner}}=10$ 和 2 车的 $M_{\text{inner}}=10$ 是一样的开销。$M_{\text{inner}}$ 控制的是均衡估计的精度，而非计算复杂度对 agent 数的缩放。

**5. 单 GPU VRAM 预算估算**

```
16 车博弈 + 3 连续任务, RTX 3090 (24 GB):

  JAX 预分配:         ~2 GB  (XLA_PYTHON_CLIENT_ALLOCATOR=platform)
  JIT 编译缓存:        ~1 GB  (所有 agent 共享, 一次编译)
  
  RNE 求解器:
    GMM 参数:          ~2 MB  (48 blocks × K=3 × D=10 × float32 × 2)
    样本缓冲区:        ~50 MB (B=200 × 48 blocks × D=10 × float32)
    MC vmap 展开:     ~500 MB (M_inner=10 × B=200 rollout 并发)
    
  索引树 × 16:
    节点统计量:        ~10 MB (16 × ~100 nodes × (3×Q + 3×n) × float32)
  
  总 VRAM:            ~4-5 GB  ← 3090 的 1/5
```

**关键前提**：这些节省依赖"建模良好"——macro-action 压缩（树深 $\leq 3$）、blockwise 分块（$D \leq 10$）、合理的 $M_{\text{inner}}$ 和 $B$ 设置。如果每棵树的深度上升到 15（不压缩），每 agent 的连续维度上升到 100（不 blockwise），那么确实需要更多算力。但这属于建模问题，不是框架问题。

---

## 10. 我们不是在做 MCTS：静态树上的多保真度引导采样

### 10.1 一个根本性的澄清

这个框架借用了 MCTS 的**形式**（树、Selection、Backprop、UCB 评分），但和标准 MCTS 有本质区别。理解这个区别是理解整个 §8 的 Q 值体系的前提。

**标准 MCTS 的核心假设**：
- 树是逐步构建的（expansion 产生新节点）
- 每个节点的价值 = 从该节点出发的子树中 roll-out 回报的期望
- 叶节点的 roll-out 回报通过 Backprop 向上传播，节点价值是子节点价值的加权平均
- **这依赖马尔可夫性**：子树的回报分布只取决于当前节点状态，与到达路径无关

**我们的"森林"是什么**：
- 一旦建模完成，森林是**静态且完整的**——所有从根到叶的路径（离散变量组合）**从一开始就已经存在**
- 森林有多个 root，每个 root 对应一类独立的离散决策维度
- 每条完整路径 = 跨所有 root 的叶节点组合，对应一个确定的结构 $s$
- **所有 cost 对离散变量都是非马尔可夫的**：启动费、最小启停时间、储能 SoC 递推——一条路径的 cost 取决于其**整条路径上的决策序列**，不能分解为逐节点求和

这意味着：**不存在"从节点 $i$ 出发的子树价值"这个概念**。因为节点 $i$ 的价值依赖于它是通过哪条路径到达的（非马尔可夫），以及它通往的叶节点的完整决策序列。节点的 Q 值不是子节点 Q 值的平均——**它根本不是树结构上的可加性量**。

### 10.2 正确模型：有限路径集上的多保真度 Bandit

把我们的问题还原到最简单的形式：

**给定**：有限条路径（离散决策组合）$\mathcal{P} = \{p_1, p_2, \dots, p_M\}$。$M$ 可能很大（$2^{100}$）但通过分块 + 宏决策压缩后可控。

**目标**：找到 $p^* = \arg\min_{p \in \mathcal{P}} \text{true\_cost}(p, x^*(p))$，其中 $x^*(p)$ 是路径 $p$ 对应结构下的最优连续参数（由 IGO 求解）。

**手段**：有限评估预算。可以对任意 $p$ 进行三种保真度的评估：

| 保真度 mode | 评估方式 | 返回 | 耗时 |
|------------|---------|------|------|
| none | 给定 $p$ 的结构 $s$ + 固定连续参数 $x_0$，算一次 Constran cost | $R_q^{\text{none}}(p)$ | μs |
| light | 给定 $p$ 的结构 $s$，MGIGO T=100 优化 $x$ | $R_q^{\text{light}}(p)$ | 3-5s |
| full | 给定 $p$ 的结构 $s$，MGIGO T=300 + MC=100 优化 $x$ | $R_q^{\text{full}}(p)$ | 15-30s |

每种评估是对**同一个底层真值** $\text{true\_cost}(p)$ 的不同精度观测。none 噪声大但有偏（连续参数未优化），full 噪声小且接近真值（充分优化）。

**"树"的作用**：树不是用来传播价值的，而是用来**组织 $\mathcal{P}$ 的索引结构和共享统计量**。同一个节点下的所有路径共享该节点对应的离散决策前缀，方便做 UCB 式的 exploration-exploitation——但这纯粹是索引优化，不改变"我们在对有限个路径做多保真度评估"的底层模型。

### 10.3 在这个模型下，$Q_j^{\text{mode}}$ 是什么

$Q_j^{\text{mode}}$ 不再对应标准 MCTS 的"节点 $j$ 的价值"，而是：

> **$Q_j^{\text{mode}}$ 是对"经过分支 $j$ 的那些路径"的集合，在 mode 保真度下的**加权经验平均 cost。

因为所有经过分支 $j$ 的路径共享分支 $j$ 对应的离散选择前缀，它们的 cost 在统计上有相关性（虽然不是 IID）。当我们沿着某条路径 $p$ 做了一次 mode 评估，得到 $R_q^{\text{mode}}(p)$ 后，Backprop 沿路径经过的每个节点，更新对应分支的 $Q_j^{\text{mode}}$。这不是在传播"价值"，而是在**累计该分支对应路径集合的评估统计量**。

**关键后果**：$Q_j^{\text{mode}}$ 可以有任何内部结构（三层独立计数、增量均值、饱和嵌套组合）——只要最终的 $\widetilde{Q}_j$ 通过 $\sigma$-饱和嵌套被有界化在 $(-1, 1)$ 之内，Hoeffding 就成立。Hoeffding 只关心：① 每条臂的奖励有界 ② 计数是 $n_j$。不关心奖励的生成机制是均值还是饱和嵌套还是别的什么。

### 10.4 这意味着 §8 的 $\widetilde{Q}_j$ 公式是合法的

$$\widetilde{Q}_j = \sigma\!\Big( \sigma(Q_j^{\text{none}}) + \delta_{nl} \cdot \sigma\!\Big( \sigma(Q_j^{\text{light}}) + \delta_{lf} \cdot \sigma(Q_j^{\text{full}}) \Big) \Big)$$

- 每个 $Q_j^{\text{mode}}$ 是对路径集合的某种经验平均（增量均值）
- 每个 $Q_j^{\text{mode}}$ 先独立 $\sigma$-饱和 → 有界
- 加权嵌套组合 → 输出 $\in (-1, 1)$
- 归一化 $(\widetilde{Q}_j + 1)/2 \in (0, 1)$
- Hoeffding：有界奖励 + 计数 $n_j$ → UCB 浓度界成立

**不需要** $Q_j^{\text{mode}}$ 之间满足任何"公平比较"关系（none 的估计和 full 的估计不需要对齐），**不需要** 假设树结构上的可加性，**不需要** 马尔可夫性。所有这些东西都是标准 MCTS 需要的，但我们不做标准 MCTS。

### 10.5 "Selection"和"Backprop"在这个框架下的正确语义

**Selection**：在路径集合 $\mathcal{P}$ 中，选择下一条应该被评估的路径。通过树结构组织索引——Selection 从根走到叶，在每个节点选分支 $j$ 使得 UCT 最大。但走完整条路径后选中的是一个**完整路径** $p$，不是"一个决策"。

**Backprop**：不是传播价值。而是把这条路径 $p$ 的评估结果 $R_q^{\text{mode}}(p)$ 登记到它经过的所有节点的对应分支上。这样，下一次 Selection 走到这些节点时，看到的是"经过这个节点的那些路径最近被评估得怎么样"的统计摘要。

**树不是决策树，是索引树**。它把 $M$ 条路径组织成可搜索的结构，利用离散变量的层次性（"所有 Gen 1=ON 的路径都在这个子树下"）来高效分配评估预算。但每个叶节点代表一条完整路径，Backprop 不假设路径 cost 可以在节点间分解。

### 10.6 工程后果

**引导树可以激进，因为最终裁决是 IGO**：$\widetilde{Q}_j$ 只用于选择"下一个评估哪条路径"，不用于最终 planning。IGO 对 top-K 条路径的全量评估才是最终裁决。不是因为 $\widetilde{Q}_j$ 破坏了 Hoeffding，而是因为我们根本不需要 $\widetilde{Q}_j$ 来回答 planning 问题——那是 IGO 的工作。

**$K$ 是安全边际**：$K$ 条路径被送入 IGO 精评。如果 UCT 引导得好，$K=2\sim3$ 条就覆盖了最优路径。如果某个好路径因为早期噪声被 UCT 低估，$K$ 充当保险——IGO 会纠正。

**分块就是路径空间的 Cartesian 积分解**：当离散变量太多（$M$ 太大），把变量分成块 → 每块的路径集更小 → 跨块的联合路径是 Cartesian 积。多棵树（§9）是这个分解的自然延伸——每棵树覆盖一部分变量的路径空间。

---

## 11. 最小闭环算法评估

### 11.1 算法流程（原文）

```
Algorithm: GuideTree + IGO 最小闭环

初始化: 对所有节点 v: Q_v^mode ← 0, n_v^mode ← 0
        best_s ← None, best_C ← +∞

For phase = 1 to 3 (none, light, full):
  mode ← phase 对应的保真度
  N_iter ← phase 对应的预算

  For iter = 1 to N_iter:
    // Selection
    v ← root, path ← [root]
    While v 不是叶子:
      ∀ child v_j: UCT(v_j) = Q̃_{v_j}^norm + c·√(ln N_v / n_{v_j})
      选 j* = argmax UCT(v_j)
      v ← v_j*, path.append(v)

    // 评估
    R_raw ← IGO.evaluate(s_v, mode)
    R ← σ(R_raw)

    // 更新最优
    If R_raw < best_C: best_C ← R_raw, best_s ← s_v

    // Backprop
    For each u in path:
      Q_u^mode ← (n_u^mode·Q_u^mode + R) / (n_u^mode + 1)
      n_u^mode ← n_u^mode + 1

输出: best_s, best_C
可选: 对 best_s 再跑一次完整 IGO 做最终精搜
```

### 11.2 总体评价

**可以跑。** 这个最小闭环抓住了 §2-§10 的核心机制——静态索引树、三层保真度、UCT 引导采样、recursive σ-包裹 $\widetilde{Q}$——并且把它们组织成一个简单、可实现的顺序流程。作为"试水"原型，设计选择是合理的。

但也正因为是"最小"闭环，有几个简化在真实场景下会成为瓶颈。

### 11.3 详细评审

**① 顺序分阶段 vs 自适应交织**

当前设计：先跑完所有 $N_{\text{none}}$，再跑 $N_{\text{light}}$，再跑 $N_{\text{full}}$。

- **优点**：实现简单，每阶段内 UCT 的 $Q$ 值语义清晰（只有本阶段及之前阶段的 $Q$ 非零）
- **问题**：阶段切换时存在**冷启动**。进入 phase 2 时，所有节点的 $Q^{\text{light}} = 0$，$\widetilde{Q}$ 退化为仅依赖 $Q^{\text{none}}$ + 探索项。前 $\sim K$ 次 light 评估（$K$ 是分支数）几乎纯随机探索——相当于在 phase 1 已经获得的粗粒度认知上，又重新"盲目"了一轮。Phase 2→3 同理。

**建议**：初始原型保持顺序分阶段（简单），但预留**自适应交织**的接口。即：允许 Selection 不仅选分支 $j$，也选保真度 mode。UCD（Upper Confidence bound with Discounted fidelity）：

$$\text{UCD}(v_j, \text{mode}) = \widetilde{Q}_{v_j}^{\text{norm}} + c \sqrt{\frac{\ln N_v}{n_{v_j}}} - \lambda \cdot \text{cost}(\text{mode})$$

其中 $\text{cost}(\text{mode})$ 是该保真度的计算开销（none=0, light=3s, full=30s），$\lambda$ 平衡探索价值和计算成本。这个扩展不改变核心结构（`select_child` 多加一个 mode 维度的选择），但让预算分配更高效。

**② 阶段预算 $N_{\text{none}}, N_{\text{light}}, N_{\text{full}}$ 如何定**

没有万能的公式，但有一个下限约束和一个启发式：

- **下限**：$N_{\text{phase}} \geq K$（分支数）。否则连每个分支一次评估都做不到，UCT 完全由探索项驱动（随机游走）
- **启发式**：$N_{\text{none}} : N_{\text{light}} : N_{\text{full}} \approx K \times 5 : K \times 1 : 2\sim3$。none 最便宜（μs），多跑无害；light 次之；full 最贵，只在最有希望的 2-3 条路径上精跑
- 如果 none 和 light 共享同一个 MGIGO 的 $K$-分量并行评估（§8.4），$N_{\text{light}}$ 可以进一步减小——一次 MGIGO 运行同时评估 $K$ 个分支

**③ $\widetilde{Q}^{\text{norm}}$ 的计算公式需要在算法中显式给出**

当前伪代码写 `Q̃_{v_j}^norm` 但没有展开。应该补充：

$$\widetilde{Q}_{v_j} = \sigma\!\Big( Q_{v_j}^{\text{none}} + \sigma\!\big( Q_{v_j}^{\text{light}} + \sigma(Q_{v_j}^{\text{full}}) \big) \Big)$$

$$\widetilde{Q}_{v_j}^{\text{norm}} = (\widetilde{Q}_{v_j} + 1) / 2 \quad \in (0, 1)$$

且 $n_{v_j} = n_{v_j}^{\text{none}} + n_{v_j}^{\text{light}} + n_{v_j}^{\text{full}}$，$N_v = \sum_k n_{v_k}$。

**④ $n_{v_j} = 0$ 时的处理**

当前伪代码 `ln(N_v) / n_{v_j}` 在 $n_{v_j} = 0$ 时未定义。标准处理：

```
If n_{v_j} == 0: UCT(v_j) = +∞   // 未访问分支优先探索
```

这与 §8.5 代码中的 `if n_j == 0: return child` 一致。需要在伪代码中明确。

**⑤ $R \leftarrow \sigma(R_{\text{raw}})$ 的正确性**

I G O 返回的 $R_{\text{raw}}$ 已经是 Constran cost 的输出——本身就经过最外层 $\sigma$-饱和，$\in (-1, 1)$。再 $\sigma$ 一次是**双重饱和**，会额外压缩。参考 ConstranUser_README.md §5 的讨论：

> 双重 T 无害：小值线性区 $T(T(x)) \approx T(x)$，大值额外压缩但单调性完好

同理，双重 $\sigma$ 在小值区近似恒等，大值区额外压缩——不影响排序。对于 Backprop 中使用的 $R$（用于更新 $Q$），双重 $\sigma$ 无害。但 `best_C` 应该跟踪 $R_{\text{raw}}$（原始 Constran cost）而非 $R$（双重饱和后的值），以保留 cost 的原始尺度——当前算法已经这样做了。

**⑥ 最终精搜的必要性**

算法输出 `best_s` 是在某次评估中被标记为最优的路径。但那个评估可能只是 light 甚至 none 保真度的——不足以作为最终 planning。**可选的最终 full IGO 精搜是必需的，不是可选的**。建议改为：

```
输出:
  对 best_s 强制跑一次完整 IGO (T=300, B=200, MC=100) → 最终 (s*, x*, C*)
  如果 best_C_full > best_C (之前某次 full 评估的 cost):
    输出之前的 full 评估结果
```

### 11.4 最小闭环与完整框架的差距

| 维度 | 最小闭环 | 完整框架（本文档） | 差距 |
|------|---------|-------------------|------|
| 保真度调度 | 固定顺序三阶段 | 自适应 UCD / 按需调度 (§11.3①) | 中等（调参 vs 自动） |
| 评估并行 | 每次评估一条路径 | $K$ 分量 MGIGO 并行评估 $K$ 条路径 (§8.4) | 大（$K\times$ 加速） |
| 多树 | 单树 | 多树异构 + 联合 RNE (§9) | 大（架构级） |
| Warm Start | 无 | 时域滑窗 + GMM 继承 + 跨时步诊断 (§4.3-4.4) | MPC 场景关键 |
| 诊断回传 | 无 | IGO → 索引树诊断信号 (§2.3②) | 中（提高搜索效率） |
| 不确定处理 | 无 | Chance/DRO (§3, §7.2) | 场景相关 |

**这些差距不影响"试水"原型**——它们是在路径采样机制验证通过后逐步添加的优化。最小闭环的价值在于验证核心循环（Selection → 评估 → Backprop → $\widetilde{Q}$ 更新）是否正确工作，以及饱和嵌套 $\widetilde{Q}$ 是否确实引导到了更好的路径。

### 11.5 建议的最小实现

```python
def guide_tree_minimal(tree, igo_solver, N_none, N_light, N_full):
    # 初始化
    for node in tree.all_nodes():
        node.Q = {mode: 0.0 for mode in ['none','light','full']}
        node.n = {mode: 0   for mode in ['none','light','full']}
    best_s, best_C = None, float('inf')
    
    for phase, (mode, N_iter) in enumerate([
        (1, 'none',  N_none),
        (2, 'light', N_light),
        (3, 'full',  N_full),
    ]):
        for _ in range(N_iter):
            # Selection
            v, path = tree.root, [tree.root]
            while not v.is_leaf:
                N_v = sum(c.total_n() for c in v.children)
                best_uct, best_child = -float('inf'), None
                for child in v.children:
                    n_j = child.total_n()
                    if n_j == 0:
                        best_child = child; break  # 未访问优先
                    Q_tilde = sigma(child.Q['none'] + 
                                    sigma(child.Q['light'] + 
                                          sigma(child.Q['full'])))
                    uct = (Q_tilde + 1)/2 + (1/np.sqrt(2)) * np.sqrt(np.log(N_v)/n_j)
                    if uct > best_uct:
                        best_uct, best_child = uct, child
                v = best_child
                path.append(v)
            
            # 评估
            R_raw = igo_solver.evaluate(v.strategy, mode=mode)
            R = sigma(R_raw)  # R_raw 已饱和, 双重 σ 无害
            
            if R_raw < best_C:
                best_C, best_s = R_raw, v.strategy
            
            # Backprop
            for u in path:
                u.Q[mode] = (u.n[mode] * u.Q[mode] + R) / (u.n[mode] + 1)
                u.n[mode] += 1
    
    # 最终精搜 (必需)
    final_cost, final_x = igo_solver.evaluate_full(best_s, T=300, B=200, MC=100)
    return best_s, final_x, final_cost
```

**试水建议**：
- 先用一个已知全局最优的小 UC 问题（5-10 台机，24h）验证
- 检查 `best_s` 是否随着 phase 推进而改善（$C_{\text{after none}} \geq C_{\text{after light}} \geq C_{\text{after full}}$）
- 检查 $\widetilde{Q}$ 饱和嵌套是否产生合理的分支排序（最优分支的 $\widetilde{Q}$ 应该在 phase 3 显著高于次优分支）
- 如果 phase 2→3 的冷启动导致 UCT 在 phase 3 初期纯随机探索，考虑用 phase 2 结束时的 $Q^{\text{light}}$ 作为 phase 3 $Q^{\text{full}}$ 的 warm start（$Q^{\text{full}} \leftarrow Q^{\text{light}}$ 作为先验）
| `Energy/stochastic_uc.py` | 纯连续 IGO 滚动 MPC 实现 |
| `gmm_igo/blockwise_mgigo.py` | 分块 MGIGO 求解器（单智能体） |
| `gmm_igo/MPC_G_MS.py` | 多智能体 blockwise RNE（混合策略 Nash 均衡，§7.6, §9） |
| `gmm_igo/MPC_G_S.py` | 多智能体 RNE（每 agent 单块，§7.6） |
| `gmm_igo/MPC_G_S_V.py` | 多智能体 RNE 变体 |
| `Constraintdealer/Constran.py` | Constran 约束变换引擎（σ-饱和嵌套，§8.6 同构） |
| `Constraintdealer/ConstranUser_README.md` | Constran 用户手册（聚合语义速查 §5） |

### 11.6 Q_j 的两种更新算法

$Q_j^{\text{mode}}$ 是"经过分支 $j$ 的路径集合在 mode 保真度下的质量估计"。§8.2 建立了三层独立 $Q$ 和递归 $\sigma$-包裹 $\widetilde{Q}$ 的框架。本节给出 $Q_j^{\text{mode}}$ 本身的两种具体计算方法。

#### 算法 A：增量均值（Incremental Mean of σ(cost)）

每次评估得到 $cost_{\text{raw}} \in (-1, 1)$（Constran 输出，越小越好）。对路径上每个节点 $j$：

$$R = \sigma(cost_{\text{raw}}) \quad \in (-1, 1)$$

$$Q_j^{\text{mode}} \leftarrow \frac{n_j^{\text{mode}} \cdot Q_j^{\text{mode}} + R}{n_j^{\text{mode}} + 1}$$

$$n_j^{\text{mode}} \leftarrow n_j^{\text{mode}} + 1$$

**语义**：$Q_j$ 是经过该分支的所有评估的 $\sigma(cost)$ 经验均值。$Q_j$ 越小 = 成本越低 = 分支越好。

**性质**：
- O(1) 更新，零额外内存
- 对 cost 的绝对量值敏感——一次异常差评会拉偏均值
- 对数值尺度敏感：两条路径 cost 差 0.001，在 $\sigma$ 压缩后差异更小
- UCT 方向：$utility = 1 - \widetilde{Q}_j^{\text{norm}}$（低 cost → 低 Q → 低 $\widetilde{Q}$ → 高 utility）

#### 算法 B：精英比例（Elite Fraction）

维护每个保真度的全局精英路径列表 $\mathcal{E}_{\text{mode}}$（按 cost 升序，容量 $K$）。去重：同一路径只保留最低 cost 的条目。

每次评估得到 $(path, cost_{\text{raw}})$ 后：

1. 去重：若 $path$ 已在 $\mathcal{E}$ 中，移除旧条目
2. 二分插入 $(cost_{\text{raw}}, path)$ 到 $\mathcal{E}$
3. 裁剪 $\mathcal{E}$ 到容量 $K$
4. 全量重算所有节点的 $Q$：

$$Q_j^{\text{mode}} = \frac{|\{p \in \mathcal{E}_{\text{mode}} : j \in p\}|}{K} \quad \in \left\{0, \frac{1}{K}, \frac{2}{K}, \dots, 1\right\}$$

**语义**：$Q_j$ = top-K 精英路径中经过节点 $j$ 的比例。$Q_j = 1.0$ = 必经之路，$Q_j = 0.0$ = 死胡同。

**性质**：
- O(K·depth + log K) 更新
- 完全尺度不变——只依赖排名，不碰 cost 量值
- 对异常值免疫——一次离谱的差评不影响排名
- 对噪声鲁棒——多次噪声评估通过"投票"收敛
- UCT 方向：$utility = \widetilde{Q}_j^{\text{norm}}$（高 Q → 高 utility）
- $K$ 控制粒度：$K$ 太小 → 噪声大（一次评估就改变精英集）；$K$ 太大 → 区分度低
- 推荐 $K_{\text{none}} > K_{\text{light}} > K_{\text{full}}$（越便宜越宽松），对应 `elite_fraction_*` 参数

#### 对比

| | 算法 A (Mean) | 算法 B (Elite) |
|---|---|---|
| 更新代价 | O(1) | O(K·depth) |
| 数值依赖 | 依赖 cost 量值 | 仅依赖排名 |
| 异常值 | 敏感 | 完全免疫 |
| 噪声鲁棒 | 需要大量样本平滑 | 精英投票自然去噪 |
| Q 语义 | "平均成本多少" | "配不配做精英" |
| 信号放大 | 不放大（Δcost 小 → ΔQ 小） | 放大（排名翻转 → Q 跳变） |
| 深浅层信号传导 | 等权重 | 深层精英比例主导 $\widetilde{Q}$ |

两种算法共享相同的 $\widetilde{Q}_j$ 饱和嵌套和 UCT 框架，仅 $Q_j$ 的来源不同。工程实现中通过 `FidelityConfig.q_mode` 切换。

---

### 11.7 如何对比两种算法的寻优能力

"找到最优解"不是唯一的评价标准——两种算法在有限预算下都能找到。真正的差异在**过程质量**。以下是建议的对比维度：

#### ① 累积 Regret（核心指标）

$$R(T) = \sum_{t=1}^{T} \left( cost_{\text{best\_known}}(t) - cost_{\text{true\_optimal}} \right)$$

其中 $cost_{\text{best\_known}}(t)$ 是前 $t$ 次评估中找到的最低 cost（按 full 保真度计），$cost_{\text{true\_optimal}}$ 是暴力枚举的全局最优。Regret 越低 = 收敛越快。对比两条曲线：
- **早期**（$t < n_{\text{paths}}$）：谁更快覆盖全部可行路径？
- **中期**（$n_{\text{paths}} < t < 2 \cdot n_{\text{paths}}$）：谁更快聚焦到 top 区域？
- **后期**（$t > 2 \cdot n_{\text{paths}}$）：谁在最优路径上分配更多预算？

#### ② 分支识别精度（Root-Level Signal）

根节点的子节点中，真正最优路径所在的分支 $j^*$ 的 $Q$ 值 vs 其他分支的 $Q$ 值：

$$\text{Signal}(t) = \widetilde{Q}_{j^*}^{\text{norm}}(t) - \max_{k \neq j^*} \widetilde{Q}_{k}^{\text{norm}}(t)$$

Signal > 0 且单调递增 → 算法正确识别了最优分支。Signal 的**稳定速度**和**最大幅度**是区分算法的关键。

#### ③ 噪声放大 / 抑制比

对同一路径重复评估 $m$ 次（噪声 $\sigma$ 独立同分布），测量 $Q_j$ 的方差：

$$\text{NoiseSuppression} = \frac{\sigma^2}{\text{Var}[Q_j \text{ after } m \text{ evals}]}$$

值越大 = 去噪越好。对 mean 模式，$\text{Var}[Q] = \sigma^2 / m$（经典 $1/\sqrt{m}$ 收敛）。对 elite 模式，$Q$ 的变化只来自排名翻转——翻转概率随 $m$ 增大指数衰减。

#### ④ 预算分配效率

对每条路径 $p$，统计评估次数 $n_p$。理想情况下，$n_p$ 应该与路径质量负相关（更好的路径获得更多评估）。

$$\text{AllocEfficiency} = \text{Spearman}\rho\left( \{n_p\}, \{-cost_{\text{full}}(p)\} \right)$$

$\rho$ 越接近 -1 = 预算越集中在低成本路径上。

#### ⑤ 跨保真度信号传导

在三保真度场景中，测量 $\widetilde{Q}$ 的组成：

$$\widetilde{Q} = \sigma(Q^{\text{none}} + \sigma(Q^{\text{light}} + \sigma(Q^{\text{full}})))$$

分别观察 $Q^{\text{none}}$、$\sigma(Q^{\text{light}} + \sigma(Q^{\text{full}}))$、和完整 $\widetilde{Q}$ 各自的排名正确率。关键问题：
- none 结束后：$Q^{\text{none}}$ 能区分最优分支吗？（预期：微弱）
- light 结束后：内层信号是否增强了区分？（预期：明显改善）
- full 结束后：最内层是否结构性主导了 $\widetilde{Q}$？（预期：full 信号穿过最多层 σ，以几乎未衰减的形式到达输出）

#### ⑥ 实验设计建议

| 变量 | 设置 |
|------|------|
| 问题规模 | 2 机 (4 路径) → 5 机 (32 路径) → 10 机 (1024 路径，需分块) |
| 噪声水平 | σ ∈ {0, 0.01, 0.05, 0.10, 0.20} × 保真度递减 |
| 预算 | $N_{\text{total}} \in \{2n_{\text{paths}}, 5n_{\text{paths}}, 10n_{\text{paths}}\}$ |
| 重复次数 | ≥ 30 次（不同随机种子），报告均值和置信区间 |

**最小可行对比**：在 5 机 32 路径问题上，固定 $N_{\text{total}} = 120$，噪声 $\sigma_{\text{none}}=0.15, \sigma_{\text{light}}=0.08, \sigma_{\text{full}}=0.04$，跑 30 个随机种子，画两条 cumulative regret 曲线。这足以揭示两种算法的本质差异。

代码实现参考 `MCTSIGO/trial_5gen_multifidelity.py`（单次运行）和 `MCTSIGO/trial_uc_constran.py`（小规模验证）。

---

## 12. 模板化树结构设计：对标 Constran 的 Builder 模式

### 12.1 最小闭环 vs 原始方案：理论无差别，工程有差距

在进入模板设计之前，先回答一个关键问题：§11 的最小闭环和本文档 §2-§10 的完整方案之间，**是否存在理论上的差别？**

**没有。** 理论核心完全一致：
- 静态索引树 → 预建路径集 → 多保真度 bandit（§10）
- 递归 $\sigma$-包裹 $\widetilde{Q}_j$（§8.2）
- UCT 引导路径采样 → IGO 评估 → Backprop 登记统计量（§8.5, §2.3）
- IGO 做最终 planning，引导树不裁决（§10.5）

**工程上的差距**（§11.4 表已列）都是"加功能"而非"改理论"：自适应保真度调度替代固定三阶段、$K$ 分量 MGIGO 并行评估替代串行评估、多树异构替代单树、Warm Start 支持 MPC、诊断回传提高搜索效率。这些不影响核心循环的正确性——原型可以先跑，再逐层加功能。

### 12.2 设计目标：像 Constran 一样 `build()` 就能用

Constran 的用户体验：

```python
constraints = [Deterministic(g1, mode='hard'), Chance(g2, ...)]
cost_fn = build(my_obj, constraints)      # ← 一行组装
solver(..., fitness_fn_total=cost_fn)      # ← 直接塞给求解器
```

引导树应该提供同等级别的抽象：

```python
tree = GuideTreeBuilder(problem)           # ← 从问题描述构建树
solver = GuideTreeSolver(tree, igo)        # ← 绑定求解器
result = solver.solve(budget_config)       # ← 跑闭环
```

### 12.3 核心抽象

**`DecisionUnit`** — 单个离散决策变量：

```python
@dataclass
class DecisionUnit:
    """
    问题中的一个离散决策维度。
    
    例:
      UC:   DecisionUnit(name="Gen3", choices=["ON", "OFF"])
      驾驶: DecisionUnit(name="LaneAtIntersectionX", 
                         choices=["LEFT", "STRAIGHT", "RIGHT"])
      储能: DecisionUnit(name="Storage1_mode", 
                         choices=["CHARGE", "DISCHARGE", "IDLE"])
    """
    name: str
    choices: List[str]                     # 合法取值
    macro_encoder: Optional[Callable] = None  
    # macro_encoder: 将 choice 映射为时域约束
    #   例: "ON" → ON_interval(start_hour=0, end_hour=24)
    #       "LEFT" → LaneConstraint(lane=LEFT, from_t, to_t)
```

**`IndexTree`** — 静态索引树：

```python
@dataclass
class IndexTree:
    """
    静态预建的路径索引树。
    
    构建时:
      1. 枚举所有 DecisionUnit 的合法组合 → 路径集 P
      2. 通过宏决策压缩减少路径数
      3. 建树: 每层 = 一个 DecisionUnit, 分支 = 该 Unit 的 choices
      4. 若路径数过大, 自动触发 blockwise 分解 (§2.2)
    
    树是静态的: 构建后不再增删节点。
    Backprop 只更新统计量 (Q, n), 不改树结构。
    """
    root: TreeNode
    leaves: List[TreeNode]                 # 每条路径对应一个叶节点
    path_to_strategy: Dict[TreeNode, Any]  # 叶节点 → 完整离散策略 s
    
    @staticmethod
    def from_decision_units(
        units: List[DecisionUnit],
        macro_compress: bool = True,
        max_paths: int = 10000,
    ) -> 'IndexTree':
        """从 DecisionUnit 列表构建索引树"""
        ...
    
    @staticmethod
    def from_macro_actions(
        actions: List[MacroAction],
        horizon: int,
    ) -> 'IndexTree':
        """从时域宏决策列表构建索引树（MPC 场景）"""
        ...
```

**`FidelityConfig`** — 三层保真度参数：

```python
@dataclass
class FidelityConfig:
    """三层保真度的评估参数和预算"""
    
    # none: Constran 快评, 无 IGO
    none_enabled: bool = True
    
    # light: 轻量 IGO
    light_enabled: bool = True
    light_T: int = 100
    light_B: int = 30
    light_MC: int = 30
    
    # full: 精评 IGO
    full_enabled: bool = True
    full_T: int = 300
    full_B: int = 200
    full_MC: int = 100
    
    # 预算 (若为 None, 使用启发式 auto-budget)
    budget_none: Optional[int] = None
    budget_light: Optional[int] = None
    budget_full: Optional[int] = None
    
    # MGIGO 分量数 (≥ 分支数时一次性并行评估所有分支)
    K_components: int = 3
```

**`GuideTreeSolver`** — 闭环求解器：

```python
class GuideTreeSolver:
    """
    引导树 + IGO 闭环求解器。
    
    用法:
      solver = GuideTreeSolver(tree, igo_solver, fidelity_config)
      result = solver.solve()
    """
    
    def __init__(
        self,
        tree: IndexTree,
        igo_solver: Callable,              # IGO.evaluate(strategy, mode) -> cost
        fidelity_config: FidelityConfig = FidelityConfig(),
        uct_c: float = 1.0 / np.sqrt(2),   # UCT 探索系数
    ):
        ...
    
    def solve(self) -> GuideTreeResult:
        """运行最小闭环 (§11)"""
        ...
    
    def solve_adaptive(self) -> GuideTreeResult:
        """运行自适应保真度版本 (UCD, §11.3①)"""
        ...
```

**`GuideTreeResult`** — 求解结果：

```python
@dataclass
class GuideTreeResult:
    best_strategy: Any          # 最优离散策略 s*
    best_cost: float            # 对应 full IGO cost
    best_continuous: Any        # 对应连续参数 x*
    tree_statistics: Dict       # 最终 Q 统计量 (用于诊断/Warm Start)
    phase_costs: List[float]    # 各 phase 结束时的 best_C (用于验证单调递减)
    n_evaluations: Dict[str, int]  # 各保真度的实际评估次数
```

### 12.4 Builder 模式：从问题到求解器的一行组装

对标 Constran 的 `build(objective, constraints) → cost_fn`：

```python
def build_guide_tree_solver(
    problem: ProblemSpec,
    igo_solver: Optional[Callable] = None,
    fidelity_config: Optional[FidelityConfig] = None,
) -> GuideTreeSolver:
    """
    从问题描述一行构建引导树求解器。
    
    problem: ProblemSpec — 描述离散决策空间和 cost 函数
      - decision_units: List[DecisionUnit]
      - cost_fn: (strategy, ctx) -> float  (Constran 构建的 cost)
      - horizon: int (MPC 场景, 可选)
    
    Returns: GuideTreeSolver — 配置好、可以直接 .solve() 的求解器
    """
    # ① 从 DecisionUnit 构建索引树
    tree = IndexTree.from_decision_units(
        problem.decision_units,
        macro_compress=True,
    )
    
    # ② 若未提供 IGO 求解器, 使用默认 blockwise MGIGO
    if igo_solver is None:
        igo_solver = create_default_mgigo_solver(
            total_dim=problem.continuous_dim,
            n_blocks=estimate_blocks(problem),
        )
    
    # ③ 若未提供保真度配置, 使用启发式 auto-config
    if fidelity_config is None:
        K = sum(len(u.choices) for u in problem.decision_units)
        fidelity_config = auto_fidelity_config(
            n_paths=tree.n_paths,
            n_branches=K,
        )
    
    return GuideTreeSolver(tree, igo_solver, fidelity_config)
```

**使用示例 — UC 问题**：

```python
from guide_tree import build_guide_tree_solver, DecisionUnit
from Constraintdealer.Constran import build, Deterministic, Chance

# ① 定义离散决策空间（模板化）
decision_units = [
    DecisionUnit(name=f"Gen{g}", choices=["ON", "OFF"],
                 macro_encoder=uc_on_off_encoder(g, horizon=24))
    for g in range(10)
] + [
    DecisionUnit(name=f"Storage{s}", choices=["CHARGE", "DISCHARGE", "IDLE"],
                 macro_encoder=storage_mode_encoder(s, horizon=24))
    for s in range(2)
]

# ② Constran 构建 cost_fn
cost_fn = build(
    objective=uc_fuel_cost,
    constraints=[
        Deterministic(power_balance, mode='hard', priority=1),
        Chance(power_balance_stochastic, noise_fn=load_noise, 
               mode='hard', priority=2),
        Deterministic(ramp_limit, mode='tunable', priority=3),
    ]
)

# ③ 一行组装 + 求解
solver = build_guide_tree_solver(
    ProblemSpec(decision_units=decision_units, cost_fn=cost_fn)
)
result = solver.solve()
print(f"Best cost: {result.best_cost}")
```

**使用示例 — 驾驶路口决策**：

```python
# ① 离散决策空间
decision_units = [
    DecisionUnit(name="Intersection", choices=["LEFT", "STRAIGHT", "RIGHT"],
                 macro_encoder=lane_choice_encoder(intersection_id=0)),
    DecisionUnit(name="LaneChange_t2", choices=["LEFT_LANE", "STAY", "RIGHT_LANE"],
                 macro_encoder=lane_change_encoder(t_min=2, t_max=4)),
    DecisionUnit(name="YieldToCarB", choices=["YIELD", "GO_FIRST"],
                 macro_encoder=yield_encoder(other_agent="B")),
]

# ② Constran cost（驾驶特化，§7.1）
cost_fn = build(
    objective=progress_cost,  # 尽快到达目标
    constraints=[
        Deterministic(collision_avoidance, mode='hard', priority=1, delta=5.0),
        Deterministic(lane_bounds, mode='hard', priority=1, delta=3.0),
        Chance(safety_margin, noise_fn=perception_noise, mode='tunable', priority=2),
        Deterministic(comfort_jerk, mode='soft', priority=3),
    ]
)

# ③ 一行求解
solver = build_guide_tree_solver(
    ProblemSpec(decision_units=decision_units, cost_fn=cost_fn,
                horizon=8)  # MPC 时域
)
result = solver.solve()
```

### 12.5 与现有代码的对应关系

引导树模板化后，调用链清晰：

```
build_guide_tree_solver(problem)
  │
  ├─ IndexTree.from_decision_units(units)
  │   └─ 枚举路径 → 建静态树 → 若过大则 blockwise 分解
  │
  └─ GuideTreeSolver(tree, igo, config)
      │
      ├─ .solve()
      │   ├─ Selection: select_child() [§8.5]
      │   │   └─ Q̃_j = σ(Q_none + σ(Q_light + σ(Q_full)))  [§8.2]
      │   │
      │   ├─ Evaluation: igo_solver.evaluate(strategy, mode)
      │   │   ├─ mode='none':  Constran cost_fn(strategy, x0)  [μs]
      │   │   ├─ mode='light': MPCsolverM22(T=100, B=30, ...) [3-5s]
      │   │   └─ mode='full':  MPCsolverM22(T=300, B=200, ...)[15-30s]
      │   │
      │   └─ Backprop: Q_u^mode += (R - Q_u^mode) / (n_u^mode + 1)  [§8.2]
      │
      └─ .solve_adaptive()  [§11.3①, 后续迭代]
          └─ UCD: UCT 基础上加 -λ·cost(mode) 项

文件位置建议:
  gmm_igo/
    guide_tree.py          ← IndexTree, GuideTreeSolver, build_guide_tree_solver
    decision_unit.py       ← DecisionUnit, MacroAction, ProblemSpec
    fidelity_config.py     ← FidelityConfig, auto_fidelity_config
```

### 12.6 试水的具体步骤

1. **先写 `decision_unit.py`** — 定义 `DecisionUnit` 和 `ProblemSpec` 数据结构。这是纯数据类，无外部依赖
2. **再写 `guide_tree.py`** — 实现 `IndexTree.from_decision_units()` 和 `GuideTreeSolver.solve()`（§11.5 的 Python 实现）。依赖 `decision_unit.py` + `sigma` 函数（从 Constran 复用）
3. **在已知最优解的小问题上验证** — 用一个 5 台机 24h 的 UC 问题（最优解可通过 Gurobi 或暴力枚举获得），检查：
   - 路径枚举是否正确（所有合法启停组合都在树里）
   - `best_cost` 是否随 phase 单调递减
   - 最终 `best_strategy` 是否等于已知最优
4. **然后写 `build_guide_tree_solver()`** — 加上 auto-config 逻辑，对标 Constran 的 `build()` 接口
5. **最后加工程优化** — $K$ 分量并行 MGIGO（§8.4）、自适应 UCD（§11.3①）、Warm Start（§4.3）
| `gmm_igo/blockwise_mgigo.py` | 分块 MGIGO 求解器 |
| `Constraintdealer/Constran.py` | Constran 约束变换引擎 |
| `Constraintdealer/ConstranUser_README.md` | Constran 用户手册（聚合语义速查 §5） |
