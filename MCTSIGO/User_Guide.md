# Forest + IGO 用户指南

**离散组合搜索框架——物理语义驱动的索引结构，而非任意离散变量设置。**

---

## 1. 这不是什么

Forest 不是：

- **不是 MCTS**——没有动态扩展、没有 rollout、没有 Markov 价值传播。Forest 是静态预建的组合索引。
- **不是时序展开**——路径不是 $s_0 \to s_1 \to s_2$。路径是一个完整组合，如 `(Gen0=FULL_DAY, Gen1=PEAK_ONLY, Storage=CHARGE)`。
- **不是任意离散变量**——每个 root、每个 choice 必须对应物理/语义上真实的决策。Forest 的结构由问题决定，不由建模直觉决定。

Forest 是：

> **组合标签索引**。所有可能的离散决策组合预建为静态森林，UCT 引导搜索，IGO 优化连续参数，非马尔可夫 cost 整体评估。

---

## 2. 数据结构

### 2.1 DecisionUnit — 一个决策维度

```python
from MCTSIGO.decision_unit import DecisionUnit

# 每台机组的 commitment 模式是一个 DecisionUnit
DecisionUnit(name="Gen0", choices=["FULL_DAY", "PEAK_ONLY", "OFF"])
DecisionUnit(name="Gen1", choices=["FULL_DAY", "PEAK_ONLY", "OFF"])
DecisionUnit(name="Storage", choices=["CHARGE", "DISCHARGE", "IDLE"])
```

**关键约束**：
- `choices` 必须 ≥ 2 个，不能重复
- 每个 choice 对应真实的物理/语义决策，**不能凭空造选项**
- `macro_encoder`（可选）将 choice 映射为时域约束

### 2.2 IndexTree — 索引森林

```python
from MCTSIGO.guide_tree import IndexTree

# 方式 1: 单 root (所有决策在一个树里)
tree = IndexTree.from_decision_units([
    DecisionUnit("Gen0", ["ON", "OFF"]),
    DecisionUnit("Gen1", ["ON", "OFF"]),
])

# 方式 2: 多 root 森林 (每类决策独立 root)
forest = IndexTree.from_forest([
    ("Gen0", [DecisionUnit("Gen0", ["FULL_DAY", "PEAK_ONLY", "OFF"])]),
    ("Gen1", [DecisionUnit("Gen1", ["FULL_DAY", "PEAK_ONLY", "OFF"])]),
    ("Gen3", [DecisionUnit("Gen3", ["FULL_DAY", "OFF"])]),
])
```

**森林结构**：
- 每个 root 是一棵独立的索引树（有自己的 root → children → leaves）
- 一条完整路径 = 跨所有 root 的叶节点组合
- 路径总数 = ∏ 每个 root 的叶节点数
- 同层 root 之间无顺序依赖，通过联合 cost 隐式协调

**关键属性**：
- `roots: List[TreeNode]` — 所有 root 节点
- `leaves: List[TreeNode]` — 所有叶节点
- `n_paths: int` — 完整路径总数（跨 root 乘积）
- `all_nodes()` — DFS 遍历所有节点
- `add_root(root, leaves)` / `remove_root(root)` — 动态增减 root

### 2.3 TreeNode — 森林节点

```python
class TreeNode:
    name: str         # 对应 DecisionUnit.name
    choice: str       # 本节点的选择（如 "FULL_DAY"）
    parent: TreeNode  # 父节点
    children: list    # 子节点列表
    is_leaf: bool     # 是否叶节点
    strategy: dict    # 叶节点: 完整策略 {unit_name: choice}

    # 三层保真度统计量
    Q: Dict[mode, float]  # 'none', 'light', 'full'
    n: Dict[mode, int]    # 对应访问次数

    # UCT 计算
    compute_Q_tilde()      # σ(Q_none + σ(Q_light + σ(Q_full)))
    total_n()              # n_none + n_light + n_full
```

### 2.4 FidelityConfig — 保真度配置

```python
from MCTSIGO.fidelity_config import FidelityConfig

config = FidelityConfig(
    none_enabled=True,        # P1 结构可行性 (μs 级)
    light_enabled=True,       # IGO 轻量优化 (T=100)
    full_enabled=False,       # IGO 精评 (T=300, MC=100)

    budget_none=20,           # none 评估次数
    budget_light=12,          # light 评估次数
    budget_full=0,            # full 评估次数

    q_mode="elite",           # Q 计算: 'elite' | 'mean'
    elite_fraction_none=0.2,  # none 精英比例 → K_none
    elite_fraction_light=0.1, # light 精英比例 → K_light

    uct_c=0.7071,            # UCT 探索系数 (= 1/√2)
)
```

---

## 3. 信息流动

### 3.1 完整循环

```
Forest Selection          IGO Evaluation           Backprop / Vote
─────────────────       ─────────────────        ─────────────────

每个 root 独立 UCT:     联合 strategy 送入:        cost 回传所有 root:
  root A → child → leaf    igo_evaluate(s, mode)     tracker.update(cost, path)
  root B → child → leaf         │                    Q_j = votes_j / total_evals
  root C → child → leaf         ▼                    或 Q_j = mean(σ(cost))
       │                  IGO 优化连续参数
       ▼                  Constran 计算 cost
  strategy = {A:...,        (非马尔可夫, 整体评估)
              B:...,
              C:...}
```

### 3.2 UCT Selection

每个 root 独立运行：

```python
for child in node.children:
    if child.total_n() == 0:
        return child  # 未访问优先

    Q̃ = σ(Q_none + σ(Q_light + σ(Q_full)))
    utility = Q̃_norm           # elite 模式 (Q 越大越好)
    # 或 utility = 1 - Q̃_norm   # mean 模式 (Q 越小越好)

    UCT = utility + c · √(ln N_parent / n_child)

return argmax(UCT)
```

### 3.3 Evaluation

```python
def igo_evaluate(strategy: Dict[str, str], mode: str) -> float:
    """
    strategy: {"Gen0": "FULL_DAY", "Gen1": "PEAK_ONLY", ...}
    mode: 'none' | 'light' | 'full'
    returns: cost (越小越好), Q 自动处理方向
    """
```

**三种保真度的实现**：

| mode | 做什么 | 耗时 |
|------|--------|------|
| `none` | Constran 快评 (x=x₀, 不优化连续参数) | μs |
| `light` | IGO T=100, B≈60-80 | 0.1-0.5s |
| `full` | IGO T=300, B≈120-200, MC=30-100 | 0.5-2s |

### 3.4 Q 更新（两种模式）

**Elite 模式（增量投票）**：
```python
# 每次评估 = 一票
if path_is_in_top_K(elite_costs):
    for node in path:
        votes[node] += 1
Q_j = votes_j / total_evals  # ∈ [0,1], 越大越好
```

**Mean 模式（增量均值）**：
```python
R = σ(cost_raw)
Q_j = (n_j · Q_j + R) / (n_j + 1)  # ∈ (-1,1), 越小越好
```

---

## 4. 求解器调用

### 4.1 Cost 函数（Constran）

```python
from Constraintdealer.Constran import build, Deterministic

def obj_fn(x, ctx):
    """x ∈ R^D, ctx 包含 strategy + 场景信息"""
    p = gens.decode_power(x)  # Random Keys 编码
    on = gens.on_mask(p)
    return fuel_cost(p, on) + startup_cost(on, ctx["prev_on"])

def viol_power_balance(x, ctx):
    return max(0, demand - sum(gens.decode_power(x))) / demand

cost_fn = build(obj_fn, [
    Deterministic(viol_pmax,  mode="hard",    priority=1, delta=3.0),
    Deterministic(viol_under, mode="tunable", priority=2,
                  delta_soft=2.0, beta=0.05),  # ← β=0.05 关键!
], k_inner=0.1, jit_cost=False)

@jax.jit
def jit_cost_fn(x, ctx):
    return cost_fn(x, ctx) + 1e-5 * jnp.sum(x)  # tiebreaker
```

**关键**：
- 功率平衡用 Tunable + β=0.05（拐点 20MW，全范围在线性区）
- 物理约束用 Hard
- `jnp.where` 切断梯度——不要用在连续约束上
- `jit_cost=False` 避免嵌套 JIT，手动 `@jax.jit` 包装

### 4.2 IGO 调用（对标 Energy 用法）

```python
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc

M, D = 1, N_GEN
K, B, B0 = 3, 60, 28
dims = (D,)  # 单 block, 或 (6,6,6,6,6) 多 block

mu = jnp.zeros((M, K, D), dtype=jnp.float32)
for c in range(K):
    mu = mu.at[0, c, :].set(random.uniform(key, (D,),
        minval=-2.0, maxval=2.0))           # ← sigmoid 编码需要 [-2,2]
Li = jnp.eye(D)[None, None, :, :] * 0.5    # ← Li=0.5I, σ_sampling≈2
v  = jnp.zeros((M, K-1))

fm, fL, fp = mmog_igo_optimizer_mpc(
    key, T=150, dt=0.15, M=1, K=K, B=B, B0=B0,
    dims=dims, T_0=60,
    fitness_fn_total=jit_cost_fn,  # ← 直接传, 不用 lambda!
    initial_mu_k=mu, initial_L_inv_k=Li, initial_v_k=v,
    context=ctx,
)
```

**块分解规则（UC_README §5）**：
- 单块维度 ≤ 10
- B_total ≈ 3~4 × 决策变量总数
- B0 < B/2
- T=300, T_0=100, dt=0.15（小规模）
- K=3（GMM 分量数）

### 4.3 组装

```python
from MCTSIGO.guide_tree import GuideTreeSolver

solver = GuideTreeSolver(
    forest,           # IndexTree (森林)
    igo_evaluate,     # (strategy, mode) -> float
    fidelity_config,  # FidelityConfig
)
result = solver.solve()
print(result.best_strategy)  # {"Gen0": "FULL_DAY", "Gen1": "PEAK_ONLY", ...}
print(result.best_cost)       # +0.1038
```

---

## 5. 组合索引的哲学（最重要）

### 5.1 离散变量不是随便设置的

**错误**：
```python
# ❌ 凭空编造——没有物理意义
DecisionUnit("X1", ["A", "B", "C", "D", "E"])
DecisionUnit("X2", ["MODE_0", "MODE_1"])
```

**正确**：
```python
# ✅ 每个 choice 对应真实的物理/语义决策
# 机组 commitment 模式——受最小启停时间约束
DecisionUnit("Gen0", ["FULL_DAY", "PEAK_ONLY", "OFF"])
# 储能模式——物理上不可同时充放
DecisionUnit("Storage", ["CHARGE", "DISCHARGE", "IDLE"])
```

### 5.2 森林结构由问题决定

**层数 = 决策类别数，深度浅，但宽**。

```
错误的"直觉"结构（时序展开）:          正确的组合索引结构:
  t=0 → t=1 → t=2 → ...              Gen0 ─┬─ FULL_DAY
  每个时步一层，深度 = H                    ├─ PEAK_ONLY
  组合爆炸 + 非马尔可夫性破坏               └─ OFF
                                      Gen1 ─┬─ FULL_DAY
                                            ├─ PEAK_ONLY
                                            └─ OFF
                                      Storage ─ CHARGE / DISCHARGE / IDLE

  路径数 = 2^(N·H) ← 不可解             路径数 = 3×3×3 = 27 ← 可解
```

**核心原则**：Forest 不是时序展开，是**组合空间的索引**。每个 root 对应一类独立的决策维度。路径是完整组合，不是时序序列。

### 5.3 哪些决策该进 Forest，哪些不

| 进 Forest（真正的离散决策） | 不进 Forest（连续优化涌现） |
|---------------------------|--------------------------|
| 机组 commitment 模式（FULL_DAY/PEAK_ONLY/OFF） | 机组出力水平 (MW) |
| 储能模式（CHARGE/DISCHARGE/IDLE） | 电池充放电功率 |
| 多 agent 策略选择 | 连续轨迹参数 |
| 宏决策（启停区间、变道时机） | 速度剖面 |
| 拓扑结构选择 | 调度细节 |

**判断标准**：这个决策能不能从连续优化中"涌现"？能 → 不进 Forest。不能（受物理/逻辑约束，如 min up/down、互斥）→ 进 Forest。

### 5.4 非马尔可夫 cost 是默认假设

Cost 函数看到的是**整个组合 + 所有连续参数**：

```python
def total_cost(x, ctx):
    # x = 所有时步的连续参数拼接
    # ctx = {strategy, demand_curve, prev_state, ...}

    for h in range(H):
        fuel += fuel_cost(x_h, strategy)        # 当前时步
        startup += startup_cost(on_h, on_{h-1})  # 非马尔可夫!
        min_up += viol_if_turned_off_too_soon()   # 非马尔可夫!
        soc = battery_step(x_batt, prev_soc)      # 非马尔可夫!

    return total / H
```

**不需要**把 cost 分解到每个节点、每个时步。Forest 只管索引，cost 整体评估。

---

## 6. 完整示例

```python
from MCTSIGO.decision_unit import DecisionUnit
from MCTSIGO.guide_tree import IndexTree, GuideTreeSolver
from MCTSIGO.fidelity_config import FidelityConfig
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from Constraintdealer.Constran import build, Deterministic

# ① 定义离散决策 (物理驱动)
forest = IndexTree.from_forest([
    ("Gen0", [DecisionUnit("Gen0", ["FULL_DAY", "PEAK_ONLY", "OFF"])]),
    ("Gen1", [DecisionUnit("Gen1", ["FULL_DAY", "PEAK_ONLY", "OFF"])]),
    ("Storage", [DecisionUnit("Storage", ["CHARGE", "DISCHARGE", "IDLE"])]),
])

# ② Cost 函数 (非马尔可夫, 整体评估)
def obj_fn(x, ctx):
    p = gens.decode_power(x)  # Random Keys
    on = gens.on_mask(p)
    return fuel(p, on) + startup(on, ctx["prev_on"])

cost_fn = build(obj_fn, [
    Deterministic(viol_under, mode="tunable", priority=1,
                  delta_soft=2.0, beta=0.05),
], jit_cost=False)

@jax.jit
def jit_cost(x, ctx):
    return cost_fn(x, ctx) + 1e-5 * jnp.sum(x)

# ③ IGO 评估器
def igo_evaluate(strategy, mode):
    ctx = {"strategy": strategy, "prev_on": prev_state}
    if mode == "none":
        return float(jit_cost(x0, ctx))
    # light/full: IGO 优化
    result = mmog_igo_optimizer_mpc(
        key, T=150, dt=0.15, M=1, K=3, B=60, B0=28,
        dims=(N_GEN,), T_0=60,
        fitness_fn_total=jit_cost, ...
    )
    return best_cost

# ④ 组装 + 运行
config = FidelityConfig(
    budget_none=20, budget_light=12,
    q_mode="elite",
    elite_fraction_none=0.2, elite_fraction_light=0.1,
)
solver = GuideTreeSolver(forest, igo_evaluate, config)
result = solver.solve()
print(result.best_strategy)  # {"Gen0": "FULL_DAY", "Gen1": "PEAK_ONLY", ...}
```

---

## 7. 相关文件

| 文件 | 内容 |
|------|------|
| `MCTSIGO/decision_unit.py` | DecisionUnit, MacroAction, ProblemSpec |
| `MCTSIGO/guide_tree.py` | IndexTree (Forest), TreeNode, GuideTreeSolver, EliteTracker |
| `MCTSIGO/fidelity_config.py` | FidelityConfig, auto_fidelity_config |
| `MCTSIGO/MCTSforMPC.md` | 完整设计文档（森林模型、饱和嵌套、MPC 特化） |
| `MCTSIGO/trial_uc_forest.py` | UC 示例：5 机 72 路径 × 30D 连续，非马尔可夫 cost |
| `Energy/UC_README.md` | IGO 用法指南：Random Keys, β 参数, 块分解 |
| `Constraintdealer/ConstranUser_README.md` | Constran 用户手册 |
| `gmm_igo/MPCsolverM22.py` | 主求解器（blockwise MGIGO） |
