# 两阶段优化架构设计

> **状态**: 设计文档 — 大部分组件已实现, 两阶段编排待实现。
> **参考**: 单阶段实现见 `Carreadme.md`。本文档聚焦两阶段架构的设计与验证路径。

## 目录

1. [动机: 为什么需要两阶段](#1-动机为什么需要两阶段)
2. [总体架构](#2-总体架构)
3. [正反变换在架构中的位置](#3-正反变换在架构中的位置)
4. [Phase 1: 探索阶段](#4-phase-1-探索阶段)
5. [Phase 2: 精炼阶段](#5-phase-2-精炼阶段)
6. [外部参考→Frenet 的入口](#6-外部参考frenet-的入口)
7. [实现状态](#7-实现状态)
8. [验证路径](#8-验证路径)
9. [附录](#9-附录)
10. [多 Agent 环岛博弈](#10-多-agent-环岛博弈)
11. [实验验证记录](#11-实验验证记录)

---

## 1. 动机: 为什么需要两阶段

### 1.1 单阶段现状

当前 `Simple.py` 使用**单个 IGO 求解器**同时处理全局路径规划、精细跟踪和物理约束:

```
make_objective (Lyapunov cost, α=0 解耦)
    +
make_constraints (obs/lane/speed/acc/jerk, σ 嵌套)
    +
build_solver (solver='m22', T=300, K=3, B=64)
    ↓
一个 IGO 优化 → 执行轨迹
```

这个方案在简单场景（空直路、单障碍物）工作良好, 但在复杂场景下可能遇到困难:

| 困难 | 原因 |
|------|------|
| 多模态 (左绕 vs 右绕) | 单个 GMM 在探索和收敛之间难以平衡 |
| 全局路径 + 局部精度 | 同一组 cost 嵌套方法同时服务两个目标 |
| 约束压力不均 | 障碍物约束和 jerk 约束不同嵌套导致有限步和有限探索能力内，IGO搜的结果质量很难保证。 |

### 1.2 单阶段的关键设计决策 (已确认)

从单阶段实验中确认的结论, 两阶段继承:

- **α=0 解耦**: K 矩阵的非对角元压低最小特征值, 两通道互相拖累。解耦后各通道独立最速收敛。
- **C2 不夹紧**: C2 夹紧 + jerk 约束 (|j|≤2.0) 锁死初始横向加速度。问题是物理性的, 不是优化性的。
- **C0+C1 夹紧**: B-spline 的 P0 (位置) 和 P1 (速度) 从当前车辆状态夹紧, 保证轨迹连续性。
- **5 次 B 样条, 10 控制点, 10s 时域, 100 采样点**: 当前基配置不变。

### 1.3 两阶段的核心假设

> **假设**: 将"探索可行空间"和"精细跟踪参考"拆成两个阶段, 各自用最合适的 cost 和约束配置, 总收敛质量优于单阶段。

具体来说:
- **Phase 1** 用小 dt (0.15) 保持 GMM 多模态 → 左绕/右绕并行探索 → cost gap 自然淘汰
- **Phase 2** 用大 dt (0.30) 快速收敛 → 锁死 Phase 1 选定的方向 → Lyapunov 纯跟踪

## 2. 总体架构

### 2.1 框架图

```
  ┌─────────────┐    ┌───────────┐
  │ ReferencePath│    │  Scenario │   地图 & 场景
  │  道路几何    │    │ 障碍物+参数│
  └──────┬──────┘    └─────┬─────┘
         │                 │
         ▼                 ▼
  ┌──────────────────────────────────────────────┐
  │ frenet_traj  (frenet_traj.py)                 │
  │  to_vehicle_states    正向: Frenet → 车辆运动学│  正反变换
  │  from_vehicle_states  反向: 车辆运动学 → Frenet│  (外部参考入口)
  │  make_frenet_reference maneuver → z_ref       │
  └──────────────────────┬───────────────────────┘
         │               │
         ▼               ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Phase 1: 行为决策 (每 MPC 步)                             │
  │   solver mode: 'active' / 'aggressive'  (或 MPC_G_MS)    │
  │   warmstart: 地图多车道 (左/中/右 lane center)             │
  │   cost: 采样点 Cartesian 距离 → 目标 lane 最小化           │
  │   约束嵌套: obs ⊃ lane ⊃ speed ⊃ acc  (松约束)            │
  │   IGO: dt=0.15, K=3, GMM 各分量对应不同车道                │
  │   输出: B-spline ctrl (Frenet)                            │
  │         ├─ gen.evaluate() → z_ref ──────────→ Phase 2     │
  │         └─ to_vehicle_states → y_ref (约束检查/诊断)       │
  └──────────────────────┬──────────────────────────────────┘
         │  z_ref (Frenet, 同基直接传递, 不需要转换)
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Phase 2: 轨迹精炼 (同一 MPC 步内)                          │
  │   solver mode: 'standard' / 'conservative'               │
  │   cost: Lyapunov 纯跟踪 z_ref (8 个参考量)                │
  │   约束嵌套: jerk ⊃ acc ⊃ speed ⊃ lane ⊃ obs  (紧约束)   │
  │   IGO: dt=0.30, K=3, T_0 大 (不重置, 锁定方向)           │
  │   输出: B-spline ctrl → 最终执行轨迹                      │
  └──────────────────────┬──────────────────────────────────┘
         │
         ▼  result.x → ctrl_s, ctrl_d
  ┌──────────────────┐
  │ execute_perfect   │  直接用 plan 的 t=1 状态
  │ _tracking         │  作为下一步初始状态
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ reporting / plot  │
  │ diagnose / eval   │
  └──────────────────┘
```

### 2.2 MPC 步内流程

两个 Phase 在**同一 MPC 步内**顺序执行 — 不是两个独立进程。
两个 Phase 使用**同一套 B-spline 基** (10 控制点, 5 次, 10s 时域),
Phase 1 的 Frenet 输出**直接**作为 Phase 2 的 z_ref — 不需要 `from_vehicle_states` 转换。

```python
# 每个 MPC 步 (0.1s):
ctx  = build_context(state, ...)
mu   = warmstart_multilane(gen, state, map_lanes)  # K=3 分量各追一条车道

# Phase 1: 行为决策
#   cost: 采样点 Cartesian 距离 → 目标 lane 最小化 + Constran 约束
result_p1 = modes.solve('active', key1, ctx, mu)

# z_ref 直接由 B-spline evaluate 得到 (同基, 不需要转换)
s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
    result_p1.x[:n], result_p1.x[n:],
    ctx['s0'], ctx['s_dot0'], ctx['s_ddot0'],
    ctx['d0'], ctx['d_dot0'], ctx['d_ddot0'],
)
z_ref = {
    's_ref': s, 's_dot_ref': s_dot, 's_ddot_ref': s_ddot, 's_dddot_ref': s_dddot,
    'd_ref': d, 'd_dot_ref': d_dot, 'd_ddot_ref': d_ddot, 'd_dddot_ref': d_dddot,
}

# Phase 2: 轨迹精炼 — 同基 warmstart
ctx_p2 = {**ctx, 'z_ref': z_ref}
result_p2 = modes.solve('standard', key2, ctx_p2, result_p1)

# 执行
state = execute_perfect_tracking(result_p2.x, gen, ctx)
```

**关键**: Phase 1 的 `result_p1` (GMM 状态) 直接作为 Phase 2 的 warmstart。
两个 Phase 的 B-spline ctrl 在同一 Frenet 空间 — ctrl 向量可以直接传递, z_ref 直接可得。

### 2.3 时序预算

| 配置 | P1 (探索) | P2 (精炼) | 合计 |
|------|----------|----------|------|
| 紧凑 | T=200, B=96 | T=150, B=64 | ~544ms |
| 标准 | T=300, B=128 | T=150, B=64 | ~650ms |

544ms 在 600ms 预算内（100ms MPC 步长 + 500ms 求解余量）。

## 3. 正反变换在架构中的位置

### 3.1 两个变换的角色

正反变换实现在 [core/frenet_traj.py](core/frenet_traj.py):

| 变换 | 方向 | 在两阶段中的角色 |
|------|------|-----------------|
| `to_vehicle_states` (line 132) | Frenet → `[T, 9]` (x,y,v,ψ,a_long,a_lat,j_long,j_lat,steer) | **两个 Phase 都用**: 把 Frenet 轨迹转成物理量, 供约束检查 (v/a/j 必须满足物理极限) |
| `from_vehicle_states` (line 202) | `[T, 9]` → Frenet (s,d,s_dot,d_dot,s_ddot,d_ddot,s_dddot,d_dddot) | **外部参考→Frenet 的入口**: 把地图 waypoints、GPS、其他 planner 输出、maneuver 构建的车辆级参考转成 Frenet z_ref |

### 3.2 为什么 Phase 1→Phase 2 不需要 from_vehicle_states

Phase 1 和 Phase 2 使用**同一套 Frenet B-spline 基**。Phase 1 的 B-spline ctrl 在 Frenet 空间,
`gen.evaluate(ctrl)` 直接产出 Frenet 状态 `(s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot)` —
这些就是 z_ref。不需要先转成车辆状态再反解回来。

```
Phase 1:
  B-spline ctrl (Frenet 空间)
    ├─ gen.evaluate() → z_ref ──────────────→ Phase 2 跟踪
    └─ to_vehicle_states → y_ref [T×9] (约束检查 + 诊断)
```

### 3.3 from_vehicle_states 的正确使用场景

`from_vehicle_states` 在以下场景需要 — 当参考来自**外部、非 Frenet 空间**时:

| 场景 | 输入 | 流程 |
|------|------|------|
| 地图 waypoints / GPS | Cartesian (x,y,v,ψ,…) | `from_vehicle_states(x,y,v,ψ,…) → z_ref` |
| `make_frenet_reference` | maneuver 描述 | `_build_vehicle_reference(maneuver) → y_ref → from_vehicle_states → z_ref` |
| MPC_G_MS.py 博弈输出 | 取决于求解器输出空间 | 如果在 Cartesian/动作空间 → 需要 `from_vehicle_states` |
| 上层 planner 输出 | 车辆级轨迹 | `from_vehicle_states(y_ref) → z_ref` |

### 3.4 为什么必须走 vehicle→Frenet 管道（外部参考场景）

车辆运动学的基本关系:

```
v² = (1 − d·κ_r)² · s_dot² + d_dot²
```

如果直接在 Frenet 空间写 `s_dot_ref = v_ref` 同时 `d_dot_ref ≠ 0` (变道有横向速度),
实际车速 `v_actual = √(v_ref² + d_dot_ref²) > v_ref` — 参考本身就违反物理约束。
弯道上还有 `(1−d·κ_r)` 的修正。

**外部参考的正确管道** (所有模式统一):

```
外部参考 (GPS, 地图, maneuver 描述, 博弈输出)
    │
    ▼
构建 vehicle-level 参考 y_ref(t) = (x,y,v,ψ,a_long,a_lat,…)
    │  考虑路径几何 θ_r(s), κ_r(s)
    │  保证 v² = (1−d·κ_r)²·s_dot² + d_dot²
    ▼
from_vehicle_states(y_ref) → z_ref = (s_ref, s_dot_ref, …, d_ref, d_dot_ref, …)
    │
    ▼
Phase 2 Lyapunov cost 跟踪 z_ref
```

### 3.5 已知近似及其影响

`to_vehicle_states` 中的近似 ([frenet_traj.py:132](core/frenet_traj.py#L132)):

| 近似 | 影响 |
|------|------|
| κ_r' = 0 (忽略曲率沿弧长导数) | 直路和圆弧精确, 回旋线有误差 |
| 简化 jerk 旋转 (忽略 `2·κ_r·v·a_long` 离心 jerk 耦合项) | 弯道上 jerk 的车辆级投影有小误差 |
| 运动学自行车转向模型 (忽略轮胎侧偏) | 极限工况下转向角不精确 |

`_build_vehicle_reference` 中的近似 ([frenet_traj.py:321](core/frenet_traj.py#L321)):

| 近似 | 影响 |
|------|------|
| κ_r 在 s0 处取一次, 全时域复用 | 常数曲率路径精确, 变曲率路径有累积误差 |

**缓解**: `from_vehicle_states` 在管道末端纠正残差 — `_build_vehicle_reference` 构建的 `y_ref` 经 `from_vehicle_states` 反解后, `z_ref` 与原始 `y_ref` 的运动学关系是自洽的（因为反解用的是同一套 `to_vehicle_states` 公式的逆）。

### 3.6 在两阶段管线中的位置

```
Phase 1 (Frenet 空间, 行为决策)
    │  B-spline ctrl → gen.evaluate() → s, d, s_dot, d_dot, ...
    │  to_vehicle_states: s,d,... → [T,9] 车辆状态 → 约束评估
    │  输出: z_ref (Frenet, 同基直接传递)
    │        y_ref (车辆级, 约束检查+诊断)
    ▼  z_ref 直接传递 (不需要转换 — 同基)
Phase 2 (Frenet 空间, 轨迹跟踪)
    │  B-spline ctrl → gen.evaluate() → s, d, s_dot, d_dot, ...
    │  Lyapunov cost: (s−s_ref)², (d−d_ref)², ... 纯跟踪 z_ref
    │  to_vehicle_states: s,d,... → [T,9] 车辆状态 → 约束评估
    │  输出: 执行轨迹
```

**外部参考路径** (当参考来自非 Frenet 空间时):

```
外部输入 (GPS, 地图, maneuver, 博弈输出)
    │
    ▼
from_vehicle_states → z_ref
    │
    ▼
Phase 1 或 Phase 2 (直接跟踪)
```

## 4. Phase 1: 行为决策

### 4.1 本质

Phase 1 的**本质是行为决策**，不是轨迹优化。它回答"做什么"的问题：
- 保持当前车道还是变道？
- 走左边还是右边绕过障碍物？
- 目标速度是多少？

Phase 2 负责"怎么做"——把决策转成物理可行的轨迹。

### 4.2 输入与输出

| | 内容 | 来源 |
|------|------|------|
| **输入** | 当前车辆状态 (FrenetState) | 上一步执行结果 |
| | 地图 (ReferencePath + 多车道中心线) | 场景定义 |
| | 障碍物位置/速度 | Scenario |
| | 其他 Agent 状态 (交互场景) | 感知/预测 |
| **输出** | z_ref (8 个 Frenet 参考量) | B-spline evaluate 直接产出 |
| | y_ref [T×9] (车辆级, 约束检查+诊断) | to_vehicle_states(z_ref) |
| | 模态选择 (左/中/右车道, cost gap 决定) | GMM π 分布 |

### 4.3 Warmstart: 地图多车道

Phase 1 的 warmstart 来自**地图提供的车道中心线**。每个 GMM 分量 (K=3) 初始化为不同车道:

```python
# 每个模态对应一条车道
d_lanes = [-3.5, 0.0, 3.5]  # 左/中/右车道中心线 (Frenet d 坐标)

for k in range(K):
    ctrl_d[k] = [d_lanes[k]] * n_free    # 全车道常数 d
    ctrl_s[k] = s0 + v0 * greville        # 匀速外推
```

**为什么不用 ramp 外推**: ramp 外推 (手工构造 d 从当前值渐变到目标) → 数值爆炸, cost 无法区分模态。
全车道 warmstart → B-spline C0/C1 夹紧自动处理从当前 `d0` 到目标车道的过渡。

参考: [warmstart.py:26](planning/warmstart.py#L26) `tangent_warmstart` 已实现 Greville 匀速外推。
需要新增: `build_multilane_mu` — 多车道 GMM 初始化。

### 4.4 Cost 设计: 地图引导

Phase 1 的 cost 是**采样点到目标 lane 的 Cartesian 距离最小化**，
所有几何可行性（障碍物、车道边界）由 Constran 的 σ 嵌套约束自动处理。

```
cost_P1 = Σ_w_i · distance(y_sample_i, lane_center)²
        + Σ (s_dot − v_target)²                      ← 最轻速度引导
        + Constran σ 嵌套约束                         ← 几何可行性自动处理
```

**核心原则**: **不手工拼权重**。手工加权和 → 7 位有效数字不够用 → 数值发散。
障碍物/车道边界的安全性由 Constran 分层约束保证 (obs=`hard`, lane/speed/acc/jerk=`soft`)。

**与 Phase 2 cost 的本质区别**:

| | Phase 1 cost | Phase 2 cost |
|------|-------------|-------------|
| 目标 | 选车道, 定方向 | 跟踪 z_ref, 产轨迹 |
| 空间 | Cartesian (地图几何) | Frenet (Lyapunov 跟踪) |
| 约束角色 | 约束定义可行空间, cost 只做最轻引导 | cost 主导收敛, 约束保证物理可行 |

### 4.5 约束嵌套方向

```
obs ⊃ lane ⊃ speed ⊃ acc ⊃ jerk
```

- **障碍物在最外层** (priority=1, mode='hard'): 先确保绕行空间存在
- **内层 jerk/acc 松** (ACC=8~10, JERK=5~8): 不限制探索, 允许粗糙轨迹
- Phase 2 会重新施加紧约束 — Phase 1 的粗糙在 Phase 2 被修正

当前 `constraints.py` 的嵌套方向 (obs outer) 已满足 Phase 1 需求, 不需要反转。

### 4.6 Solver 配置

**单 Agent 场景** — 使用 IGO (通过 `solver_modes.py`):

| 模式 | T | dt | B | B0 | ACC_MAX | JERK_MAX |
|------|---|-----|---|----|---------|----------|
| `active` | 300 | 0.25 | 96 | 40 | 7.0 | 3.0 |
| `aggressive` | 400 | 0.30 | 128 | 50 | 10.0 | 5.0 |

- 小 dt (0.15~0.30): GMM 各分量移动慢 → 多模态保持 (左/中/右并行探索)
- B 大 (96~128): 更多样本覆盖多车道空间
- K=3: 三个 GMM 分量对应左/中/右车道

**交互博弈场景** — 使用 `MPC_G_MS.py`:

当场景涉及多 Agent 交互 (十字路口、匝道合流、窄路会车),
Phase 1 替换为博弈优化器:

```
Phase 1 (交互):
  MPC_G_MS.py — mixed-strategy Nash equilibrium
    ├─ 每个 Agent 一个策略块 (Block)
    ├─ 联合 cost = 各自目标 + 碰撞惩罚
    └─ 输出: 每个 Agent 的决策 (可能在 Cartesian/动作空间)
         │
         ▼ from_vehicle_states (如果输出非 Frenet)
       z_ref → Phase 2
```

MPC_G_MS.py 位置: [gmm_igo/MPC_G_MS.py](../gmm_igo/MPC_G_MS.py)。参考用例: [MultipleTest/Testgame.py](../MultipleTest/Testgame.py)。

### 4.7 多模态淘汰机制

- GMM K=3, 各分量初始化为不同车道 (左/中/右)
- IGO 自然淘汰: cost 低的模态 π 增大, 差的 π → 0
- 实验确认: T=300, B=128 足以让正确模态 dominate
- 每步重新评估 — 环境变化自动切换模态 (如新障碍物出现)

## 5. Phase 2: 精炼阶段

### 5.1 目标

跟踪 Phase 1 产出的 `z_ref`, 产出**物理可行**的最终执行轨迹:
- 满足紧约束 (jerk ≤ 2.0, acc ≤ 5.0)
- Lyapunov 收敛到 `z_ref`
- 锁定 Phase 1 选定的绕行方向

### 5.2 Cost: Lyapunov 纯跟踪

**当前状态**: Cost 函数在内部硬编码 `s_ref` 和 `d_ref=0`, 不支持外部 `z_ref`。

- `cost.py:make_objective` (line 55): K 矩阵耦合版本, `s_ref` = 指数速度曲线, `d_ref` = 0
- `cost_transform.py:make_objective_cross_order` (line 95): 跨阶耦合版本, 同样硬编码 `s_ref`

**需要的改动**: 增加可选 `z_ref` 参数, `None` 时回退到当前硬编码行为:

```python
def make_objective_cross_order(gen, omega_z=1.0, omega_w=4.0,
                                C_ba=None, C_ab=None,
                                z_ref=None):   # ← 新增
    ...
    def obj_fn(theta, ctx):
        ...
        if z_ref is not None:
            # 使用外部 z_ref — 纯跟踪
            s_ref, s_dot_ref, s_ddot_ref = \
                z_ref['s_ref'], z_ref['s_dot_ref'], z_ref['s_ddot_ref']
            d_ref, d_dot_ref, d_ddot_ref = \
                z_ref['d_ref'], z_ref['d_dot_ref'], z_ref['d_ddot_ref']
        else:
            # 退化: 当前硬编码行为
            s_ref, s_dot_ref, s_ddot_ref = _hardcoded_ref(ctx, t_arr, omega_z)
            d_ref, d_dot_ref, d_ddot_ref = 0, 0, 0
        ...
```

### 5.3 约束嵌套方向

**设计假设** (需要验证):

```
jerk ⊃ acc ⊃ speed ⊃ lane ⊃ obs
```

**与 Phase 1 相反**:
- **jerk/acc 在最外层** (priority=1~2): 先保证物理可行
- **障碍物在内层** (priority=5): `z_ref` 已解决几何, obs 仅作最后防线

**当前状态**: [constraints.py:56](planning/constraints.py#L56) `make_constraints` 的嵌套方向固定为 obs-outer。
**原型阶段**: 两个 Phase 可先用同一方向 (obs-outer)，验证两阶段架构可行后再做对照实验决定是否需要反转。

### 5.4 Solver 配置

从 `solver_modes.py` 选择:

| 模式 | T | dt | B | B0 | ACC_MAX | JERK_MAX |
|------|---|-----|---|----|---------|----------|
| `standard` | 300 | 0.20 | 64 | 30 | 5.0 | 2.0 |
| `conservative` | 300 | 0.15 | 64 | 30 | 3.0 | 1.5 |

**选型理由**:
- 大 dt (0.20~0.30): GMM 分布快速收敛到局部最优
- T_0 大 (不重置): 锁死在 Phase 1 选定的绕行方向
- B 适中 (64): 不需要多模态探索, 精细搜索即可
- 紧约束: ACC=3~5, JERK=1.5~2.0

### 5.5 同基 Warmstart

Phase 1 的 `result_p1` (GMM 状态) 直接作为 Phase 2 的 `warm_start`:

```python
result_p2 = modes.solve('standard', key2, ctx_p2, warm_start=result_p1)
```

两个 Phase 使用同一套 B-spline 基 → ctrl 维度相同 → GMM 的 μ, L, π 直接兼容。
参考: [solver_builder](gmm_igo/solver_builder.py) 的 `warm_start` 参数支持 GMM 状态继承。

### 5.6 模态切换不发散

每步 Phase 1 重新评估 — 环境变化自动切换 `z_ref`。
同一 MPC 步内 Phase 2 从当前车辆状态直接跟踪新 `z_ref`:
- Lyapunov cost 产生收敛力（类似弹簧-阻尼系统跟随移动目标）
- B-spline C1 夹紧 + jerk 约束保证轨迹连续, 无需显式平滑过渡
- 若新 `z_ref` 几何不可达, Phase 2 自然牺牲跟踪精度保约束

## 6. 外部参考→Frenet 的入口

### 6.1 什么时候需要转换

Phase 1 和 Phase 2 共用同一套 Frenet B-spline 基 — **Phase 1 的 Frenet 输出直接就是 z_ref**，
不需要任何转换。

`from_vehicle_states` 和 `make_frenet_reference` 只在以下场景需要 —
当参考来自**外部、非 Frenet 空间**时:

| 场景 | 输入 | 转换方式 |
|------|------|---------|
| Phase 1→Phase 2 (同基) | Frenet ctrl | **不需要** — z_ref 直接可得 |
| 地图 waypoints / GPS | Cartesian (x,y,v,ψ,…) | `from_vehicle_states(x,y,v,ψ,…) → z_ref` |
| Maneuver 描述 | `{type: 'lane_change', d_end: 3.5, …}` | `make_frenet_reference(maneuver) → z_ref` (内部走 `_build_vehicle_reference` + `from_vehicle_states`) |
| MPC_G_MS.py 博弈输出 | 取决于输出空间 | 如果在 Cartesian/动作空间 → `from_vehicle_states` 转换 |

### 6.2 from_vehicle_states — 底层反解

实现在 [frenet_traj.py:202](core/frenet_traj.py#L202), 逐层反解:

```
层1 (位置):  (x, y) → ref_path.cartesian_to_frenet → (s, d)
层2 (速度):  (v, ψ) → Δψ → (vt, vn) → s_dot = vt/(1−d·κ_r), d_dot = vn
层3 (加速度): (a_long, a_lat) → 旋转回 Frenet → 剥离离心/Coriolis → (s_ddot, d_ddot)
层4 (jerk):   (j_long, j_lat) → 旋转回 Frenet → (s_dddot, d_dddot)
```

### 6.3 make_frenet_reference — 高层封装

实现在 [frenet_traj.py:261](core/frenet_traj.py#L261)。
从 maneuver 描述生成 z_ref 的标准管道:

```
maneuver (如 "d → 3.5m, v → 20m/s")
    │
    ▼
_build_vehicle_reference(gen, ctx, maneuver)
    │  构建 [T, 9] 车辆级参考
    │  含曲率修正、解析 jerk
    │  保证 v² = (1−d·κ_r)²·s_dot² + d_dot²
    ▼
from_vehicle_states(y_ref) → z_ref dict (8 个参考量)
```

支持三种 maneuver 类型: `lane_change`, `cruise`, `external`。

### 6.4 Round-trip 自洽性

`to_vehicle_states` 和 `from_vehicle_states` 互为逆变换。在已知近似范围内, round-trip 精确:

```
Frenet → to_vehicle_states → vehicle [T×9] → from_vehicle_states → Frenet'
|Frenet − Frenet'| ≈ 0   (在近似精度内)
```

弯道场景 (κ_r ≠ 0): 由于 κ_r' = 0 和简化 jerk 旋转, 有轻微误差。详见 §3.5 已知近似。

### 6.5 直路退化

直路 (StraightReference): κ_r = 0, θ_r = 0, s = x, d = y。
`to_vehicle_states` 和 `from_vehicle_states` 退化到平凡形式:
- vt = s_dot, vn = d_dot
- a_t = s_ddot, a_n = d_ddot
- j_long = s_dddot, j_lat = d_dddot

## 7. 实现状态

### 7.1 已完成

| 组件 | 文件 | 状态 |
|------|------|------|
| 正变换 `to_vehicle_states` | [frenet_traj.py:132](core/frenet_traj.py#L132) | ✅ 含曲率耦合, 用于约束检查 |
| 反变换 `from_vehicle_states` | [frenet_traj.py:202](core/frenet_traj.py#L202) | ✅ 逐层反解, Round-trip 自洽 |
| 参考生成 `make_frenet_reference` | [frenet_traj.py:261](core/frenet_traj.py#L261) | ✅ 支持 lane_change / cruise / external |
| 车辆级参考构建 `_build_vehicle_reference` | [frenet_traj.py:321](core/frenet_traj.py#L321) | ✅ 含曲率修正, 解析 jerk |
| Solver 模式预编译 `SolverModes` | [solver_modes.py](planning/solver_modes.py) | ✅ 5 个模式 (conservative~emergency) |
| 跨阶耦合 Cost | [cost_transform.py](planning/cost_transform.py) | ✅ `make_objective_cross_order` + `template_coupling` |
| T 变换 Cost (退化版) | [cost_transform.py:44](planning/cost_transform.py#L44) | ✅ `make_objective_transform` |
| K 矩阵耦合 Cost (旧) | [cost.py](planning/cost.py) | ✅ α=0 解耦为默认 |
| Warmstart (Greville + GMM 继承) | [warmstart.py](planning/warmstart.py) | ✅ `tangent_warmstart` + `mpc_warmstart` |
| 约束构建 (固定嵌套) | [constraints.py](planning/constraints.py) | ✅ `make_constraints(acc_max, jerk_max)` |
| 场景配置 | [scenario.py](planning/scenario.py) | ✅ SINGLE_OFFSET, THREE_BLOCKING, EMPTY |
| 单阶段 Demo | [Simple.py](Simple.py) | ✅ 单 solver, 单 cost |
| Frenet 正反变换测试 (16 个) | [test_frenet_invert.py](eval/test_frenet_invert.py) | ✅ |

### 7.2 待实现

| # | 任务 | 涉及文件 | 依赖 |
|---|------|---------|------|
| 1 | Cost 函数支持外部 `z_ref` (Phase 2 纯跟踪) | `cost_transform.py` | 无 |
| 2 | Phase 1 cost: 地图引导版本 (Cartesian→lane 距离) | 新文件 `planning/cost_phase1.py` | 无 |
| 3 | 多车道 warmstart (`build_multilane_mu`) | `warmstart.py` 扩展 | 无 |
| 4 | 两阶段 MPC 步编排 (同基直接传递 z_ref) | 新文件 `Simple_two_phase.py` | #1, #2, #3 完成后 |
| 5 | Phase 1 + MPC_G_MS.py 集成 (交互博弈场景) | `Simple_two_phase.py` 或新文件 | #4 完成后 |
| 6 | `carreadme.md` 框架图更新 | `carreadme.md` | #4 验证后 |

### 7.3 详细改动说明

#### 任务 1: Cost 支持外部 z_ref

**文件**: `Cartest/planning/cost_transform.py`

`make_objective_cross_order` 增加可选参数 `z_ref: dict | None = None`:
- `z_ref` 为 None → 回退当前硬编码行为 (向后兼容, 单阶段仍可用)
- `z_ref` 提供时 → 使用外部参考, 不再内部生成 `s_ref` / `d_ref`

Phase 2 通过 `ctx['z_ref']` 传入, cost 内部读取:
```python
if z_ref is not None:
    s_ref, d_ref = z_ref['s_ref'], z_ref['d_ref']
    s_dot_ref, d_dot_ref = z_ref['s_dot_ref'], z_ref['d_dot_ref']
    ...
```

同样改动 `make_objective_transform`。

#### 任务 2: Phase 1 cost — 地图引导

**新文件**: `Cartest/planning/cost_phase1.py`

Phase 1 的 cost 是采样点的 Cartesian 坐标到目标 lane 中心线的距离最小化。
不手工拼权重 — 几何可行性由 Constran 约束自动处理。

```python
def make_objective_phase1(gen, lane_centers: list[float]):
    """Phase 1: 地图引导 — 采样点 Cartesian 距离→lane 最小化.

    Args:
        gen: FrenetBSplineTrajectory
        lane_centers: Frenet d 坐标列表, 如 [-3.5, 0.0, 3.5]
                      每个 GMM 分量追一条车道
    """
    def obj_fn(theta, ctx):
        s, d, s_dot, d_dot, ... = _eval_all(gen, theta, ctx)
        # Cartesian 位置 (用于障碍物距离计算在约束中)
        # cost 只做最轻引导: 速度 + 车道吸引
        v_tgt = ctx["v_ref"][0]
        return jnp.sum((s_dot - v_tgt) ** 2)
        # 车道选择由 warmstart 的 GMM 分量 + Constran 约束实现
        # 不手工加 d→lane 的 penalty — 交给 IGO 自然淘汰

    return obj_fn
```

**为什么不在 cost 中显式加 lane distance penalty**: 
实验表明手工加权 → 7 位有效数字不够用 → 数值发散。车道选择通过以下机制实现:
1. **Warmstart**: GMM K=3 分量各初始化到不同车道 → 每个分量探索自己车道附近
2. **Constran 约束**: lane 约束 (`|d| ≤ lane_hw`) 定义可行空间
3. **IGO 自然淘汰**: cost gap 淘汰不可行的分量

#### 任务 3: 多车道 Warmstart

**文件**: `Cartest/planning/warmstart.py`

```python
def build_multilane_mu(gen, s0, s_dot0, d_lanes, K=3):
    """GMM initial mu with each component on a different lane.

    Args:
        d_lanes: list of Frenet d coordinates, e.g. [-3.5, 0.0, 3.5]
        K: number of GMM components (must match len(d_lanes))
    """
    ctrl_s_base, _ = tangent_warmstart(gen, s0, s_dot0, 0.0)
    mu_list = []
    for d_lane in d_lanes[:K]:
        ctrl_d = jnp.full((gen.n_free,), d_lane)
        mu_list.append(jnp.stack([ctrl_s_base, ctrl_d]))
    return jnp.stack(mu_list, axis=0).astype(jnp.float32)
```

#### 任务 4: 两阶段 Demo

**新文件**: `Cartest/Simple_two_phase.py`

参考 `Simple.py` 的结构, MPC 步内改为 (伪代码见 §2.2):
1. `modes.solve('active', key1, ctx, mu)` → Phase 1 (行为决策)
2. `gen.evaluate(result_p1.x)` → z_ref (同基直接可得, **不需要** from_vehicle_states)
3. `modes.solve('standard', key2, ctx_p2, result_p1)` → Phase 2 (轨迹精炼)

#### 任务 5: 交互博弈集成

**涉及文件**: `Simple_two_phase.py` 或新文件

当场景有多个 Agent 时, Phase 1 替换为 MPC_G_MS.py:
```python
if n_agents > 1:
    # Phase 1: 博弈决策
    result_p1 = mmog_igo_rne_solver(joint_cost, joint_constraints, ...)
    # 每个 Agent 的输出可能不在 Frenet → 需要 from_vehicle_states
    z_ref = from_vehicle_states(result_p1[agent_i])
else:
    # Phase 1: 单 Agent 地图引导
    result_p1 = modes.solve('active', key1, ctx, mu)
    z_ref = gen.evaluate(result_p1.x)  # 同基, 直接可得
```

## 8. 验证路径

### Step 1: 单阶段基线

**目标**: 建立性能基线, 作为两阶段的对比参照。

**操作**: 跑 `Simple.py` 在以下场景, 记录关键指标:

| 场景 | 文件 | 参数 |
|------|------|------|
| 空直路巡航 | `EMPTY` | v=12→18, d=0 |
| 直路变道 | `EMPTY` + d=−3→0 | v=12→18 |
| 单障碍绕行 | `SINGLE_OFFSET` | d=−3→0, v=12→18 |
| 三障碍密集 | `THREE_BLOCKING` | d=−3→0, v=12→18 |

**记录指标**:
- 收敛时间 (d 进入 ±0.1m 目标的时间)
- d 超调量 (m)
- v 终值 (m/s)
- 约束违反 (obs/lane/acc/jerk g 值)
- 每步耗时 (ms)

### Step 2: 单阶段超参扫描

**目标**: 确认单阶段是否调参后已足够。

**操作**: 在 `THREE_BLOCKING` 场景下扫描:

| 超参 | 扫描范围 |
|------|---------|
| T (IGO 迭代) | 300, 500, 1000 |
| B (样本数) | 64, 128, 256 |
| K (GMM 模态数) | 3, 5 |
| dt (学习率) | 0.15, 0.20, 0.30 |

**决策点**: 如果增大 T/B 后单阶段能达到满意效果 (收敛时间 < 3s, 超调 < 0.5m, 无约束违反), **两阶段可能是不必要的复杂度**。跳过后续步骤, 继续优化单阶段。

### Step 3: 最小两阶段原型

**前置条件**: Step 2 确认单阶段不够。

**操作**: 实现任务 #1~#4 的最小版本, 在 `THREE_BLOCKING` 场景下跑通。

**原型简化**:
- Phase 1 cost: 地图引导 (最轻速度引导, 车道选择由 warmstart + Constran 实现)
- z_ref: Phase 1 的 B-spline evaluate 直接产出 → Phase 2 (**不经过** from_vehicle_states)
- 约束: 两个 Phase 都用当前 obs-outer 方向 (Phase 2 暂不反转)
- 不实现 MPC_G_MS.py 集成

**通过标准**: 两阶段原型不崩溃, 产出合理轨迹, 每步耗时 < 1s。
Phase 1 的 GMM 多模态能正确淘汰 (cost gap 区分左/中/右车道)。

### Step 4: 两阶段 vs 单阶段对比

**操作**: 在全部 4 个场景下对比。

**对比指标** (与 Step 1 基线对比):

| 指标 | 目标 |
|------|------|
| 收敛时间 | ≤ 单阶段的 80% |
| d 超调量 | ≤ 单阶段 |
| v 终值 | ≥ 单阶段的 90% |
| 约束违反次数 | ≤ 单阶段 |
| 每步耗时 | < 600ms |
| 多模态正确率 | Phase 1 选对车道 > 90% |

**对照实验**:
- 约束嵌套方向 (Phase 1 obs-outer, Phase 2 physics-outer) vs 两阶段都用同一方向
- 同基 warmstart vs 独立 warmstart

### Step 5: 正反变换精度

**目标**: 验证 `from_vehicle_states` 在**外部参考→Frenet**场景下的精度（不是 Phase 间桥接）。

**操作**:

1. **Forward round-trip**:
   ```
   Frenet → to_vehicle_states → vehicle → from_vehicle_states → Frenet'
   误差 = |Frenet − Frenet'| / |Frenet|
   ```
   场景: 直路 (StraightReference) + 圆弧 (CircularReference)

2. **Inverse round-trip**:
   ```
   vehicle → from_vehicle_states → Frenet → to_vehicle_states → vehicle'
   误差 = |vehicle − vehicle'| / |vehicle|
   ```

3. **Maneuver→z_ref 精度** (验证 `make_frenet_reference` 管道):
   ```
   make_frenet_reference(maneuver) → z_ref
   to_vehicle_states(z_ref) → y_ref'
   检查 y_ref' 是否满足 v² = (1−d·κ_r)²·s_dot² + d_dot²
   ```

**通过标准**:
- 直路 round-trip 误差 < 1%
- 圆弧 round-trip 误差 < 5%
- Maneuver→z_ref 的 v 分解误差 < 1%

### Step 6: Phase 1 决策质量迭代

**前置条件**: Step 4 验证两阶段可行。

**操作**: 从简单地图引导迭代到更智能的决策:

1. **基线 (地图引导)**: Phase 1 的 cost = 最轻速度引导, 车道选择完全由 warmstart + Constran 实现
2. **显式车道选择**: Phase 1 cost 加入 lane affinity term — 根据场景（障碍物分布、交通规则）偏置某些车道
3. **交互博弈**: 引入 MPC_G_MS.py 处理多 Agent 场景

**通过标准**: 迭代后的决策质量在复杂场景 (多障碍物、多 Agent) 下优于基线。

### 关键决策点汇总

```
Step 2 结束 → 单阶段是否够?
  ├─ 是 → 放弃两阶段, 继续优化单阶段
  └─ 否 → 继续 Step 3

Step 4 结束 → 两阶段是否优于单阶段?
  ├─ 是 → 继续 Step 5-6, 完善两阶段
  └─ 否 → 重新审视 Phase 1 cost 和 warmstart 设计

Step 6 结束 → 决策质量是否满足需求?
  ├─ 是 → 采用当前 Phase 1 设计
  └─ 否 → 探索 MPC_G_MS.py 博弈集成或更复杂决策逻辑
```

## 9. 附录

### A. Constructive Lyapunov 原理摘要

当前 cost 是二阶 **Constructive Lyapunov Function (CLF)**。
"Constructive" 的含义是从低阶到高阶逐层构造, 每层引入更高阶导数作为"虚拟控制输入"。

**构造层次**:

```
层0 (位置):     V₀ = ||e||²                           ← 纯几何误差
层1 (速度):     V₁ = V₀ + ||ė + K·e||²                ← ė 作为"虚拟控制"驱动 e→0
层2 (加速度):   V₂ = V₁ + ||ë + 2K·ė + K²·e||²       ← ë 作为"虚拟控制"驱动 ė→−K·e
```

每一层引入的"虚拟控制" `v_k = e^(k) + k·K·e^(k−1) + … + K^k·e` 把上一层的收敛速率绑定到 K 的特征值。

**为什么 α=0 解耦**: `K = [[ω_s, α], [α, ω_d]]`。α>0 时 s 和 d 通道互相耦合 — 纵向速度误差影响横向 cost, 优化器被迫折中。α=0.5 时 v 终值只能到目标的 80%。

### B. 误差空间线性变换摘要

当前 framework 提供两种耦合方案 ([cost_transform.py](planning/cost_transform.py)):

#### B.1 T ∈ GL(2) 同阶耦合

```
T = [[1,  α],
     [β,  1]],     det(T) = 1 − αβ > 0
```

变换后误差 `z̃ = e_s + α·e_d`, `w̃ = β·e_s + e_d`。
在 (z̃, w̃) 空间中 Lyapunov 层完全解耦 (K̃ = diag(ω_z, ω_w))。

**关键特性**: 耦合在误差定义层实现, 不在 K 矩阵层 → λ_min 不退化 → 不牺牲收敛速率。

#### B.2 跨阶耦合

将两通道的误差向量展开到 3 阶 (位置、速度、加速度), 用 6×6 分块矩阵做跨阶混合:

```
[𝐳_A]   [ I      C_{B→A} ] [𝐞_A]      𝐞_A = [e_s, ė_s, ë_s]ᵀ
[𝐳_B] = [C_{A→B}   I      ] [𝐞_B]      𝐞_B = [e_d, ė_d, ë_d]ᵀ
```

例如 `C_{B→A}[0,1] ≠ 0` (横向速度→纵向位置): 变道时横向速度峰值出现, 纵向通道"预知"横向正在运动, 提前调整。

**稳定性条件**: `det(I − C_{B→A}·C_{A→B}) ≠ 0`

**退化到同阶耦合**: `C_{B→A} = α·I`, `C_{A→B} = β·I`

### C. solver_modes 配置速查

| 模式 | T | dt | B | B0 | ω_z | ω_w | C_ba | C_ab | ACC | JERK |
|------|---|-----|---|----|-----|-----|------|------|-----|------|
| `conservative` | 300 | 0.15 | 64 | 30 | 1.0 | 4.0 | 0 | 0 | 3.0 | 1.5 |
| `standard` | 300 | 0.20 | 64 | 30 | 1.0 | 4.0 | 0 | 0 | 5.0 | 2.0 |
| `active` | 300 | 0.25 | 96 | 40 | 1.5 | 6.0 | 轻度 | 轻度 | 7.0 | 3.0 |
| `aggressive` | 400 | 0.30 | 128 | 50 | 2.0 | 8.0 | 中度 | 中度 | 10.0 | 5.0 |
| `emergency` | 500 | 0.35 | 128 | 60 | 2.0 | 8.0 | 全下三角 | 中度 | 15.0 | 8.0 |

两阶段使用建议:
- **Phase 1**: `active` 或 `aggressive` (探索, 松约束)
- **Phase 2**: `standard` 或 `conservative` (精炼, 紧约束)

### D. 文件索引

| 文件 | 角色 |
|------|------|
| [core/frenet_traj.py](core/frenet_traj.py) | B-spline 轨迹评估, 正反变换, 参考生成 |
| [core/reference_path.py](core/reference_path.py) | 参考线 (StraightReference, CircularReference) |
| [core/vehicle_model.py](core/vehicle_model.py) | PointMassModel (摩擦圆积分) |
| [planning/cost.py](planning/cost.py) | 旧 Lyapunov cost (K 矩阵耦合) |
| [planning/cost_transform.py](planning/cost_transform.py) | 新 Lyapunov cost (T 变换 + 跨阶耦合) |
| [planning/constraints.py](planning/constraints.py) | 约束构建 (obs/lane/speed/acc/jerk) |
| [planning/solver_modes.py](planning/solver_modes.py) | 5 个预编译 solver 模式 |
| [planning/warmstart.py](planning/warmstart.py) | Greville warmstart + GMM 继承 |
| [planning/scenario.py](planning/scenario.py) | 场景配置 |
| [execution/execute.py](execution/execute.py) | FrenetState, execute_perfect_tracking |
| [eval/test_frenet_invert.py](eval/test_frenet_invert.py) | 正反变换 16 个测试 |
| [Simple.py](Simple.py) | 单阶段 MPC demo |

---

## 10. 多 Agent 环岛博弈

> **状态**: 设计文档 — 尚未实现。依赖 `CircularReference`（未实现）、MPC_G_MS.py（已实现）。

### 10.1 为什么是环岛

环岛场景的独特价值在于 **κ_r ≠ 0 全程非零** — 弯道是常态，不是特殊情况。
相比左转十字路口（可用 StraightReference 近似），环岛强制检验每个组件的弯道行为：

| 组件 | 环岛检验什么 |
|------|------------|
| **`CircularReference`** | κ_r = 1/R 常数。s→(x,y,θ,κ)→cartesian_to_frenet round-trip（**尚未实现！**） |
| **`to_vehicle_states`** | Jacobian `(1-d·κ_r)` 在 d 变化时非平凡。离心项 `κ_r·vt·s_dot`、Coriolis 项始终活跃 |
| **`from_vehicle_states`** | 反向解耦：剥离离心和 Coriolis。与 forward 的 round-trip 精度在弯道上 |
| **B-spline (Frenet)** | s 沿弧线、d 横向偏移。C0+C1 夹紧在 κ_r≠0 时仍正确 |
| **Constran lane** | `|d| ≤ lane_hw` 在弯道上 — d=0 是环道中心线（不是被 κ_r 扭曲的 Cartesian y） |
| **Constran speed/acc/jerk** | v/a/j 通过 `to_vehicle_states` → 曲率耦合自动进入物理约束 |
| **Phase 1 cost** | Cartesian 空间中距目标车道 — 目标车道在弯道上，必须通过 ref_path 正确映射 |
| **MPC_G_MS** | 多 Agent 在共享 CircularReference 上博弈：同时采样、同时评估、同时更新 |
| **GMM warmstart** | K=3 模态 = {加速进入, 匀速等待, 减速让行}，在弯道 Frenet 空间中初始化 |

如果环岛跑通了，每个之前设计的细节都被验证了。

### 10.2 场景设计

#### 几何

```
          入口A (ego 进入)
            │
            ▼
      ╭──────────╮
     ╱            ╲
    │   环岛中心    │  ← CircularReference, 半径 R=20m
     ╲            ╱
      ╰──────────╯
            ▲
            │
          入口B (Agent 2 进入)

       Agent 1 已在环岛内环行 (逆时针)
```

#### Agent 角色

| Agent | 参考路径 | 行为 | 初始状态 |
|-------|---------|------|---------|
| **Ego** (Agent 0) | 入口直路 → CircularReference → 出口直路 | 进入环岛 → 环行 → 离开 | 入口直路上, v=8m/s, 准备进入 |
| **环行车** (Agent 1) | CircularReference | 已在环内，沿圆环行 | 环道上, v=10m/s, s=πR (ego 对侧) |
| **入口B车** (Agent 2) | 入口直路 → CircularReference | 从另一入口进入 | 入口直路上, v=8m/s, 准备进入 |

#### 简化假设（原型阶段）

1. **所有 Agent 使用同一个 `CircularReference`** — 入口/出口简化为环道上特定 s 位置的 d 方向偏移
2. **Agent 1 始终在环行** — 不离开环岛，简化博弈
3. **不考虑 s 参数的周期性** — B-spline 时域内 s 单调递增（不绕满一圈）

### 10.3 CircularReference

`CircularReference` 当前**未实现**（`reference_path.py` 只有 `StraightReference`）。
规格如下：

```python
class CircularReference(ReferencePath):
    """圆弧参考路径, 弧长参数化.

    Args:
        R:     圆弧半径 (m)
        cx, cy: 圆心 Cartesian 坐标
    """

    def evaluate(self, s):
        """Path geometry at arc-length s.

        s=0 在圆心正上方 (cx, cy+R), 逆时针递增。
        """
        θ = s / self.R                    # 弧长参数化
        x = self.cx + self.R * jnp.cos(θ - jnp.pi/2)
        y = self.cy + self.R * jnp.sin(θ - jnp.pi/2)
        θ_r = θ                            # 切线方向 = 角度本身
        κ_r = jnp.full_like(s, 1.0 / self.R)  # 常数曲率
        return x, y, θ_r, κ_r

    def cartesian_to_frenet(self, x, y):
        """Cartesian → Frenet 闭式反解.

        angle = atan2(y-cy, x-cx) + π/2   → 归一化到 [0, 2π)
        s = R · angle
        d = R − √((x−cx)² + (y−cy)²)      ← 正d = 环内
        """
        dx = x - self.cx
        dy = y - self.cy
        angle = jnp.arctan2(dy, dx) + jnp.pi / 2
        angle = angle % (2 * jnp.pi)
        s = self.R * angle
        dist = jnp.sqrt(dx**2 + dy**2)
        d = self.R - dist
        return s, d
```

**关键特性**:
- `κ_r = 1/R` 常数 — 所有曲率耦合项非零且恒定
- `cartesian_to_frenet` 闭式解 — 不需要 Newton 迭代
- `s` 以 `2πR` 为周期 — 需要在 B-spline 评估时处理 wrap-around

### 10.4 Block 结构与 Joint Sample

使用 MPC_G_MS.py 的 Blocks 版本（`mmog_igo_rne_blocks_solver`）。

```
N_blocks = 6   (3 agents × 2 channels: s + d)
M_agent  = 3

block_to_agent_idx = [0, 0,    ← Agent 0 (ego):     s-ctrl, d-ctrl
                       1, 1,    ← Agent 1 (环行车):   s-ctrl, d-ctrl
                       2, 2]    ← Agent 2 (入口B车):  s-ctrl, d-ctrl

dims = [gen.n_free] × 6   (所有 Block 维度相同)
```

**Joint sample 内存布局** (长度 = 6 × n_free):

```
offset 0:           Agent 0 s-channel ctrl  ← Block 0
offset n_free:      Agent 0 d-channel ctrl  ← Block 1
offset 2*n_free:    Agent 1 s-channel ctrl  ← Block 2
offset 3*n_free:    Agent 1 d-channel ctrl  ← Block 3
offset 4*n_free:    Agent 2 s-channel ctrl  ← Block 4
offset 5*n_free:    Agent 2 d-channel ctrl  ← Block 5
```

**解码函数**:

```python
def decode_joint_to_per_agent(joint_x_flat, block_to_agent_idx, dims, n_free):
    """将扁平 joint sample 解码为 per-agent (ctrl_s, ctrl_d)."""
    ctrls = {}  # agent_idx → {'s': ctrl_s, 'd': ctrl_d}
    offset = 0
    for block_idx, agent_idx in enumerate(block_to_agent_idx):
        ctrl = joint_x_flat[offset:offset + n_free]
        if agent_idx not in ctrls:
            ctrls[agent_idx] = {}
        if block_idx % 2 == 0:  # s-channel
            ctrls[agent_idx]['s'] = ctrl
        else:                    # d-channel
            ctrls[agent_idx]['d'] = ctrl
        offset += n_free
    return ctrls
```

参考: [MPC_G_MS.py:184](../gmm_igo/MPC_G_MS.py#L184) `mmog_igo_rne_blocks_solver`。
参考用例: [Trackgame.py](../MultipleTest/Trackgame.py) ego 的 2-block 设计 (acc + steer)。

### 10.5 Cost 构造

**核心原则**: **不手工拼权重**。所有约束（包括他车碰撞）全部走 Constran 自动 σ 嵌套。

```
cost_agent_i = Constran.build(
    objective_fn = goal_cost_i,    ← 裸目标 (无权重)
    constraints  = [
        lane_g,                    ← priority=1 (内层)
        speed_g,                   ← priority=2
        acc_g,                     ← priority=3
        jerk_g,                    ← priority=4
        collision_g,               ← priority=5 (外层: 与他车的碰撞)
    ]
)
```

**嵌套语义**:
- 内层 (lane/speed/acc/jerk): 物理可行约束 — 满足时才谈得上安全
- 外层 (collision): 与他车的安全距离 — 在物理可行的前提下避免碰撞
- Constran 自动处理 σ 嵌套: 外层满足时内层信号无损; 外层违反时压制内层

#### goal_cost_i: 时空联合 + Lyapunov 指导

**核心设计**: Phase 1 的 goal_cost 是**时空联合**的 — 它描述"在什么时间、到达什么位置"。
s(t) 和 d(t) 构成一条时空曲线。Lyapunov 理论保证这条曲线不振荡（用到层0+1），但不要求
加速度/jerk 质量（那是 Phase 2 + Constran 紧约束的职责）。

```
goal_cost_P1 = Σ e_d²              ← 空间 (层0): 横向位置正确
             + Σ e_t²              ← 时间 (层0): 进度速率正确
             + Σ (d_dot + ω_d·e_d)² ← 层1: d 通道不振荡 (Lyapunov 虚拟控制)
```

**不包含加速度/jerk 误差** — Phase 1 不关心"如何实现"，只关心"去哪、多快"。
加速度/jerk 由 Constran loose 约束 (ACC=8~10, JERK=5~8) 兜底。

**s 通道**: 不追求精确位置跟踪。s 通道的"目标"是时间进度 — `s_dot ≈ v_ring`
代表"在合理时间内到达"。精确的 s 位置由 Phase 2 的 Lyapunov cost 负责。

**d 通道**: 使用 Lyapunov 层1 虚拟控制 `v1_d = d_dot + ω_d·e_d`。
保证横向收敛不振荡。ω_d 应 ≤ Phase 2 的 ω_d (通常 4.0~6.0)，
确保 Phase 1 不要求 Phase 2 做不到的横向收敛速率。

**Ego** (进入→环行→离开):

```python
def goal_cost_ego(theta, ctx, gen):
    """时空联合 goal_cost: 进入环道 → 环行 → 离开.

    空间 (d): 进入期 d→0, 环行期 d=0, 离开期 d→d_exit
    时间 (s_dot): 全程保持 v_ring 的进度速率
    Lyapunov 层1: 保证 d 收敛不振荡
    """
    s, d, s_dot, d_dot, s_ddot, d_ddot, ... = gen.evaluate(
        theta[:n], theta[n:], ...)

    T = len(s)
    t_arr = jnp.arange(T) * gen.dt
    t_enter = int(T * 0.3)
    t_exit  = int(T * 0.7)

    # ── z_target(t): 时空目标曲线 ──
    d_target = jnp.where(t_arr < t_enter * gen.dt, 0.0,      # 进入: d→0
                jnp.where(t_arr < t_exit * gen.dt, 0.0,       # 环行: d=0
                          ctx['d_exit']))                      # 离开: d→d_exit
    s_dot_target = jnp.full(T, ctx['v_ring'])                  # 匀速进度

    # ── 空间误差 (层0) ──
    e_d = d - d_target

    # ── 时间误差 (层0) ──
    # s_dot 偏离 v_ring → 进度太快或太慢
    e_t = s_dot - s_dot_target

    # ── Lyapunov 层1: d 通道虚拟控制 ──
    # v1_d = d_dot + ω_d·e_d → 0 保证横向收敛不振荡
    omega_d = ctx.get('omega_d', 4.0)  # ≤ Phase 2 的 ω_d
    v1_d = d_dot + omega_d * e_d

    return (jnp.mean(e_d**2)     # 空间
          + jnp.mean(e_t**2)     # 时间
          + jnp.mean(v1_d**2))   # 收敛光滑性
```

**环行车** (Agent 1):

```python
def goal_cost_circulating(theta, ctx, gen):
    """时空联合 goal_cost: 保持车道, 匀速环行."""
    s, d, s_dot, d_dot, ... = gen.evaluate(theta[:n], theta[n:], ...)

    e_d = d - ctx['d_lane']
    e_t = s_dot - ctx['v_ring']
    omega_d = ctx.get('omega_d', 4.0)
    v1_d = d_dot + omega_d * e_d

    return (jnp.mean(e_d**2)
          + jnp.mean(e_t**2)
          + jnp.mean(v1_d**2))
```

**入口B车** (Agent 2): 与 ego 同构，不同初始 s 和进入时机。

#### 分量可行性评估

Phase 1 的 GMM (K=3) 优化结束后：
- 每个分量 k 有 cost_k 和混合权重 π_k
- **cost 越小 → π 越大**（IGO 的自然淘汰机制）
- 直接选 **max-π 分量**的 μ 作为 z_ref → Phase 2

不需要多指标综合判断 — 好的 cost 构造（时空联合 + Constran 约束）保证
π 分布直接反映可行性。

#### collision_g: 几何 RSS 碰撞约束 (Constran Deterministic)

**核心思路**: 不依赖 Frenet s/d 解析分离，直接在 B-spline 的 T=100 个 Cartesian 采样点上做时空重叠检测。
利用已有采样点评估冲突区域时间，不需要额外解析求解。

```
对每个采样时刻 t (T=100):
  dist[t] = √((x_i−x_j)² + (y_i−y_j)²)                          ← Cartesian 距离
  safe[t] = v_i·ρ + v_i²/(2a) + v_j·ρ + v_j²/(2a) + margin     ← RSS 制动距离
  violation[t] = max(0, safe[t] − dist[t])                       ← 违反量
```

**为什么用几何 RSS 而非 Frenet RSS**:
- Cartesian 距离自动处理环岛 s 周期性和曲率 — 不需要手动 `mod 2πR`
- 速度相关的 `safe[t]` 比固定 `safe_s`/`safe_d` 更物理（高速需要更大安全距离）
- B-spline 已经产出了所有采样点的 Cartesian 坐标和速度 — 直接复用，零额外开销

```python
def make_collision_g_geometric(gen, agent_idx, all_init_states,
                                rho=0.1, a_brake=8.0, margin=2.0):
    """几何 RSS 碰撞约束 — Cartesian 采样点 + 冲突区域评估.

    Args:
        rho:       RSS 反应时间 (s)
        a_brake:   最大制动减速度 (m/s²)
        margin:    最小静态安全距离 (m)

    作为 Constran Deterministic 约束:
      - mode='hard': 安全底线
      - priority=5: 最外层
      - aggregate='max': 任何时刻违反即触发
    """
    n_free = gen.n_free

    def collision_g(joint_x, ctx):
        # ── 1. 解码 + 评估所有 Agent 的 Cartesian 轨迹 + 速度 ──
        positions = {}  # agent_idx → [T, 2]
        speeds = {}     # agent_idx → [T]
        for i in range(3):
            base = i * 2 * n_free
            ctrl_s = joint_x[base : base + n_free]
            ctrl_d = joint_x[base + n_free : base + 2 * n_free]
            s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
                ctrl_s, ctrl_d,
                all_init_states[i].s0, all_init_states[i].s_dot0,
                all_init_states[i].s_ddot0,
                all_init_states[i].d0, all_init_states[i].d_dot0,
                all_init_states[i].d_ddot0,
            )
            vehicle = gen.to_vehicle_states(
                s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot)
            x, y = gen.to_cartesian(s, d)
            positions[i] = jnp.stack([x, y], axis=-1)  # [T, 2]
            speeds[i] = vehicle[:, 2]                    # [T]

        # ── 2. 本车 vs 他车, 逐采样点 ──
        pos_i = positions[agent_idx]  # [T, 2]
        spd_i = speeds[agent_idx]     # [T]

        all_violations = []
        for j in range(3):
            if j == agent_idx:
                continue
            pos_j = positions[j]  # [T, 2]
            spd_j = speeds[j]     # [T]

            # 逐采样点 Cartesian 距离
            dist = jnp.sqrt(jnp.sum((pos_i - pos_j)**2, axis=-1))  # [T]

            # RSS 安全距离: 两车制动距离之和 + 静态 margin
            safe = (spd_i * rho + spd_i**2 / (2 * a_brake)
                  + spd_j * rho + spd_j**2 / (2 * a_brake)
                  + margin)

            # 违反量: 距离低于安全阈值
            violation = jnp.maximum(0.0, safe - dist)  # [T]
            all_violations.append(violation)

        # 取所有他车中最严重的时刻
        return jnp.max(jnp.stack(all_violations, axis=0), axis=0)  # [T]

    return collision_g
```

**可选的冲突区域限制** (减少远离环岛的 false positive):

```python
# 在 collision_g 内部添加:
def in_conflict_zone(pos, center, radius):
    return jnp.sqrt(jnp.sum((pos - center)**2, axis=-1)) < radius

in_zone = (in_conflict_zone(pos_i, center=jnp.array([0.,0.]), radius=25.0) &
           in_conflict_zone(pos_j, center=jnp.array([0.,0.]), radius=25.0))
violation = jnp.where(in_zone, jnp.maximum(0.0, safe - dist), 0.0)
```

#### 升级选项: Chance/Robust 约束

Constran 已支持非确定性约束类型（[Constran.py](../Constraintdealer/Constran.py)），
可直接替换 `Deterministic` 包装应对外部不确定性：

| 类型 | 用法 | 适用场景 |
|------|------|---------|
| `Chance(collision_g, alpha=0.05, n_samples=100)` | P(碰撞) < 5% | 其他 Agent 位姿有感知噪声 |
| `Robust(collision_g, uncertainty_set=...)` | 最差情况无碰撞 | `to_vehicle_states` 模型近似 (κ_r'=0, 简化 jerk) |

**注意**: Phase 1 的 MPC_G_MS 已通过 RNE 期望评估处理了其他 Agent 的**策略不确定性**。
`Chance`/`Robust` 应用于 RNE 无法覆盖的不确定性源（感知噪声、模型误差），
不要重复已有的博弈采样。否则外层 B×M_inner × 内层 n_samples → 开销爆炸。

**建议**: 先用 `Deterministic` 几何 RSS 跑通原型。如果感知噪声或模型误差导致实际碰撞，
再针对性升级为 `Chance`/`Robust`。

#### Constran 集成

每个 Agent 独立 `Constran.build()`，碰撞约束作为外层嵌入:

```python
def build_agent_constran(gen, agent_idx, all_init_states, goal_cost_fn,
                         lane_hw=3.5, R=20.0, omega_d=4.0):
    """为一个 Agent 构建 Constran — 包含 lane/speed/acc/jerk/collision.

    goal_cost_fn: (theta, ctx) → scalar — 时空联合 goal_cost
      gen 和 omega_d 通过闭包捕获 (Constran build 需要 obj_fn(x, ctx) 签名)
    """
    n_free = gen.n_free

    # 将 gen 和 omega_d 闭包进 goal_cost
    def objective_fn(theta, ctx):
        return goal_cost_fn(theta, {**ctx, 'omega_d': omega_d}, gen)

    constraints = [
        # 内层: 物理约束 (Phase 1 用松约束 — 只兜底, 不要求舒适)
        *make_constraints(gen, lane_hw, obs_safe_dist=0.1,
                          acc_max=8.0, jerk_max=5.0),  # 松!
        # 外层: 他车碰撞 (priority > 物理约束 → 更外层)
        Deterministic(
            make_collision_g_geometric(gen, agent_idx, all_init_states),
            mode='hard',      # 安全底线
            priority=5,       # 最外层 (lane=1, speed=2, acc=3, jerk=4)
            aggregate='max',
            transform='hard',
        ),
    ]

    return Constran.build(objective_fn=objective_fn, constraints=constraints)
```

> **实验验证** (§11.3 2b): `Constran.build()` 方案已在 `Simple_game_2b_constran.py` 验证。
> 对称场景下 mixed-strategy Nash 自发打破, 零手工调参。碰撞作为 `Deterministic(mode='hard', priority=5)` 自动平衡。

**关键设计点**:
- **goal_cost 是时空联合的** (空间 + 时间 + Lyapunov 层1)，不加手工权重
- **Constran 约束是松的** (acc=8.0, jerk=5.0) — 只兜底防数值爆炸，不追求舒适
- 碰撞 vs 车道 vs 速度之间的权衡全部由 Constran 的 σ 嵌套自动处理
- Phase 1 的松约束产出的 z_ref 会有粗糙的 acc/jerk → Phase 2 用紧约束精炼

### 10.6 fitness_fn_j 完整伪代码

```python
def make_fitness_fn_j(gen, all_init_states, lane_hw=3.5, R=20.0):
    """构建 MPC_G_MS 的 fitness_fn_j 接口.

    每个 Agent 有自己的 Constran (含碰撞约束), 在初始化时预编译。
    fitness_fn_j 只负责调用预编译的 constran_fn。
    """
    n_free = gen.n_free

    # 预编译每个 Agent 的 Constran cost function
    constran_fns = {}
    for agent_idx in range(3):
        # goal_cost_fn 根据 Agent 角色选择
        if agent_idx == 0:
            goal_fn = goal_cost_ego
        elif agent_idx == 1:
            goal_fn = goal_cost_circulating
        else:
            goal_fn = goal_cost_ego

        # Constran.build: goal_cost + lane/speed/acc/jerk/collision → 单一 cost
        constran_fns[agent_idx] = build_agent_constran(
            gen, agent_idx, all_init_states, goal_fn, lane_hw, R)

    def fitness_fn_j(agent_idx, joint_x_flat, ctx):
        """MPC_G_MS 调用的 fitness function.

        joint_x_flat: [6 * n_free] — 所有 Agent 的拼接 ctrl
        返回: 本 Agent 的 scalar cost (含 Constran 自动嵌套的目标+约束)
        """
        # Constran-built cost 已包含 goal + lane/speed/acc/jerk + collision
        # 直接调用, 不需要手写任何 penalty
        return constran_fns[agent_idx](joint_x_flat, ctx)

    return fitness_fn_j
```

**关键简化**: 相比之前的版本，去掉了:
- `decode_joint_to_per_agent` (解码现在在 collision_g 内部)
- `trajs` 字典 (轨迹评估在 Constran g_fn 内部按需进行)
- 手动 `collision_penalty` 循环
- 手动 `goal_cost + constran_cost + collision_cost` 加法

所有约束通过 `Constran.build()` 自动 σ 嵌套为单一 scalar cost。
MPC_G_MS 看到的只是一个黑盒 `(agent_idx, joint_x, ctx) → scalar`。

**context 结构** (精简 — Constran 内部已封装大部分逻辑):

```python
context = {
    'R': 20.0,                  # 环岛半径 (collision_g 计算 s 周期用)
    'v_ring': 10.0,             # 环行目标速度 (m/s)
    'd_exit': 3.5,              # 出口车道 d 偏移
    'd_lane': 0.0,              # 当前车道 d 偏移 (环行车用)
}
```

### 10.7 Warmstart: 博弈意图的 GMM 初始化

每个 Agent 的 GMM (K=3) 分量代表不同的**博弈意图**，在弯道 Frenet 空间中初始化:

```python
def build_roundabout_warmstart(gen, agent_states, K=3):
    """为环岛 3 Agent 构建 GMM 初始 mu.

    每个 Agent 的 3 个模态对应:
      - k=0: 激进 (加速/抢先)
      - k=1: 中性 (匀速/正常)
      - k=2: 保守 (减速/让行)

    每个 Agent 有 2 个 Block (s-channel + d-channel).
    """
    n_free = gen.n_free
    N_blocks = 6  # 3 agents × 2 channels
    D_max = n_free

    mu_init = jnp.zeros((N_blocks, K, D_max))

    for agent_idx in range(3):
        state = agent_states[agent_idx]
        # s-channel: 不同速度对应不同 s_dot
        v_options = {
            0: state['v_nominal'] * 1.2,   # 激进: 加速 20%
            1: state['v_nominal'],           # 中性: 匀速
            2: state['v_nominal'] * 0.8,    # 保守: 减速 20%
        }
        for k in range(K):
            v_k = v_options[k]
            # s-channel ctrl: 匀速外推
            ctrl_s = state['s0'] + v_k * gen.greville[2:gen.n_ctrl]
            mu_init = mu_init.at[agent_idx * 2, k].set(ctrl_s)

            # d-channel ctrl: 目标车道常数
            ctrl_d = jnp.full(n_free, state['d_target'])
            mu_init = mu_init.at[agent_idx * 2 + 1, k].set(ctrl_d)

    return mu_init
```

**为什么在弯道 Frenet 空间中 warmstart 仍然有效**:
GMM 采样在 Frenet ctrl 空间中进行。`gen.evaluate()` 和 `gen.to_vehicle_states()` 自动处理
κ_r ≠ 0 的曲率耦合。warmstart 不需要知道曲率 — 它只是给 GMM 一个合理的初始搜索区域。

参考: [warmstart.py:26](planning/warmstart.py#L26) `tangent_warmstart`。

### 10.8 环岛暴露的边界条件

#### s 参数周期性

`CircularReference` 的 s 以 2πR 为周期。B-spline 时域 (10s × v_max ≈ 200m) 可能超过
环岛周长 (2π × 20m ≈ 126m)，导致 s 绕回。

**解法**: B-spline 时域内 s 单调递增（不 wrap），参考路径的 s 参数做 `mod 2πR`。
即 B-spline 产出的 Frenet s 在 [0, 200m] 范围，但传给 `ref_path.evaluate()` 时
对 2πR 取模。

#### 入口/出口过渡

从直路进入环道 → 参考路径从 StraightReference 切换到 CircularReference。
在 Frenet 空间中这是 s 的连续过渡，但 κ_r 从 0 → 1/R 跳变。

**解法** (原型阶段): 简化 — 所有 Agent 始终使用 CircularReference。
入口/出口建模为环道上特定 s 位置的 d 方向偏移（d 从 -3.5 变到 0 表示进入，
d 从 0 变到 3.5 表示离开）。

#### Cartesian vs Frenet 碰撞检测

环道上两车: Cartesian 距离是弦长，Frenet 弧距是弧长。在碰撞约束中:
- **Frenet (s, d)**: `collision_g` 在 Frenet 空间做 RSS 判断 (s/d 分离, 含周期性)
- **Cartesian (x, y)**: 用于最终 safety check 和可视化

**当前方案**: `collision_g` (Constran Deterministic) 在 Frenet 空间计算 RSS 碰撞风险。
Cartesian 距离用于 `compute_summary` 的诊断输出。

### 10.9 实现状态

#### 已完成

| 组件 | 文件 | 状态 |
|------|------|------|
| MPC_G_MS Blocks 求解器 | [gmm_igo/MPC_G_MS.py](../gmm_igo/MPC_G_MS.py) | ✅ `mmog_igo_rne_blocks_solver` |
| 3-Agent 参考实现 (Trackgame) | [MultipleTest/Trackgame.py](../MultipleTest/Trackgame.py) | ✅ ego 2-block + 其他 1-block |
| Frenet B-spline 基 | [Cartest/core/frenet_traj.py](core/frenet_traj.py) | ✅ evaluate, to_vehicle_states |
| Constran per-agent build | [Constraintdealer/Constran.py](../Constraintdealer/Constran.py) | ✅ `build_multi_agent` |
| Constran 约束 (lane/speed/acc/jerk) | [Cartest/planning/constraints.py](planning/constraints.py) | ✅ `make_constraints` |
| StraightReference | [Cartest/core/reference_path.py](core/reference_path.py) | ✅ |

#### 待实现

| # | 任务 | 涉及文件 | 依赖 |
|---|------|---------|------|
| 1 | **`CircularReference`** | `reference_path.py` | 无 |
| 2 | **环岛 scenario 定义** | 新文件 `planning/scenario_roundabout.py` | #1 |
| 3 | **decode_joint_to_per_agent** | 新文件或 `fitness_fn_j` 内部 | 无 |
| 4 | **环岛 goal_cost** (进入/环行/离开) | `fitness_fn_j` 内部 | #1, #2 |
| 5 | **环岛 collision_penalty** (Frenet RSS) | `fitness_fn_j` 内部 | #1 |
| 6 | **环岛 warmstart** (博弈意图) | `warmstart.py` 扩展 | #1, #2 |
| 7 | **MPC_G_MS + Constran + B-spline 集成** | 新文件 `Simple_roundabout_game.py` | #1~#6 |
| 8 | **CircularReference round-trip 测试** | `test_frenet_invert.py` 扩展 | #1 |

### 10.10 验证路径

框架太大，一步到位不可能。下面是一条**递增验证阶梯**：每层只加一个新东西，
通过后再进入下一层。每层有独立的可跑代码，失败可回退。

```
Level 0: 组件独立验证 (现在就能做, ~35 行新代码)
  ├─ 0a: CircularReference 几何 (30行)
  └─ 0b: 单Agent在CircularReference上跑通 (5行)

Level 1: 两阶段 + 弯道 (~50 行新代码)
  └─ 1a: 单Agent, CircularReference, Phase1+2

Level 2: B-spline + 博弈 (~150 行新代码) ← 最关键的集成
  ├─ 2a: 2Agent, StraightReference, B-spline + MPC_G_MS
  └─ 2b: + collision_g 几何RSS

Level 3: 环岛博弈 (~100 行新代码)
  ├─ 3a: 2Agent 环岛 (ego + 环行车)
  └─ 3b: 3Agent 完整环岛
```

---

#### Level 0a — CircularReference 几何

**加什么**: `CircularReference` 类

**操作**: 在 `reference_path.py` 中实现, 在 `test_frenet_invert.py` 中加测试。

**测试**:
```python
R, cx, cy = 20.0, 0.0, 0.0
ref = CircularReference(R, cx, cy)

# s=0 → (0, 20): 圆心正上方
x, y, θ, κ = ref.evaluate(jnp.array([0.0]))
assert abs(x) < 1e-6 and abs(y - 20.0) < 1e-6
assert abs(κ - 1/20.0) < 1e-6

# s=πR → (0, -20): 半圈后在圆心正下方
x, y, θ, κ = ref.evaluate(jnp.array([jnp.pi * R]))
assert abs(x) < 1e-6 and abs(y + 20.0) < 1e-6

# round-trip: cartesian_to_frenet ∘ evaluate = identity
s_in = jnp.linspace(0, 2*jnp.pi*R, 100)
x, y, _, _ = ref.evaluate(s_in)
s_out, d_out = ref.cartesian_to_frenet(x, y)
assert max(|s_out - s_in|) < 1e-6
assert max(|d_out|) < 1e-6
```

**通过标准**: round-trip 误差 < 1e-6。如果不过 → 公式写错，不回退（必须过）。

---

#### Level 0b — 单 Agent 在 CircularReference 上跑通

**加什么**: κ_r ≠ 0 的 B-spline + IGO + Constran

**操作**: 复制 `Simple.py`→`Simple_circle.py`，只改一行:
```python
ref_path = CircularReference(20.0, 0.0, 0.0)  # 替 StraightReference()
```
跑 `--steps 50 --no-plot`。

**通过标准**:
1. 不崩溃（无 NaN/Inf）
2. 轨迹的 Cartesian (x, y) 近似在圆上
3. `g_lane` ≈ 0（车道约束满足）
4. v/a/j 物理量在合理范围（离心项未导致数值爆炸）

**如果失败**: 查 `to_vehicle_states` 的 Jacobian/离心/Coriolis 项。κ_r=1/20≈0.05 — 离心项 `κ_r·vt·s_dot` 在 v=10 时约 0.05×100=5 m/s²，应在合理范围。

---

#### Level 1a — 单 Agent, CircularReference, Phase1+2

**加什么**: 两阶段编排（同基 B-spline, z_ref 直接传递）

**操作**: 在 Level 0b 基础上，MPC 步内拆成两段:
```python
# Phase 1: 行为决策 (松约束, 地图引导 warmstart)
result_p1 = modes.solve('active', key1, ctx, mu)
z_ref = gen.evaluate(result_p1.x[:n], result_p1.x[n:], ...)  # 同基直接可得

# Phase 2: 轨迹跟踪 (紧约束, Lyapunov 跟踪 z_ref)
ctx_p2 = {**ctx, 'z_ref': z_ref}
result_p2 = modes.solve('standard', key2, ctx_p2, result_p1)
```

**通过标准**:
1. Phase 1 的 GMM 各分量 π 分布合理（有一个明显胜出）
2. Phase 2 的轨迹物理可行（acc ≤ 5, jerk ≤ 2）
3. 弯道上 κ_r ≠ 0 全程不崩溃
4. 收敛质量不低于单阶段基线（Level 0b）

**如果失败**: 查 Phase 1 warmstart 是否覆盖了正确车道，goal_cost 时空联合是否合理。

---

#### Level 2a — 2 Agent, StraightReference, B-spline + MPC_G_MS

**加什么**: B-spline 和 MPC_G_MS Blocks 求解器的**第一次集成**

**操作**: 最简单的 B-spline 博弈 — 两车在直路上各走各的车道（暂不碰撞）:
- 2 Agent, 各 2 blocks (s+d)
- fitness_fn_j = goal_cost + Constran(lane, speed, acc, jerk)（不含 collision）
- 用 StraightReference — 先排除弯道变量

```python
# 最小博弈: 两车直路变道
Agent 0: d0=-3.5, 目标 d=0   (向右变道)
Agent 1: d0=+3.5, 目标 d=0   (向左变道)
# 两车不会碰 — 纯验证 B-spline + RNE 能一起工作
```

**通过标准**:
1. RNE 收敛（mean_fitness 下降, π 稳定）
2. 两车的 B-spline 轨迹在各自车道内
3. Constran 约束满足（lane/speed/acc/jerk g≈0）
4. 每步求解 < 2s

**如果失败**: 这是最可能的瓶颈。查:
- Block dims 和 joint_x 布局是否与 MPC_G_MS 的 `block_to_agent_idx` 一致
- GMM 初始 mu 是否在合理范围（Greville 匀速外推）
- fitness_fn 的梯度是否正常（是否因 B-spline 基导致梯度消失）

---

#### Level 2b — 加 collision_g

**加什么**: 几何 RSS 碰撞约束

**操作**: 在 Level 2a 的 Constran 中加 `collision_g`（`Deterministic`, priority=5）。
改成碰撞场景: 两车在同一车道相向而行 → 博弈迫使一车让行。

```python
Agent 0: d0=0, 目标 d=0,  s0=0,   方向 +s
Agent 1: d0=0, 目标 d=0,  s0=100, 方向 −s
# 两车在同一车道相向 → 必须博弈让行
```

**通过标准**:
1. 两车不碰（collision_g 违规 → 0）
2. 博弈 π 分布有意义: 一车的"让行"分量 π 高，另一车的"通过"分量 π 高
3. 不让行时 collision_g 确实产生违规（验证约束有效）

**如果失败**: 查 collision_g 的 RSS 参数（rho, a_brake, margin）和 Constran priority 嵌套。

---

#### Level 3a — 2 Agent 环岛

**加什么**: CircularReference + 博弈

**操作**: 把 Level 2b 的 StraightReference 换成 CircularReference:
- Agent 0 (ego): 初始在入口 (d=-3.5, s=0), 目标进入环道 (d=0)
- Agent 1 (环行车): 初始在环内 (d=0, s=πR), 目标保持环行

**通过标准**:
1. 弯道 + 博弈同时工作（κ_r ≠ 0 无问题）
2. ego 选择合适的进入时机: 环行车远 → 进入; 环行车近 → 等待
3. collision_g 在 Cartesian 空间正确处理弯道碰撞

---

#### Level 3b — 3 Agent 完整环岛

**加什么**: 第三个 Agent

**操作**: 在 Level 3a 基础上加 Agent 2 (入口B):
- Agent 2: 从另一入口进入 (d=-3.5, s=πR/2)

**通过标准**:
1. 三方安全共存，无碰撞
2. 每车 π 分布合理（激进/保守分量按场景淘汰）
3. 所有 Constran 约束满足

---

#### 复杂度控制原则

- **每层只加一个变量**: 弯道 → 两阶段 → 博弈 → 碰撞 → 环岛 → 三车
- **失败可回退**: 每层有独立可跑的脚本
- **可并行**: Level 0 跑通后，1a 和 2a 可以同时推进（不同人在不同层上工作）
- **不跳级**: 2a（B-spline+博弈）是最大风险点，不能跳过。如果 2a 不过，3a 不可能过

---

## 11. 实验验证记录

> 记录了从 Level 0 到 Level 2 的逐步验证过程和关键发现。
> 对应脚本: `Simple_circle.py`, `Simple_circle_two_phase.py`, `Simple_game_2a.py`, `Simple_game_2b.py`。

### 11.1 Level 0: 组件独立验证

#### 0a — CircularReference 几何

**文件**: `core/reference_path.py` (新增 30 行), `eval/test_frenet_invert.py` (新增 2 测试)

**关键修正**:
- **Heading 符号**: `x = R·sin(θ), y = R·cos(θ)` 的切线方向 = `−θ mod 2π`, 不是 `θ`
- **d 符号**: Frenet 基类约定 `d > 0` = 圆外 (左法线方向)。`cartesian_to_frenet` 的 `d = dist − R` 与此一致
- **s 周期性**: `s` 以 `2πR` 为周期。Round-trip 测试需允许 `s ≡ s + 2πR`

**结果**: 18/18 测试通过 (16 个已有 + 2 个新增)

#### 0b — 单 Agent 在 CircularReference 上跑通

**文件**: `Simple_circle.py` (复制 `Simple.py`, 改 1 行 import)

**关键发现**:

| R | v | a_lat = v²/R | d 收敛 | 结论 |
|---|----|-------------|--------|------|
| 20m | 12→18 | 7.2~16.2 ❌ | d 从 −3 漂到 −4.8 | 约束冲突: IGO 发现 d<0 增大 R_eff 降低 a_lat, 形成错误局部最优 |
| 20m | 8→10 | 3.2~5.0 ✅ | −3→−0.01, 3.7s | 约束满足时完美收敛 |
| 100m | 12→18 | 1.4~3.2 ✅ | −3→0.04, 3.9s | 全程安全, 无超调 |

**物理规律**: `a_lat = v²/R`。`ACC_MAX=5.0` → `v_max = sqrt(5R)`。R=100m 时 v_max=22.4 m/s (80 km/h), 城市驾驶全程安全。R=20m 时 v_max=10 m/s, 初始 12 m/s 即违规。

**结论**: Frenet 跟踪在弯道上与直路行为一致 (R=100m 时 d 收敛时间、超调量与 StraightReference 完全相同)。问题不是代码 bug, 是真实的物理约束——单 IGO 在 cost 和约束冲突时找到数学对但物理错的局部最优。这正是两阶段设计的动机。

### 11.2 Level 1: 两阶段 + 弯道

#### 1a — 单 Agent, CircularReference, Phase1+2

**文件**: `Simple_circle_two_phase.py` (新增 140 行), `cost.py` (增加 ctx z_ref 支持)

**初始问题**: Phase 1 松约束 (acc=8.0) → v_up = sqrt(8R) 仍超 Phase 2 紧约束 → Phase 2 跟踪时 a_lat 违规。

**修正**: Phase 1 和 Phase 2 使用**相同的物理约束** (acc=5.0)。区别在**探索策略**:
- Phase 1: B=128, T_0=100 (大样本 + 周期重置 → 广探索)
- Phase 2: B=64, T_0=300 (小样本 + 不重置 → 深精炼)

**结果**:

| 指标 | 单阶段 R=100m | 两阶段 R=100m | 两阶段 R=50m |
|------|-------------|-------------|-------------|
| d 收敛 | −3→0.04, 3.9s | −3→0.00, 3.3s | −3→0.15, 3.7s |
| v 终值 | 14.5 | 17.8 | **15.7** |
| a_lat_max | 3.3 | 3.4 | **4.8** |
| 耗时 | ~420ms | P1+P2≈870ms | P1+P2≈872ms |

**关键发现**:
- 两阶段 d 收敛快 15% (3.3s vs 3.9s), v 更接近目标 (17.8 vs 14.5)
- R=50m 上 v 自动限制在 15.7 ≈ sqrt(5×50), a_lat ≤ 5.0——物理约束被尊重
- z_ref 通过 ctx 传递 (cost.py 中 `ctx.get('z_ref')`), 同基 B-spline 直接 evaluate 即可, **不需要 from_vehicle_states 桥接**

### 11.3 Level 2: B-spline + 博弈

#### 2a — B-spline + MPC_G_MS 第一次集成

**文件**: `Simple_game_2a.py` (新增 250 行)

**集成要点**:
- 每个 Agent = 2 blocks (s-channel + d-channel), N_blocks=4
- MPC_G_MS 的 `block_to_agent_idx = (0,0,1,1)`
- `fitness_fn_j` 使用 `lax.cond` 实现 JIT 兼容的 per-agent dispatch
- 动态 ctx 通过 `game_ctx` dict 传入 solver (不能闭包静态 ctx)

**结果**: 两车直路变道 (无碰撞):

| Agent | d 收敛 | v 终值 | π |
|-------|--------|-------|---|
| 0 | −3→0.06, 1.6s | 17.6 | 从 comp0 切换到 comp1 |
| 1 | +3→−0.06, 1.9s | 15.8 | 均匀 (三个分量代价接近) |

**关键发现**:
- B-spline + RNE 成功集成, 每步 ~150ms
- Agent 1 的 π 均匀是因为三分量都是合理的变道策略 (代价接近)——不是 bug, 是 GMM 正确识别了多个等价好方案
- **动态 ctx 是关键**: 初始实现用闭包静态 ctx → v 降到 1m/s (cost 跟踪错误参考)。改为每步更新 game_ctx → 行为正常

#### 2b — 碰撞博弈 (Constran 自动 σ 嵌套)

**文件**: `Simple_game_2b_constran.py`

**场景**: 
- 对称: 两车同 s=0, 同 v=14, 同 d=−3 (同时变道到 d=0)
- 不对称: Agent 0 从后方快速接近 (v=16, s=0, d=−3), Agent 1 在前方慢行 (v=8, s=8, d=0)

**碰撞实现**: `Constran.build()` — 碰撞作为 `Deterministic(collision_g, mode='hard', priority=5)` 嵌入 σ 嵌套最外层。**零手工权重**。

```python
constraints = [
    Deterministic(lane_g,  mode='soft', priority=1),   # 内层
    Deterministic(speed_g, mode='soft', priority=2),
    Deterministic(acc_g,   mode='soft', priority=3),
    Deterministic(jerk_g,  mode='soft', priority=4),
    Deterministic(collision_g, mode='hard', priority=5), # 外层: 碰撞硬底线
]
cost = Constran.build(goal_cost, constraints)
```

**碰撞检测**: 利用 B-spline 的 T=100 时间同步采样点, Cartesian 距离 < 3m → 违规。
采样点天然时间同步 → 碰撞检测自动是时空联合的 (同时间 + 同位置才算冲突)。

**对称场景结果**:

| Agent | d | v | 行为 |
|-------|---|---|------|
| 0 | −3→−2.05 | 14→22.5 | 靠向车道中心 (先行) |
| 1 | −3→−5.15 | 14→31.7 | 退到更外侧 (让行) |

**对称自发打破！** Mixed-strategy Nash + Constran `mode='hard'` 迫使一车退让、一车先行。RNE 的随机采样提供了打破对称的初始扰动。

**不对称场景结果**:

| Agent | d | v | 行为 |
|-------|---|---|------|
| 0 | −3→−3.44 | 16→16.7 | 保持在原车道 (不并入, 让行) |
| 1 | 0→0.33 | 8→20.2 | 加速通过 |

Agent 0 (并入车) 选择不并入——Agent 1 已在车道内, collision_g 的 hard 约束迫使 Agent 0 保持 d<0。物理上合理: 已在车道内的车有优先权。

> **与手写 weight 的对比**: 手写版 (`Simple_game_2b.py`) 用 `own_cost + 5000 * collision_penalty` 需要反复试 magic number, 对称场景无法打破。Constran 版零调参, σ 嵌套自动平衡。

**已知问题**: 对称场景速度偏高 (22-32 m/s, 超过 v_target=18)。`mode='hard'` 的碰撞惩罚让退让车加速逃逸——后续通过收紧 V_MAX 或调整 speed_g priority 解决。

#### Level B — 对抗鲁棒性

**测试 1** (手写版): 固定 Agent 1 为激进策略 (v=12, 永不让行), Ego 单边 IGO 优化应对。
Ego **延迟变道** (d 保持 −2.7, 不并入), 最小距离 2.7m。✅

**测试 2** (Constran 版): 对称场景下自发打破——不需要固定对手策略, RNE 自己产生不对称均衡。✅

### 11.4 关键设计经验

1. **R=100m 是合理的城市弯道参数**: v=18 m/s (65 km/h) 时 a_lat=3.2, 远低于 ACC_MAX=5.0。R=20m 只适合低速场景 (v≤10 m/s)。

2. **同基 B-spline → z_ref 不需要 from_vehicle_states**: Phase 1 的 `gen.evaluate()` 直接产出 Frenet 状态就是 z_ref, 无需车辆级→Frenet 的转换。

3. **Phase 1 + Phase 2 的约束应相同**: 区别在探索策略 (B, T_0, dt) 而非约束松紧。物理极限 (向心加速度) 不能"松"。

4. **动态 ctx 是博弈的关键**: MPC_G_MS 的 fitness_fn 必须通过 context 参数接收当前状态, 不能用闭包静态捕获。

5. **碰撞博弈需要足够的 GMM 覆盖**: 如果一个分量都不包含"让行"行为 (低速), RNE 无法发现让行均衡。

6. **博弈合理性不能只用自洽性验证**: RNE 收敛只是数学不动点。需要对抗测试 (固定对方策略) 和重放测试验证鲁棒性。

7. **碰撞必须走 Constran.build(), 不能手写 weight**: ✅ 已验证。`Constran.build(goal_cost, [lane, speed, acc, jerk, Deterministic(collision_g, mode='hard')])` — 零调参, `mode='hard'` 自动保证碰撞是硬底线, σ 嵌套自动平衡碰撞 vs 速度。B-spline 采样点时间同步 → 碰撞检测天然时空联合, 不需要额外处理。

8. **Mixed-strategy Nash 能自发打破对称**: RNE 的随机采样 + Constran `mode='hard'` 强碰撞约束 → 对称场景下自动产生"一车先行、一车让行"的非对称均衡。不需要手工设计不对称场景。
