# Cartest — Frenet B-Spline Trajectory MPC

基于 Frenet 坐标系 + 五次 B 样条的 MPC 轨迹规划器。IGO 黑箱优化 + Constran 自相似 σ 嵌套约束。

## 架构

```
                          ┌─────────────────────┐
                          │    ReferencePath     │  地图输入
                          │  evaluate(s) → x,y,θ,κ │
                          │  frenet→cartesian    │
                          └──────────┬──────────┘
                                     │
  ┌──────────┐    ┌──────────┐       ▼
  │  spline  │───▶│ frenet   │───▶ [s,d,ḃ,ḋ,s̈,d̈,s⃛,d⃛](t)  ← 规划
  │  B,dB,   │    │ _traj    │         │
  │ d2B,d3B  │    └──────────┘         │
  └──────────┘           │             │
                         │             ▼
                    ┌────┴─────┐  ┌──────────┐    ┌──────────┐
                    │ warmstart│  │   cost   │    │constraints│
                    │ Greville │  │ Σd²+Σv²  │    │P5→P1 嵌套│
                    └──────────┘  └────┬─────┘    └────┬─────┘
                                       │               │
                                       ▼               ▼
                                  ┌──────────────────────┐
                                  │   Constran.build()   │  ← 代价函数
                                  │  σ 自相似饱和嵌套     │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  IGO 黑箱优化器       │  ← 求解
                                  │  M=2 blocks, K=3     │
                                  └──────────┬───────────┘
                                             │
                                             ▼
  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐
  │ vehicle  │◀───│ execute  │◀───│  plan: s̈_cmd,d̈_cmd   │  ← 执行
  │ _model   │    │ extract  │    │  (B-spline t=0)      │
  │ Euler+clip│   │ _command │    └──────────────────────┘
  └──────────┘    └──────────┘
```

### 核心思想：Frenet 下的线性运动学

Cartesian B-spline 的致命问题：ctrl → jerk/acc 经过 `arctan2` + 叉乘投影 + 曲率除法，控制点微扰被级联放大。

Frenet 下：

```
ctrl_s → s̈ = d2B·ctrl_s    ← 纵向加速度，线性
ctrl_s → s⃛ = d3B·ctrl_s    ← 纵向 jerk，线性
ctrl_d → d̈ = d2B·ctrl_d    ← 横向加速度，线性
ctrl_d → d⃛ = d3B·ctrl_d    ← 横向 jerk，线性
```

**每个物理量 = 基函数矩阵 × 控制点，一步到位。** 参考线承担曲率、`arctan2`、叉乘投影——这些全在给定的参考线上，不是优化变量。

## 文件结构

```
Cartest/
├── spline.py            # B 样条基函数预计算 (B, dB, d2B, d3B, d4B)
├── frenet_traj.py       # Frenet B 样条轨迹生成器
├── reference_path.py    # 参考线抽象 (Straight + 弯道接口)
├── warmstart.py         # Warm-start (Greville-based)
├── cost.py              # 目标函数 (速度 + 横向跟踪)
├── constraints.py       # 约束构建 (积分链 P5→P1)
├── execute.py           # 执行层: plan → vehicle model → state
├── vehicle_model.py     # 车辆模型 (点质量 + Euler + 限幅)
├── Simple.py            # MPC demo
└── bspline_basis.npz    # 预计算基函数矩阵
```

## 1. 地图输入：ReferencePath

参考线 = 弧长参数化的光滑中心线。`ReferencePath` 是抽象基类，两个核心方法：

```python
class ReferencePath:
    def evaluate(self, s) -> (x_r, y_r, θ_r, κ_r)
    def frenet_to_cartesian(self, s, d) -> (x, y)
```

### 内置：StraightReference

```python
from Cartest.reference_path import StraightReference

ref = StraightReference()
# x = s, y = d, θ = 0, κ = 0
```

### 自定义弯道

继承 `ReferencePath`，实现 `evaluate(s)`：

```python
class CircularReference(ReferencePath):
    def __init__(self, radius=100.0):
        self.R = radius

    def evaluate(self, s):
        θ = s / self.R                          # 弧长 → 角度
        x_r = self.R * jnp.sin(θ)               # 圆心在原点
        y_r = self.R * (1 - jnp.cos(θ))
        θ_r = θ                                  # 切线方向
        κ_r = jnp.full_like(s, 1.0 / self.R)    # 恒定曲率
        return x_r, y_r, θ_r, κ_r
```

参考线只用于**避障约束**的 Frenet→Cartesian 映射和车辆状态的 heading/曲率计算。规划本身完全在 (s, d) 空间进行。

## 2. B 样条轨迹

5 次 (quintic) B 样条，12 个控制点，10 秒规划时域，C⁴ 连续。

```
t_eval ∈ [0, 10]s, dt = 0.1s, T = 100 个采样点
```

### 夹紧边界条件（C0/C1/C2）

前 3 个控制点夹紧，保证从当前状态出发：

```
P0 = x0                                    → C0: 位置连续
P1 = P0 + (Δt_knot/5) · v0                → C1: 速度连续
P2 = 3·P1 − 2·P0 + (Δt²_knot/10) · a0   → C2: 加速度连续
```

后 9 个控制点是自由的（优化变量），θ = [ctrl_s(9) | ctrl_d(9)]，共 18 维。

### 基函数矩阵

预计算 `spline.py` 生成 `bspline_basis.npz`：

```bash
uv run python Cartest/spline.py
```

生成 B, dB, d2B, d3B, d4B 矩阵（各 [100, 12]）和 Greville 横坐标。

## 3. 车辆模型

```python
from Cartest.vehicle_model import FrenetVehicleModel

vehicle = FrenetVehicleModel(acc_max=5.0, dt=0.1)

# 每步执行:
s_new, d_new, s_dot_new, d_dot_new = vehicle.step(
    s0, d0, s_dot0, d_dot0,          # 当前状态
    s_ddot_cmd, d_ddot_cmd,           # plan 的期望加速度 (t=0)
)
```

当前是点质量 + Euler 积分 + 加速度限幅。可替换为 kinematic bicycle 或更复杂的模型——只需实现相同的 `step()` 接口。

**执行不直接从 plan 读状态**（那是 "完美跟踪" 假设），而是读 plan 的加速度指令，经车辆模型仿真出实际状态。

## 4. 约束：按积分链组织

物理量的因果积分链：

```
jerk (s⃛) ──∫──▶ acc (s̈) ──∫──▶ speed (ḃ) ──∫──▶ position (s,d)
 控制输入         中间量         中间量           输出
 最外层           ...            ...            最内层
 P5               P4             P3             P2, P1
```

约束按这个链从外到内排列（Constran 的 self-similar σ 嵌套）：

| Priority | 层 | 约束 | 来源 | 坐标 |
|----------|----|------|------|------|
| P1 (内) | 避障 | 穿透深度 | `d2B·ctrl→Cartesian` | Cartesian |
| P2 | 车道 | `|d| ≤ lane_hw` | `B·ctrl_d` | Frenet 直出 |
| P3 | 速度 | `V_min ≤ ḃ ≤ V_max` | `dB·ctrl_s` | Frenet 直出 |
| P4 | 加速度 | `√(s̈²+d̈²) ≤ ACC_MAX` | `d2B·ctrl` | Frenet 直出 |
| P5 (外) | jerk | `√(s⃛²+d⃛²) ≤ JERK_MAX` | `d3B·ctrl` | Frenet 直出 |

### 为什么这个顺序？


- 积分链条因果影响顺序：jerk -> acc -> velocity -> state -> distances to obstacles and tracking cost

- 按照积分链搞，一般外面物理约束搞对了，里面避障需要的时候，规划出来的才能执行，
不然就是瞎搞

### 模式选择

- **P1 (obstacle)**: `mode='hard'` → baseline=2.0 → 即使无违规也有惩罚地板 → 绝对优先
- **P2-P5 (comfort)**: `mode='soft'` → baseline=0 → 无违规时 Φ=0 → σ 层透明 → 目標信号完全恢复

### 约束函数示例

```python
def jerk_g(theta, ctx):
    _, _, _, _, _, _, s_dddot, d_dddot = _eval_traj(theta, ctx, gen)
    jm = jnp.sqrt(s_dddot**2 + d_dddot**2)   # jerk 幅值
    return jnp.maximum(0., jm - JERK_MAX)     # 精确罚函数 (P.S 可以直接换成q90 峰值大就大了)

def lane_g(theta, ctx):
    _, d, _, _, _, _, _, _ = _eval_traj(theta, ctx, gen)
    return jnp.maximum(0., jnp.abs(d) - ctx["lane_hw"])
```

所有运动学约束（P2-P5）直接用 Frenet 量——不需要 Cartesian 投影。**只罚违规 (max(0, |x|-limit))，从不推向零。**

## 5. IGO 优化器

18 维搜索空间 (2 blocks × 9 控制点)，M=2, K=3。

```python
mu_k, L_k, pi_k = mmog_igo_optimizer_mpc(
    key, 500, 0.15,          # IGO steps, dt
    M=2, K=3,                # blocks, populations
    B=64, B0=20, T0=250,    # samples, elite, reset
    dims=(9, 9),             # block dimensions
    cost_fn,                 # Constran-built
    mu_init, L_inv, v_init, ctx,
)
```

IGO 是黑箱全局优化——不需要梯度，只要求值。Frenet 下 ctrl→物理量接近线性，搜索空间条件数好，收敛快。

## 6. 运行

```bash
# 1. 生成基函数矩阵 (只需一次)
uv run python Cartest/spline.py

# 2. 运行 MPC demo
uv run python Cartest/Simple.py --steps 150 --seed 0

# 跳过绘图
uv run python Cartest/Simple.py --steps 50 --no-plot
```

输出 `Cartest/frenet_demo.gif`（执行轨迹 vs 规划轨迹的动画）。

## 7. 关键设计决策

| 决策 | 理由 |
|------|------|
| Frenet 坐标 | ctrl→jerk/acc 线性，消除 Cartesian 的非线性放大 |
| 5 次 B 样条 | C⁴ 连续：jerk 连续，bang 有界 |
| 单侧夹紧 | 初始状态精确匹配；终端自由（MPC 只执行第一步） |
| 9 个自由控制点 | 18 维搜索空间，IGO 可处理 |
| Greville-based warm-start | 精确常速轨迹：jerk=0, acc=0 |
| soft constraints (P2-P5) | 无违规时透明，目标信号不被 baseline 淹没 |
| hard constraint (P1) | 安全底线，baseline=2.0 不可协商 |
| plan→model→execute | 规划/仿真解耦，车辆模型可替换 |

## 依赖

- `jax[cuda12]` — B 样条矩阵运算
- `numpy`, `scipy` — 基函数预计算
- `matplotlib` — 可视化
- `gmm_igo` — IGO 优化器
- `Constraintdealer.Constran` — σ 嵌套约束引擎
