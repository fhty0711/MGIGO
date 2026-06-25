# Unit Commitment via Pure Continuous IGO — 方法指南

## 1. What This Is

用 MGIGO 纯连续黑箱优化破解机组组合（Unit Commitment）问题。
**不需要离散变量、不需要 MILP、不需要线性化成本函数。**

核心思想（来自 Hybrid_test_README §9 光滑版公式）：

$$\mathcal{L} = \sigma\!\left(\frac{v_1}{S_1} + \delta_1\sigma(\beta_1 v_1) + \sigma\!\left(\frac{v_2}{S_2} + \delta_2\sigma(\beta_2 v_2) + \sigma\!\left(\frac{f}{S_3}\right)\right)\right)$$

没有 `jnp.where` — 所有层级永远可见，violation 通过 σ 函数连续变化来惩罚。

## 2. 5-Generator Proof of Concept — 结果

| 指标 | 单个时步 | 24h 滚动 MPC |
|------|---------|-------------|
| 最大功率失衡 | 0.83 MW | 5.0 MW (2.5%) |
| vs 贪心 | -18.5% 成本 | — |
| 求解时间/步 | 2.0s (含 JIT) | 0.16s (稳态) |
| p_min/p_max 合规 | ✓ | ✓ |
| 爬坡率合规 | ✓ | ✓ |

**MPC 求解器参数**：
```
M=1, K=3, B=60, B0=28, T=300, T_0=100, dt=0.15
单块维度 D = N_GEN (≤ 10 必须)
β1 = 0.05  ← 关键！让 0-200MW 全范围在 σ 线性区
β2 = 5.0
δ1 = 2.0, δ2 = 0.8
L_inv 初始 = 0.5*I (即 σ_sampling ≈ 2，覆盖 on/off 两个区域)
```

**运行**：
```bash
uv run python Functiontest/UnitCommitment_test.py --mode single   # 单步测试
uv run python Functiontest/UnitCommitment_test.py --mode mpc      # 24h 滚动MPC
```

## 3. 关键设计决策

### 3.1 编码方案：Random Keys

```
每个发电机 i → 1 个连续变量 x_i ∈ ℝ
解码: p_i = P_max_i · sigmoid(x_i)
  x_i → -∞  ⇒  p_i → 0      (关机)
  x_i → +∞  ⇒  p_i → P_max  (满发)
```

不需要显式离散变量。GMM 自然收敛到 Dirac delta — 优化后期 `x_i` 要么 >>0 要么 <<0。

**Tiebreak**：`ε · Σ x_i` (ε=1e-6) 确保 float32 可区分性。当所有发电机都关机时 (x_i 全部 <<0)，不同的 x_i 组合产生相同的 p_i ≈ 0 — tiebreak 打破平坦地形。

### 3.2 为什么不用 jnp.where（关键教训）

UC 的资源平衡约束与 Hybrid 的几何约束根本不同：

- **几何约束**（障碍物）：trajectory 要么撞要么不撞 — `jnp.where(viol>0, …)` 的硬分支合理
- **资源平衡**（功率）：viol 从 0 连续变化到 200MW — 硬分支会切断底层信号

当 `jnp.where(viol_balance > 0, …)` 激活时：
- 所有有 violation 的样本 cost ≈ `σ(δ1) ≈ 0.89`（几乎相同）
- 经济调度信号完全丢失 → solver 无法分辨"差"与"更差"
- **结果**：solver 坍缩到"全部关机，接受最大 violation"的退化解

### 3.3 β 参数的选择（最关键的调参）

```
β1 = 0.05  ← L1 功率平衡：拐点在 1/β1 = 20MW
             让 0-200MW 全范围在 σ 的线性区
             viol=0 vs viol=50 vs viol=200 有清晰区分

β2 = 5.0   ← L2 物理约束：拐点在 0.2MW
             物理约束是准二值的 (合规/违规) → 允许更陡
```

**β 太大会怎样**：拐点位移到 <1MW → 几乎所有 violation 立刻饱和 → 重蹈 jnp.where 的覆辙

**β 太小会怎样**：σ 的线性区太宽 → δ·σ(β·viol) ≈ 0 对所有 viol → L1→L3 层级关系消失 → 全部关机更便宜

### 3.4 MPC 策略：Fresh Init vs Warm Start

```
Warm-start (carry mu):  快速 (0.16s) 但锁死 generator count
Fresh init (random mu): 同样快 (0.16s, JIT已编译) 但避免盆地锁定
```

UC 的特殊性：**负荷变化时，最优发电机数量会变**。Warm-start 把 solver 锁在上一步的 generator-count 盆地。对于 UC，每步 fresh init 是正确的选择。

对于其他问题（如 trajectory tracking，步间变化小），warm-start carry mu 是更好的选择。

## 4. 扩展到更大规模（50 发电机）

### 4.1 块分解策略

```
50台机 = 5 blocks × 每块 10 台发电机 (D=10 ≤ 极限)

Block 0: Gen 0-9    (D=10)
Block 1: Gen 10-19  (D=10)
Block 2: Gen 20-29  (D=10)
Block 3: Gen 30-39  (D=10)
Block 4: Gen 40-49  (D=10)

M=5, K=3
每块 B=40 → 总 B_effective = 5×40 = 200
B0 ≈ 90 (总精英数的一半减一点)
```

**为什么可行**：每块只需决定 10 台发电机的启停+功率 (2^10=1024 种组合)，而不是全局 2^50。块间耦合通过 cost function 中的功率平衡项处理 — solver 通过自然梯度学习"我这块多发/少发对总 cost 的影响"。

### 4.2 Cost Function 注意事项

功率平衡是所有块的联合约束：
```python
total_gen = sum(
    block_i_contribution(block_i_samples)
    for i in range(M)
)
viol_balance = |total_gen - demand|
```

每块独立更新时，块间耦合通过精英样本的选择隐式处理：只有那些"配合得好"的块组合会被选为精英。

### 4.3 预期问题

| 问题 | 症状 | 缓解 |
|------|------|------|
| 块间振荡 | 块1减发→块2不得不增发→cost来回跳 | 增大B (更多样本→更稳定梯度估计) |
| 启动成本协调 | 各块独立决定启停，付出多余启动成本 | 启动成本入 L2 或 L1，提高其惩罚 |
| 对称性/多峰 | 相同参数的发电机互换，多个等价解 | K≥3 的 GMM 自然处理 |
| 采样覆盖不足 | D=10 时 2^10=1024 组合，B=40 只覆盖 4% | 增大 B 到 60-80；光滑 σ 提供梯度引导 |

### 4.4 参数起点

```python
# 50-generator parameter template
M = 5                    # number of blocks
K = 3                    # GMM components per block
dims = (10, 10, 10, 10, 10)  # 10 gens per block
B = 80                   # total sample count (16 per block on average)
B0 = 35                  # elite count < B/2
T = 300                  # total iterations
T_0 = 100                # restart period
dt = 0.15                # step size

# Initial exploration
L_inv = 0.5 * I          # sigma_sampling ≈ 2
mu_init = uniform(-2, 2) # covers both on and off regions

# Smooth formulation parameters
beta_1 = 0.05            # knee at 20 MW for power balance
beta_2 = 5.0             # knee at 0.2 MW for physical limits
delta_1 = 2.0            # L1 jump
delta_2 = 0.8            # L2 jump
```

## 5. Solver 参数经验法则

来自用户实践经验的硬约束：
- **单块维度 ≤ 10**（实时求解要求）（当然；如果规模实在是非常大的话 1min以内求解也可以；你也可以将一个问题类似于MPC那样；虽然问题不是MPC 但是可以通过不停地启动求解器寻优）
- **采样数 B_total ≈ 3~4 × 决策变量总数**
- **精英数 B0 < B/2**（略小于半数）
- **T=300, T_0=100, dt=0.15**（小规模推荐，T_0 需整除 T）
- **K=3**（3个 GMM 分量，平衡探索/开发）
- **L_inv 初始 = c·I, c ∈ [0.3, 1.0]**（c 越小 σ_sampling 越大，探索越广）

## 6. 为什么这比 Gurobi/MILP 有优势

| 方面 | Gurobi/MILP | IGO (本方法) |
|------|------------|-------------|
| 目标函数 | 必须分段线性化 (SOS2) | 任意非线性 f(x) |
| 约束 | 必须线性/二阶锥 | 任意可求值的违反量 |
| Legacy 代码 | 必须重新公式化 | 直接包进 cost function |
| Ad-hoc 规则 | 大M + 辅助变量 | 几行 if-else (通过 σ 光滑化) |
| 不确定性 | 场景树爆炸 | MC 采样，天然支持 |
| 最优性保证 | 有 (MIP gap) | 无 (局部最优) |
| 大规模离散 | Branch-and-bound | 取决于编码和分解 |

IGO 的价值主张不是"比 Gurobi 更快更准"，而是**"能处理 Gurobi 不能/不便公式化的问题"**。

## 7. 相关文件

| 文件 | 内容 |
|------|------|
| `Functiontest/UnitCommitment_test.py` | 5-gen UC 实现 (单步 + MPC) |
| `Functiontest/Hybridsystemtest.py` | 饱和嵌套的参考实现 |
| `Functiontest/Hybrid_test_README.md` | 饱和理论、auto-scaling、float32 分析 |
| `gmm_igo/MPCsolverM22.py` | 多块 GMM-IGO 求解器 |
| `Functiontest/RobustConstraints.py` | 鲁棒约束（非凸不确定性集的扫描范式） |
| `Functiontest/Constraints.py` | 机会约束、混合不确定性 |
