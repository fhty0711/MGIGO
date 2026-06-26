"""
Forest + IGO — 非马尔可夫 UC (commitment modes × 24h dispatch)
===============================================================

Forest: 5 gens × 2~3 commitment modes = 72 路径
  FULL_DAY: 0-23h ON  (baseload)
  PEAK_ONLY: 8-20h ON (peak)
  OFF:      0-23h OFF

IGO:  给定 commitment, 优化 5 机 × 24h 连续出力 (120D → 5 blocks × 24D)
Cost: 燃料 + 启动费 + 最小启停 + 爬坡 + 供电不足 (全非马尔可夫, 整体评估)

用法:
  uv run python MCTSIGO/trial_uc_forest.py
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax, jax.numpy as jnp
import numpy as np
from jax import random
from itertools import product

from Constraintdealer.Constran import build, Deterministic
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from Energy.Generator import make_5gen_system, LOAD_5GEN
from MCTSIGO.decision_unit import DecisionUnit
from MCTSIGO.guide_tree import IndexTree, GuideTreeSolver
from MCTSIGO.fidelity_config import FidelityConfig

# ══════════════════════════════════════════════════════════════════════════════
gens = make_5gen_system()
N_GEN = gens.N
# Compressed: 6 time blocks × 4h each
H = 6
# Sample demand at 4h intervals
_full_load = np.array(LOAD_5GEN)
DEMAND = np.array([np.mean(_full_load[i*4:(i+1)*4]) for i in range(H)])
PEAK_START = 2  # block 2 = hours 8-12 (peak starts)

GEN_MODES = [
    ["FULL_DAY", "PEAK_ONLY", "OFF"],   # Gen0 50MW
    ["FULL_DAY", "PEAK_ONLY", "OFF"],   # Gen1 80MW
    ["PEAK_ONLY", "OFF"],               # Gen2 30MW
    ["FULL_DAY", "OFF"],                # Gen3 100MW
    ["PEAK_ONLY", "OFF"],               # Gen4 40MW
]
GEN_NAMES = [f"Gen{i}" for i in range(N_GEN)]
n_paths = 1
for m in GEN_MODES: n_paths *= len(m)

print(f"UC: {N_GEN} gens × {H}h, {n_paths} paths")
print(f"  caps: {[float(g) for g in gens.p_max]} MW")
print(f"  demand: {float(DEMAND.min()):.0f}~{float(DEMAND.max()):.0f} MW")

# ══════════════════════════════════════════════════════════════════════════════
# Forest
# ══════════════════════════════════════════════════════════════════════════════

forest = IndexTree.from_forest([
    (name, [DecisionUnit(name, modes)])
    for name, modes in zip(GEN_NAMES, GEN_MODES)
])
print(f"Forest: {forest.describe()}")

# ══════════════════════════════════════════════════════════════════════════════
# Commitment → 24h on/off mask
# ══════════════════════════════════════════════════════════════════════════════

def commit_mask(strategy):
    """返回 (N_GEN, H) 的 on/off mask. H=6 time blocks."""
    mask = np.zeros((N_GEN, H))
    for i in range(N_GEN):
        mode = strategy[GEN_NAMES[i]]
        if mode == "FULL_DAY":
            mask[i, :] = 1.0
        elif mode == "PEAK_ONLY":
            mask[i, PEAK_START:PEAK_START+3] = 1.0  # blocks 2,3,4
        elif mode == "OFF":
            mask[i, :] = 0.0
    return jnp.array(mask)

# ══════════════════════════════════════════════════════════════════════════════
# Cost: 24h 全时域, 非马尔可夫
# ══════════════════════════════════════════════════════════════════════════════

# Per-generator params
p_max = gens.p_max; p_min = gens.p_min
a_arr, b_arr, c_arr = gens.a, gens.b, gens.c
startup_arr = gens.startup
ramp_arr = gens.ramp

def full_cost_fn(x_flat, ctx):
    """x: (N_GEN * H,) → 解码为 (N_GEN, H) 出力矩阵.
    整体评估: 燃料 + 启动费 + 最小启停 + 爬坡 + 供需平衡.
    """
    x = x_flat.reshape(N_GEN, H)
    commit = ctx["commit_mask"]  # (N_GEN, H)
    prev_on = ctx.get("prev_on", jnp.zeros(N_GEN))

    total = jnp.array(0.0)
    prev_p = jnp.zeros(N_GEN)
    on_hours = jnp.zeros(N_GEN)     # 连续开机时数
    off_hours = jnp.zeros(N_GEN)    # 连续关机时数

    for h in range(H):
        p_h = x[:, h]
        # 强制 OFF 机组出力=0 (通过 commit mask)
        p_h = p_h * commit[:, h]
        on_h = (p_h > p_min * 0.5).astype(jnp.float32)

        # --- Fuel ---
        fuel = jnp.sum(a_arr * p_h**2 + b_arr * p_h + c_arr * on_h)

        # --- Startup (非马尔可夫) ---
        if h == 0:
            startup = jnp.sum(startup_arr * on_h * (1.0 - prev_on))
        else:
            prev_on_h = (prev_p > p_min * 0.5).astype(jnp.float32)
            startup = jnp.sum(startup_arr * on_h * (1.0 - prev_on_h))

        # --- Ramp ---
        ramp_viol = jnp.sum(jnp.maximum(0.0,
            jnp.abs(p_h - prev_p) - ramp_arr * commit[:, h]))

        # --- Min up/down (非马尔可夫, compressed: 1 block ≈ 4h) ---
        on_hours = jnp.where(on_h > 0.5, on_hours + 1, 0.0)
        off_hours = jnp.where(on_h < 0.5, off_hours + 1, 0.0)
        turned_off = (prev_p > p_min * 0.5).astype(jnp.float32) * (1.0 - on_h)
        min_up_viol = jnp.sum(turned_off * jnp.maximum(0.0, 1.0 - on_hours))
        turned_on = (1.0 - (prev_p > p_min * 0.5).astype(jnp.float32)) * on_h
        min_down_viol = jnp.sum(turned_on * jnp.maximum(0.0, 1.0 - off_hours))

        # --- Power balance ---
        supplied = jnp.sum(p_h)
        under = jnp.maximum(0.0, DEMAND[h] - supplied)

        total += (fuel + startup) / 500.0
        total += 0.05 * ramp_viol
        total += 0.5 * (min_up_viol + min_down_viol)
        total += 2.0 * under / max(DEMAND[h], 1.0)

        prev_p = p_h

    return total / H


_base = build(full_cost_fn, [], k_inner=0.1, jit_cost=False)

@jax.jit
def cost_fn(x, ctx):
    return _base(x, ctx) + 1e-5 * jnp.sum(x)

# ══════════════════════════════════════════════════════════════════════════════
# IGO
# ══════════════════════════════════════════════════════════════════════════════

IGO_KEY = random.PRNGKey(42)
D_CONT = N_GEN * H  # 30D total
# Block decompose: 每台机独立 block (6D ≤ 10D hard constraint)
DIMS = tuple([H] * N_GEN)  # (6, 6, 6, 6, 6)
M, K_IGO = 1, 3
B, B0 = 120, 55  # 总样本, B ≈ 4×D


def igo_evaluate(strategy, mode):
    global IGO_KEY
    cmask = commit_mask(strategy)
    ctx = {"commit_mask": cmask}

    if mode == "none":
        # x=0 for ON hours, x=-5 for OFF hours
        x_init = jnp.where(cmask > 0.5, 0.0, -5.0)
        return float(cost_fn(x_init.ravel(), ctx))

    IGO_KEY, skey, ikey = random.split(IGO_KEY, 3)
    mu = jnp.zeros((M, K_IGO, D_CONT), dtype=jnp.float32)
    for c in range(K_IGO):
        mu = mu.at[0, c, :].set(random.uniform(
            random.fold_in(ikey, c), (D_CONT,),
            minval=-2.0, maxval=2.0))
    Li = jnp.tile(jnp.eye(D_CONT, dtype=jnp.float32)[None, None, :, :],
                   (M, K_IGO, 1, 1)) * 0.5
    v = jnp.zeros((M, K_IGO - 1), dtype=jnp.float32)

    T_igo = 100 if mode == "light" else 300
    fm, fL, fp = mmog_igo_optimizer_mpc(
        skey, T_igo, 0.15, M, K_IGO, B, B0, DIMS, 50,
        fitness_fn_total=cost_fn,
        initial_mu_k=mu, initial_L_inv_k=Li, initial_v_k=v,
        context=ctx,
    )
    jax.block_until_ready((fm, fp))
    mc = fm[0]
    return float(min(cost_fn(mc[k], ctx) for k in range(K_IGO)))


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Forest + IGO ({n_paths} paths, {D_CONT}D continuous)")
print(f"{'='*60}")

config = FidelityConfig(
    none_enabled=True, light_enabled=True, full_enabled=False,
    budget_none=20, budget_light=12,
    q_mode="elite",
    elite_fraction_none=0.2, elite_fraction_light=0.1,
)

solver = GuideTreeSolver(forest, igo_evaluate, config)
t0 = time.perf_counter()
result = solver.solve(final_full_refine=False)
elapsed = time.perf_counter() - t0

print(f"\n  耗时: {elapsed:.1f}s")
print(f"  最优策略:")
if result.best_strategy:
    for g in GEN_NAMES:
        print(f"    {g}: {result.best_strategy[g]}")
    print(f"  cost: {result.best_cost:+.4f}")

# Per-root Q convergence
print(f"\n  各 root 的 Q̃ 偏好:")
for root in forest.roots:
    ranked = sorted(root.children, key=lambda c: c.compute_Q_tilde(), reverse=True)
    for child in ranked[:2]:
        qt = child.compute_Q_tilde()
        print(f"    [{root.name}] {child.choice:>12s}: Q̃={qt:+.3f} n={child.total_n()}")
