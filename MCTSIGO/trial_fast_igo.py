"""
Forest + 快速 IGO — 比照 Energy/UnitCommitment_test.py
=======================================================

Forest: 1 root × 3 choices (宽浅)
IGO:   对标 solve_one_step (sigmoid 编码, dims=(D,), mu∈[-2,2], Li=0.5I)
Cost:  Constran 包含非马尔可夫启动费

用法:
  uv run python MCTSIGO/trial_fast_igo.py
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax, jax.numpy as jnp
import numpy as np
from jax import random

from Constraintdealer.Constran import build, Deterministic
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from Energy.Generator import make_5gen_system
from MCTSIGO.decision_unit import DecisionUnit
from MCTSIGO.guide_tree import IndexTree, GuideTreeSolver
from MCTSIGO.fidelity_config import FidelityConfig

gens = make_5gen_system()
N_GEN, DEMAND = gens.N, 150.0
PREV_ON = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0])
print(f"5-gen, demand={DEMAND}MW, prev_on={np.array(PREV_ON).astype(int).tolist()}")

# ── Forest ──
forest = IndexTree.from_decision_units([
    DecisionUnit("Storage", ["CHARGE", "DISCHARGE", "IDLE"]),
])
print(f"Forest: {forest.describe()}")

# ── Cost (对标 UnitCommitment_test + 非马尔可夫启动费) ──
p_max, p_min, a_arr, b_arr, c_arr = gens.p_max, gens.p_min, gens.a, gens.b, gens.c
startup_arr = gens.startup

# Storage mode → numeric encoding (JAX 不能 trace 字符串)
ST_MODE_MAP = {"CHARGE": 0, "DISCHARGE": 1, "IDLE": 2}
ST_EFFECTS  = jnp.array([0.02, -0.01, 0.0])  # CHARGE, DISCHARGE, IDLE

def obj_fn(x, ctx):
    p = gens.decode_power(x)
    on = gens.on_mask(p)
    prev_on = ctx["prev_on"]
    st_mode = ctx["st_mode"]  # int: 0=CHARGE, 1=DISCHARGE, 2=IDLE
    fuel = jnp.sum(a_arr * p**2 + b_arr * p + c_arr * on)
    startup = jnp.sum(startup_arr * on * (1.0 - prev_on))
    st_effect = ST_EFFECTS[st_mode]
    return (fuel + startup) / DEMAND + st_effect

def viol_under(x, ctx):
    p = gens.decode_power(x)
    on = gens.on_mask(p)
    return jnp.maximum(0.0, DEMAND - jnp.sum(p * on)) / DEMAND

def viol_pmax(x, ctx):
    return jnp.sum(jnp.maximum(0.0, gens.decode_power(x) - p_max)) / DEMAND

def viol_over(x, ctx):
    p = gens.decode_power(x)
    on = gens.on_mask(p)
    return jnp.maximum(0.0, jnp.sum(p * on) - DEMAND) / DEMAND

_base_cost = build(obj_fn, [
    Deterministic(viol_pmax,  mode="hard",    priority=1, delta=3.0),
    Deterministic(viol_under, mode="hard",    priority=2, delta=3.0),
    Deterministic(viol_over,  mode="tunable", priority=3, delta_soft=1.0, beta=0.05),
], k_inner=0.1, jit_cost=False)  # jit_cost=False: 避免嵌套 JIT

@jax.jit
def cost_fn(x, ctx):
    return _base_cost(x, ctx) + 1e-5 * jnp.sum(x)  # 手动 JIT + 正则化

# ── IGO Evaluate (精确对标 solve_one_step) ──
IGO_KEY = random.PRNGKey(42)
M, D = 1, N_GEN
K, B, B0 = 3, 60, 28
DIMS = (D,)
# 按保真度分级: none=0, light=T=100, full=T=300
IGO_PARAMS = {
    "none":  {"T": 0,   "T0": 0,   "dt": 0.15},
    "light": {"T": 100, "T0": 40,  "dt": 0.15},
    "full":  {"T": 300, "T0": 100, "dt": 0.15},
}

def igo_evaluate(strategy, mode):
    global IGO_KEY
    st_mode = ST_MODE_MAP[strategy["Storage"]]
    ctx = {"prev_on": PREV_ON, "st_mode": st_mode}

    if mode == "none":
        x = jnp.zeros(D)
        return float(cost_fn(x, ctx))

    IGO_KEY, skey, ikey = random.split(IGO_KEY, 3)
    mu = jnp.zeros((M, K, D), dtype=jnp.float32)
    for c in range(K):
        mu = mu.at[0, c, :].set(random.uniform(
            random.fold_in(ikey, c), (D,), minval=-2.0, maxval=2.0))
    Li = jnp.tile(jnp.eye(D, dtype=jnp.float32)[None,None,:,:], (M,K,1,1)) * 0.5
    v  = jnp.zeros((M, K-1), dtype=jnp.float32)

    cfg = IGO_PARAMS[mode]
    fm, fL, fp = mmog_igo_optimizer_mpc(
        skey, cfg["T"], cfg["dt"], M, K, B, B0, DIMS, cfg["T0"],
        fitness_fn_total=cost_fn,  # 直接传，避免 lambda 导致重编译
        initial_mu_k=mu, initial_L_inv_k=Li, initial_v_k=v,
        context=ctx,
    )
    jax.block_until_ready((fm, fp))
    mc = fm[0]
    return float(min(cost_fn(mc[k], ctx) for k in range(K)))

# ── Run ──
print(f"\n{'='*50}")
print(f"Forest + IGO (none=1, light=4)")
print(f"{'='*50}")

config = FidelityConfig(
    none_enabled=True, light_enabled=True, full_enabled=False,
    budget_none=1, budget_light=4,
    q_mode="elite", elite_fraction_none=0.5, elite_fraction_light=0.5,
)

solver = GuideTreeSolver(forest, igo_evaluate, config)
t0 = time.perf_counter()
result = solver.solve(final_full_refine=False)
elapsed = time.perf_counter() - t0

print(f"耗时: {elapsed:.1f}s")
print(f"最优: {result.best_strategy}  cost={result.best_cost:+.4f}")

# IGO 逐策略验证
print(f"\nIGO light 逐策略验证 (每个跑 1 次 IGO):")
for st in ["DISCHARGE", "CHARGE", "IDLE"]:
    s = {"Storage": st}
    t0 = time.perf_counter()
    c = igo_evaluate(s, "light")
    t1 = time.perf_counter() - t0
    found_st = result.best_strategy["Storage"] if result.best_strategy else "?"
    mark = " ← 最优" if st == found_st else ""
    print(f"  {st:>10s}: {c:+.4f}  ({t1:.1f}s/IGO){mark}")
