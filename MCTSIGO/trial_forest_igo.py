"""
Forest + IGO + Warm-Start — 层宽浅深，非马尔可夫 cost
=====================================================

设计:
  Layer 0 (粗): Strategy   — ECO / BALANCED / AGGRESSIVE
  Layer 1 (中): Baseload   — (Gen0,Gen3) 4种组合
  Layer 2 (细): Peaker     — (Gen1,Gen2,Gen4) 8种组合  ← IGO 优化连续出力

  Total: 3×4×8 = 96 路径

  none:  Constran P1 结构可行性 (x=warm)
  light: IGO T=100, 用 none 的 x 做 warm-start
  full:  IGO T=300, MC=30, 用 light 的 x 做 warm-start

用法:
  uv run python MCTSIGO/trial_forest_igo.py
"""

import sys, time, argparse
from pathlib import Path
from typing import Dict, List, Optional
import jax, jax.numpy as jnp
import numpy as np
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Constraintdealer.Constran import build, Deterministic
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from Energy.Generator import make_5gen_system
from MCTSIGO.decision_unit import DecisionUnit
from MCTSIGO.guide_tree import IndexTree, GuideTreeSolver, sigma
from MCTSIGO.fidelity_config import FidelityConfig

# ══════════════════════════════════════════════════════════════════════════════
# 1. Setup
# ══════════════════════════════════════════════════════════════════════════════

gens = make_5gen_system()
N_GEN = gens.N  # 5
DEMAND = 180.0   # MW — 需要开启多台机组

# Generator grouping for forest layers
BASELOAD_GENS = [0, 3]   # Gen0 (50MW), Gen3 (100MW)
PEAKER_GENS  = [1, 2, 4] # Gen1 (80MW), Gen2 (30MW), Gen4 (40MW)

# Baseload combinations
BASELOAD_CHOICES = [
    f"G0_{s0}_G3_{s3}"
    for s0 in ["ON", "OFF"] for s3 in ["ON", "OFF"]
]

# Peaker combinations
PEAKER_CHOICES = [
    f"G1_{s1}_G2_{s2}_G4_{s4}"
    for s1 in ["ON", "OFF"] for s2 in ["ON", "OFF"] for s4 in ["ON", "OFF"]
]

print(f"Forest: 3 layers × 96 paths")
print(f"  Layer 0: Strategy (ECO/BALANCED/AGGRESSIVE)")
print(f"  Layer 1: Baseload — {len(BASELOAD_CHOICES)} combos")
print(f"  Layer 2: Peaker  — {len(PEAKER_CHOICES)} combos")
print(f"  Demand: {DEMAND}MW")
print(f"  Gens: p_max={[float(g) for g in gens.p_max]}, total_cap={float(gens.p_max.sum())}MW")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Forest with layers
# ══════════════════════════════════════════════════════════════════════════════

forest = IndexTree.from_forest(
    [
        ("Strategy", [DecisionUnit("Strategy",
                     ["ECO", "BALANCED", "AGGRESSIVE"])]),
        ("Baseload", [DecisionUnit("Baseload", BASELOAD_CHOICES)]),
        ("Peaker",  [DecisionUnit("Peaker", PEAKER_CHOICES)]),
    ],
    layers=[["Strategy"], ["Baseload"], ["Peaker"]],
)
print(f"  Forest: {forest.describe()}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Decode strategy → on_mask + cost adjustments
# ══════════════════════════════════════════════════════════════════════════════

def decode_strategy(strategy: Dict[str, str]):
    """Decode forest strategy to on_mask (5,) and strategy weight."""
    on_mask = np.zeros(N_GEN)

    def _parse_combo(combo_str, gen_indices):
        # combo_str like "G0_ON_G3_OFF" → paired: (G0,ON), (G3,OFF)
        parts = combo_str.split("_")
        for i in range(0, len(parts), 2):
            g_name = parts[i]       # "G0"
            state  = parts[i+1]     # "ON" or "OFF"
            g_idx = int(g_name[1:]) # 0
            if g_idx in gen_indices:
                on_mask[g_idx] = 1.0 if state == "ON" else 0.0

    _parse_combo(strategy["Baseload"], BASELOAD_GENS)
    _parse_combo(strategy["Peaker"], PEAKER_GENS)

    strat = strategy["Strategy"]
    strat_weight = {"ECO": 1.3, "BALANCED": 1.0, "AGGRESSIVE": 0.7}[strat]

    return jnp.array(on_mask), strat_weight


# ══════════════════════════════════════════════════════════════════════════════
# 4. Constran cost functions (per fidelity)
# ══════════════════════════════════════════════════════════════════════════════

def _make_cost_fn(fidelity: str):
    """Build Constran cost. Fidelity controls which constraints are active."""
    D = N_GEN
    pmax = gens.p_max
    a_arr, b_arr, c_arr = gens.a, gens.b, gens.c

    def uc_objective(x, ctx):
        on = ctx["on_mask"]
        sw = ctx.get("strat_weight", 1.0)
        fuel = jnp.sum(on * (a_arr * x**2 + b_arr * x + c_arr))
        return sw * fuel / DEMAND

    def viol_struct(x, ctx):
        on = ctx["on_mask"]
        mp = jnp.sum(on * pmax)
        return jnp.maximum(0.0, DEMAND - mp) / DEMAND

    def viol_pmax(x, ctx):
        return jnp.sum(jnp.maximum(0.0, x - pmax)) / DEMAND

    def viol_under_gen(x, ctx):
        on = ctx["on_mask"]
        tp = jnp.sum(on * x)
        return jnp.maximum(0.0, DEMAND - tp) / DEMAND

    def viol_over_gen(x, ctx):
        on = ctx["on_mask"]
        tp = jnp.sum(on * x)
        return jnp.maximum(0.0, tp - DEMAND) / DEMAND

    constraints = []
    # P1: always — structural feasibility
    constraints.append(Deterministic(viol_struct, mode="hard", priority=1, delta=5.0))

    if fidelity in ("light", "full"):
        constraints.append(Deterministic(viol_pmax, mode="hard", priority=2, delta=3.0))
        constraints.append(Deterministic(viol_under_gen, mode="hard", priority=3, delta=3.0))

    if fidelity == "full":
        constraints.append(Deterministic(viol_over_gen, mode="tunable", priority=4,
                                         delta_soft=1.0, beta=0.05))

    return build(uc_objective, constraints, k_inner=0.1, jit_cost=False)


cost_none  = _make_cost_fn("none")
cost_light = _make_cost_fn("light")
cost_full  = _make_cost_fn("full")

# ══════════════════════════════════════════════════════════════════════════════
# 5. IGO Evaluate with warm-start
# ══════════════════════════════════════════════════════════════════════════════

IGO_KEY = random.PRNGKey(123)

# Warm-start store: {(phase, strategy_key): best_mu}
warm_start_mu: Dict[tuple, jnp.ndarray] = {}


def _strategy_key(s: Dict[str, str]) -> tuple:
    return tuple(sorted(s.items()))


def _run_igo(subkey, cost_fn, ctx, T, B, B0, warm_mu: Optional[jnp.ndarray] = None):
    """Run IGO optimization. If warm_mu provided, use as initial mean."""
    D = N_GEN
    K = 3
    mu = jnp.zeros((1, K, D), dtype=jnp.float32)

    if warm_mu is not None:
        # Warm-start: center all components around previous best
        mu = mu.at[0, :, :].set(warm_mu[None, :])
    else:
        # Fresh init
        pmax_np = np.array(gens.p_max)
        for k in range(K):
            mu = mu.at[0, k, :].set(jnp.array([
                random.uniform(random.fold_in(subkey, k * D + d), (),
                               minval=0.0, maxval=float(pmax_np[d]))
                for d in range(D)
            ]))

    Li = jnp.tile(jnp.eye(D, dtype=jnp.float32)[None, None, :, :],
                   (1, K, 1, 1)) * 1.0
    v = jnp.zeros((1, K - 1), dtype=jnp.float32)

    fm, fL, fp = mmog_igo_optimizer_mpc(
        subkey, T, 0.15,
        M=1, K=K, B=B, B0=B0,
        dims=(D,), T_0=max(20, T // 4),
        fitness_fn_total=lambda x, c: cost_fn(x, c),
        initial_mu_k=mu, initial_L_inv_k=Li, initial_v_k=v,
        context=ctx,
    )
    jax.block_until_ready((fm, fp))

    mc = fm[0]
    fvs = jnp.array([cost_fn(mc[k], ctx) for k in range(K)])
    best_k = int(jnp.argmin(fvs))
    best_mu = mc[best_k]
    best_cost = float(fvs[best_k])

    return best_cost, best_mu


def igo_evaluate(strategy: Dict[str, str], mode: str) -> float:
    """Forest 评估接口 with warm-start between fidelities."""
    global IGO_KEY
    on_mask, strat_weight = decode_strategy(strategy)
    ctx = {"demand": DEMAND, "on_mask": on_mask, "strat_weight": strat_weight}
    sk = _strategy_key(strategy)

    if mode == "none":
        x = warm_start_mu.get(("light", sk),
               warm_start_mu.get(("none", sk),
                jnp.zeros(N_GEN)))
        return float(cost_none(x, ctx))

    elif mode == "light":
        IGO_KEY, skey = random.split(IGO_KEY)
        warm = warm_start_mu.get(("none", sk))
        cost, best_mu = _run_igo(skey, cost_light, ctx,
                                 T=100, B=80, B0=40, warm_mu=warm)
        warm_start_mu[("light", sk)] = best_mu
        return cost

    elif mode == "full":
        IGO_KEY, skey = random.split(IGO_KEY)
        warm = warm_start_mu.get(("light", sk))
        cost, best_mu = _run_igo(skey, cost_full, ctx,
                                 T=300, B=120, B0=60, warm_mu=warm)
        warm_start_mu[("full", sk)] = best_mu
        return cost

    raise ValueError(f"Unknown mode: {mode}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Brute force (heuristic x, noiseless) — ground truth
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Brute Force (heuristic x, full cost, noiseless)")
print(f"{'='*60}")

from itertools import product

bf_results = []
for strat in ["ECO", "BALANCED", "AGGRESSIVE"]:
    for bl in BASELOAD_CHOICES:
        for pk in PEAKER_CHOICES:
            s = {"Strategy": strat, "Baseload": bl, "Peaker": pk}
            on_mask, sw = decode_strategy(s)
            ctx = {"demand": DEMAND, "on_mask": on_mask, "strat_weight": sw}

            # Heuristic x: ON gens share demand proportionally
            on_idx = [i for i in range(N_GEN) if on_mask[i] > 0.5]
            x = np.zeros(N_GEN)
            if on_idx:
                total_cap = sum(float(gens.p_max[i]) for i in on_idx)
                for i in on_idx:
                    x[i] = min(float(gens.p_max[i]),
                               DEMAND * float(gens.p_max[i]) / max(1e-3, total_cap))
            c = float(cost_full(jnp.array(x), ctx))
            bf_results.append((s, c, on_mask))

bf_results.sort(key=lambda r: r[1])
best_s, best_c, best_mask = bf_results[0]

feasible = sum(1 for _, c, _ in bf_results if c < 0.5)
print(f"  Feasible: {feasible}/{len(bf_results)}")
print(f"  Best: cost={best_c:+.4f}")
print(f"  Top 5:")
for i, (s, c, _) in enumerate(bf_results[:5]):
    short = f"{s['Strategy'][:3]}|BL={s['Baseload'][:8]}|PK={s['Peaker'][:8]}"
    print(f"    #{i+1}: {short:>35s}  cost={c:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Run Forest + IGO
# ══════════════════════════════════════════════════════════════════════════════

def main(budget_mult: float = 1.0, use_warm: bool = True):
    global warm_start_mu
    warm_start_mu.clear()

    base = max(30, 96 // 3)
    n_none  = int(base * 0.5 * budget_mult)
    n_light = int(base * 0.35 * budget_mult)
    n_full  = int(base * 0.15 * budget_mult)

    print(f"\n{'='*60}")
    print(f"Forest + IGO (warm-start={'ON' if use_warm else 'OFF'}, budget={budget_mult:.1f}x)")
    print(f"  none={n_none}, light={n_light}, full={n_full}")
    print(f"{'='*60}")

    if not use_warm:
        warm_start_mu.clear()  # Disable warm-start by always clearing

    config = FidelityConfig(
        none_enabled=True, light_enabled=True, full_enabled=True,
        budget_none=n_none, budget_light=n_light, budget_full=n_full,
        q_mode="elite",
        elite_fraction_none=0.25,
        elite_fraction_light=0.15,
        elite_fraction_full=0.08,
    )

    solver = GuideTreeSolver(forest, igo_evaluate, config)
    t0 = time.perf_counter()
    result = solver.solve(final_full_refine=False)
    elapsed = time.perf_counter() - t0

    found = result.best_strategy
    if found:
        _, _ = decode_strategy(found)
        true_cost = next((c for s, c, _ in bf_results if s == found), 999)
    else:
        true_cost = 999
    match = (found == best_s)
    gap = true_cost - best_c if not match else 0

    print(f"\n  耗时: {elapsed:.1f}s")
    print(f"  找到: {found['Strategy'][:3]}|{found['Baseload'][:10]}|{found['Peaker'][:10]}" if found else "  None")
    print(f"  found cost: {true_cost:+.4f}  |  best cost: {best_c:+.4f}  |  match: {'✅' if match else f'Δ={gap:+.3f}'}")

    # Per-layer analysis
    print(f"\n  各 Layer 的 Q̃ 偏好:")
    for li, layer in enumerate(forest.layers):
        for root in layer:
            ranked = sorted(root.children, key=lambda c: c.compute_Q_tilde(), reverse=True)
            best_child = ranked[0]
            qt = best_child.compute_Q_tilde()
            truth = "✓" if found and found[root.name] == best_child.choice else ""
            print(f"    L{li} [{root.name:>12s}] → {best_child.choice:>20s}  "
                  f"Q̃={qt:+.4f}  n={best_child.total_n()}  {truth}")

    return result, match, elapsed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--no-warm", action="store_true")
    args = p.parse_args()
    main(budget_mult=args.budget, use_warm=not args.no_warm)
