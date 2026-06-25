"""
Unit Commitment — 50 Generators, Incremental Encoding + Block-Decomposed IGO
============================================================================

50 generators, 1 variable each → 5 blocks × D=10.

Incremental encoding:
  Δp_i = ramp_i · tanh(x_i)  ∈ [-ramp, +ramp]  ← guaranteed
  p_i  = clip(prev_p_i + Δp_i, 0, P_MAX_i)

On/off emerges from C_COST. Ramp is built-in — no penalty needed.
"""

import sys, time
from pathlib import Path
import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import random, jit, vmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from Constraintdealer.Constran import (build, Deterministic, quick_check)
from Energy.Generator import make_50gen_system, LOAD_50GEN

# ======================================================================
# 1. Problem
# ======================================================================

gens = make_50gen_system(seed=42)
_LOAD = LOAD_50GEN
N_GEN = gens.N
N_BLOCKS = 5
D_PER_BLOCK = N_GEN // N_BLOCKS  # 10

DEMAND = 1800.0

print(f"50-Generator System (incremental encoding):")
print(f"  Baseload (200-500MW): 15 | Mid-merit (80-200MW): 20 | Peakers (20-80MW): 15")
print(f"  Total capacity: {gens.total_capacity:.0f} MW | Demand: {DEMAND:.0f} MW")
print(f"  Blocks: {N_BLOCKS} × D={D_PER_BLOCK}")


# ======================================================================
# 2. Objective & constraints — incremental, no ramp penalty
# ======================================================================

def uc_objective(x, ctx):
    demand, prev_power = ctx
    p = gens.decode_incremental(x, prev_power)
    on = gens.on_mask(p)
    prev_on = gens.on_mask(prev_power)
    return gens.operating_cost(p, on, prev_on)


def viol_under_gen(x, ctx):
    demand, prev_power = ctx
    p = gens.decode_incremental(x, prev_power)
    on = gens.on_mask(p)
    return jnp.maximum(0.0, demand - jnp.sum(p * on))


def viol_pmax(x, ctx):
    demand, prev_power = ctx
    p = gens.decode_incremental(x, prev_power)
    return jnp.sum(jnp.maximum(0.0, p - gens.p_max))


def viol_over_gen(x, ctx):
    demand, prev_power = ctx
    p = gens.decode_incremental(x, prev_power)
    on = gens.on_mask(p)
    return jnp.maximum(0.0, jnp.sum(p * on) - demand)


# ======================================================================
# 3. Build cost via Constran
# ======================================================================

_base_uc_cost = build(
    uc_objective,
    [
        Deterministic(viol_pmax, mode='hard', priority=1),
        # Under-gen: tunable (not hard) — needed for 50-gen exploration.
        # Hard mode creates flat terrain when no initial sample is feasible.
        Deterministic(viol_under_gen, mode='tunable', priority=2,
                      delta_soft=10.0, beta=0.005),
        Deterministic(viol_over_gen, mode='tunable', priority=3,
                      delta_soft=3.0, beta=0.02),
    ],
    k_inner=0.1,
    jit_cost=False,
)


@jit
def uc_cost_fn(x, ctx):
    return _base_uc_cost(x, ctx) + 1e-5 * jnp.sum(x)


# ======================================================================
# 4. Solver
# ======================================================================

def solve_one_step(key, demand, prev_power,
                   T=800, dt=0.15, K=3, B=300, B0=140, T_0=100):
    M = N_BLOCKS
    dims = tuple([D_PER_BLOCK] * M)
    D_max = D_PER_BLOCK

    key, ki = random.split(key)
    mu = jnp.zeros((M, K, D_max), dtype=jnp.float32)
    for c in range(K):
        for m in range(M):
            mu = mu.at[m, c, :].set(random.uniform(
                random.fold_in(ki, c * M + m),
                (D_max,), minval=-2.0, maxval=2.0))
    Li = jnp.tile(jnp.eye(D_max, dtype=jnp.float32)[None, None, :, :],
                  (M, K, 1, 1)) * 0.3
    v = jnp.zeros((M, K - 1), dtype=jnp.float32)
    ctx = (jnp.array(demand, dtype=jnp.float32),
           jnp.array(prev_power, dtype=jnp.float32))

    t0 = time.perf_counter()
    fm, fL, fp = mmog_igo_optimizer_mpc(
        key, T, dt, M, K, B, B0, dims, T_0,
        fitness_fn_total=uc_cost_fn,
        initial_mu_k=mu, initial_L_inv_k=Li, initial_v_k=v, context=ctx)
    jax.block_until_ready((fm, fL, fp))
    et = time.perf_counter() - t0

    # Extract best: evaluate all K^M component-mean combinations
    best_cost = jnp.inf
    best_x = jnp.zeros(N_GEN, dtype=jnp.float32)
    for combo in range(K ** M):
        parts = []
        tmp = combo
        for m in range(M):
            c = tmp % K
            tmp //= K
            parts.append(fm[m, c])
        x_cand = jnp.concatenate(parts)
        cost_cand = uc_cost_fn(x_cand, ctx)
        if cost_cand < best_cost:
            best_cost = cost_cand
            best_x = x_cand

    bp = np.array(gens.decode_incremental(best_x, jnp.array(prev_power)))
    bo = gens.on_mask(bp)
    bp_d = bp * np.array(bo)

    fuel = np.sum(np.array(gens.a) * bp**2 + np.array(gens.b) * bp +
                   np.array(gens.c) * bo)
    po = np.array(prev_power) > np.array(gens.p_min) * 0.5
    su = np.sum(np.array(gens.startup) * bo * (~po))

    return bp_d, bo, fuel + su, et


# ======================================================================
# 5. Initial state — greedy dispatch for single-step test
# ======================================================================

def _make_initial_state(demand):
    """Minimal initial state: a few baseload generators at p_min.

    Represents day-ahead commitments — not a full dispatch.
    The solver must ramp them up + start additional units to meet demand.
    """
    prev = np.zeros(N_GEN)
    # Turn on first 6 baseload generators at p_min (~1200 MW total capacity)
    committed = 0
    for i in range(15):  # baseload are indices 0-14
        if committed >= 6:
            break
        if gens.p_min[i] > 0:
            prev[i] = float(gens.p_min[i])
            committed += 1
    return prev


# ======================================================================
# 6. Test
# ======================================================================

def test_single_step():
    prev = _make_initial_state(DEMAND)
    total_prev = np.sum(prev)
    on_prev = np.sum(gens.on_mask(prev))

    print(f"\n{'='*60}")
    print(f"50-Gen UC — Single Step (demand={DEMAND:.0f} MW)")
    print(f"Initial: {total_prev:.0f} MW from {on_prev}/50 gens")
    print(f"Incremental: ramp built-in | T=800 B=300 | {N_BLOCKS} blocks")
    print(f"{'='*60}")

    key = random.PRNGKey(123)
    bp, bo, cost, t = solve_one_step(key, DEMAND, prev)

    total = np.sum(bp)
    n_on = np.sum(bo)
    slack = total - DEMAND

    print(f"\n  Total: {total:.1f} MW ≥ {DEMAND:.0f} MW  "
          f"{'✓' if total >= DEMAND - 1 else '✗ UNDER!'}")
    print(f"  Slack: {slack:+.1f} MW | On: {n_on}/50 (was {on_prev})")
    print(f"  Cost: ${cost:,.0f} | Time: {t:.1f}s")

    # Type breakdown
    for name, start, end in [('baseload', 0, 15), ('mid-merit', 15, 35), ('peaker', 35, 50)]:
        on_now = np.sum(bo[start:end])
        on_was = np.sum(gens.on_mask(prev)[start:end])
        print(f"  {name}: {on_now}/{end-start} on (was {on_was})")

    # Ramp check: only for generators that were already running.
    # Startup jumps (0→p_min) are not ramp-limited.
    was_on = prev > np.array(gens.p_min) * 0.5
    ramp_viols = []
    for i in range(N_GEN):
        if was_on[i]:
            delta = abs(bp[i] - prev[i])
            if delta > gens.ramp[i] + 0.5:
                ramp_viols.append(f"G{i}: |{bp[i]:.1f}-{prev[i]:.1f}|={delta:.1f} > ramp={gens.ramp[i]:.1f}")
    if ramp_viols:
        print(f"  Ramp violations (running gens): {len(ramp_viols)}")
        for v in ramp_viols[:3]:
            print(f"    {v}")
    else:
        print(f"  Ramp: all running gens within limits  ✓")

    assert total >= DEMAND - 2.0, f"Under-gen: {total:.1f} < {DEMAND:.1f}"
    assert np.all(bp <= np.array(gens.p_max) + 0.1), "p_max violation"
    assert len(ramp_viols) == 0, f"Ramp violations: {len(ramp_viols)}"
    print("  ✓ All checks passed")


if __name__ == "__main__":
    test_single_step()
