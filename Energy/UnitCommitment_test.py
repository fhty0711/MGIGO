"""
Unit Commitment via Continuous IGO + Constran — Incremental Encoding
====================================================================

Incremental encoding: ramp is GUARANTEED by construction.
  Δp_i = ramp_i · tanh(x_i)  ∈ [-ramp, +ramp]
  p_i  = clip(prev_p_i + Δp_i, 0, P_MAX_i)

On/off emerges from C_COST: p < p_min/2 → on=0 → no no-load cost.
Natural startup/shutdown dynamics via ramp limits — no instantaneous jumps.

Constraint hierarchy (Constran):
  P1 (hard):  p_max
  P2 (hard):  under-generation (Σp ≥ demand)
  P3 (tunable): over-generation slack
  (No ramp constraint — guaranteed by encoding)
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
from Energy.Generator import make_5gen_system, LOAD_5GEN

# ======================================================================
# 1. Problem
# ======================================================================

gens = make_5gen_system()
_LOAD = LOAD_5GEN
N_GEN = gens.N

# ======================================================================
# 2. Objective & constraints (incremental encoding)
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
        Deterministic(viol_under_gen, mode='hard', priority=2),
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
# 4. Solver wrapper
# ======================================================================

def solve_one_step(key, demand, prev_power):
    M, D = 1, N_GEN
    dims = (D,)
    K, B, B0 = 3, 60, 28
    T, T0, dt = 300, 100, 0.15

    key, ki = random.split(key)
    mu = jnp.zeros((M, K, D), dtype=jnp.float32)
    for c in range(K):
        mu = mu.at[0, c, :].set(random.uniform(
            random.fold_in(ki, c), (D,), minval=-2.0, maxval=2.0))
    Li = jnp.tile(jnp.eye(D, dtype=jnp.float32)[None,None,:,:], (M,K,1,1)) * 0.5
    v  = jnp.zeros((M, K-1), dtype=jnp.float32)
    ctx = (jnp.array(demand, dtype=jnp.float32),
           jnp.array(prev_power, dtype=jnp.float32))

    t0 = time.perf_counter()
    fm, fL, fp = mmog_igo_optimizer_mpc(
        key, T, dt, M, K, B, B0, dims, T0,
        fitness_fn_total=uc_cost_fn,
        initial_mu_k=mu, initial_L_inv_k=Li, initial_v_k=v, context=ctx)
    jax.block_until_ready((fm, fL, fp))
    et = time.perf_counter() - t0

    mc = fm[0]; fv = vmap(lambda m: uc_cost_fn(m, ctx))(mc)
    best_x = mc[jnp.argmin(fv)]
    bp = np.array(gens.decode_incremental(best_x, jnp.array(prev_power)))
    bo = gens.on_mask(bp)
    bp_d = bp * np.array(bo)

    fuel = np.sum(np.array(gens.a)*bp**2 + np.array(gens.b)*bp + np.array(gens.c)*bo)
    po = np.array(prev_power) > np.array(gens.p_min)*0.5
    su = np.sum(np.array(gens.startup) * bo * (~po))

    return bp_d, bo, fuel + su, et


# ======================================================================
# 5. Single-step test
# ======================================================================

def test_single_step():
    d = float(_LOAD[12])
    print("="*60)
    print(f"UC Single Step | Gens:{N_GEN} | Demand:{d:.0f}MW")
    print(f"Incremental encoding: ramp built-in, on/off from C_COST")
    print("="*60)

    key = random.PRNGKey(42)
    # Start from a feasible state (Gen 0,1,3 already running at ~greedy dispatch)
    prev = np.array([15.0, 80.0, 0.0, 100.0, 0.0])
    bp, bo, cost, t = solve_one_step(key, d, prev)

    print(f"\n  {'Gen':>5s} {'Power':>8s} {'On':>5s} "
          f"{'P_min':>6s} {'P_max':>6s} {'Ramp':>6s}")
    for i in range(N_GEN):
        print(f"  {i:5d} {bp[i]:8.2f} {str(bo[i]):>5s} "
              f"{gens.p_min[i]:6.1f} {gens.p_max[i]:6.1f} {gens.ramp[i]:6.1f}")
    total = np.sum(bp)
    print(f"  Total: {total:.1f} MW ≥ {d:.1f} MW  "
          f"{'✓' if total>=d-0.5 else '✗ UNDER!'}")
    print(f"  Slack: {total-d:+.1f} MW | Cost: ${cost:.2f} | {t:.2f}s")

    assert total >= d - 1.0, f"Under-generation: {total:.1f} < {d:.1f}"
    for i in range(N_GEN):
        if bo[i]:
            assert bp[i] <= gens.p_max[i]+1e-4, f"G{i} exceeds p_max"
    # Ramp check: |p - prev| ≤ ramp for each generator
    for i in range(N_GEN):
        delta = abs(bp[i] - prev[i])
        assert delta <= gens.ramp[i] + 1.0, \
            f"G{i} ramp violation: Δ={delta:.1f} > ramp={gens.ramp[i]:.1f}"
    print("  ✓ All checks (incl. ramp) passed")


# ======================================================================
# 6. MPC
# ======================================================================

def run_mpc(T_steps=24):
    print("\n"+"="*60)
    print(f"UC MPC — {T_steps} steps | Incremental | Ramp built-in")
    print("="*60)

    M, K, D = 1, 3, N_GEN
    dims = (N_GEN,)
    key = random.PRNGKey(123)
    To, dt, B, B0, T0 = 300, 0.15, 60, 28, 100
    prev = np.zeros(N_GEN)
    hist = {'p':[], 'on':[], 'd':[], 'c':[], 'slack':[], 't':[]}

    for s in range(T_steps):
        d = float(_LOAD[s])
        key, ks, ki = random.split(key, 3)

        mu = jnp.zeros((M,K,D), dtype=jnp.float32)
        for c in range(K):
            mu = mu.at[0,c,:].set(random.uniform(
                random.fold_in(ki,c), (D,), minval=-2.0, maxval=2.0))
        Li = jnp.tile(jnp.eye(D,dtype=jnp.float32)[None,None,:,:],(M,K,1,1))*0.5
        v  = jnp.zeros((M,K-1), dtype=jnp.float32)
        ctx = (jnp.array(d,dtype=jnp.float32), jnp.array(prev,dtype=jnp.float32))

        t0 = time.perf_counter()
        fm,_,fp = mmog_igo_optimizer_mpc(
            ks,To,dt,M,K,B,B0,dims,T0,
            fitness_fn_total=uc_cost_fn,
            initial_mu_k=mu,initial_L_inv_k=Li,initial_v_k=v,context=ctx)
        jax.block_until_ready((fm,fp))
        et = time.perf_counter()-t0

        mc = fm[0]; fv = vmap(lambda m: uc_cost_fn(m,ctx))(mc)
        best_x = mc[jnp.argmin(fv)]
        bp = np.array(gens.decode_incremental(best_x, jnp.array(prev)))
        bo = gens.on_mask(bp)
        bp_d = bp * np.array(bo)

        fuel = np.sum(np.array(gens.a)*bp**2+np.array(gens.b)*bp+np.array(gens.c)*bo)
        po = prev > np.array(gens.p_min)*0.5
        su = np.sum(np.array(gens.startup)*bo*(~po))
        slack = np.sum(bp_d) - d

        # Ramp check
        ramp_viol = np.max(np.abs(bp_d - prev))
        max_ramp = np.max(np.array(gens.ramp))

        hist['p'].append(bp_d); hist['on'].append(bo)
        hist['d'].append(d); hist['c'].append(fuel+su)
        hist['slack'].append(slack); hist['t'].append(et)
        prev = bp_d

        print(f" Step{s:2d} | d={d:6.1f} | gen={np.sum(bp_d):6.1f} | "
              f"on={np.sum(bo):1.0f}/{N_GEN} | slack={slack:+.1f} | "
              f"Δmax={ramp_viol:.1f}/{max_ramp:.0f} | ${fuel+su:8.1f} | {et:.2f}s")

    tt = hist['t']; sl = np.array(hist['slack'])
    print(f"\n{'='*60}")
    print(f"Summary: {T_steps} steps | {sum(tt):.1f}s (avg {np.mean(tt):.2f}s)")
    print(f"Cost: ${sum(hist['c']):,.0f}")
    print(f"Slack: min={min(sl):+.1f} max={max(sl):+.1f}")
    under = np.sum(sl < -0.5)
    print(f"Under-gen steps: {under}/{T_steps}")
    print(f"{'='*60}")
    _plot(hist)
    return hist


def _plot(hist):
    P = np.array(hist['p']); T = P.shape[0]
    d = np.array(hist['d']); t = np.arange(T)
    colors = plt.cm.tab10(np.arange(N_GEN))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0,0]; b = np.zeros(T)
    for i in range(N_GEN):
        ax.bar(t, P[:,i], bottom=b, color=colors[i],
               label=f'G{i}')
        b += P[:,i]
    ax.plot(t, d, 'k-o', lw=2, ms=4); ax.legend(fontsize=5, ncol=2)
    ax.set_title('Dispatch (incremental)'); ax.grid(alpha=0.3)

    ax = axes[0,1]
    on = np.array(hist['on'])
    ax.imshow(on.T, aspect='auto', cmap='RdYlGn', origin='lower', interpolation='none')
    ax.set_yticks(range(N_GEN)); ax.set_title('On/Off')

    ax = axes[1,0]
    tg = np.sum(P, axis=1)
    ax.plot(t, tg, 'b-', lw=2, label='Gen')
    ax.plot(t, d, 'k--', lw=2, label='Demand')
    ax.fill_between(t, tg, d, where=(tg<d), alpha=0.3, color='red')
    ax.set_title('Power Balance'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1,1]
    c = np.array(hist['c'])
    ax.bar(t, c, color='steelblue', alpha=0.7)
    ax.axhline(np.mean(c), color='red', ls='--', label=f'Mean ${np.mean(c):.0f}')
    ax.set_title(f'Cost (Total ${np.sum(c):,.0f})'); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('UC: Incremental Encoding (ramp built-in)', fontweight='bold')
    fig.tight_layout()
    out = Path("output_igo_test"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/"uc_incr_mpc.png", dpi=150); plt.close(fig)
    print(f"Plot: {out/'uc_incr_mpc.png'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='single', choices=['single','mpc'])
    args = p.parse_args()
    if args.mode == 'single':
        test_single_step()
    else:
        run_mpc()
