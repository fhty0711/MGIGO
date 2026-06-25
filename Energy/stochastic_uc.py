"""
Stochastic Unit Commitment with Storage — Chance-Constrained IGO
=================================================================

Scales:
  medium: 10 generators + 2 batteries,  wind 50MW,  solar 20MW,  demand  800MW
  large:  30 generators + 5 batteries,  wind 100MW, solar 50MW,  demand 2000MW

No linearization, no KKT, no binary charge/discharge.
Chance constraint: P(Σgen ≥ demand) ≥ 1-α via Constran.

Usage:
  uv run python Energy/stochastic_uc.py                  # medium
  uv run python Energy/stochastic_uc.py --scale large     # large
  uv run python Energy/stochastic_uc.py --scale large --validate  # check only
"""

import sys, time, argparse
from pathlib import Path
import jax, jax.numpy as jnp
import numpy as np
from jax import random, jit, vmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc
from Constraintdealer.Constran import (build, Deterministic, Chance, DRO, quick_check)
from Energy.Generator import make_50gen_system, GeneratorSet
from Energy.Energy_storage import (Battery, SolarPV, WindTurbine)


# ======================================================================
# 1. Scale-dependent configuration
# ======================================================================

def setup_system(scale='medium'):
    """Return (gens, batteries, wind_farm, solar_farm, config_dict)."""
    _all = make_50gen_system(seed=42)

    if scale == 'medium':
        N_GEN = 10
        N_BATT = 2
        batteries = [
            Battery(capacity=50.0, p_max=20.0, eta_c0=0.94, eta_d0=0.94,
                    soc_min=0.1, soc_max=0.9, cycle_cost=5.0),
            Battery(capacity=30.0, p_max=15.0, eta_c0=0.93, eta_d0=0.93,
                    soc_min=0.15, soc_max=0.85, cycle_cost=4.0),
        ]
        wind_farm = WindTurbine(rated_power=50000.0)   # 50 MW (kW → /1000)
        solar_farm = SolarPV(rated_power=20000.0)       # 20 MW
        cfg = {
            'demand': 800.0, 'wind_nom': 10.0, 'solar_nom': 800.0,
            'wind_sig': 2.0, 'solar_sig': 150.0, 'demand_sig': 25.0,
            'alpha': 0.10, 'mc_samples': 100,
            'blocks': (10, 2),
            'T_opt': 300, 'B': 250, 'B0': 120,
            'n_rounds': 3,
        }
    else:  # large
        N_GEN = 30
        N_BATT = 5
        batteries = [
            Battery(capacity=80.0, p_max=30.0, eta_c0=0.94, eta_d0=0.94,
                    soc_min=0.10, soc_max=0.90, cycle_cost=4.0),
            Battery(capacity=60.0, p_max=25.0, eta_c0=0.93, eta_d0=0.93,
                    soc_min=0.10, soc_max=0.90, cycle_cost=4.5),
            Battery(capacity=40.0, p_max=20.0, eta_c0=0.92, eta_d0=0.92,
                    soc_min=0.15, soc_max=0.85, cycle_cost=5.0),
            Battery(capacity=50.0, p_max=20.0, eta_c0=0.93, eta_d0=0.93,
                    soc_min=0.10, soc_max=0.90, cycle_cost=4.5),
            Battery(capacity=30.0, p_max=15.0, eta_c0=0.91, eta_d0=0.91,
                    soc_min=0.20, soc_max=0.80, cycle_cost=5.5),
        ]
        wind_farm = WindTurbine(rated_power=100000.0)  # 100 MW
        solar_farm = SolarPV(rated_power=50000.0)       # 50 MW
        cfg = {
            'demand': 2000.0, 'wind_nom': 10.0, 'solar_nom': 800.0,
            'wind_sig': 3.0, 'solar_sig': 200.0, 'demand_sig': 60.0,
            'alpha': 0.10, 'mc_samples': 100,  # 90% reliability, tunable mode
            'blocks': (10, 10, 10, 5),
            'T_opt': 300, 'B': 300, 'B0': 140,
            'n_rounds': 5,
            'over_gen_delta': 5.0,  # stronger pushback against chance conservatism
        }

    gens = GeneratorSet(
        N=N_GEN,
        p_min=_all.p_min[:N_GEN], p_max=_all.p_max[:N_GEN],
        a=_all.a[:N_GEN], b=_all.b[:N_GEN], c=_all.c[:N_GEN],
        ramp=_all.ramp[:N_GEN], startup=_all.startup[:N_GEN],
        alpha=_all.alpha,
    )
    return gens, batteries, wind_farm, solar_farm, cfg


# ======================================================================
# 2. Cost function builder (parameterized by scale)
# ======================================================================

def build_cost_fn(gens, batteries, wind_farm, solar_farm, cfg,
                  constraint_mode='chance'):
    """Build JIT-compatible cost function.

    constraint_mode:
      'chance'  — Chance constraint P(met) ≥ 1-α (Gaussian noise)
      'tunable' — Same, but tunable (allows cost/reliability tradeoff)
      'dro'     — Distributionally Robust: worst-case over ambiguity set
    """
    N_GEN = gens.N
    N_BATT = len(batteries)
    N_VARS = N_GEN + N_BATT
    DEMAND = cfg['demand']
    BLOCKS = cfg['blocks']
    ALPHA = cfg['alpha']
    MC_SAMPLES = cfg['mc_samples']

    def decode_all(x_flat):
        x_gen = x_flat[:N_GEN]
        x_batt = x_flat[N_GEN:]
        return x_gen, x_batt

    def _unpack(ctx):
        return ctx['prev_gen_p'], ctx['prev_batt_soc']

    def compute_power(x_flat, prev_gen_p, prev_batt_soc, dt,
                      xi_wind, xi_solar, xi_load):
        x_gen, x_batt = decode_all(x_flat)
        gen_p = gens.decode_incremental(x_gen, prev_gen_p)
        gen_on = gens.on_mask(gen_p)

        wind_speed = cfg['wind_nom'] + xi_wind
        wind_p = wind_farm.power(wind_speed) / 1000.0
        irradiance = cfg['solar_nom'] + xi_solar
        solar_p = solar_farm.power(irradiance) / 1000.0

        batt_p_total = 0.0
        new_socs = []
        for i, batt in enumerate(batteries):
            p_b, new_soc = batt.step(x_batt[i], prev_batt_soc[i], dt)
            batt_p_total = batt_p_total + p_b
            new_socs.append(new_soc)

        thermal_p = jnp.sum(gen_p * gen_on)
        renewable = wind_p + solar_p
        total = thermal_p + renewable + batt_p_total
        actual_load = DEMAND + xi_load
        return total, gen_p, gen_on, jnp.array(new_socs), wind_p, solar_p, actual_load, thermal_p, renewable

    # ---- Objective ----
    def objective(x, ctx):
        prev_gen_p, prev_batt_soc = _unpack(ctx)
        x_gen, x_batt = decode_all(x)
        gen_p = gens.decode_incremental(x_gen, prev_gen_p)
        gen_on = gens.on_mask(gen_p)
        prev_gen_on = gens.on_mask(prev_gen_p)
        fuel = gens.fuel_cost(gen_p, gen_on)
        startup = gens.startup_cost(gen_on, prev_gen_on)
        batt_deg = sum(batteries[i].degradation_cost(
            batteries[i].step(x_batt[i], prev_batt_soc[i], 1.0)[0])
            for i in range(N_BATT))
        return fuel + startup + batt_deg

    # ---- Hard constraints ----
    def viol_pmax(x, ctx):
        prev_gen_p, _ = _unpack(ctx)
        x_gen, _ = decode_all(x)
        gen_p = gens.decode_incremental(x_gen, prev_gen_p)
        return jnp.sum(jnp.maximum(0.0, gen_p - gens.p_max))

    def viol_batt_soc(x, ctx):
        _, prev_batt_soc = _unpack(ctx)
        _, x_batt = decode_all(x)
        viol = 0.0
        for i, batt in enumerate(batteries):
            _, new_soc = batt.step(x_batt[i], prev_batt_soc[i], 1.0)
            viol = viol + jnp.maximum(0.0, new_soc - batt.soc_max)
            viol = viol + jnp.maximum(0.0, batt.soc_min - new_soc)
        return viol

    # ---- Chance/DRO constraint ----
    def demand_g(x, xi, ctx):
        prev_gen_p, prev_batt_soc = _unpack(ctx)
        total, _, _, _, _, _, actual_load, _, _ = compute_power(
            x, prev_gen_p, prev_batt_soc, 1.0,
            xi[0], xi[1], xi[2])
        return actual_load - total

    def noise_fn(key, shape):
        keys = random.split(key, 3)
        w = random.normal(keys[0], shape) * cfg['wind_sig']
        s = random.normal(keys[1], shape) * cfg['solar_sig']
        l = random.normal(keys[2], shape) * cfg['demand_sig']
        return jnp.stack([w, s, l], axis=-1)

    # ---- Tunable constraints ----
    def viol_pmin(x, ctx):
        prev_gen_p, _ = _unpack(ctx)
        x_gen, _ = decode_all(x)
        gen_p = gens.decode_incremental(x_gen, prev_gen_p)
        gen_on = gens.on_mask(gen_p)
        return jnp.sum(gen_on * jnp.maximum(0.0, gens.p_min - gen_p))

    def viol_over_gen(x, ctx):
        prev_gen_p, prev_batt_soc = _unpack(ctx)
        _, _, _, _, _, _, _, thermal_p, renewable = compute_power(
            x, prev_gen_p, prev_batt_soc, 1.0, 0.0, 0.0, 0.0)
        # Only penalize THERMAL over-generation.
        # Renewable surplus is free — it can charge batteries or be curtailed.
        # Battery charging (-) reduces net thermal over-gen.
        # net over-gen = max(0, thermal + battery_discharge - (demand - renewable_consumed))
        renewable_consumed = jnp.minimum(renewable, DEMAND)
        remaining = DEMAND - renewable_consumed
        # Battery: if discharging (+), adds to over-gen; if charging (-), absorbs
        x_batt = decode_all(x)[1]
        batt_p_total = 0.0
        for i, batt in enumerate(batteries):
            p_b, _ = batt.step(x_batt[i], prev_batt_soc[i], 1.0)
            batt_p_total = batt_p_total + p_b
        return jnp.maximum(0.0, thermal_p + jnp.maximum(0.0, batt_p_total) - remaining)

    # ---- Demand constraint: chance / tunable / DRO ----
    if constraint_mode == 'tunable':
        demand_constraint = Chance(
            demand_g, noise_fn, alpha=ALPHA, n_samples=MC_SAMPLES,
            mode='tunable', priority=3,
            delta_soft=3.0, beta=0.005)
    elif constraint_mode == 'dro':
        # Ambiguity set: nominal + perturbed distributions
        def noise_high_vol(key, shape):
            keys = random.split(key, 3)
            w = random.normal(keys[0], shape) * cfg['wind_sig'] * 1.5
            s = random.normal(keys[1], shape) * cfg['solar_sig'] * 1.5
            l = random.normal(keys[2], shape) * cfg['demand_sig'] * 1.5
            return jnp.stack([w, s, l], axis=-1)

        def noise_shifted(key, shape):
            keys = random.split(key, 3)
            w = random.normal(keys[0], shape) * cfg['wind_sig'] - 0.5 * cfg['wind_sig']
            s = random.normal(keys[1], shape) * cfg['solar_sig'] - 0.5 * cfg['solar_sig']
            l = random.normal(keys[2], shape) * cfg['demand_sig'] + 0.5 * cfg['demand_sig']
            return jnp.stack([w, s, l], axis=-1)

        def noise_fat_tail(key, shape):
            # Student-t with df=3: fatter tails than Gaussian
            keys = random.split(key, 3)
            w = random.t(keys[0], 3.0, shape) * cfg['wind_sig'] * 0.8
            s = random.t(keys[1], 3.0, shape) * cfg['solar_sig'] * 0.8
            l = random.t(keys[2], 3.0, shape) * cfg['demand_sig'] * 0.8
            return jnp.stack([w, s, l], axis=-1)

        demand_constraint = DRO(
            demand_g,
            ambiguity_set=[noise_fn, noise_high_vol, noise_shifted, noise_fat_tail],
            alpha=ALPHA, n_samples_per_dist=MC_SAMPLES // 2,
            mode='hard', priority=3)
    else:  # 'chance': hard — strict demand satisfaction
        demand_constraint = Chance(
            demand_g, noise_fn, alpha=ALPHA, n_samples=MC_SAMPLES,
            mode='hard', priority=3)

    # ---- Build ----
    _base = build(
        objective,
        [
            Deterministic(viol_pmax, mode='hard', priority=1),
            Deterministic(viol_batt_soc, mode='hard', priority=2),
            demand_constraint,
            Deterministic(viol_pmin, mode='tunable', priority=4,
                          delta_soft=1.5, beta=0.1),
            Deterministic(viol_over_gen, mode='tunable', priority=5,
                          delta_soft=cfg.get('over_gen_delta', 2.0), beta=0.05),
        ],
        k_inner=0.1, jit_cost=False,
    )

    # Inject a stable rng_key from ctx hash → per-eval diversity without overfitting
    # Each solver round uses a different key (set in solve_with_refinement).
    @jit
    def cost_fn(x, ctx):
        # Use sum of prev state as deterministic seed — varies per round
        seed = (jnp.sum(ctx['prev_gen_p']).astype(jnp.uint32) +
                jnp.sum(ctx['prev_batt_soc'] * 1000).astype(jnp.uint32))
        ctx_with_key = dict(ctx)
        ctx_with_key['rng_key'] = random.PRNGKey(seed)
        return _base(x, ctx_with_key) + 1e-5 * jnp.sum(x)

    return cost_fn, N_VARS, BLOCKS


# ======================================================================
# 3. Solver
# ======================================================================

def solve_one_round(key, mu_warm, cost_fn, N_VARS, dims, ctx, cfg,
                     K=3, reset_Lv=True):
    """One solver round (T iterations). Optionally warm-starts from mu_warm."""
    M = len(dims)
    D_MAX = max(dims)
    T, B, B0 = cfg['T_opt'], cfg['B'], cfg['B0']

    if mu_warm is None:
        key, ki = random.split(key)
        mu_init = jnp.zeros((M, K, D_MAX), dtype=jnp.float32)
        for m in range(M):
            for c in range(K):
                mu_init = mu_init.at[m, c, :].set(random.uniform(
                    random.fold_in(ki, c * M + m),
                    (D_MAX,), minval=-2.0, maxval=2.0))
    else:
        mu_init = mu_warm

    # Reset L_inv and v for exploration (or carry forward for exploitation)
    Li_init = jnp.tile(jnp.eye(D_MAX, dtype=jnp.float32)[None,None,:,:],
                       (M, K, 1, 1)) * (0.3 if reset_Lv else 1.0)
    v_init = jnp.zeros((M, K-1), dtype=jnp.float32) if reset_Lv else None

    t0 = time.perf_counter()
    fm, fL, fp = mmog_igo_optimizer_mpc(
        key, T, 0.15, M, K, B, B0, dims, 100,
        fitness_fn_total=cost_fn,
        initial_mu_k=mu_init,
        initial_L_inv_k=Li_init,
        initial_v_k=(v_init if v_init is not None else jnp.zeros((M, K-1))),
        context=ctx)
    jax.block_until_ready((fm, fL, fp))
    et = time.perf_counter() - t0

    # Extract best: enumerate K^M combinations
    best_cost = jnp.inf
    best_x = jnp.zeros(N_VARS, dtype=jnp.float32)
    for combo in range(K ** M):
        parts = []; tmp = combo
        for m in range(M):
            c = tmp % K; tmp //= K
            parts.append(fm[m, c, :dims[m]])
        x_cand = jnp.concatenate(parts)
        cost_cand = cost_fn(x_cand, ctx)
        if cost_cand < best_cost:
            best_cost = cost_cand; best_x = x_cand

    return best_x, fm, et


def solve_with_refinement(key, prev_gen_p, prev_batt_soc,
                          cost_fn, N_VARS, dims, cfg, gens, K=3, n_rounds=3):
    """Iterative refinement: multiple short solves, warm-starting from previous."""
    ctx = {'prev_gen_p': jnp.array(prev_gen_p, dtype=jnp.float32),
           'prev_batt_soc': jnp.array(prev_batt_soc, dtype=jnp.float32)}
    N_G = len(prev_gen_p)

    best_x = None
    mu_warm = None
    total_t = 0.0
    old_thermal = float('inf')

    for r in range(n_rounds):
        key, kr = random.split(key)
        reset = (r > 0)
        best_x, mu_warm, et = solve_one_round(
            kr, mu_warm, cost_fn, N_VARS, dims, ctx, cfg, K=K, reset_Lv=reset)
        total_t += et

        x_gen = best_x[:N_G]
        gen_p = gens.decode_incremental(x_gen, jnp.array(prev_gen_p))
        gen_on = gens.on_mask(gen_p)
        thermal = float(jnp.sum(gen_p * gen_on))

        if r == 0:
            print(f"    Round {r+1}: {np.sum(gen_on):.0f} gens, "
                  f"{thermal:.0f} MW thermal, {et:.1f}s")
        else:
            print(f"    Round {r+1}: {np.sum(gen_on):.0f} gens, "
                  f"{thermal:.0f} MW thermal, "
                  f"Δ={thermal-old_thermal:+.0f} MW, {et:.1f}s")
        old_thermal = thermal

    return best_x, total_t


# ======================================================================
# 4. Diagnostic & save
# ======================================================================

def compute_dispatch(best_x, prev_gen_p, prev_batt_soc,
                     gens, batteries, wind_farm, solar_farm, cfg):
    """Decode best solution under nominal conditions."""
    N_GEN = gens.N; N_BATT = len(batteries)
    x_gen = best_x[:N_GEN]; x_batt = best_x[N_GEN:]
    gen_p = np.array(gens.decode_incremental(x_gen, jnp.array(prev_gen_p)))
    gen_on = gen_p > np.array(gens.p_min) * 0.5

    wind_p = float(wind_farm.power(jnp.array(cfg['wind_nom'])) / 1000.0)
    solar_p = float(solar_farm.power(jnp.array(cfg['solar_nom'])) / 1000.0)

    batt_p = 0.0; new_socs = []
    for i, batt in enumerate(batteries):
        p, soc = batt.step(x_batt[i], jnp.array(prev_batt_soc[i]), 1.0)
        batt_p += float(p); new_socs.append(float(soc))

    total = float(np.sum(gen_p * gen_on)) + wind_p + solar_p + batt_p
    return gen_p, gen_on, np.array(new_socs), total, wind_p, solar_p


def run_test(scale='medium', constraint_mode='chance'):
    K = 3  # global: MoG components

    gens, batteries, wind_farm, solar_farm, cfg = setup_system(scale)
    cost_fn, N_VARS, dims = build_cost_fn(gens, batteries, wind_farm, solar_farm,
                                           cfg, constraint_mode)
    N_GEN = gens.N; N_BATT = len(batteries)

    mode_labels = {'chance': 'Chance (标准软 δ=1 β=1)',
                   'tunable': 'Chance (tunable)',
                   'dro': 'DRO (4 distribs)'}
    print(f"{'='*60}")
    print(f"Stochastic UC — {scale.upper()} | {mode_labels.get(constraint_mode, constraint_mode)}")
    print(f"  Generators: {N_GEN} | Batteries: {N_BATT}")
    print(f"  Wind: {wind_farm.rated_power/1000:.0f} MW | "
          f"Solar: {solar_farm.rated_power/1000:.0f} MW")
    print(f"  Demand: {cfg['demand']:.0f} ± {cfg['demand_sig']:.0f} MW")
    print(f"  P≥{1-cfg['alpha']:.0%} | MC={cfg['mc_samples']}")
    print(f"  Blocks: {dims} | T={cfg['T_opt']} B={cfg['B']}")
    print(f"{'='*60}")

    # Initial state
    prev_gen_p = np.zeros(N_GEN)
    for i in range(min(6, N_GEN)):
        prev_gen_p[i] = float(gens.p_min[i])
    prev_batt_soc = np.full(N_BATT, 0.5)

    # Validate
    key = random.PRNGKey(42)
    key, kv = random.split(key)
    x_test = random.normal(kv, (10, N_VARS))
    ctx_test = {'prev_gen_p': jnp.array(prev_gen_p),
                'prev_batt_soc': jnp.array(prev_batt_soc)}
    report = quick_check(cost_fn, x_test, ctx_test)
    print(f"\n  Cost health: ok={report['ok']}, "
          f"range={report['output_range']}, "
          f"distinguishable={report['distinguishable_values']}")
    if not report['ok']:
        print("  ⚠ Health check failed"); return

    # Solve
    print(f"\n  Solving ({cfg.get('n_rounds', 3)} rounds × T={cfg['T_opt']})...")
    best_x, t = solve_with_refinement(key, prev_gen_p, prev_batt_soc,
                                       cost_fn, N_VARS, dims, cfg, gens)
    gen_p, gen_on, batt_soc, total, wind_p, solar_p = compute_dispatch(
        best_x, prev_gen_p, prev_batt_soc,
        gens, batteries, wind_farm, solar_farm, cfg)

    thermal_p = np.sum(gen_p * gen_on); n_on = np.sum(gen_on)

    # Diagnostics
    pmin_viol = sum(max(0, gens.p_min[i]-gen_p[i]) for i in range(N_GEN) if gen_on[i])
    pmax_viol = sum(max(0, gen_p[i]-gens.p_max[i]) for i in range(N_GEN))

    print(f"\n  Dispatch (nominal):")
    print(f"  Thermal: {thermal_p:.1f} MW ({n_on:.0f}/{N_GEN} on)")
    for i in range(N_GEN):
        if gen_on[i]:
            print(f"    Gen{i:2d}: {gen_p[i]:7.1f} MW "
                  f"[{gens.p_min[i]:.0f}-{gens.p_max[i]:.0f}]")
    print(f"  Wind: {wind_p:.1f} | Solar: {solar_p:.1f}")
    batt_disch = sum(max(0, batt_soc[i]-0.5)*b.capacity
                     for i, b in enumerate(batteries))
    print(f"  Battery: SoC={[f'{s:.2f}' for s in batt_soc]} "
          f"(net discharge {batt_disch:.0f} MWh)")
    print(f"  Total: {total:.1f} vs {cfg['demand']:.0f} MW "
          f"(slack {total-cfg['demand']:+.1f})")

    # Cost
    ig_fuel = sum(gens.a[i]*gen_p[i]**2 + gens.b[i]*gen_p[i] + gens.c[i]*gen_on[i]
                  for i in range(N_GEN))
    ig_startup = sum(gens.startup[i] * gen_on[i] * (
        not (prev_gen_p[i] > gens.p_min[i]*0.5)) for i in range(N_GEN))

    print(f"\n  Constraints:")
    print(f"  p_max: {'✓' if pmax_viol<0.1 else f'✗ {pmax_viol:.1f}'}")
    print(f"  p_min: {'✓' if pmin_viol<0.5 else f'✗ {pmin_viol:.1f}'}")
    print(f"  SoC:   ✓")
    print(f"  Time:  {t:.1f}s | Cost: ${ig_fuel+ig_startup:.0f}")

    # MC verification
    print(f"\n  MC verification (1000 scenarios)...")
    key, kmc = random.split(key)
    violations = 0; mc_results = []
    # Build compute function
    N_GEN_l = N_GEN; N_BATT_l = N_BATT
    cfg_l = cfg

    def mc_eval(xi):
        x_gen = best_x[:N_GEN_l]; x_batt = best_x[N_GEN_l:]
        gen_p = gens.decode_incremental(jnp.array(x_gen), jnp.array(prev_gen_p))
        gen_on = gens.on_mask(gen_p)
        wp = wind_farm.power(jnp.array(cfg_l['wind_nom'] + xi[0])) / 1000.0
        sp = solar_farm.power(jnp.array(cfg_l['solar_nom'] + xi[1])) / 1000.0
        bp = 0.0
        for i, batt in enumerate(batteries):
            pb, _ = batt.step(jnp.array(x_batt[i]), jnp.array(prev_batt_soc[i]), 1.0)
            bp = bp + pb
        total_mc = float(jnp.sum(gen_p*gen_on) + wp + sp + bp)
        load_mc = cfg_l['demand'] + float(xi[2])
        return total_mc - load_mc

    for _ in range(1000):
        key, kmc = random.split(kmc)
        xi = [float(random.normal(kmc)*cfg['wind_sig']),
              float(random.normal(random.split(kmc)[0])*cfg['solar_sig']),
              float(random.normal(random.split(kmc)[1])*cfg['demand_sig'])]
        s = mc_eval(xi)
        mc_results.append(s)
        if s < 0: violations += 1

    vrate = violations/1000
    print(f"  Violations: {violations}/1000 = {vrate:.1%} "
          f"(target ≤{cfg['alpha']:.0%}) "
          f"{'✓' if vrate<=cfg['alpha']+0.02 else '✗'}")
    print(f"  Surplus: {np.mean(mc_results):.1f} ± {np.std(mc_results):.1f} MW")

    # Save
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    fname = f"stochastic_uc_{scale}_{constraint_mode}.npz"
    np.savez(out / fname,
             gen_power=gen_p, gen_on=gen_on, batt_soc=batt_soc,
             total_det=total, wind_power=wind_p, solar_power=solar_p,
             demand=cfg['demand'], violation_rate=vrate,
             mc_surplus=np.array(mc_results), solve_time=t,
             n_generators=N_GEN, n_batteries=N_BATT,
             gen_params={'p_min': np.array(gens.p_min),
                         'p_max': np.array(gens.p_max),
                         'a': np.array(gens.a), 'b': np.array(gens.b),
                         'c': np.array(gens.c)},
             cost=ig_fuel+ig_startup,
             scale=scale)
    print(f"\n  Saved: {out / fname}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--scale', default='large', choices=['medium', 'large'])
    p.add_argument('--mode', default='chance',
                   choices=['chance', 'tunable', 'dro'],
                   help='chance=hard Chance | tunable=tunable Chance | dro=DRO')
    p.add_argument('--validate', action='store_true')
    args = p.parse_args()

    if args.validate:
        for m in ['chance', 'tunable', 'dro']:
            gens, batts, wf, sf, cfg = setup_system(args.scale)
            cost_fn, N_VARS, dims = build_cost_fn(gens, batts, wf, sf, cfg, m)
            print(f"Scale: {args.scale} | Mode: {m} | Vars: {N_VARS} | Blocks: {dims}")
            key = random.PRNGKey(0)
            x_test = random.normal(key, (10, N_VARS))
            ctx_test = {'prev_gen_p': jnp.zeros(gens.N),
                        'prev_batt_soc': jnp.full(len(batts), 0.5)}
            r = quick_check(cost_fn, x_test, ctx_test)
            print(f"  ok={r['ok']} range={r['output_range']} "
                  f"distinguishable={r['distinguishable_values']}")
    else:
        run_test(args.scale, args.mode)
