"""
Objective Transform Numerical Experiment
========================================

Compare three approaches for composing sub-objectives into a scalar cost:

  A. Manual weights:   w1·track + w2·final + w3·ctrl
  B. log + dynamic k:  σ(Σ σ_k(log(term_i), k=ctx['k_i']))
  C. T_alpha + fixed k: σ(Σ σ_k(T_alpha(term_i), k=0.2))

Goal: show that T_alpha + fixed k (Scheme C) maintains sensitivity
across all orders of magnitude WITHOUT depending on noisy elite-solution k.

No solver needed — pure static numerical analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Constraintdealer.ObjectiveComposer import (
    compose_objective_semantic, OBJ_K_SAT,
)
from Constraintdealer.Constran import (
    sigma_k, log_transform, T_alpha,
    OBJECTIVE_KNOTS_G, OBJECTIVE_KNOTS_T,
)


# Pre-recorded k-history from actual composed-mode run (populated by experiment_3)
COMPOSED_K_HISTORY = [
    # (step, k_tracking, k_final, k_control)
    (0,  0.1199, 0.1190, 10.4921),
    (1,  0.1213, 0.1220, 0.7400),
    (2,  0.1215, 0.1221, 0.9532),
    (3,  0.1217, 0.1222, 1.1595),
    (4,  0.1219, 0.1223, 1.2358),
    (20, 0.1367, 0.1582, 0.3482),
    (40, 2.2554, 10.4921, 0.2089),
    (60, 7.0291, 10.4921, 0.2089),
]


# ===========================================================================
# Three Schemes
# ===========================================================================

# Scheme A: Manual weights (mimicking typical Hybridsystemtest weights)
def scheme_a_weights(tracking_raw, final_raw, control_raw):
    """Standard weighted sum — raw = unbounded."""
    return 1.0 * tracking_raw + 15.0 * final_raw + 0.1 * control_raw


# Scheme B: log_transform + sigma_k with given k values
def scheme_b_compose(tracking_raw, final_raw, control_raw,
                     k_track=1.0, k_final=1.0, k_ctrl=1.0, k_outer=1.0):
    """log_transform → sigma_k(k_i) → sum → sigma_k(k_outer)."""
    t_track = log_transform(jnp.array(tracking_raw))
    t_final = log_transform(jnp.array(final_raw))
    t_ctrl  = log_transform(jnp.array(control_raw))

    total = (sigma_k(t_track, k=k_track) +
             sigma_k(t_final, k=k_final) +
             sigma_k(t_ctrl,  k=k_ctrl))
    return float(sigma_k(total, k=k_outer))


# Scheme C: T_alpha + fixed k (= compose_objective_semantic kernel)
def scheme_c_talpha(tracking_raw, final_raw, control_raw,
                    kg=OBJECTIVE_KNOTS_G, kT=OBJECTIVE_KNOTS_T, k_sat=OBJ_K_SAT):
    """T_alpha → sigma_k(k_sat) → sum → sigma_k(k_sat).

    This is exactly the kernel of compose_objective_semantic() in
    ObjectiveComposer — one table, one k, zero parameters.
    """
    t_track = T_alpha(jnp.array(tracking_raw), kg, kT)
    t_final = T_alpha(jnp.array(final_raw),  kg, kT)
    t_ctrl  = T_alpha(jnp.array(control_raw),  kg, kT)

    total = (sigma_k(t_track, k=k_sat) +
             sigma_k(t_final, k=k_sat) +
             sigma_k(t_ctrl,  k=k_sat))
    return float(sigma_k(total, k=k_sat))


# ===========================================================================
# Experiment 1: Static Sensitivity Scan
# ===========================================================================

def experiment_1():
    print("=" * 70)
    print("Experiment 1: Static Sensitivity Scan — per-term, per-magnitude")
    print("=" * 70)

    tracking_range = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    final_range    = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    control_range  = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]

    # --- Per-term: how does each transform behave in isolation? ---
    print("\n--- 1a: Per-term transform comparison ---")
    print(f"{'Term':<10} {'Raw':>10} {'log(T)':>10} {'T_alpha':>10} "
          f"{'σ(log, k=1)':>13} {'σ(T_α, k=0.2)':>13}")
    print("-" * 70)

    all_ranges = [
        ("track", tracking_range),
        ("final",  final_range),
        ("ctrl",   control_range),
    ]

    for label, rng in all_ranges:
        for raw in rng:
            lt = float(log_transform(jnp.array(raw)))
            ta = float(T_alpha(jnp.array(raw), OBJECTIVE_KNOTS_G, OBJECTIVE_KNOTS_T))
            sl = float(sigma_k(jnp.array(lt), k=1.0))
            st = float(sigma_k(jnp.array(ta), k=0.2))
            print(f"{label:<10} {raw:>10.4f} {lt:>10.4f} {ta:>10.4f} "
                  f"{sl:>13.4f} {st:>13.4f}")

    # --- 1b: Full composite — scan tracking, hold others at typical ---
    print("\n--- 1b: Composite output vs tracking (final=0.1, ctrl=1.0) ---")
    print(f"{'track':>10} {'A:weighted':>12} {'B:log+k=1':>12} "
          f"{'B:log+k=0.5':>12} {'B:log+k=0.2':>12} {'C:T_alpha':>12}")
    print("-" * 70)
    for track in tracking_range:
        a = scheme_a_weights(track, 0.1, 1.0)
        b1 = scheme_b_compose(track, 0.1, 1.0, 1.0, 1.0, 1.0)
        b05 = scheme_b_compose(track, 0.1, 1.0, 0.5, 0.5, 0.5)
        b02 = scheme_b_compose(track, 0.1, 1.0, 0.2, 0.2, 0.2)
        c  = scheme_c_talpha(track, 0.1, 1.0)
        print(f"{track:>10.4f} {a:>12.4f} {b1:>12.4f} {b05:>12.4f} {b02:>12.4f} {c:>12.4f}")

    # --- 1c: Full composite — scan all three simultaneously (corner cases) ---
    print("\n--- 1c: Corner cases — all three vary together ---")
    print(f"{'track':>10} {'final':>10} {'ctrl':>10} {'A':>12} {'B(k=1)':>10} {'B(k=0.2)':>10} {'C':>10}")
    print("-" * 70)
    scenarios = [
        (500.0, 50.0,  2.0,   "far"),
        (50.0,  5.0,   5.0,   "mid"),
        (0.1,   0.01,  10.0,  "near"),
        (0.01,  0.001, 0.1,   "arrived"),
        (1000,  100,    0.01,  "bad"),
    ]
    for track, final, ctrl, label in scenarios:
        a  = scheme_a_weights(track, final, ctrl)
        b1 = scheme_b_compose(track, final, ctrl, 1.0, 1.0, 1.0)
        b02 = scheme_b_compose(track, final, ctrl, 0.2, 0.2, 0.2)
        c  = scheme_c_talpha(track, final, ctrl)
        print(f"{track:>10.2f} {final:>10.4f} {ctrl:>10.2f} "
              f"{a:>12.2f} {b1:>10.4f} {b02:>10.4f} {c:>10.4f}  ({label})")

    # --- 1d: Sensitivity metric — min gap between adjacent magnitudes ---
    print("\n--- 1d: Sensitivity (min gap between adjacent raw values) ---")
    for label, rng in all_ranges:
        # Scheme B with k=1
        b_vals = [scheme_b_compose(r, 0.1, 1.0, 1.0, 1.0, 1.0) for r in rng]
        b_gaps = [abs(b_vals[i] - b_vals[i-1]) for i in range(1, len(b_vals))]
        # Scheme C
        c_vals = [scheme_c_talpha(r, 0.1, 1.0) for r in rng]
        c_gaps = [abs(c_vals[i] - c_vals[i-1]) for i in range(1, len(c_vals))]
        # Scheme A
        a_vals = [scheme_a_weights(r, 0.1, 1.0) for r in rng]
        a_gaps = [abs(a_vals[i] - a_vals[i-1]) for i in range(1, len(a_vals))]

        print(f"  {label:8s}: A min_gap={min(a_gaps):.4f}  "
              f"B(k=1) min_gap={min(b_gaps):.6f}  "
              f"C min_gap={min(c_gaps):.6f}  "
              f"C/A={min(c_gaps)/max(min(a_gaps), 1e-12):.2f}x")


# ===========================================================================
# Experiment 2: Simulated MPC Trajectory Sensitivity
# ===========================================================================

def experiment_2():
    print("\n" + "=" * 70)
    print("Experiment 2: MPC Trajectory — cost drift across optimization stages")
    print("=" * 70)

    # Simulate three MPC stages
    stages = {
        "Stage1 (far)":    {"track": 500.0, "final": 50.0,  "ctrl": 2.0},
        "Stage2 (mid)":    {"track": 50.0,  "final": 5.0,   "ctrl": 5.0},
        "Stage3 (near)":   {"track": 0.1,   "final": 0.01,  "ctrl": 10.0},
    }

    # For each stage, evaluate cost with small perturbations (±10%)
    # to measure local sensitivity
    print(f"\n{'Stage':<20} {'Scheme':<12} {'Cost':>10} {'Cost(+10%)':>12} "
          f"{'Δ':>10} {'log10(Δ)':>10} {'Sensitive?':>12}")
    print("-" * 80)

    for stage_name, vals in stages.items():
        track, final, ctrl = vals["track"], vals["final"], vals["ctrl"]
        dp = 1.10  # +10%

        # A: weighted
        a0 = scheme_a_weights(track, final, ctrl)
        a1 = scheme_a_weights(track * dp, final * dp, ctrl * dp)
        da = abs(a1 - a0)

        # B: log + dynamic-k (simulate elite updates)
        # Simulate 3 possible elite qualities → 3 different k sets
        k_sets = {
            "elite_good":  (0.15, 0.12, 0.9),   # elite near optimum → small raw → large k
            "elite_ok":    (0.5,  0.4,  0.5),   # elite mid-range
            "elite_bad":   (2.0,  1.5,  0.2),   # elite far → large raw → small k
        }

        print(f"\n  {stage_name}:")
        print(f"  {'A: weighted':>20s}  {a0:>10.4f} {a1:>12.4f} {da:>10.4f} {np.log10(max(da,1e-16)):>10.1f}")

        for ek, (kt, kf, kc) in k_sets.items():
            b0 = scheme_b_compose(track, final, ctrl, kt, kf, kc)
            b1 = scheme_b_compose(track * dp, final * dp, ctrl * dp, kt, kf, kc)
            db = abs(b1 - b0)
            sensitive = "YES" if db > 1e-4 else "FLAT"
            print(f"  {'B(' + ek + ')':>20s}  {b0:>10.4f} {b1:>12.4f} {db:>10.6f} {np.log10(max(db,1e-16)):>10.1f} {sensitive:>12s}")

        # C: T_alpha + fixed k
        c0 = scheme_c_talpha(track, final, ctrl)
        c1 = scheme_c_talpha(track * dp, final * dp, ctrl * dp)
        dc = abs(c1 - c0)
        print(f"  {'C: T_alpha':>20s}  {c0:>10.4f} {c1:>12.4f} {dc:>10.6f} {np.log10(max(dc,1e-16)):>10.1f} {'YES' if dc > 1e-4 else 'FLAT':>12s}")

    # --- 2b: k-value jitter from elite quality ---
    print("\n--- 2b: Dynamic k jitter — same raw value, different elite → different k ---")
    print(f"{'Term':<10} {'Raw':>10} {'k(elite_good)':>14} {'k(elite_ok)':>14} "
          f"{'k(elite_bad)':>14} {'k_range':>10} {'k_std/k_mean':>14}")
    print("-" * 80)

    from Constraintdealer.ObjectiveComposer import knee_to_k

    for label, raw in [("track", 500.0), ("final", 50.0), ("ctrl", 2.0)]:
        # Simulate: elite sees different raw values → different k
        raw_elites = {
            "good": raw * 0.1,   # elite is 10x better than current
            "ok":   raw * 1.0,   # elite is at current level
            "bad":  raw * 3.0,   # elite is 3x worse
        }
        ks = {}
        for ek, eraw in raw_elites.items():
            knee = max(eraw * 3.0, 0.1)  # multiplier=3, min_knee=0.1
            ks[ek] = float(knee_to_k(knee))

        k_vals = list(ks.values())
        k_range = max(k_vals) - min(k_vals)
        k_mean = np.mean(k_vals)
        k_std  = np.std(k_vals)
        print(f"{label:<10} {raw:>10.1f} {ks['good']:>14.4f} {ks['ok']:>14.4f} "
              f"{ks['bad']:>14.4f} {k_range:>10.4f} {k_std/max(k_mean,1e-12):>14.2%}")


# ===========================================================================
# Experiment 3: Compare with actual Hybridsystemtest results
# ===========================================================================

def experiment_3():
    print("\n" + "=" * 70)
    print("Experiment 3: Analysis of actual Hybridsystemtest composed-mode data")
    print("=" * 70)

    k_tracking_vals = [r[1] for r in COMPOSED_K_HISTORY]
    k_final_vals    = [r[2] for r in COMPOSED_K_HISTORY]
    k_ctrl_vals     = [r[3] for r in COMPOSED_K_HISTORY]

    print("\n--- 3a: k-value evolution across MPC steps ---")
    print(f"{'Step':>6} {'k_tracking':>12} {'k_final':>12} {'k_control':>12}")
    print("-" * 48)
    for step, kt, kf, kc in COMPOSED_K_HISTORY:
        print(f"{step:>6} {kt:>12.4f} {kf:>12.4f} {kc:>12.4f}")

    print(f"\n  k_tracking: range=[{min(k_tracking_vals):.4f}, {max(k_tracking_vals):.4f}] "
          f"ratio={max(k_tracking_vals)/max(min(k_tracking_vals),1e-12):.1f}x")
    print(f"  k_final:    range=[{min(k_final_vals):.4f}, {max(k_final_vals):.4f}] "
          f"ratio={max(k_final_vals)/max(min(k_final_vals),1e-12):.1f}x")
    print(f"  k_control:  range=[{min(k_ctrl_vals):.4f}, {max(k_ctrl_vals):.4f}] "
          f"ratio={max(k_ctrl_vals)/max(min(k_ctrl_vals),1e-12):.1f}x")

    # --- 3b: What if we used T_alpha instead? ---
    print("\n--- 3b: T_alpha output at key MPC stages ---")
    # Simulate tracking values from composed-mode run
    # (estimated from the robot's distance to target at key steps)
    tracking_estimates = {
        "Step 0 (far)":    1400.0,
        "Step 20 (mid)":   500.0,
        "Step 40 (near)":  0.19,
        "Step 60 (arrived)": 0.05,
    }
    print(f"{'Stage':<20} {'Raw_track':>12} {'log(raw)':>10} {'T_alpha(raw)':>14} "
          f"{'σ(log,k=1)':>12} {'σ(T_α,k=0.2)':>14}")
    print("-" * 75)
    for label, raw in tracking_estimates.items():
        lt = float(log_transform(jnp.array(raw)))
        ta = float(T_alpha(jnp.array(raw), OBJECTIVE_KNOTS_G, OBJECTIVE_KNOTS_T))
        sl = float(sigma_k(jnp.array(lt), k=1.0))
        st = float(sigma_k(jnp.array(ta), k=0.2))
        print(f"{label:<20} {raw:>12.1f} {lt:>10.4f} {ta:>14.4f} "
              f"{sl:>12.4f} {st:>14.4f}")

    # --- 3c: Composite cost sensitivity at the oscillation phase ---
    print("\n--- 3c: Sensitivity at oscillation phase (Step 40-70) ---")
    print("Small perturbation δtrack=±0.01 around tracking=0.05 (arrived state):")
    print()

    # Scheme B: with the k values from step 40
    kt, kf, kc = 2.2554, 10.4921, 0.2089
    b_center = scheme_b_compose(0.05, 0.01, 10.0, kt, kf, kc)
    b_plus   = scheme_b_compose(0.06, 0.01, 10.0, kt, kf, kc)
    b_minus  = scheme_b_compose(0.04, 0.01, 10.0, kt, kf, kc)
    db_plus  = abs(b_plus - b_center)
    db_minus = abs(b_minus - b_center)

    # Scheme C
    c_center = scheme_c_talpha(0.05, 0.01, 10.0)
    c_plus   = scheme_c_talpha(0.06, 0.01, 10.0)
    c_minus  = scheme_c_talpha(0.04, 0.01, 10.0)
    dc_plus  = abs(c_plus - c_center)
    dc_minus = abs(c_minus - c_center)

    print(f"  Scheme B (k_track={kt:.2f}): cost={b_center:.6f}  "
          f"Δ(+0.01)={db_plus:.2e}  Δ(-0.01)={db_minus:.2e}")
    print(f"  Scheme C (T_alpha, k=0.2): cost={c_center:.6f}  "
          f"Δ(+0.01)={dc_plus:.2e}  Δ(-0.01)={dc_minus:.2e}")
    print(f"  C/B sensitivity ratio: {dc_plus/max(db_plus,1e-16):.1f}x")

    # --- 3d: Summary table ---
    print("\n--- 3d: Summary — Scheme C vs Scheme B ---")
    print(f"  {'Metric':<40} {'B (dynamic k)':>16} {'C (T_alpha)':>16}")
    print("-" * 74)
    print(f"  {'Depends on elite quality?':<40} {'YES — k from elite':>16} {'NO — fixed table':>16}")
    print(f"  {'JAX recompilation risk?':<40} {'NO (k in ctx)':>16} {'NO (fixed k)':>16}")
    print(f"  {'Sensitivity 1e-6..1e6 range?':<40} {'depends on k':>16} {'YES — floor at T(0⁺)':>16}")
    print(f"  {'k jitter across steps?':<40} {f'{max(k_tracking_vals)/max(min(k_tracking_vals),1e-12):.0f}x range':>16} {'0 — constant':>16}")
    print(f"  {'Works at optimization start?':<40} {'NO — elite is noise':>16} {'YES — static':>16}")


# ===========================================================================
# Visualization
# ===========================================================================

def make_plots():
    """Generate comparison plots saved to ObjectiveTransform_test/."""
    out = Path("ObjectiveTransform_test")
    out.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Transform comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    raw_grid = np.logspace(-4, 6, 500)
    log_t = np.array([float(log_transform(jnp.array(r))) for r in raw_grid])
    ta_t  = np.array([float(T_alpha(jnp.array(r), OBJECTIVE_KNOTS_G, OBJECTIVE_KNOTS_T)) for r in raw_grid])

    for ax, title in zip(axes, ["T(x) vs |x|", "σ(T(x), k=1) for log", "σ(T(x), k=0.2) for T_alpha"]):
        pass  # filled below

    ax = axes[0]
    ax.plot(raw_grid, log_t, label="log_transform", linewidth=2)
    ax.plot(raw_grid, ta_t, label="T_alpha (obj_standard)", linewidth=2)
    ax.axhline(y=0.7, color='gray', ls='--', alpha=0.5, label="T_alpha floor=0.7")
    ax.set_xscale('log')
    ax.set_xlabel("|x| (raw value)")
    ax.set_ylabel("T(x)")
    ax.set_title("Transform Functions")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(raw_grid, [float(sigma_k(jnp.array(lt), k=1.0)) for lt in log_t],
            linewidth=2, color='tab:orange')
    ax.axhline(y=0.707, color='gray', ls='--', alpha=0.5, label="knee (0.707)")
    ax.axhspan(0.2, 0.8, alpha=0.1, color='green', label="sensitive zone")
    ax.set_xscale('log')
    ax.set_xlabel("|x| (raw value)")
    ax.set_ylabel("σ(T(x))")
    ax.set_title("Scheme B: σ(log(x), k=1.0)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(raw_grid, [float(sigma_k(jnp.array(t), k=0.2)) for t in ta_t],
            linewidth=2, color='tab:green')
    ax.axhline(y=0.707, color='gray', ls='--', alpha=0.5, label="knee (0.707)")
    ax.axhspan(0.2, 0.8, alpha=0.1, color='green', label="sensitive zone")
    ax.set_xscale('log')
    ax.set_xlabel("|x| (raw value)")
    ax.set_ylabel("σ(T(x))")
    ax.set_title("Scheme C: σ(T_alpha(x), k=0.2)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Transform + Saturation: log vs T_alpha", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out / "transform_comparison.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Composite cost across MPC stages ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Simulate tracking from 1000 → 0.001
    track_grid = np.logspace(-3, 3, 200)
    final_fixed, ctrl_fixed = 0.1, 1.0

    b_costs = np.array([scheme_b_compose(t, final_fixed, ctrl_fixed, 1.0, 1.0, 1.0)
                        for t in track_grid])
    c_costs = np.array([scheme_c_talpha(t, final_fixed, ctrl_fixed)
                        for t in track_grid])
    a_costs = np.array([scheme_a_weights(t, final_fixed, ctrl_fixed)
                        for t in track_grid])

    ax = axes[0]
    ax.plot(track_grid, a_costs, label="A: weighted", linewidth=2, alpha=0.7)
    ax.plot(track_grid, b_costs, label="B: log + σ(k=1)", linewidth=2)
    ax.plot(track_grid, c_costs, label="C: T_alpha + σ(k=0.2)", linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel("Tracking error (raw)")
    ax.set_ylabel("Composite cost")
    ax.set_title("Cost vs Tracking Error")
    ax.legend()
    ax.grid(alpha=0.3)

    # Sensitivity: derivative-like Δcost/Δtrack
    ax = axes[1]
    b_sens = np.abs(np.diff(b_costs) / np.diff(track_grid))
    c_sens = np.abs(np.diff(c_costs) / np.diff(track_grid))
    a_sens = np.abs(np.diff(a_costs) / np.diff(track_grid))
    ax.plot(track_grid[1:], a_sens, label="A: weighted", linewidth=2, alpha=0.7)
    ax.plot(track_grid[1:], b_sens, label="B: log + σ(k=1)", linewidth=2)
    ax.plot(track_grid[1:], c_sens, label="C: T_alpha + σ(k=0.2)", linewidth=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Tracking error (raw)")
    ax.set_ylabel("|Δcost/Δtrack| (sensitivity)")
    ax.set_title("Local Sensitivity")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Composite Cost: Three Schemes Across 6 Orders of Magnitude", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out / "composite_sensitivity.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: k-value jitter from actual composed run ---
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = [r[0] for r in COMPOSED_K_HISTORY]
    k_t_vals = [r[1] for r in COMPOSED_K_HISTORY]
    k_f_vals = [r[2] for r in COMPOSED_K_HISTORY]
    k_c_vals = [r[3] for r in COMPOSED_K_HISTORY]
    ax.plot(steps, k_t_vals, 'o-', label='k_tracking', linewidth=2)
    ax.plot(steps, k_f_vals, 's-', label='k_final', linewidth=2)
    ax.plot(steps, k_c_vals, '^-', label='k_control', linewidth=2)
    ax.axhline(y=0.2, color='gray', ls='--', alpha=0.5, label='C: fixed k=0.2')
    ax.set_xlabel("MPC Step")
    ax.set_ylabel("k value")
    ax.set_title("Scheme B: Dynamic k Jitter Across MPC Steps (from actual run)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "dynamic_k_jitter.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {out}/")
    print(f"  - transform_comparison.png")
    print(f"  - composite_sensitivity.png")
    print(f"  - dynamic_k_jitter.png")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
    make_plots()
    print("\n" + "=" * 70)
    print("All experiments complete.")
    print("=" * 70)
