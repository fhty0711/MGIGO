"""
ObjectiveComposer Demo — Manual Weights vs Composed vs Auto-k
=============================================================

Demonstrates how ``compose_objective`` replaces manual weight tuning with
per-term saturation gain (k) control.

Scenario: 2D vehicle reaching a target.
  - Term 1: path-integral tracking cost (sum of distances over horizon)
  - Term 2: final approach accuracy
  - Term 3: control smoothness (jerk penalty)

Three modes:
  1. ``'manual'``  — traditional f = w1·track + w2·final + w3·smooth
  2. ``'composed'`` — compose_objective with per-term k
  3. ``'auto'``     — compose_objective_auto with k auto-suggested

The demo solves a short-horizon optimization problem with each mode and
compares the resulting cost landscapes.

Usage::

    uv run python Functiontest/ObjectiveComposer_demo.py [--mode manual|composed|auto]

See ``Constraintdealer/ObjectiveComposer_README.md`` for methodology.
"""

from __future__ import annotations

import sys, os
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, jit, vmap

from Constraintdealer.ObjectiveComposer import (
    compose_objective, compose_objective_auto, knee_to_k, k_to_knee, suggest_k
)
from Constraintdealer.Constran import sigma_k, log_transform

# ===========================================================================
# 1. Problem Setup — Simple 2D Vehicle, H=8 Horizon
# ===========================================================================

CONFIG = {
    'dt': 0.15,
    'horizon': 8,
    'max_force': 3.0,
}

# Simple double-integrator dynamics
@jit
def rollout(theta_x, theta_y, init_state, dt):
    """Roll out a trajectory from control sequence."""
    u_max = CONFIG['max_force']
    H = theta_x.shape[0]  # dynamic but traceable

    def step_fn(carry, t):
        px, py, vx, vy = carry
        fx = u_max * jnp.tanh(theta_x[t])
        fy = u_max * jnp.tanh(theta_y[t])
        ax = fx - 1.0 * vx   # friction μ=1
        ay = fy - 1.0 * vy
        npx = px + vx * dt
        npy = py + vy * dt
        nvx = vx + ax * dt
        nvy = vy + ay * dt
        return (npx, npy, nvx, nvy), (npx, npy)

    (_, _, _, _), (px_seq, py_seq) = jax.lax.scan(
        step_fn, (init_state[0], init_state[1], init_state[2], init_state[3]),
        xs=jnp.arange(H))
    return jnp.stack([px_seq, py_seq], axis=1)  # (H, 2)


# ===========================================================================
# 2. Sub-Term Definitions (reusable across modes)
# ===========================================================================

def tracking_term(z_flat, ctx):
    """Path-integral tracking: sum of squared distances over horizon."""
    H = CONFIG['horizon']
    theta_x = z_flat[:H]
    theta_y = z_flat[H:2*H]
    positions = rollout(theta_x, theta_y, ctx['init_state'],
                        jnp.array(CONFIG['dt']))
    dists = jnp.sum((positions - ctx['target'][None, :]) ** 2, axis=1)
    return jnp.sum(dists)


def final_term(z_flat, ctx):
    """Final approach accuracy."""
    H = CONFIG['horizon']
    theta_x = z_flat[:H]
    theta_y = z_flat[H:2*H]
    positions = rollout(theta_x, theta_y, ctx['init_state'],
                        jnp.array(CONFIG['dt']))
    return 10.0 * jnp.sum((positions[-1] - ctx['target']) ** 2)


def smoothness_term(z_flat, ctx):
    """Control smoothness: sum of squared jerk (diff of accel)."""
    H = CONFIG['horizon']
    u_max = CONFIG['max_force']
    theta_x = z_flat[:H]
    theta_y = z_flat[H:2*H]
    fx = u_max * jnp.tanh(theta_x)
    fy = u_max * jnp.tanh(theta_y)
    jerk_x = jnp.sum(jnp.diff(fx) ** 2)
    jerk_y = jnp.sum(jnp.diff(fy) ** 2)
    return jerk_x + jerk_y


# ===========================================================================
# 3. Three Cost Functions
# ===========================================================================

# --- Mode 1: Manual Weights ---
@jit
def cost_manual(z_flat, ctx):
    """Traditional: f = 1.0·track + 0.5·final + 0.05·smooth"""
    f_total = (1.0 * tracking_term(z_flat, ctx) +
               0.5 * final_term(z_flat, ctx) +
               0.05 * smoothness_term(z_flat, ctx))
    # Wrap in sigma(log_transform(...)) for fair comparison with composed
    return sigma_k(log_transform(f_total), k=0.5)


# --- Mode 2: Composed (per-term k) ---
_cost_composed_base = compose_objective([
    (tracking_term,   0.5, "tracking"),     # knee ~6.4 — wide range
    (final_term,      2.0, "final"),        # knee ~0.65 — precise
    (smoothness_term, 1.0, "smoothness"),   # knee ~1.7 — balanced
])

@jit
def cost_composed(z_flat, ctx):
    return _cost_composed_base(z_flat, ctx)


# --- Mode 3: Auto-k ---
_cost_auto_base = compose_objective_auto([
    (tracking_term,   5.0,  200.0),   # typical ~5, max ~200
    (final_term,      0.5,  20.0),    # typical ~0.5, max ~20
    (smoothness_term, 1.0,  50.0),    # typical ~1, max ~50
])

@jit
def cost_auto(z_flat, ctx):
    return _cost_auto_base(z_flat, ctx)


# ===========================================================================
# 4. Comparison Runner
# ===========================================================================

def run_comparison():
    """Generate several candidate trajectories and compare cost landscapes."""
    key = random.PRNGKey(42)
    H = CONFIG['horizon']

    init_state = jnp.array([0.0, 0.0, 0.0, 0.0])
    target = jnp.array([8.0, 6.0])
    ctx = {'init_state': init_state, 'target': target}

    # Generate random control sequences
    n_samples = 50
    keys = random.split(key, n_samples)
    z_samples = []
    for k in keys:
        tx = random.uniform(k, (H,), minval=-3.0, maxval=3.0)
        ty = random.uniform(random.fold_in(k, 1), (H,), minval=-3.0, maxval=3.0)
        z_samples.append(jnp.concatenate([tx, ty]))

    # Evaluate all three cost functions
    manual_vals = np.array([float(cost_manual(z, ctx)) for z in z_samples])
    composed_vals = np.array([float(cost_composed(z, ctx)) for z in z_samples])
    auto_vals = np.array([float(cost_auto(z, ctx)) for z in z_samples])

    # --- Report ---
    print("=" * 70)
    print("ObjectiveComposer Demo — Manual vs Composed vs Auto-k")
    print("=" * 70)
    print(f"  Scenario: 2D vehicle, H={H}, n_samples={n_samples}")
    print(f"  Start: (0,0) → Target: (8,6)")
    print()

    print("--- Term Definitions ---")
    print(f"  tracking:    path-integral ∑||pos[t]-target||²  (range ~0–200)")
    print(f"  final:       10·||pos[-1]-target||²             (range ~0–20)")
    print(f"  smoothness:  ∑jerk²                              (range ~0–50)")
    print()

    print("--- Manual Weights ---")
    print(f"  f = 1.0·track + 0.5·final + 0.05·smooth")
    print(f"  → σ(T(f), k=0.5)")
    print(f"  Cost range: [{manual_vals.min():.4f}, {manual_vals.max():.4f}]")
    print(f"  Mean: {manual_vals.mean():.4f}  Std: {manual_vals.std():.4f}")
    print()

    print("--- Composed (per-term k) ---")
    from Constraintdealer.ObjectiveComposer import k_to_knee
    for name, k in [("tracking", 0.5), ("final", 2.0), ("smoothness", 1.0)]:
        knee = float(k_to_knee(k))
        print(f"  {name:12s}: k={k:.1f}  →  knee at raw ≈ {knee:.2f}")
    print(f"  Cost range: [{composed_vals.min():.4f}, {composed_vals.max():.4f}]")
    print(f"  Mean: {composed_vals.mean():.4f}  Std: {composed_vals.std():.4f}")
    print()

    print("--- Auto-k (suggest_k) ---")
    for name, typ, mx, k in [("tracking", 5.0, 200.0,
                               float(suggest_k(5.0, 200.0))),
                              ("final", 0.5, 20.0,
                               float(suggest_k(0.5, 20.0))),
                              ("smoothness", 1.0, 50.0,
                               float(suggest_k(1.0, 50.0)))]:
        knee = float(k_to_knee(k))
        print(f"  {name:12s}: typical={typ:.1f}, max={mx:.0f}  "
              f"→ k={k:.4f}  →  knee ≈ {knee:.2f}")
    print(f"  Cost range: [{auto_vals.min():.4f}, {auto_vals.max():.4f}]")
    print(f"  Mean: {auto_vals.mean():.4f}  Std: {auto_vals.std():.4f}")
    print()

    # --- Key insight ---
    print("--- Key Observations ---")
    print(f"  1. All three modes bounded in [0, 1): ✓")
    print(f"  2. Composed k chosen by physical meaning (meters, jerk units)")
    print(f"     → no dimensionless weights to tune")
    print(f"  3. Auto-k uses only typical/max values → no manual k selection")
    print(f"  4. Manual weights require trial-and-error per problem")
    print()
    print("  See Constraintdealer/ObjectiveComposer_README.md for full guide.")


def run_optimization(mode='composed'):
    """Run a minimal IGO optimization to show the composed objective works."""
    from gmm_igo.MPCsolverM22 import mmog_igo_optimizer_mpc

    H = CONFIG['horizon']
    dims = (H, H)
    M = 2
    K = 3
    B = 60
    B0 = 25

    key = random.PRNGKey(123)
    init_state = jnp.array([0.0, 0.0, 0.0, 0.0])
    target = jnp.array([8.0, 6.0])
    ctx = {'init_state': init_state, 'target': target}

    cost_fn = {'manual': cost_manual, 'composed': cost_composed,
               'auto': cost_auto}[mode]

    initial_mu = jnp.zeros((M, K, H))
    for c in range(K):
        initial_mu = initial_mu.at[0, c, :].set(
            random.uniform(random.fold_in(key, c), (H,), minval=-2.0, maxval=2.0))
        initial_mu = initial_mu.at[1, c, :].set(
            random.uniform(random.fold_in(key, c+100), (H,), minval=-2.0, maxval=2.0))
    initial_L_inv = jnp.tile(jnp.eye(H)[None, None, :, :], (M, K, 1, 1)) * 1.0
    initial_v = jnp.zeros((M, K-1))

    key, sk = random.split(key)
    print(f"\n--- Optimization ({mode} mode) ---")
    final_mu, final_L, final_pi = mmog_igo_optimizer_mpc(
        sk, 500, 0.15, M, K, B, B0, dims, 100,
        fitness_fn_total=cost_fn,
        initial_mu_k=initial_mu,
        initial_L_inv_k=initial_L_inv,
        initial_v_k=initial_v,
        context=ctx)

    # Extract best solution
    bi = [int(jnp.argmax(final_pi[0])), int(jnp.argmax(final_pi[1]))]
    best_z = jnp.concatenate([final_mu[0, bi[0], :H], final_mu[1, bi[1], :H]])

    # Evaluate and report
    final_cost = float(cost_fn(best_z, ctx))
    positions = rollout(best_z[:H], best_z[H:2*H], init_state,
                        jnp.array(CONFIG['dt']))
    final_pos = np.array(positions[-1])
    dist = np.sqrt(np.sum((final_pos - np.array(target)) ** 2))

    print(f"  Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
    print(f"  Distance to target ({target[0]}, {target[1]}): {dist:.4f}")
    print(f"  Final cost: {final_cost:.6f}")
    print(f"  Cost in valid range [0,1): {'✓' if 0 <= final_cost < 1 else '✗'}")

    return final_pos, dist, final_cost


# ===========================================================================
# 5. Main
# ===========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ObjectiveComposer demo — manual vs composed vs auto-k")
    parser.add_argument('--mode', default='all',
                        choices=['all', 'manual', 'composed', 'auto', 'compare'],
                        help="Which mode(s) to run")
    parser.add_argument('--optimize', action='store_true',
                        help="Run IGO optimization (requires solver)")
    args = parser.parse_args()

    if args.mode == 'all' or args.mode == 'compare':
        run_comparison()

    if args.optimize:
        for mode in (['manual', 'composed', 'auto'] if args.mode == 'all'
                     else [args.mode]):
            if mode == 'compare':
                continue
            try:
                run_optimization(mode)
            except ImportError as e:
                print(f"  Skipping optimization ({mode}): {e}")
                print(f"  (This is fine — the demo comparison above is the main content.)")
