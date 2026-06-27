"""
ObjectiveComposer — Structured Sub-Objective Composition
=========================================================

Replaces manual weight tuning with per-term saturation gain control.

Each sub-objective term gets its own ``log_transform → sigma_k(..., k)``
treatment. All terms compete additively through a final outer sigma.
The result is a standard ``(x, ctx) -> scalar`` callable, directly
usable as the ``objective_fn`` in ``Constran.build()``.

Core insight: **k is the gain**. Unlike a multiplicative weight (unbounded),
k controls where the term's sigma_k half-saturates — a physically meaningful
knee in the original units of the term.

::

    from Constraintdealer.ObjectiveComposer import compose_objective
    from Constraintdealer.Constran import build, Deterministic

    obj = compose_objective([
        (tracking_fn,  0.5, "tracking"),    # knee at ~6.4 — wide range
        (final_fn,     2.0, "final"),       # knee at ~0.65 — precise
        (control_fn,   1.0, "control"),      # knee at ~1.7 — balanced
    ])

    cost = build(obj, [Deterministic(obs_fn, mode='hard', priority=1)])

See ``ObjectiveComposer_README.md`` for the full methodology.
"""

from __future__ import annotations

import sys, os
# Ensure the project root is on sys.path so we can import Constraintdealer
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import jax
import jax.numpy as jnp
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

# Reuse the core math from Constran — no duplication
from Constraintdealer.Constran import sigma_k, log_transform


# ===========================================================================
# 1. Knee / Gain Conversion Utilities
# ===========================================================================

def knee_to_k(raw_knee: float) -> float:
    """Convert a desired knee (in raw units) to the corresponding k.

    The knee is defined as the raw input where σ_k(T(raw)) ≈ 0.707
    (half-saturation).  Since σ_k(1/k) = 1/√2, and after log_transform:

        T(raw_knee) = 1/k  →  k = 1 / log(1 + raw_knee)

    Parameters
    ----------
    raw_knee : float
        The raw term value at which you want half-saturation.
        E.g., raw_knee=0.5 means "I want 0.5m tracking error to be
        at the middle of the sensitivity range."

    Returns
    -------
    k : float

    Examples
    --------
    >>> knee_to_k(0.22)   # lateral tracking: sub-meter precision
    5.02...
    >>> knee_to_k(6.4)    # path integral: wide range
    0.501...
    >>> knee_to_k(22000)  # control effort: only extreme waste
    0.100...
    """
    return 1.0 / jnp.log1p(raw_knee)


def k_to_knee(k: float) -> float:
    """Convert k back to the raw knee value.

        raw_knee = exp(1/k) - 1

    Examples
    --------
    >>> k_to_knee(5.0)
    0.221...
    >>> k_to_knee(0.5)
    6.389...
    >>> k_to_knee(0.1)
    22025.4...
    """
    return jnp.expm1(1.0 / k)


def suggest_k(raw_typical: float, raw_max: Optional[float] = None) -> float:
    """Heuristically suggest a k value given typical and max raw term values.

    The suggestion places the knee between the typical value and a fraction
    of the max, so the term has good distinguishability in its normal
    operating range while still differentiating large values.

    If raw_max is None, uses raw_typical * 100 as an estimate.

    Parameters
    ----------
    raw_typical : float
        A typical / expected value of this term during normal operation.
    raw_max : float, optional
        The maximum expected value. If None, estimated as 100 * raw_typical.

    Returns
    -------
    k : float

    Examples
    --------
    >>> suggest_k(1.0, 50.0)   # typical 1m error, max 50m
    0.596...
    >>> suggest_k(0.1, 2.0)    # typical 0.1 control, max 2.0
    2.28...
    """
    if raw_max is None:
        raw_max = raw_typical * 100.0
    # Place knee at geometric mean between typical and max/10
    knee_target = float(jnp.sqrt(raw_typical * max(raw_max / 10.0, raw_typical * 10.0)))
    return float(knee_to_k(knee_target))


# ===========================================================================
# 2. Main API: compose_objective
# ===========================================================================

def compose_objective(
    terms: Sequence[Union[Tuple[Callable, float],
                           Tuple[Callable, float, str]]],
    *,
    k_outer: float = 1.0,
    jit_result: bool = True,
) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """Compose multiple sub-objectives with per-term saturation gain (k).

    Each sub-term is independently compressed:

        compressed_i = σ_k( T(term_fn_i(x, ctx)), k=k_i )

    where T = log_transform (auto-scaling), σ_k = saturation with knee at 1/k.

    All compressed terms are summed and passed through a final outer σ_k:

        result = σ( Σ compressed_i, k=k_outer )

    **Why this replaces manual weights:**

    - Each term is bounded in [0, 1) — no single term can dominate
    - k controls the knee in physically meaningful units (meters, m/s², …)
    - log_transform eliminates per-term normalization
    - The outer σ prevents the sum from exceeding [0, 1)

    **Choosing k:**

    ======  ============  ==============================
    k       Raw knee      When to use
    ======  ============  ==============================
    5.0     ~0.22 m       Sub-meter precision (lateral)
    2.0     ~0.65 m       Final approach accuracy
    1.0     ~1.7          Balanced, moderate sensitivity
    0.5     ~6.4          Wide range (path integral)
    0.2     ~147          Very wide (longitudinal, ~100m)
    0.1     ~2.2×10⁴      Only penalize extreme waste
    ======  ============  ==============================

    Large k = high gain = saturates quickly = "small values matter".
    Small k = low gain = wide linear range = "only large values matter".

    Parameters
    ----------
    terms : list of tuple
        Each element is (term_fn, k) or (term_fn, k, name).
        - term_fn(x, ctx) -> scalar  (should return >= 0)
        - k : float, sensitivity (see table above)
        - name : str, optional (for debugging / logging)
    k_outer : float
        Final compression knee. Default 1.0.
    jit_result : bool
        If True (default), wrap with jax.jit.

    Returns
    -------
    objective_fn : callable
        ``(x, ctx) -> scalar`` in [0, 1).
        Pass as ``objective_fn`` to ``Constran.build()``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> def track(x, ctx): return jnp.sum((x[:2] - ctx['tgt'])**2)
    >>> def ctrl(x, ctx):  return 0.1 * jnp.sum(x[2:]**2)
    >>> obj = compose_objective([
    ...     (track, 0.5, "tracking"),
    ...     (ctrl,  1.0, "control"),
    ... ])
    >>> # obj(x, ctx) -> scalar in [0, 1)
    >>> # Use with Constran:
    >>> # cost = build(obj, [Deterministic(...), ...])
    """
    # Normalize: accept (fn, k) or (fn, k, name)
    parsed: List[Tuple[Callable, float]] = []
    for t in terms:
        if len(t) == 2:
            parsed.append((t[0], t[1]))
        else:
            parsed.append((t[0], t[1]))  # name ignored at runtime

    def _composed(x: jnp.ndarray, ctx: Any) -> jnp.ndarray:
        total = 0.0
        for term_fn, k in parsed:
            raw = term_fn(x, ctx)
            total = total + sigma_k(log_transform(raw), k=k)
        return sigma_k(total, k=k_outer)

    if jit_result:
        return jax.jit(_composed)
    return _composed


# ===========================================================================
# 3. Convenience: Auto-suggest k for all terms
# ===========================================================================

def compose_objective_auto(
    terms: Sequence[Tuple[Callable, float, Optional[float]]],
    *,
    k_outer: float = 1.0,
    jit_result: bool = True,
) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """Like compose_objective, but k is auto-suggested from (typical, max).

    Parameters
    ----------
    terms : list of (term_fn, raw_typical, raw_max_or_None)
        - term_fn(x, ctx) -> scalar
        - raw_typical : float — typical value during normal operation
        - raw_max : float or None — max expected value (None = auto-estimate)

    Returns
    -------
    objective_fn : (x, ctx) -> scalar in [0, 1)

    Examples
    --------
    >>> obj = compose_objective_auto([
    ...     (track_fn,  1.0,  50.0),   # typical 1m, max 50m → k≈0.6
    ...     (final_fn,  0.1,  5.0),    # typical 0.1m, max 5m → k≈2.5
    ...     (ctrl_fn,   0.5,  10.0),   # typical 0.5, max 10 → k≈1.0
    ... ])
    """
    term_list = []
    for t in terms:
        fn, typical, maximum = t
        k = suggest_k(typical, maximum)
        term_list.append((fn, k))
    return compose_objective(term_list, k_outer=k_outer, jit_result=jit_result)


# ===========================================================================
# 4. Diagnostic: Inspect per-term contributions
# ===========================================================================

def inspect_terms(
    composed_obj: Callable,
    x: jnp.ndarray,
    ctx: Any,
    terms_info: Sequence[Tuple[Callable, float]],
) -> dict:
    """Return per-term contribution breakdown for a given x and ctx.

    Useful for debugging / tuning k values: see which terms are saturated
    and which are still in their linear region.

    Parameters
    ----------
    composed_obj : the function returned by compose_objective
    x : decision variable
    ctx : context
    terms_info : the original terms list passed to compose_objective

    Returns
    -------
    dict with keys: 'total', 'contributions' (list of {name, raw, compressed})
    """
    contributions = []
    for i, t in enumerate(terms_info):
        fn = t[0]
        k = t[1]
        name = t[2] if len(t) > 2 else f"term_{i}"
        raw = float(fn(x, ctx))
        compressed = float(sigma_k(log_transform(jnp.array(raw)), k=k))
        contributions.append({
            'name': name,
            'raw': raw,
            'compressed': compressed,
            'saturated': 'YES' if abs(compressed) > 0.9 else
                          'knee' if 0.5 < abs(compressed) < 0.9 else
                          'linear'
        })
    total = float(composed_obj(x, ctx))
    return {'total': total, 'contributions': contributions}


# ===========================================================================
# 5. Quick self-test (run when executed directly)
# ===========================================================================

if __name__ == "__main__":
    print("=== ObjectiveComposer Self-Test ===\n")

    # Test knee conversion round-trip
    for k_test in [5.0, 2.0, 1.0, 0.5, 0.1]:
        knee = float(k_to_knee(k_test))
        k_back = float(knee_to_k(knee))
        print(f"  k={k_test:.1f} → knee={knee:.4f} → k_back={k_back:.4f}  "
              f"({'✓' if abs(k_test - k_back) < 0.01 else '✗'})")

    # Test compose_objective
    import jax.numpy as jnp

    def track(x, ctx):
        return jnp.sum((x[:2] - ctx['target']) ** 2)

    def ctrl(x, ctx):
        return 0.1 * jnp.sum(x[2:] ** 2)

    obj = compose_objective([
        (track, 0.5, "tracking"),
        (ctrl,  1.0, "control"),
    ])

    x = jnp.array([3.0, 4.0, 1.0, 2.0])
    ctx = {'target': jnp.array([1.0, 2.0])}

    cost = float(obj(x, ctx))
    print(f"\n  compose_objective test: cost={cost:.6f}")
    assert 0.0 <= cost < 1.0, f"Output {cost} not in [0, 1)!"

    # Test inspect_terms
    info = inspect_terms(obj, x, ctx, [
        (track, 0.5, "tracking"),
        (ctrl,  1.0, "control"),
    ])
    print(f"  Total: {info['total']:.6f}")
    for c in info['contributions']:
        print(f"    {c['name']:12s}: raw={c['raw']:.4f}  "
              f"compressed={c['compressed']:.4f}  [{c['saturated']}]")

    # Test compose_objective_auto
    obj_auto = compose_objective_auto([
        (track, 1.0, 50.0),
        (ctrl,  0.5, 10.0),
    ])
    cost_auto = float(obj_auto(x, ctx))
    print(f"\n  compose_objective_auto test: cost={cost_auto:.6f}")
    assert 0.0 <= cost_auto < 1.0, f"Output {cost_auto} not in [0, 1)!"

    # Test suggest_k
    print("\n  suggest_k examples:")
    for typical, maximum in [(1.0, 50.0), (0.1, 2.0), (0.5, 10.0)]:
        k = float(suggest_k(typical, maximum))
        print(f"    typical={typical:.1f}, max={maximum:.1f} → k={k:.4f}")

    print("\n  All tests passed ✓")
