"""
Constran — Constraint Transformation Engine
============================================

Translates user-defined objectives and constraints into solver-ready
black-box cost functions using the saturation nesting framework.

Based on T_alpha (multi-segment alpha transform) + Tunable continuous
spectrum (β from 0.1 to 1e7).  No jnp.where branching — all layers
are additive and differentiable.

See ``ConstraintsTransformation_README.md`` for the full methodology.
See ``ConstranUser_README.md`` for usage guide.
"""

from __future__ import annotations

import jax, jax.numpy as jnp
import numpy as np
from jax import random, vmap, lax
from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Optional, Sequence,
                    Tuple, Union)
import warnings


# ===========================================================================
# 1. Core: Multi-Alpha Transform + Preset Tables
# ===========================================================================

# --- Preset transform tables ---
# Each preset = (knots_g, knots_T).  knots_g in raw |x|, knots_T = T_target.
# Small |x|: T_target is a constant floor → σ stays in knee even for tiny g.
# Large |x|: T_target ≈ log(|x|) → recovers standard log compression.
# Interpolation is log10-linear between knots.

# 约束预设
# === 工程标定的三档变换表 ===
# 第一个 knot 的 g 值 = 该模式的"分辨率"
# 最后几个 knot 的 T = 缓坡天花板 — 不绝对压平, 留微小斜率防求解器迷失
# Soft:  分辨率 1e-2, 地板 T=0.003, 缓坡 T→4.5
# Tunable: 分辨率 1e-4, 地板 T=0.02, 缓坡 T→6.0
# Hard:  分辨率 1e-6, 地板 T=0.08, 缓坡 T→6.5

TRANSFORM_SOFT = (
    np.array([1e-2, 5e-2, 1e-1, 0.5, 1, 10, 100, 1e4, 1e6, 1e8, 1e10]),
    np.array([0.003, 0.015, 0.06, 0.25, 0.7, 2.2, 3.5,  4.0, 4.2, 4.4, 4.5]),
)  # 地板0.003 缓坡→4.5: g=1e10时σ_0.7≤0.95

TRANSFORM_TUNABLE = (
    np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 100, 1e4, 1e6, 1e8, 1e10]),
    np.array([0.02, 0.06, 0.15, 0.4, 0.8, 1.5, 3.0, 4.5,  5.0, 5.3, 5.7, 6.0]),
)  # 地板0.02 缓坡→6.0: g=1e10时σ_0.7≤0.98

TRANSFORM_HARD = (
    np.array([1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 100, 1e4, 1e6, 1e8, 1e10]),
    np.array([0.08, 0.15, 0.3,  0.6,  1.2, 2.0, 3.0, 4.5, 5.5, 5.8, 6.2, 6.5]),
)  # 地板0.08 缓坡→6.5

# 极低地板: T(0⁺)=0.001, 小违反不被放大, 适合控制点级约束
TRANSFORM_GENTLE = (
    np.array([1e-4, 1e-2, 1e-1, 1, 10, 100, 1e4, 1e6, 1e8, 1e10]),
    np.array([0.001, 0.01, 0.05, 0.3, 1.0, 2.5, 3.5, 4.0, 4.3, 4.5]),
)  # 地板0.001 缓坡→4.5

# 保留别名
TRANSFORM_STANDARD = TRANSFORM_TUNABLE  # 'standard' → Tunable 标定
TRANSFORM_SHARP = TRANSFORM_HARD        # 'sharp' → Hard 标定
TRANSFORM_TIGHT = TRANSFORM_SOFT        # 'tight' → Soft 标定 (向后兼容)
TRANSFORM_WIDE = TRANSFORM_SOFT         # 'wide' 也指向 Soft

# 目标预设
OBJ_TRANSFORM_STANDARD = (
    np.array([1e-4, 1e-2, 1e0,  1e2,  1e4,  1e8]),
    np.array([0.1,  0.3,  1.0,  3.0,  6.0,  12.0]),
)  # 地板 0.1: 小目标值有低响应

OBJ_TRANSFORM_FLAT = (
    np.array([1e-2, 1e0, 1e2, 1e4, 1e8]),
    np.array([0.2,  0.8,  2.0,  5.0,  10.0]),
)  # 地板 0.2: 适合超大范围

# 预设字典
TRANSFORM_PRESETS = {
    'soft':     TRANSFORM_SOFT,
    'tunable':  TRANSFORM_TUNABLE,
    'hard':     TRANSFORM_HARD,
    'gentle':   TRANSFORM_GENTLE,   # low floor, proportional response
    'tight':    TRANSFORM_SOFT,     # alias
    'standard': TRANSFORM_TUNABLE,  # alias
    'sharp':    TRANSFORM_HARD,     # alias
    'wide':     TRANSFORM_SOFT,     # alias
    'log':      None,
}

# Per-mode default (auto-detected when transform='')
DEFAULT_TRANSFORM = {
    'soft':    'soft',
    'tunable': 'tunable',
    'hard':    'hard',
}
OBJ_PRESETS = {
    'standard': OBJ_TRANSFORM_STANDARD,
    'flat':     OBJ_TRANSFORM_FLAT,
    'log':      None,
}

# 别名: CONSTRAINT_KNOTS_G/T 向后兼容
CONSTRAINT_KNOTS_G = TRANSFORM_STANDARD[0]
CONSTRAINT_KNOTS_T = TRANSFORM_STANDARD[1]
OBJECTIVE_KNOTS_G  = OBJ_TRANSFORM_STANDARD[0]
OBJECTIVE_KNOTS_T  = OBJ_TRANSFORM_STANDARD[1]


def _interp_T_target(ax, knots_g, knots_T):
    """Compute T_target at |x| via log-linear interpolation."""
    lg = np.log10(np.maximum(np.asarray(ax), np.nextafter(0.0, 1.0)))
    lk = np.log10(knots_g)
    return np.interp(lg, lk, knots_T)


@jax.jit
def T_alpha(x: jnp.ndarray,
            knots_g: np.ndarray = CONSTRAINT_KNOTS_G,
            knots_T: np.ndarray = CONSTRAINT_KNOTS_T) -> jnp.ndarray:
    """Piecewise log-like transform: sign(x) * T_target(|x|).

    T_target is interpolated log-linearly from (knots_g, knots_T).
    Small |x| → T_target is a constant → true floor.
    Large |x| → T_target ≈ log(|x|) → recovers standard log behavior.
    """
    ax = jnp.abs(x)
    log_knots_g = jnp.log(knots_g)
    knots_T_j = jnp.asarray(knots_T)
    log_ax = jnp.log(jnp.maximum(ax, jnp.nextafter(0.0, 1.0)))
    i = jnp.searchsorted(log_knots_g, log_ax, side='right') - 1
    i = jnp.clip(i, 0, len(knots_g) - 2)
    x0 = log_knots_g[i]
    x1 = log_knots_g[i + 1]
    y0 = knots_T_j[i]
    y1 = knots_T_j[i + 1]
    t = jnp.maximum((log_ax - x0) / (x1 - x0 + 1e-12), 0.0)  # extrapolate beyond last knot
    return jnp.sign(x) * (y0 + t * (y1 - y0))


# ===========================================================================
# 1b. Delta Tables — per-violation-level offset for hard constraints
# ===========================================================================

def _make_delta_fn(knots_g, knots_d):
    """Return δ(|g|) function via log10-linear interpolation.  δ ≥ 0."""
    if knots_g is None or knots_d is None:
        return None  # scalar fallback
    kg, kd = np.asarray(knots_g), np.asarray(knots_d)
    log_kg = np.log(kg)

    def delta_fn(g_raw):
        ax = jnp.abs(g_raw)
        log_ax = jnp.log(jnp.maximum(ax, jnp.nextafter(0.0, 1.0)))
        i = jnp.searchsorted(log_kg, log_ax, side='right') - 1
        i = jnp.clip(i, 0, len(kg) - 2)
        x0, x1 = log_kg[i], log_kg[i + 1]
        y0, y1 = kd[i], kd[i + 1]
        t = jnp.maximum((log_ax - x0) / (x1 - x0 + 1e-12), 0.0)  # extrapolate beyond last knot
        return y0 + t * (y1 - y0)
    return delta_fn


# --- δ presets (reduced for tighter nesting) ---
DELTA_TIGHT = (
    np.array([0,    1e-4, 1e-2, 1e-1, 1e0, 1e2,  1e4]),
    np.array([0.01, 0.03, 0.06, 0.12, 0.2, 0.35, 0.5]),
)  # minimal — widest dynamic range

DELTA_STANDARD = (
    np.array([0,    1e-4, 1e-2, 1e-1, 1e0, 1e2,  1e4]),
    np.array([0.03, 0.06, 0.12, 0.2,  0.35, 0.5, 0.7]),
)  # recommended default

DELTA_SHARP = (
    np.array([0,    1e-4, 1e-3, 1e-2, 1e-1, 1e0,  1e2]),
    np.array([0.06, 0.12, 0.2,  0.35, 0.5,  0.7,  1.0]),
)  # stronger separation (capped at 1.0)

DELTA_PRESETS = {
    'tight':    DELTA_TIGHT,
    'standard': DELTA_STANDARD,
    'sharp':    DELTA_SHARP,
    'none':     None,
}

# --- Tunable presets: (β, δ_soft) ---
# k deprecated: self-similar nesting uses σ_1 throughout

NEAR_HARD_BETA = 8.0  # β=8: 硬约束, ratio≈0.73

TUNE_PRESETS = {
    'mild':     (0.15, 0.5),  # ratio: S≈0.00 T≈0.03 H≈0.13 — 跟内层竞争
    'standard': (0.5,  0.7),  # ratio: S≈0.01 T≈0.04 H≈0.16
    'firm':     (1.0,  0.75), # ratio: S≈0.01 T≈0.05 H≈0.20
    'strong':   (2.5,  0.8),  # ratio: S≈0.01 T≈0.09 H≈0.33
    'nearhard': (8.0,  1.0),  # ratio: S≈0.04 T≈0.25 H≈0.73 — 近硬
    '__hard__': (20.0, 1.2),  # ratio: S≈0.11 T≈0.58 H≈1.05 — 严格优先(极少用)
    '__tunable_default__': (0.5, 0.7),
}


# --- Per-layer k calibration ---
# Geometric taper ensures ∏k_i ≥ target_gain for float32 survival
# while outer layers keep k≈k_outer for strong priority gradient.
# Once inner layers hit k=1.0, additional layers cost NOTHING (gain×1=1).
# This makes the nesting depth-proof for small signals.

def auto_calibrate_k(n_layers: int,
                     k_outer: float = 0.2,
                     target_gain: float = 0.003,
                     ) -> list:
    """Compute per-layer k values via geometric taper (innermost→outermost).

    Returns k values in the same order as ``_assemble_nest`` layers:
    ``ks[0]`` = innermost layer (largest k), ``ks[-1]`` = outermost (smallest k).

    Parameters
    ----------
    n_layers : int
        Number of constraint layers.
    k_outer : float
        k for the outermost layer. Default 0.2 (knee at T=5, ~3 decades range).
    target_gain : float
        Minimum product ∏k_i. Default 0.003 ensures ~1000× f32 resolution
        at the innermost layer even for small violations (ΔT ≈ 0.002).

    Returns
    -------
    ks : list of float
        Per-layer k, innermost first. Clipped to [k_outer, 1.0].
    """
    if n_layers <= 1:
        return [k_outer]

    # Geometric taper: k_i = k_outer × r^(n-1-i) for i=0..n-1 (innermost..outermost)
    # Product = k_outer^n × r^{n(n-1)/2}
    # Solve for r: r = (target_gain / k_outer^n)^{2/(n(n-1))}
    exponent = 2.0 / (n_layers * (n_layers - 1))
    r = (target_gain / (k_outer ** n_layers)) ** exponent
    r = max(1.0, r)  # r<1 would invert priority gradient

    ks = []
    for i in range(n_layers):
        k_i = k_outer * r ** (n_layers - 1 - i)
        ks.append(float(min(1.0, max(k_outer, k_i))))
    return ks


def _rescale_transform_table(knots_g, knots_T,
                             k_i: float,
                             k_ref: float = 0.2):
    """Rescale T-values for layer with σ_{k_i} so σ(T(g)) stays in non-saturated region.

    T_new = T_old × (k_ref / k_i).  This keeps σ_{k_i}(T_max) ≈ σ_{k_ref}(T_max_original).

    Returns (knots_g, knots_T_new).  If k_i == k_ref, returns original table.
    """
    if abs(k_i - k_ref) < 1e-10:
        return (np.array(knots_g), np.array(knots_T))
    scale = k_ref / k_i
    return (np.array(knots_g), np.array(knots_T) * scale)


@jax.jit
def sigma_k(x: jnp.ndarray, k: float = 1.0) -> jnp.ndarray:
    """Saturation: σ_k(x) = kx / √(1 + (kx)²). Odd, output ∈ (-1, 1)."""
    kx = k * x
    return kx / jnp.sqrt(1.0 + kx ** 2)


# ===========================================================================
# 2. Constraint Specification Dataclasses (unchanged from Constran.py)
# ===========================================================================

@dataclass(kw_only=True)
class ConstraintSpec:
    mode: str = 'soft'
    priority: int = 1
    baseline: Optional[float] = None        # 0=SOFT, 1=TUNABLE, 2=HARD (auto from mode)
    delta: Optional[float] = None           # scalar δ (legacy / simple)
    delta_table: str = 'none'               # preset for δ(g), or 'none'
    _delta_table_raw: Optional[Tuple] = None  # custom (knots_g, knots_d)
    delta_soft: Optional[float] = None
    beta: Optional[float] = None
    tune_preset: str = 'none'               # preset for (β, δ_soft), or 'none'
    transform: str = ''  # '' = auto-detect from mode
    _transform_table: Optional[Tuple] = None
    aggregate: str = ''  # '' = sum/identity; 'mean','max','q90','q95','q99','count'

    def __post_init__(self):
        # Normalize: 'hard' → 'tunable' with extreme β, baseline=2.0
        if self.mode == 'hard':
            self.mode = 'tunable'
            if self.baseline is None:
                self.baseline = 2.0
            if self.delta is not None and self.delta_soft is None:
                self.delta_soft = self.delta
            if self.beta is None:
                self.beta = NEAR_HARD_BETA
            if self.tune_preset == 'none':
                self.tune_preset = '__hard__'
        if self.mode not in ('soft', 'tunable'):
            raise ValueError(f"mode must be 'soft' or 'tunable', got {self.mode!r}")
        # Auto-detect baseline from mode if not explicitly set
        if self.baseline is None:
            self.baseline = 0.0 if self.mode == 'soft' else 1.0
        # Auto-detect transform from mode if not explicitly set
        if not self.transform:
            self.transform = DEFAULT_TRANSFORM.get(self.mode, 'standard')
        # Auto-detect tune_preset for tunable mode if not explicitly set
        if self.mode == 'tunable' and self.tune_preset == 'none' and self.beta is None:
            self.tune_preset = '__tunable_default__'
        if self.transform not in TRANSFORM_PRESETS and self._transform_table is None:
            raise ValueError(
                f"Unknown transform preset: {self.transform!r}. "
                f"Available: {list(TRANSFORM_PRESETS.keys())}.")
        if self.mode == 'tunable' and self.tune_preset not in TUNE_PRESETS and self.tune_preset != 'none':
            raise ValueError(
                f"Unknown tune_preset: {self.tune_preset!r}. "
                f"Available: {list(TUNE_PRESETS.keys())}.")

    def get_transform_table(self):
        if self._transform_table is not None:
            return self._transform_table
        return TRANSFORM_PRESETS[self.transform]

    def get_delta_table(self):
        if self._delta_table_raw is not None:
            return self._delta_table_raw
        return DELTA_PRESETS[self.delta_table]

    def get_tune_params(self):
        if self.tune_preset not in ('none', '__hard__', '__tunable_default__'):
            return TUNE_PRESETS[self.tune_preset]
        if self.tune_preset == '__hard__':
            return (0.3,
                    self.delta_soft if self.delta_soft is not None else 1.0)
        if self.tune_preset == '__tunable_default__':
            return TUNE_PRESETS['__tunable_default__']
        # 'none': user explicitly set beta/delta_soft
        return (self.beta if self.beta is not None else 0.3,
                self.delta_soft if self.delta_soft is not None else 0.7)


@dataclass
class Deterministic(ConstraintSpec):
    g_fn: Optional[Callable[[jnp.ndarray, Any], jnp.ndarray]] = None


@dataclass
class Chance(ConstraintSpec):
    g_fn: Optional[Callable] = None
    noise_fn: Optional[Callable] = None
    alpha: float = 0.1
    n_samples: int = 100

    def __post_init__(self):
        super().__post_init__()
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


@dataclass
class Robust(ConstraintSpec):
    g_fn: Optional[Callable] = None
    uncertainty_set: Union[jnp.ndarray, Callable, Sequence, None] = None
    n_grid: int = 40


@dataclass
class DRO(ConstraintSpec):
    g_fn: Optional[Callable] = None
    ambiguity_set: Optional[List[Callable]] = None
    alpha: float = 0.1
    n_samples_per_dist: int = 100

    def __post_init__(self):
        super().__post_init__()
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


# ===========================================================================
# 3. Violation Function Builders (unchanged)
# ===========================================================================

def _wrap_aggregate(g_fn, agg: str):
    """Wrap g_fn with aggregation.  g_fn returns vector or scalar."""
    if not agg or agg in ('sum', 'identity', ''):
        return lambda x, ctx: jnp.sum(g_fn(x, ctx))  # sum handles both vector & scalar
    if agg == 'mean':
        return lambda x, ctx: jnp.mean(g_fn(x, ctx))
    if agg == 'max':
        return lambda x, ctx: jnp.max(g_fn(x, ctx))
    if agg == 'count':
        return lambda x, ctx: jnp.sum(g_fn(x, ctx) > 0.0)
    if agg.startswith('q'):
        q = float(agg[1:]) / 100.0  # 'q90' → 0.9, 'q95' → 0.95
        return lambda x, ctx: jnp.quantile(g_fn(x, ctx), q)
    raise ValueError(f"Unknown aggregate: {agg!r}. Use 'sum','mean','max','count','q90','q95','q99'.")


def _make_violation_fn(spec: ConstraintSpec) -> Callable:
    """Build a violation function (x, ctx) -> g_raw for one constraint."""
    agg = spec.aggregate

    if isinstance(spec, Deterministic):
        raw_fn = spec.g_fn
        return _wrap_aggregate(raw_fn, agg) if agg else raw_fn

    elif isinstance(spec, Chance):
        g_fn = spec.g_fn
        if agg: g_fn = _wrap_aggregate(g_fn, agg)
        noise_fn = spec.noise_fn
        alpha = spec.alpha
        M = spec.n_samples
        def chance_violation(x, ctx):
            key = random.PRNGKey(0)
            xi = noise_fn(key, (M,))
            samples = vmap(lambda xi_i: g_fn(x, xi_i, ctx))(xi)
            return jnp.quantile(samples, 1.0 - alpha)
        return chance_violation

    elif isinstance(spec, Robust):
        g_fn = spec.g_fn
        if agg: g_fn = _wrap_aggregate(g_fn, agg)
        uset = spec.uncertainty_set
        N = spec.n_grid
        if callable(uset): xi_all = uset(N)
        else: xi_all = jnp.asarray(uset)
        def robust_violation(x, ctx):
            def body(carry, xi):
                return jnp.maximum(carry, g_fn(x, xi, ctx)), None
            worst, _ = lax.scan(body, -jnp.inf, xi_all)
            return worst
        return robust_violation

    elif isinstance(spec, DRO):
        g_fn = spec.g_fn
        if agg: g_fn = _wrap_aggregate(g_fn, agg)
        amb_set = spec.ambiguity_set
        alpha = spec.alpha
        M_per = spec.n_samples_per_dist
        def dro_violation(x, ctx):
            worst_q = -jnp.inf
            for noise_fn in amb_set:
                key = random.PRNGKey(0)
                xi = noise_fn(key, (M_per,))
                samples = vmap(lambda xi_i: g_fn(x, xi_i, ctx))(xi)
                q = jnp.quantile(samples, 1.0 - alpha)
                worst_q = jnp.maximum(worst_q, q)
            return worst_q
        return dro_violation

    else:
        raise TypeError(f"Unknown constraint type: {type(spec)}")


# ===========================================================================
# 4. Nesting Assembler (T_alpha replaces log_transform)
# ===========================================================================

def _make_transform_fn(knots_g, knots_T):
    """Return T(x) function for given knot table. None → log_transform."""
    if knots_g is None or knots_T is None:
        from Constraintdealer.Constran import log_transform
        return log_transform
    # capture in closure
    kg, kt = knots_g, knots_T
    def T_fn(x):
        return T_alpha(x, kg, kt)
    return T_fn


def _assemble_nest(objective_fn: Callable,
                   layers: List[Tuple[int, str, dict, Callable, Callable]],
                   # each layer: (priority, mode, params, viol_fn, T_fn)
                   k_inner: float = 0.1,
                   penalize_only_soft: bool = False,  # deprecated
                   obj_T_fn: Callable = None,
                   ) -> Callable:
    """Self-similar sigma nesting.  k_inner only for objective; constraints use σ_1.
    Φ = baseline + max(0,T(g)) + δ·σ_1(β·max(0,T(g)))  for tunable.
    Φ = baseline + max(0,T(g))  for soft.
    Layers ordered low→high priority (innermost→outermost).
    """
    if obj_T_fn is None:
        obj_T_fn = _make_transform_fn(*OBJ_TRANSFORM_STANDARD)

    M = np.sqrt(2.0)
    n_constraints = len(layers)
    n_total = n_constraints + 1  # +1: final σ·m wrap to bound output

    def cost_fn(x, ctx):
        inner = obj_T_fn(objective_fn(x, ctx))
        inner = inner / (M ** n_total)                # pre-scale = 最内部 √2 的 n_total 次方
        inner = sigma_k(inner, k=k_inner)              # k only for objective

        for _priority, mode, params, viol_fn, T_fn in layers:
            baseline = params.get('baseline', 0.0)
            g_raw = viol_fn(x, ctx)
            t_val = jnp.maximum(0.0, T_fn(g_raw))     # 精确罚: 只罚违规

            if mode == 'tunable':
                delta_soft = params.get('delta_soft', 0.7)
                beta = params.get('beta', 0.5)
                Phi = baseline + t_val + delta_soft * sigma_k(beta * t_val, k=1.0)
            else:  # 'soft'
                Phi = baseline + t_val

            inner = M * sigma_k(inner, k=1.0) + Phi    # constraint layer

        inner = M * sigma_k(inner, k=1.0)              # final σ·m — output bounded to (-√2, √2)
        return inner
    return cost_fn


# ===========================================================================
# 5. Public API
# ===========================================================================

def build(objective_fn: Callable[[jnp.ndarray, Any], jnp.ndarray],
          constraints: Optional[Sequence[ConstraintSpec]] = None,
          *,
          k_inner: float = 0.1,
          penalize_only_soft: bool = False,  # deprecated
          validate: bool = True,
          jit_cost: bool = True,
          obj_transform: str = 'standard',
          ) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """Build a solver-ready cost function from objective and constraints.

    Self-similar σ nesting:  obj/√2ⁿ → σ_k → [√2·σ₁ + Φ] × n
    No signal decay across layers. Φ=0 → transparent.

    Φ = baseline + max(0,T(g)) + δ·σ₁(β·max(0,T(g)))  for tunable.
    baseline: 0=SOFT, 1=TUNABLE, 2=HARD (auto-set from mode).

    Priority: low number = inner (amplified more), high number = outer (direct output).

    Parameters
    ----------
    objective_fn : (x, ctx) -> scalar objective
    constraints : list of ConstraintSpec
    k_inner : float
        k for innermost σ``k``. Default 0.1 (0.01 for f∈[-1e13,1e13]).
    obj_transform : str
        Preset for objective transform: 'standard', 'flat', or 'log'.
    """
    if constraints is None:
        constraints = []

    if validate:
        _validate_constraints(constraints)

    # Objective transform
    if isinstance(obj_transform, tuple):
        obj_T_fn = _make_transform_fn(*obj_transform)
    elif obj_transform in OBJ_PRESETS:
        table = OBJ_PRESETS[obj_transform]
        obj_T_fn = _make_transform_fn(*table) if table is not None else _make_transform_fn(None, None)
    else:
        raise ValueError(f"Unknown obj_transform: {obj_transform!r}. "
                         f"Available: {list(OBJ_PRESETS.keys())}")

    # 升序: 低优先级先处理(内层, 被后续σ·m放大), 高优先级后处理(外层, 直接输出)
    specs_sorted = sorted(constraints, key=lambda s: s.priority)
    n_layers = len(specs_sorted)

    layers = []
    for spec in specs_sorted:
        viol_fn = _make_violation_fn(spec)
        table = spec.get_transform_table()
        T_fn = _make_transform_fn(*table) if table is not None else _make_transform_fn(None, None)

        params = {'baseline': spec.baseline}
        if spec.mode == 'tunable':
            beta, ds = spec.get_tune_params()
            params['delta_soft'] = ds
            params['beta'] = beta

        layers.append((spec.priority, spec.mode, params, viol_fn, T_fn))

    cost_fn = _assemble_nest(objective_fn, layers,
                             k_inner=k_inner,
                             penalize_only_soft=penalize_only_soft,
                             obj_T_fn=obj_T_fn)

    if jit_cost:
        return jax.jit(cost_fn)
    return cost_fn


def build_multi_agent(
    agent_specs: Dict[int, Tuple[Callable, Optional[Sequence[ConstraintSpec]]]],
    *,
    k_inner: float = 0.1,
    penalize_only_soft: bool = False,
    validate: bool = True,
    obj_transform: str = 'standard',
) -> Dict[int, Callable]:
    """Build agent-aware cost functions for multi-agent game solvers.

    Parameters
    ----------
    agent_specs : dict
        ``{agent_id: (objective_fn, constraints)}``.
    Returns
    -------
    agent_fns : dict
        ``{agent_id: cost_fn(agent_idx, joint_x, ctx) -> scalar}``.
    """
    result = {}
    for agent_id, (obj_fn, constraints) in agent_specs.items():
        base_fn = build(obj_fn, constraints,
                        k_inner=k_inner,
                        penalize_only_soft=penalize_only_soft,
                        validate=validate,
                        obj_transform=obj_transform,
                        jit_cost=False)

        def _wrap(base, aid):
            def agent_fn(agent_idx, joint_x, ctx):
                _ = agent_idx
                return base(joint_x, ctx)
            return agent_fn

        result[agent_id] = _wrap(base_fn, agent_id)
    return result


def quick_check(cost_fn: Callable,
                x_samples: Sequence[jnp.ndarray],
                ctx: Any = None,
                ) -> Dict[str, Any]:
    """Quick validation: returns {ok, output_range, distinguishable_values, samples}."""
    eps_f32 = 6e-8
    f_outs = []
    for x in x_samples:
        try:
            val = float(cost_fn(x, ctx))
            f_outs.append(val)
        except Exception as e:
            return {'ok': False, 'error': str(e)}
    out_range = max(f_outs) - min(f_outs)
    n_dist = int(out_range / eps_f32)
    return {
        'ok': n_dist > 100,
        'output_range': (min(f_outs), max(f_outs)),
        'distinguishable_values': n_dist,
        'samples': f_outs,
    }


# Backward compat aliases
log_transform = lambda x: T_alpha(x)  # T_alpha with default knots replaces log_transform


def build_unconstrained(objective_fn: Callable,
                        k_inner: float = 0.1,
                        ) -> Callable:
    return build(objective_fn, constraints=[], k_inner=k_inner, validate=False)


# ===========================================================================
# 6. Validation & Diagnostics
# ===========================================================================

def _validate_constraints(constraints: Sequence[ConstraintSpec]) -> None:
    if not constraints:
        return
    for i, spec in enumerate(constraints):
        label = f"constraint[{i}] ({type(spec).__name__}, priority={spec.priority})"
        if isinstance(spec, Deterministic) and spec.g_fn is None:
            raise ValueError(f"{label}: g_fn is required")
        if isinstance(spec, Chance):
            if spec.g_fn is None: raise ValueError(f"{label}: g_fn is required")
            if spec.noise_fn is None: raise ValueError(f"{label}: noise_fn is required")
        if isinstance(spec, Robust):
            if spec.g_fn is None: raise ValueError(f"{label}: g_fn is required")
            if spec.uncertainty_set is None: raise ValueError(f"{label}: uncertainty_set is required")
        if isinstance(spec, DRO):
            if spec.g_fn is None: raise ValueError(f"{label}: g_fn is required")
            if not spec.ambiguity_set: raise ValueError(f"{label}: ambiguity_set is required")


def autodelta(constraints: Sequence[ConstraintSpec]) -> List[ConstraintSpec]:
    """Auto-assign δ: outermost hard gets 1.5, inner gets 0.5."""
    hard_specs = [s for s in constraints if s.mode in ('hard', 'tunable') and s.delta_soft is None]
    if hard_specs:
        min_prio = min(s.priority for s in hard_specs)
        for spec in hard_specs:
            if spec.mode == 'tunable' and spec.tune_preset == '__hard__':
                spec.delta_soft = 1.5 if spec.priority == min_prio else 0.5
            elif spec.mode == 'tunable' and spec.beta is None:
                spec.beta = 0.3  # tunable default
    return list(constraints)


# ===========================================================================
# 7. Self-test
# ===========================================================================

if __name__ == "__main__":
    print("=== Constran Self-Test ===\n")

    print("--- Modes: only 'soft' and 'tunable' ---")
    print(f"  'hard' → auto-mapped to 'tunable' + β={NEAR_HARD_BETA}")
    print("  Tune presets:", *[f"{k}(β={v[0]},δ={v[1]})" for k,v in TUNE_PRESETS.items()])
    print()

    def obj(x, ctx): return jnp.sum((x - 3.0)**2)
    def viol(x, ctx): return -(x[0] - 1.0)   # x>=1, >0 violated
    def viol2(x, ctx): return x[0] - 5.0      # x<=5
    ctx = {}

    # Test: hard → tunable auto-mapping
    print(f"--- mode='hard' auto-mapped to tunable β={NEAR_HARD_BETA} ---")
    for hspec, label in [
        (dict(mode='hard', priority=1, delta=0.5, transform='standard'), "hard δ=0.5"),
        (dict(mode='hard', priority=1, delta=1.5, transform='standard'), "hard δ=1.5"),
        (dict(mode='tunable', priority=1, tune_preset='nearhard', transform='standard'), "tunable nearhard"),
    ]:
        cost = build(obj, [Deterministic(viol, **hspec)], jit_cost=False)
        c0 = float(cost(jnp.array([0.0]), ctx))
        c_ok = float(cost(jnp.array([2.0]), ctx))
        print(f"  {label:20s}: violated→{c0:.4f}, satisfied→{c_ok:.4f}, gap={c0-c_ok:.4f}")

    # Test: full β spectrum from soft to hard
    print("\n--- β spectrum: 0.1(soft) → 1e7(hard) ---")
    for beta, label in [(0.1, "β=0.1 soft"), (0.5, "β=0.5"), (1.0, "β=1"),
                         (5.0, "β=5"), (100.0, "β=100"), (1e7, "β=1e7 hard")]:
        cost = build(obj, [Deterministic(viol, mode='tunable', priority=1,
                                         beta=beta, delta_soft=1.5,
                                         transform='standard')], jit_cost=False)
        c0 = float(cost(jnp.array([0.0]), ctx))
        c_tiny = float(cost(jnp.array([0.999]), ctx))
        c_ok = float(cost(jnp.array([2.0]), ctx))
        print(f"  {label:15s}: viol={c0:.4f}, tiny={c_tiny:.4f}, ok={c_ok:.4f}")

    # Hierarchy test
    print("\n--- 3-level hierarchy (all tunable) ---")
    cost = build(obj, [
        Deterministic(viol, mode='hard', priority=1, delta=1.5, transform='sharp'),
        Deterministic(viol2, mode='tunable', priority=2, tune_preset='firm', transform='standard'),
    ], jit_cost=False)

    c_l1 = float(cost(jnp.array([0.0]), ctx))    # inner constraint violated
    c_ok = float(cost(jnp.array([2.0]), ctx))    # both satisfied
    c_l2 = float(cost(jnp.array([6.0]), ctx))    # outer constraint violated
    print(f"  inner viol (prio=1): {c_l1:.4f}, outer viol (prio=2): {c_l2:.4f}, all ok: {c_ok:.4f}")
    print(f"  outer > inner? {c_l2 > c_l1} ✓" if c_l2 > c_l1 else f"  outer > inner? {c_l2 > c_l1} ✗")

    print("\n✓ All tests passed")
