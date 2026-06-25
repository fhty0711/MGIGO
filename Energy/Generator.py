"""
Generator Models & Parameters for Unit Commitment Benchmarks
=============================================================

Provides:
  - GeneratorSet: container for generator parameters + encoding/decoding
  - make_5gen_system(): small 5-generator test case
  - make_50gen_system(): realistic 50-generator system (baseload / mid-merit / peaker mix)
  - LOAD_5GEN, LOAD_50GEN: 24h load curves

Usage:
  from Energy.Generator import make_5gen_system

  gens = make_5gen_system()
  p = gens.decode_power(x)       # x ∈ ℝ^N → p ∈ [0, p_max]^N
  on = gens.on_mask(p)           # which generators are "on"
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorSet:
    """Immutable container for a set of generators.

    All fields are JAX arrays of shape (N_GEN,).
    """

    N: int                # number of generators
    p_min: jnp.ndarray    # minimum output (MW) — below this = inefficient
    p_max: jnp.ndarray    # maximum output (MW) — above this = violation
    a: jnp.ndarray        # quadratic fuel cost coeff ($/MW²/h)
    b: jnp.ndarray        # linear fuel cost coeff ($/MWh)
    c: jnp.ndarray        # no-load cost ($/h, only when on)
    ramp: jnp.ndarray     # max ramp rate (MW/h)
    startup: jnp.ndarray  # startup cost ($, charged on off→on transition)
    alpha: float = 5.0    # sigmoid sharpness: α↑ → sharper on/off transition

    @property
    def total_capacity(self) -> float:
        return float(jnp.sum(self.p_max))

    # ---- Encoding / Decoding (JIT'd by outer cost function) ----

    def decode_power(self, x: jnp.ndarray) -> jnp.ndarray:
        """Absolute encoding: x_i → p_i = p_max_i · σ(α·x_i).

        Clean on/off via sharp sigmoid. Best for single-step / no-ramp problems.
        """
        return self.p_max * jax.nn.sigmoid(self.alpha * x)

    def decode_incremental(self, x: jnp.ndarray,
                           prev_p: jnp.ndarray) -> jnp.ndarray:
        """Incremental encoding with startup jump.

        Running generators: Δp_i = ramp_i · tanh(x_i) ∈ [-ramp, +ramp]
        Off generators:    can jump to p_min in one step (startup is discrete)

        p_i = clip(prev_p + ramp·tanh(x) + startup_boost, 0, P_MAX)
        where startup_boost = max(0, p_min·σ(α·x) - prev_p) when off.
        """
        delta = self.ramp * jnp.tanh(x)
        p_raw = prev_p + delta

        # Startup boost: if off and x positive, jump to at least p_min
        off = (prev_p < self.p_min * 0.5).astype(jnp.float32)
        startup_target = self.p_min * jax.nn.sigmoid(self.alpha * x)
        startup_boost = off * jnp.maximum(0.0, startup_target - p_raw)

        return jnp.clip(p_raw + startup_boost, 0.0, self.p_max)

    def on_mask(self, p: jnp.ndarray) -> jnp.ndarray:
        """Generator is "on" if output exceeds half of p_min."""
        return (p > self.p_min * 0.5).astype(jnp.float32)

    # ---- Economic calculations ----

    def fuel_cost(self, p: jnp.ndarray, on: jnp.ndarray) -> jnp.ndarray:
        """Fuel cost: a·p² + b·p + c (c only when on)."""
        return jnp.sum(self.a * p**2 + self.b * p + self.c * on)

    def startup_cost(self, on: jnp.ndarray,
                     prev_on: jnp.ndarray) -> jnp.ndarray:
        """Startup cost: charged only on off→on transition."""
        return jnp.sum(self.startup * on * (1.0 - prev_on))

    def operating_cost(self, p: jnp.ndarray, on: jnp.ndarray,
                       prev_on: jnp.ndarray) -> jnp.ndarray:
        """Total operating cost = fuel + startup."""
        return self.fuel_cost(p, on) + self.startup_cost(on, prev_on)


# ======================================================================
# Factory functions
# ======================================================================

def make_5gen_system() -> GeneratorSet:
    """Small 5-generator test case for quick validation."""
    params = jnp.array([
        # p_min, p_max,  a,      b,    c,   ramp, startup
        [10.0,  50.0,  0.002,  12.0,  80.0, 20.0, 200.0],   # Gen 0: small coal
        [20.0,  80.0,  0.0015, 10.0, 120.0, 30.0, 350.0],   # Gen 1: medium coal
        [5.0,   30.0,  0.003,  15.0,  50.0, 15.0,  80.0],   # Gen 2: gas peaker
        [30.0, 100.0,  0.001,   8.0, 200.0, 25.0, 500.0],   # Gen 3: large baseload
        [8.0,   40.0,  0.0025, 13.0,  60.0, 18.0, 150.0],   # Gen 4: medium gas
    ])
    return GeneratorSet(
        N=5,
        p_min=params[:, 0], p_max=params[:, 1],
        a=params[:, 2], b=params[:, 3], c=params[:, 4],
        ramp=params[:, 5], startup=params[:, 6],
        alpha=5.0,
    )


def make_50gen_system(seed: int = 42) -> GeneratorSet:
    """Realistic 50-generator system with baseload / mid-merit / peaker mix.

    Distribution:
      - 15 baseload (large coal/nuclear): 200-500 MW, cheap, slow ramp, high startup
      - 20 mid-merit (CC gas): 80-200 MW, medium cost, medium ramp
      - 15 peakers (gas turbine): 20-80 MW, expensive, fast ramp, low startup
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Baseload
    for _ in range(15):
        pmax = rng.uniform(200, 500)
        pmin = pmax * rng.uniform(0.4, 0.6)
        a = rng.uniform(0.0003, 0.0008) / pmax
        b = rng.uniform(5, 12)
        c = b * pmin * rng.uniform(0.5, 1.0)
        ramp = pmax * rng.uniform(0.02, 0.05)
        startup = c * rng.uniform(1.5, 3.0)
        rows.append([pmin, pmax, a, b, c, ramp, startup])

    # Mid-merit
    for _ in range(20):
        pmax = rng.uniform(80, 200)
        pmin = pmax * rng.uniform(0.35, 0.55)
        a = rng.uniform(0.0005, 0.0015) / pmax
        b = rng.uniform(10, 25)
        c = b * pmin * rng.uniform(0.6, 1.2)
        ramp = pmax * rng.uniform(0.04, 0.10)
        startup = c * rng.uniform(1.0, 2.0)
        rows.append([pmin, pmax, a, b, c, ramp, startup])

    # Peakers
    for _ in range(15):
        pmax = rng.uniform(20, 80)
        pmin = pmax * rng.uniform(0.3, 0.5)
        a = rng.uniform(0.002, 0.006) / pmax
        b = rng.uniform(20, 50)
        c = b * pmin * rng.uniform(0.3, 0.7)
        ramp = pmax * rng.uniform(0.10, 0.30)
        startup = c * rng.uniform(0.3, 1.0)
        rows.append([pmin, pmax, a, b, c, ramp, startup])

    arr = jnp.array(rows)
    return GeneratorSet(
        N=50,
        p_min=arr[:, 0], p_max=arr[:, 1],
        a=arr[:, 2], b=arr[:, 3], c=arr[:, 4],
        ramp=arr[:, 5], startup=arr[:, 6],
        alpha=5.0,
    )


# ======================================================================
# Load curves (24 hours)
# ======================================================================

# 5-gen: peak ~200 MW
LOAD_5GEN = jnp.array([
    120, 110, 105, 100,  98, 105, 130, 155, 170, 180, 185, 190,
    195, 200, 198, 190, 180, 175, 185, 200, 195, 180, 160, 140,
], dtype=jnp.float32)

# 50-gen: scaled ~10×, peak ~2000 MW
LOAD_50GEN = jnp.array([
    1200, 1100, 1050, 1000,  980, 1050, 1300, 1550, 1700, 1800, 1850, 1900,
    1950, 2000, 1980, 1900, 1800, 1750, 1850, 2000, 1950, 1800, 1600, 1400,
], dtype=jnp.float32)
