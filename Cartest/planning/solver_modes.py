"""Pre‑built solver modes for runtime scenario switching.

Each mode bundles:
  - cost function (with coupling template)
  - constraint limits (ACC_MAX, JERK_MAX)
  - IGO hyper‑parameters (T, dt, K, B, …)

Modes are pre‑compiled at initialisation so that switching at runtime
is instantaneous (no JIT recompilation, no solver rebuild).

Usage::

    modes = SolverModes(gen)
    result = modes[key].solve(ctx, mu_init)      # or
    result = modes.solve('emergency', ctx, mu)

The upper‑level planner can request a mode by name; the lower‑level MPC
executes without changing anything else.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from Cartest.planning.cost_transform import (
    make_objective_cross_order,
    template_coupling,
    build_context as _build_context,
)
from Cartest.planning.constraints import make_constraints as _make_constraints
from gmm_igo.solver_builder import build_solver


# ═══════════════════════════════════════════════════════════════════════
# Mode catalog
# ═══════════════════════════════════════════════════════════════════════

# Each entry: (cost_kwargs, constraint_kwargs, igo_kwargs)
# cost_kwargs come from template_coupling(name)
# constraint_kwargs are ACC_MAX, JERK_MAX (also from template)
# igo_kwargs are solver hyper‑params

MODE_CATALOG = {
    'conservative': dict(T=300, dt=0.15, B=64, B0=30),
    'standard':     dict(T=300, dt=0.20, B=64, B0=30),
    'active':       dict(T=300, dt=0.25, B=96, B0=40),
    'aggressive':   dict(T=400, dt=0.30, B=128, B0=50),
    'emergency':    dict(T=500, dt=0.35, B=128, B0=60),
}


class SolverMode:
    """One pre‑built solver + its bundled configuration."""

    def __init__(self, name: str, gen, lane_hw: float, safe_dist: float):
        self.name = name
        C_ba, C_ab, omega_z, omega_w, acc_max, jerk_max = template_coupling(name)
        self.acc_max = acc_max
        self.jerk_max = jerk_max

        igo = MODE_CATALOG[name]
        obj_fn = make_objective_cross_order(
            gen,
            omega_z=omega_z, omega_w=omega_w,
            C_ba=C_ba, C_ab=C_ab,
        )
        constraints = _make_constraints(gen, lane_hw, safe_dist,
                                        acc_max=acc_max, jerk_max=jerk_max)

        self._solver = build_solver(
            obj_fn,
            dims=(gen.n_free, gen.n_free),
            constraints=constraints,
            solver='m22',
            K=3, T_0=igo['T'],
            **igo,
            k_inner=1.0, obj_transform='standard',
        )

    def solve(self, key, ctx, mu_init):
        """Run one MPC step with this mode."""
        return self._solver(key, context=ctx, initial_mu=mu_init)

    def warmup(self, key, ctx, mu_init):
        """Trigger JIT compilation (call once before timing)."""
        _ = self._solver(key, context=ctx, initial_mu=mu_init)


class SolverModes:
    """Collection of pre‑built solver modes, keyed by name.

    Initialise once at startup; then switch modes at each MPC step
    by calling ``modes.solve(mode_name, ctx, mu)``.
    """

    def __init__(self, gen, lane_hw: float = 2.0, safe_dist: float = 0.1,
                 modes: list[str] | None = None):
        if modes is None:
            modes = list(MODE_CATALOG.keys())
        self._modes = {}
        self._default = modes[0]
        for name in modes:
            self._modes[name] = SolverMode(name, gen, lane_hw, safe_dist)

    def __getitem__(self, name: str) -> SolverMode:
        return self._modes[name]

    def solve(self, name: str, key, ctx, mu_init):
        """Run one MPC step with the named mode."""
        mode = self._modes.get(name, self._modes[self._default])
        return mode.solve(key, ctx, mu_init)

    def warmup_all(self, key, ctx, mu_init):
        """Pre‑compile all modes (call once at startup)."""
        for name, mode in self._modes.items():
            mode.warmup(key, ctx, mu_init)

    @property
    def names(self) -> list[str]:
        return list(self._modes.keys())
