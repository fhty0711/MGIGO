"""Frenet-frame quintic B-spline trajectory generator.

Same basis matrices as the Cartesian version (spline.py), but control
points live in (s, d) Frenet space instead of (x, y).

    s-channel:  s(t) = B·ctrl_s   ḃ = dB·ctrl_s   s̈ = d2B·ctrl_s  s⃛ = d3B·ctrl_s
    d-channel:  d(t) = B·ctrl_d   ḋ = dB·ctrl_d   d̈ = d2B·ctrl_d  d⃛ = d3B·ctrl_d

All kinematics are linear in control points — no arctan2 projection,
no curvature-division chain.  The reference path absorbs the curvature.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from pathlib import Path

from Cartest.reference_path import ReferencePath


class FrenetBSplineTrajectory:
    """Clamped quintic B-spline in Frenet (s, d) coordinates."""

    def __init__(self, basis_path: Path | str, ref_path: ReferencePath):
        data = np.load(str(basis_path))

        self.B = jnp.array(data["B"])
        self.dB = jnp.array(data["dB"])
        self.d2B = jnp.array(data["d2B"])
        self.d3B = jnp.array(data["d3B"])
        self.d4B = jnp.array(data["d4B"])
        self.greville = jnp.array(data["greville"])

        self.T = int(self.B.shape[0])
        self.n_ctrl = int(self.B.shape[1])
        self.dt = float(data["dt"])
        self.total_time = float(data["total_time"])
        self.degree = int(data["degree"])
        self.dt_knot = float(data["dt_knot"])

        self.n_free = self.n_ctrl - 3  # P0,P1,P2 clamped → 9 free
        self.ref_path = ref_path

    # ═══════════════════════════════════════════════════════════════════
    # Clamped endpoints — same formula as Cartesian, per channel
    # ═══════════════════════════════════════════════════════════════════

    def _clamped_3pts(self, x0, v0, a0):
        """P0,P1,P2 from (position, velocity, acceleration) for one channel."""
        p0 = x0
        p1 = p0 + (self.dt_knot / self.degree) * v0
        p2 = 3.0 * p1 - 2.0 * p0 + (self.dt_knot ** 2 / 10.0) * a0
        return p0, p1, p2

    # ═══════════════════════════════════════════════════════════════════
    # Trajectory evaluation
    # ═══════════════════════════════════════════════════════════════════

    def evaluate(self, ctrl_s_free, ctrl_d_free,
                 s0, s_dot0, s_ddot0,
                 d0, d_dot0, d_ddot0):
        """Evaluate Frenet trajectory from free control points + clamped state.

        Args:
            ctrl_s_free: [n_free]   free s-channel control points
            ctrl_d_free: [n_free]   free d-channel control points
            s0, s_dot0, s_ddot0:    initial s position, velocity, acceleration
            d0, d_dot0, d_ddot0:    initial d position, velocity, acceleration

        Returns:
            s, d:        [T]     position in Frenet
            s_dot, d_dot: [T]     velocity (Frenet time-derivative)
            s_ddot, d_ddot: [T]   acceleration
            s_dddot, d_dddot: [T] jerk
        """
        # Clamped endpoints for each channel
        p0_s, p1_s, p2_s = self._clamped_3pts(s0, s_dot0, s_ddot0)
        p0_d, p1_d, p2_d = self._clamped_3pts(d0, d_dot0, d_ddot0)

        ctrl_s = jnp.concatenate([
            jnp.array([p0_s]), jnp.array([p1_s]), jnp.array([p2_s]), ctrl_s_free
        ], axis=0)
        ctrl_d = jnp.concatenate([
            jnp.array([p0_d]), jnp.array([p1_d]), jnp.array([p2_d]), ctrl_d_free
        ], axis=0)

        return (
            jnp.dot(self.B,   ctrl_s),   # s
            jnp.dot(self.B,   ctrl_d),   # d
            jnp.dot(self.dB,  ctrl_s),   # ḃ
            jnp.dot(self.dB,  ctrl_d),   # ḋ
            jnp.dot(self.d2B, ctrl_s),   # s̈
            jnp.dot(self.d2B, ctrl_d),   # d̈
            jnp.dot(self.d3B, ctrl_s),   # s⃛
            jnp.dot(self.d3B, ctrl_d),   # d⃛
        )

    # ═══════════════════════════════════════════════════════════════════
    # Cartesian mapping (via reference path) — obstacle checking
    # ═══════════════════════════════════════════════════════════════════

    def to_cartesian(self, s, d):
        """Frenet (s,d) → Cartesian (x,y) through reference path."""
        return self.ref_path.frenet_to_cartesian(s, d)

    # ═══════════════════════════════════════════════════════════════════
    # Vehicle states [T, 8]:  x, y, v, ψ, a_long, a_lat, jerk_long, steer
    # ═══════════════════════════════════════════════════════════════════

    def to_vehicle_states(self, s, d, s_dot, d_dot,
                          s_ddot, d_ddot, s_dddot, d_dddot,
                          wheel_base=2.8):
        """Convert Frenet kinematics to vehicle-state vector.

        Frenet quantities are linear in control points.
        Nonlinearities (sqrt, arctan, curvature→steer) are well-conditioned
        and fine for black-box IGO optimization.
        """
        # Speed
        v = jnp.sqrt(s_dot ** 2 + d_dot ** 2)
        vs = v + 1e-6

        # Reference path geometry at s(t)
        _, _, θ_r, κ_r = self.ref_path.evaluate(s)

        # Vehicle heading: path heading + lateral-motion angle
        # ψ = θ_r(s) + arctan(ḋ / ḃ)
        psi = θ_r + jnp.arctan2(d_dot, s_dot)

        # Acceleration in Frenet frame (= vehicle-frame for small Δθ)
        # These are DIRECTLY d2B·ctrl — the key stability improvement
        a_long = s_ddot
        a_lat  = d_ddot

        # Jerk — directly d3B·ctrl
        jerk_long = s_dddot

        # Curvature of the actual vehicle trajectory
        # κ = (ḃ·κ_r + d/dt(arctan(ḋ/ḃ))) / v
        # d/dt(arctan(ḋ/ḃ)) = (ḃ·d̈ - s̈·ḋ) / v²
        # For straight reference (κ_r=0): κ = (ḃ·d̈ - s̈·ḋ) / v³
        dpsi_dt = s_dot * κ_r + (s_dot * d_ddot - s_ddot * d_dot) / (vs ** 2)
        curvature = dpsi_dt / vs
        steer = jnp.arctan(curvature * wheel_base)

        # Cartesian position (for obstacle checking / visualization)
        x, y = self.to_cartesian(s, d)

        return jnp.stack([
            x, y, v, psi, a_long, a_lat, jerk_long, steer,
        ], axis=-1)
