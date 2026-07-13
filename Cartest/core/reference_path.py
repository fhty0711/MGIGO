"""Reference path abstraction for Frenet ↔ Cartesian transforms.

A reference path is a smooth centerline parameterized by arc length s.
It provides:
  - evaluate(s) → (x_r, y_r, θ_r, κ_r)     path geometry
  - frenet_to_cartesian(s, d) → (x, y)      for obstacle checking
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class ReferencePath:
    """Abstract reference path: arc-length-parameterized smooth centerline."""

    def evaluate(self, s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Path geometry at arc-length s.

        Returns:
            x_r:   Cartesian x [...,]
            y_r:   Cartesian y [...,]
            θ_r:   heading angle [...,]
            κ_r:   curvature [...,]
        """
        raise NotImplementedError

    def frenet_to_cartesian(self, s: jnp.ndarray, d: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Convert Frenet (s, d) to Cartesian (x, y).

        x = x_r(s) - d · sin(θ_r(s))
        y = y_r(s) + d · cos(θ_r(s))
        """
        x_r, y_r, θ_r, _ = self.evaluate(s)
        x = x_r - d * jnp.sin(θ_r)
        y = y_r + d * jnp.cos(θ_r)
        return x, y

    def cartesian_to_frenet(self, x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Convert Cartesian (x, y) to Frenet (s, d). Inverse of frenet_to_cartesian.

        For general paths this requires solving the foot-of-perpendicular
        condition via 1D Newton.  The base class raises NotImplementedError;
        subclasses (including StraightReference) provide the implementation.
        """
        raise NotImplementedError


class StraightReference(ReferencePath):
    """Straight road aligned with x-axis: x=s, y=0, θ=0, κ=0."""

    def evaluate(self, s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x_r = s
        y_r = jnp.zeros_like(s)
        θ_r = jnp.zeros_like(s)
        κ_r = jnp.zeros_like(s)
        return x_r, y_r, θ_r, κ_r

    def frenet_to_cartesian(self, s: jnp.ndarray, d: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Straight road: x = s, y = d (no sin/cos needed)
        return s, d

    def cartesian_to_frenet(self, x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Straight road: s = x, d = y
        return x, y


class CircularReference(ReferencePath):
    """Circular reference path parameterized by arc length.

    s = 0 at the top (cx, cy + R), increasing clockwise (top → right → bottom → left).
    Positive d = outside the circle (standard Frenet left-normal convention).

    Because the traversal is clockwise the heading decreases with s
    (θ_r = −s/R), so the *signed* curvature is κ_r = −1/R.
    """

    def __init__(self, R: float, cx: float = 0.0, cy: float = 0.0):
        self.R = R
        self.cx = cx
        self.cy = cy

    def evaluate(self, s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        θ = s / self.R
        # s=0 at top (cx, cy+R), CW:  x = R·sin(θ), y = R·cos(θ)
        # Tangent: (cos(θ), −sin(θ)) → heading = atan2(−sin, cos) = −θ (mod 2π)
        x_r = self.cx + self.R * jnp.sin(θ)
        y_r = self.cy + self.R * jnp.cos(θ)
        θ_r = -θ % (2 * jnp.pi)
        # Signed curvature = dθ_r/ds = −1/R (clockwise ⇒ heading decreases).
        # frenet_traj consumes κ_r as signed (jac = 1 − d·κ_r, a_n = κ_r·v_t·ṡ).
        κ_r = jnp.full_like(s, -1.0 / self.R)
        return x_r, y_r, θ_r, κ_r

    def cartesian_to_frenet(self, x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Closed-form inverse for circles.

        s = R · atan2(x−cx, y−cy) mod 2πR  (swap x,y for sin/cos parameterization)
        d = √((x−cx)² + (y−cy)²) − R        (positive d = outside circle, Frenet convention)
        """
        dx = x - self.cx
        dy = y - self.cy
        angle = jnp.arctan2(dx, dy)         # swap: sin↔cos in parameterization
        angle = angle % (2 * jnp.pi)
        s = self.R * angle
        dist = jnp.sqrt(dx ** 2 + dy ** 2)
        d = dist - self.R                      # positive d = outside circle (Frenet convention)
        return s, d


class PiecewiseCurvatureReference(ReferencePath):
    """Piecewise-linear curvature reference path with clothoid transitions.

    Each segment interpolates curvature linearly from kappa_start to kappa_end,
    so dkappa/ds is constant within a segment.  A typical road layout is
    straight - clothoid - circular arc - clothoid - straight.

    Construction pre-integrates theta_r, x_r, y_r on a uniform grid with numpy
    (trapezoidal rule); evaluate() looks them up via jnp.interp.  Beyond s_max
    the position and heading are extrapolated along the terminal tangent so
    Cartesian positions keep advancing instead of freezing at the clamp point
    (which would distort collision checking on long horizons).
    """

    def __init__(self, segments, ds_grid: float = 0.01):
        """Pre-integrate the path geometry.

        Args:
            segments: list of (s_start, s_end, kappa_start, kappa_end).
                      kappa is linearly interpolated within each segment and
                      is continuous across segment boundaries.
            ds_grid:  pre-integration grid resolution [m].
        """
        self.segments = segments
        s_max = segments[-1][1]
        s_grid = np.arange(0.0, s_max + ds_grid, ds_grid)

        kappa_arr = np.zeros_like(s_grid)
        for s_start, s_end, k_start, k_end in segments:
            mask = (s_grid >= s_start) & (s_grid <= s_end)
            seg_len = s_end - s_start
            t = (s_grid[mask] - s_start) / seg_len
            kappa_arr[mask] = k_start + (k_end - k_start) * t

        # theta_r = ∫ kappa ds  (trapezoidal cumulative integral)
        theta_arr = np.concatenate([
            [0.0],
            np.cumsum((kappa_arr[:-1] + kappa_arr[1:]) * 0.5 * ds_grid),
        ])
        # x_r = ∫ cos(theta) ds,  y_r = ∫ sin(theta) ds
        th_mid = (theta_arr[:-1] + theta_arr[1:]) * 0.5
        x_arr = np.concatenate([
            [0.0],
            np.cumsum(np.cos(th_mid) * ds_grid),
        ])
        y_arr = np.concatenate([
            [0.0],
            np.cumsum(np.sin(th_mid) * ds_grid),
        ])

        self._s = jnp.array(s_grid, dtype=jnp.float32)
        self._x = jnp.array(x_arr, dtype=jnp.float32)
        self._y = jnp.array(y_arr, dtype=jnp.float32)
        self._theta = jnp.array(theta_arr, dtype=jnp.float32)
        self._kappa = jnp.array(kappa_arr, dtype=jnp.float32)

        self._s_max = float(s_max)
        self._x_end = jnp.array(x_arr[-1], dtype=jnp.float32)
        self._y_end = jnp.array(y_arr[-1], dtype=jnp.float32)
        self._theta_end = jnp.array(theta_arr[-1], dtype=jnp.float32)

    def evaluate(self, s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Look up path geometry at arc-length s.

        Returns (x_r, y_r, theta_r, kappa_r).  For s > s_max the position and
        heading are extrapolated along the terminal tangent; the terminal
        segment is usually a straight, so the extrapolation is exact there.
        """
        x_r = jnp.interp(s, self._s, self._x)
        y_r = jnp.interp(s, self._s, self._y)
        theta_r = jnp.interp(s, self._s, self._theta)
        kappa_r = jnp.interp(s, self._s, self._kappa)

        overflow = s > self._s_max
        ds = s - self._s_max
        x_r = jnp.where(overflow, self._x_end + ds * jnp.cos(self._theta_end), x_r)
        y_r = jnp.where(overflow, self._y_end + ds * jnp.sin(self._theta_end), y_r)
        return x_r, y_r, theta_r, kappa_r

    def cartesian_to_frenet(self, x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Invert (x, y) -> (s, d) via nearest-grid lookup + normal projection.

        Finds the pre-integrated grid point closest to (x, y), uses its s as the
        foot of the perpendicular, then projects the residual onto the path
        normal n = (-sin theta_r, cos theta_r) to recover d.  Accuracy is
        bounded by ds_grid (default 1 cm), well below collision tolerances.
        """
        x = jnp.asarray(x, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.float32)
        shape = x.shape
        xf = x.ravel()
        yf = y.ravel()

        # [Q, G] pairwise squared distances to the pre-integrated grid
        dx = self._x[None, :] - xf[:, None]
        dy = self._y[None, :] - yf[:, None]
        idx = jnp.argmin(dx * dx + dy * dy, axis=1)

        s = self._s[idx]
        xr = self._x[idx]
        yr = self._y[idx]
        tr = self._theta[idx]
        # frenet_to_cartesian:  x-x_r = -d sin theta,  y-y_r = d cos theta
        d = -(xf - xr) * jnp.sin(tr) + (yf - yr) * jnp.cos(tr)
        return s.reshape(shape), d.reshape(shape)
