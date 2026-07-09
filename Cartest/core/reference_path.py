"""Reference path abstraction for Frenet ↔ Cartesian transforms.

A reference path is a smooth centerline parameterized by arc length s.
It provides:
  - evaluate(s) → (x_r, y_r, θ_r, κ_r)     path geometry
  - frenet_to_cartesian(s, d) → (x, y)      for obstacle checking
"""

from __future__ import annotations

import jax.numpy as jnp


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

    s = 0 at the top (cx, cy + R), increasing counter-clockwise.
    Positive d = outside the circle (standard Frenet left-normal convention).
    """

    def __init__(self, R: float, cx: float = 0.0, cy: float = 0.0):
        self.R = R
        self.cx = cx
        self.cy = cy

    def evaluate(self, s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        θ = s / self.R
        # s=0 at top (cx, cy+R), CCW:  x = R·sin(θ), y = R·cos(θ)
        # Tangent: (cos(θ), −sin(θ)) → heading = atan2(−sin, cos) = −θ (mod 2π)
        x_r = self.cx + self.R * jnp.sin(θ)
        y_r = self.cy + self.R * jnp.cos(θ)
        θ_r = -θ % (2 * jnp.pi)
        κ_r = jnp.full_like(s, 1.0 / self.R)
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
