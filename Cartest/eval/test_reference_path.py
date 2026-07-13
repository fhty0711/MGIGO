"""Unit tests for the PiecewiseCurvatureReference (clothoid centerline)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Cartest.core.reference_path import PiecewiseCurvatureReference

# R = 40 m: straight - clothoid - arc - clothoid - straight, ~150 deg total.
SEGMENTS = [
    (0.0, 40.0, 0.0, 0.0),
    (40.0, 65.0, 0.0, 1.0 / 40.0),
    (65.0, 145.0, 1.0 / 40.0, 1.0 / 40.0),
    (145.0, 170.0, 1.0 / 40.0, 0.0),
    (170.0, 250.0, 0.0, 0.0),
]
REF = PiecewiseCurvatureReference(SEGMENTS)


def test_evaluate_returns_four_tuple():
    out = REF.evaluate(jnp.array([10.0, 100.0]))
    assert len(out) == 4, "evaluate must return (x_r, y_r, theta_r, kappa_r)"


def test_curvature_by_section():
    x_r, y_r, theta_r, kappa_r = REF.evaluate(jnp.array([20.0, 100.0, 200.0]))
    assert_allclose(float(kappa_r[0]), 0.0, atol=1e-6)        # straight
    assert_allclose(float(kappa_r[1]), 1.0 / 40.0, atol=1e-4)  # arc
    assert_allclose(float(kappa_r[2]), 0.0, atol=1e-6)        # straight


def test_curvature_is_continuous_at_clothoid_ends():
    # Clothoid meets arc at s=65: kappa should be 1/40 on both sides.
    ka = float(REF.evaluate(jnp.array([64.99]))[3][0])
    kb = float(REF.evaluate(jnp.array([65.01]))[3][0])
    assert_allclose(ka, kb, atol=1e-3)


def test_extrapolation_advances_past_s_max():
    s_max = SEGMENTS[-1][1]
    x1, y1, _, _ = REF.evaluate(jnp.array([s_max]))
    x2, y2, _, _ = REF.evaluate(jnp.array([s_max + 50.0]))
    assert abs(float(x1[0]) - float(x2[0])) > 1e-3
    assert abs(float(y1[0]) - float(y2[0])) > 1e-3


def test_frenet_cartesian_round_trip():
    s_q = jnp.array([10.0, 50.0, 100.0, 180.0, 240.0])
    d_q = jnp.array([1.5, -2.0, 0.5, 3.0, -1.0])
    x, y = REF.frenet_to_cartesian(s_q, d_q)
    s_back, d_back = REF.cartesian_to_frenet(x, y)
    assert_allclose(np.asarray(s_back), np.asarray(s_q), atol=2e-2)  # ds_grid
    assert_allclose(np.asarray(d_back), np.asarray(d_q), atol=1e-3)


def test_origin_geometry():
    x_r, y_r, theta_r, kappa_r = REF.evaluate(jnp.array([0.0]))
    assert_allclose(float(x_r[0]), 0.0, atol=1e-6)
    assert_allclose(float(y_r[0]), 0.0, atol=1e-6)
    assert_allclose(float(theta_r[0]), 0.0, atol=1e-6)


if __name__ == "__main__":
    test_evaluate_returns_four_tuple(); print("✓ test_evaluate_returns_four_tuple")
    test_curvature_by_section(); print("✓ test_curvature_by_section")
    test_curvature_is_continuous_at_clothoid_ends(); print("✓ test_curvature_is_continuous_at_clothoid_ends")
    test_extrapolation_advances_past_s_max(); print("✓ test_extrapolation_advances_past_s_max")
    test_frenet_cartesian_round_trip(); print("✓ test_frenet_cartesian_round_trip")
    test_origin_geometry(); print("✓ test_origin_geometry")
    print("\nAll tests passed.")
