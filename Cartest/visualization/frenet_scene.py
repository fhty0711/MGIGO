"""Scene adapter for Frenet MPC visualization.

This module owns coordinate-system decisions. Cartest reports store executed
history as Frenet ``s/d`` samples while planned paths are already Cartesian;
renderers should not need to know that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from Cartest.core.reference_path import StraightReference


@dataclass(frozen=True)
class CircleObstacle:
    center: tuple[float, float]
    radius: float
    safe_radius: float


@dataclass(frozen=True)
class FrenetScene:
    history_xy: np.ndarray
    plan_xy: np.ndarray
    road_lines: tuple[np.ndarray, ...]
    obstacles: tuple[CircleObstacle, ...]
    ego_center: tuple[float, float] | None
    ego_heading: float
    xlim: tuple[float, float]
    ylim: tuple[float, float]


def road_bounds(scenario: Mapping) -> tuple[float, float]:
    """Return lower/upper lateral Frenet road bounds for a scenario."""
    road = scenario.get("road", {})
    if "lane_bounds_d" in road:
        low, high = road["lane_bounds_d"]
        return float(low), float(high)
    lane_hw = float(road.get("lane_hw", 4.0))
    return -lane_hw, lane_hw


def default_straight_scenario(lane_hw: float = 4.0) -> dict:
    """Return a minimal scenario for legacy plotting calls."""
    return {
        "ref_path": StraightReference(),
        "road": {"lane_hw": float(lane_hw)},
    }


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def frenet_to_cartesian(ref_path, s_values, d_values) -> np.ndarray:
    """Convert Frenet arrays to an ``[N, 2]`` Cartesian array."""
    s = _as_float_array(s_values)
    d = _as_float_array(d_values)
    x, y = ref_path.frenet_to_cartesian(s, d)
    return np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])


def cartesian_to_frenet(ref_path, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert an ``[N, 2]`` Cartesian array to Frenet arrays."""
    arr = np.asarray(xy, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    s, d = ref_path.cartesian_to_frenet(arr[:, 0], arr[:, 1])
    return np.asarray(s, dtype=float), np.asarray(d, dtype=float)


def _heading_from_xy(xy: np.ndarray) -> float:
    arr = np.asarray(xy, dtype=float)
    if len(arr) < 2:
        return 0.0
    delta = arr[-1] - arr[-2]
    if float(np.linalg.norm(delta)) < 1e-9:
        return 0.0
    return math.atan2(float(delta[1]), float(delta[0]))


def _obstacle_s_values(ref_path, obstacles: Sequence[Mapping]) -> np.ndarray:
    if not obstacles:
        return np.zeros(0, dtype=float)
    xy = np.asarray([[float(o["x"]), float(o["y"])] for o in obstacles], dtype=float)
    s, _ = cartesian_to_frenet(ref_path, xy)
    return s


def _sample_road_lines(
    ref_path,
    scenario: Mapping,
    s_values: Sequence[float],
    *,
    samples: int = 160,
) -> tuple[np.ndarray, ...]:
    low, high = road_bounds(scenario)
    s_arr = np.asarray(s_values, dtype=float)
    if s_arr.size == 0:
        s_arr = np.asarray([0.0, 30.0], dtype=float)
    s_min = float(np.nanmin(s_arr)) - 12.0
    s_max = float(np.nanmax(s_arr)) + 18.0
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
        s_min, s_max = 0.0, 30.0
    s_grid = np.linspace(max(0.0, s_min), max(1.0, s_max), samples)
    d_values = [low, high]
    if low < 0.0 < high:
        d_values.append(0.0)
    return tuple(
        frenet_to_cartesian(ref_path, s_grid, np.full_like(s_grid, d))
        for d in d_values
    )


def _view_limits(
    history_xy: np.ndarray,
    plan_xy: np.ndarray,
    road_lines: Sequence[np.ndarray],
    obstacles: Sequence[CircleObstacle],
) -> tuple[tuple[float, float], tuple[float, float]]:
    arrays = [history_xy, plan_xy, *road_lines]
    if obstacles:
        obstacle_xy = np.asarray([obs.center for obs in obstacles], dtype=float)
        arrays.append(obstacle_xy)
    points = np.vstack([arr for arr in arrays if np.asarray(arr).size])
    if points.size == 0:
        return (-10.0, 30.0), (-8.0, 8.0)
    x_min = float(np.min(points[:, 0])) - 8.0
    x_max = float(np.max(points[:, 0])) + 8.0
    y_min = float(np.min(points[:, 1])) - 5.0
    y_max = float(np.max(points[:, 1])) + 5.0
    if x_max - x_min < 20.0:
        pad = 0.5 * (20.0 - (x_max - x_min))
        x_min -= pad
        x_max += pad
    if y_max - y_min < 12.0:
        pad = 0.5 * (12.0 - (y_max - y_min))
        y_min -= pad
        y_max += pad
    return (x_min, x_max), (y_min, y_max)


def build_frenet_scene(
    report,
    *,
    scenario: Mapping,
    obstacles: Sequence[Mapping] = (),
    obs_safe_dist: float = 0.0,
) -> FrenetScene:
    """Build a renderer-ready Cartesian scene from a Cartest StepReport."""
    ref_path = scenario["ref_path"]
    history_s = _as_float_array(report.hx)
    history_d = _as_float_array(report.hy)
    history_xy = frenet_to_cartesian(ref_path, history_s, history_d)
    plan_xy = np.column_stack([
        _as_float_array(report.px),
        _as_float_array(report.py),
    ])

    plan_s, _ = cartesian_to_frenet(ref_path, plan_xy)
    obstacle_s = _obstacle_s_values(ref_path, obstacles)
    road_s = np.concatenate([history_s, plan_s, obstacle_s])
    road_lines = _sample_road_lines(ref_path, scenario, road_s)

    circle_obstacles = tuple(
        CircleObstacle(
            center=(float(obs["x"]), float(obs["y"])),
            radius=float(obs["r"]),
            safe_radius=float(obs["r"]) + float(obs_safe_dist),
        )
        for obs in obstacles
    )
    xlim, ylim = _view_limits(history_xy, plan_xy, road_lines, circle_obstacles)
    ego_center = None
    if len(history_xy):
        ego_center = (float(history_xy[-1, 0]), float(history_xy[-1, 1]))
    return FrenetScene(
        history_xy=history_xy,
        plan_xy=plan_xy,
        road_lines=road_lines,
        obstacles=circle_obstacles,
        ego_center=ego_center,
        ego_heading=_heading_from_xy(history_xy),
        xlim=xlim,
        ylim=ylim,
    )
