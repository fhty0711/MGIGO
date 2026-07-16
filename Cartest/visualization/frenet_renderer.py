"""Frenet trajectory scene renderer for Cartest MPC demos."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from Cartest.visualization.frenet_scene import build_frenet_scene
from Cartest.visualization.scene_renderer import (
    CircleLayer,
    LineLayer,
    PolygonLayer,
    SceneRenderSpec,
    TextLayer,
    VehicleLayer,
    render_scene,
)


ROAD_BG = "#0a1120"
ROAD_FILL = "#10192b"
LANE_FILL = "#18243a"
LANE_EDGE = "#d7dee8"
LANE_DIVIDER = "#cbd5e1"
EXEC_CLR = "#ffd166"
PLAN_CLR = "#7aa2ff"
EGO_CLR = "#2ecc71"
OBSTACLE_CLR = "#fb7185"
SAFE_CLR = "#fca5a5"
TEXT_CLR = "#edf3ff"

VEHICLE_LENGTH = 4.8
VEHICLE_WIDTH = 2.0


def _obstacle_layers(scene):
    patches = []
    for obs in scene.obstacles:
        x, y = obs.center
        patches.append(
            CircleLayer(
                center=(float(x), float(y)),
                radius=float(obs.safe_radius),
                facecolor="none",
                edgecolor=SAFE_CLR,
                linewidth=1.1,
                alpha=0.55,
                zorder=4,
                gid="safety_envelope",
            )
        )
        patches.append(
            CircleLayer(
                center=(float(x), float(y)),
                radius=float(obs.radius),
                facecolor=OBSTACLE_CLR,
                edgecolor="#fecdd3",
                linewidth=0.8,
                alpha=0.38,
                zorder=5,
                gid="obstacle",
            )
        )
    return patches


def _road_lines(scene):
    lines = []
    for idx, road_line in enumerate(scene.road_lines):
        if len(road_line) < 2:
            continue
        is_center = idx == 2
        lines.append(
            LineLayer(
                x=tuple(float(v) for v in road_line[:, 0]),
                y=tuple(float(v) for v in road_line[:, 1]),
                color=LANE_DIVIDER if is_center else LANE_EDGE,
                linewidth=1.0 if is_center else 1.7,
                linestyle=(0, (7, 6)) if is_center else "-",
                alpha=0.72 if is_center else 0.95,
                zorder=3,
                gid="lane_divider" if is_center else "road_boundary",
            )
        )
    return lines


def _road_patches(scene):
    if len(scene.road_lines) < 2:
        return []
    lower = np.asarray(scene.road_lines[0], dtype=float)
    upper = np.asarray(scene.road_lines[1], dtype=float)
    if len(lower) < 2 or len(upper) < 2:
        return []
    polygon = np.vstack([lower, upper[::-1]])
    return [
        PolygonLayer(
            xy=tuple((float(x), float(y)) for x, y in polygon),
            facecolor=ROAD_FILL,
            edgecolor=LANE_EDGE,
            linewidth=1.0,
            alpha=0.96,
            zorder=1,
            gid="road_surface",
        )
    ]


def render_frenet_panel(
    ax,
    report,
    *,
    scenario: Mapping,
    obstacles: Sequence[Mapping] = (),
    obs_safe_dist: float = 0.0,
):
    """Render a top-down Frenet MPC panel for a single StepReport."""
    ax.cla()
    scene = build_frenet_scene(
        report,
        scenario=scenario,
        obstacles=obstacles,
        obs_safe_dist=obs_safe_dist,
    )

    patches = _road_patches(scene)
    patches.extend(_obstacle_layers(scene))
    lines = _road_lines(scene)
    if len(scene.history_xy) >= 2:
        lines.append(
            LineLayer(
                x=tuple(float(v) for v in scene.history_xy[:, 0]),
                y=tuple(float(v) for v in scene.history_xy[:, 1]),
                color=EXEC_CLR,
                linewidth=2.4,
                alpha=0.96,
                zorder=6,
                gid="executed_history",
            )
        )
    if len(scene.plan_xy) >= 2:
        lines.append(
            LineLayer(
                x=tuple(float(v) for v in scene.plan_xy[:, 0]),
                y=tuple(float(v) for v in scene.plan_xy[:, 1]),
                color=PLAN_CLR,
                linewidth=2.0,
                linestyle="--",
                alpha=0.92,
                zorder=6,
                gid="planned_path",
            )
        )

    vehicles = ()
    if scene.ego_center is not None:
        vehicles = (
            VehicleLayer(
                center=scene.ego_center,
                heading=scene.ego_heading,
                length=VEHICLE_LENGTH,
                width=VEHICLE_WIDTH,
                facecolor=EGO_CLR,
                edgecolor="#dcfce7",
                linewidth=1.0,
                alpha=0.98,
                zorder=8,
                gid="ego_vehicle",
            ),
        )

    g = getattr(report, "g_values", {}) or {}
    summary = (
        f"step {int(report.step)}  v={float(report.hv[-1]):.1f}m/s  "
        f"obs={float(report.min_obs):.1f}m  solve={float(report.solve_ms):.0f}ms\n"
        f"cost={float(report.cost):.3g}  "
        f"g lane={float(g.get('lane', 0.0)):.2f} obs={float(g.get('obs', 0.0)):.2f} "
        f"acc={float(g.get('acc', 0.0)):.2f} jerk={float(g.get('jerk', 0.0)):.2f}"
    )

    render_scene(
        ax,
        SceneRenderSpec(
            facecolor=ROAD_BG,
            xlim=scene.xlim,
            ylim=scene.ylim,
            aspect="equal",
            patches=tuple(patches),
            lines=tuple(lines),
            vehicles=vehicles,
            texts=(
                TextLayer(
                    x=0.015,
                    y=0.975,
                    text=summary,
                    color=TEXT_CLR,
                    fontsize=8.0,
                    ha="left",
                    va="top",
                    transform="axes",
                    bbox={
                        "boxstyle": "round,pad=0.35,rounding_size=0.16",
                        "facecolor": "#020617",
                        "edgecolor": "#334155",
                        "alpha": 0.82,
                    },
                    gid="status_text",
                ),
            ),
            hide_ticks=True,
            hide_spines=True,
        ),
    )
    return ax
