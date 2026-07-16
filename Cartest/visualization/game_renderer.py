"""Renderer for Cartest multi-agent game rollouts.

All multi-agent game demos should render through this module instead of
defining local Matplotlib animation helpers.  It consumes report dicts
produced by the unified runner in ``Cartest/Simple.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np

from Cartest.visualization.scene_renderer import (
    LineLayer,
    RectLayer,
    SceneRenderSpec,
    VehicleLayer,
    render_scene,
)


ROAD_BG = "#10192b"
ROAD_FILL = "#253244"
LANE_FILL_A = "#2f3d52"
LANE_FILL_B = "#29384d"
LANE_EDGE = "#e5e7eb"
DIVIDER = "#cbd5e1"
HIST_CLR = "#ffd166"
COLORS = ("#2ecc71", "#1f77b4", "#d1495b", "#f4a261", "#b388eb")
DARK_COLORS = ("#1a8a4a", "#13496e", "#8a2e38", "#9d5f28", "#6d4ca2")


def _road_bounds(road):
    road = road or {}
    if "lane_bounds_d" in road:
        return tuple(float(v) for v in road["lane_bounds_d"])
    lane_centers = tuple(float(v) for v in road.get("lane_centers_d", (0.0, 3.5)))
    lane_width = float(road.get("lane_width", 3.5))
    return min(lane_centers) - 0.5 * lane_width, max(lane_centers) + 0.5 * lane_width


def compute_game_limits(reports, road=None, x_window=44.0, lead=8.0,
                        y_margin=0.8, agent_margin=3.0):
    """Compute a local view window that keeps current agents visible.

    The window follows ego by default, but expands and shifts when the current
    executed positions span more than the default MS-igo-style view. Predicted
    horizons intentionally do not affect the limits, so long plans cannot pull
    the camera back into an unreadable panorama.
    """
    last_report = reports[-1]
    history_xy = np.asarray(last_report["history_xy"], dtype=float)
    ego_x = float(history_xy[-1, 0, 0])
    current_x = history_xy[-1, :, 0]

    desired_min = float(np.min(current_x)) - float(agent_margin)
    desired_max = float(np.max(current_x)) + float(agent_margin)
    window = max(float(x_window), desired_max - desired_min)
    x_min = ego_x - float(lead)
    x_max = x_min + window
    if desired_min < x_min:
        x_min = desired_min
        x_max = x_min + window
    if desired_max > x_max:
        x_max = desired_max
        x_min = x_max - window
    road_min, road_max = _road_bounds(road)
    return x_min, x_max, road_min - y_margin, road_max + y_margin


def _draw_road(ax, road, limits):
    x_min, x_max, y_min, y_max = limits
    road_min, road_max = _road_bounds(road)
    lane_centers = tuple(float(v) for v in (road or {}).get("lane_centers_d", (0.0, 3.5)))
    lane_width = float((road or {}).get("lane_width", 3.5))
    patches = [
        RectLayer(
            xy=(float(x_min), float(road_min)),
            width=float(x_max - x_min),
            height=float(road_max - road_min),
            facecolor=ROAD_FILL,
            edgecolor=LANE_EDGE,
            linewidth=1.4,
            zorder=1,
            gid="road_surface",
        )
    ]
    for idx, lane_d in enumerate(lane_centers):
        patches.append(
            RectLayer(
                xy=(float(x_min), float(lane_d - 0.5 * lane_width)),
                width=float(x_max - x_min),
                height=lane_width,
                facecolor=LANE_FILL_A if idx % 2 == 0 else LANE_FILL_B,
                alpha=0.62,
                zorder=2,
                gid="lane_fill",
            )
        )
    lines = [
        LineLayer((float(x_min), float(x_max)), (float(road_min), float(road_min)),
                  color=LANE_EDGE, linewidth=2.0, zorder=4, gid="road_boundary"),
        LineLayer((float(x_min), float(x_max)), (float(road_max), float(road_max)),
                  color=LANE_EDGE, linewidth=2.0, zorder=4, gid="road_boundary"),
    ]
    for lower, upper in zip(lane_centers[:-1], lane_centers[1:]):
        lane_mid = 0.5 * (lower + upper)
        x = x_min
        while x < x_max:
            lines.append(
                LineLayer(
                    (float(x), float(min(x + 4.5, x_max))),
                    (float(lane_mid), float(lane_mid)),
                    color=DIVIDER,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.72,
                    zorder=5,
                    gid="lane_divider",
                )
            )
            x += 9.0
    render_scene(
        ax,
        SceneRenderSpec(
            facecolor=ROAD_BG,
            xlim=(float(x_min), float(x_max)),
            ylim=(float(y_min), float(y_max)),
            aspect="equal",
            patches=tuple(patches),
            lines=tuple(lines),
            hide_ticks=True,
            hide_spines=True,
        ),
    )


def _smooth_xy(points, samples_per_segment=8):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return pts
    padded = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for u in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            u2 = u * u
            u3 = u2 * u
            out.append(0.5 * ((2.0 * p1) + (-p0 + p2) * u
                              + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                              + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3))
    out.append(pts[-1])
    return np.asarray(out)


def _visible_xy(points, limits, pad=6.0):
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    x_min, x_max, y_min, y_max = limits
    mask = ((arr[:, 0] >= x_min - pad) & (arr[:, 0] <= x_max + pad)
            & (arr[:, 1] >= y_min - pad) & (arr[:, 1] <= y_max + pad))
    return arr[mask]


def render_game_frame(ax, report, road=None, limits=None):
    """Render one multi-agent game report on an axis."""
    road = road or {}
    ax.clear()
    limits = limits or compute_game_limits([report], road=road)
    _draw_road(ax, road, limits)

    history_xy = np.asarray(report["history_xy"])
    agent_names = report.get(
        "agent_names", [f"agent{i}" for i in range(history_xy.shape[1])]
    )
    predicted = report.get("predicted_xy", [])
    pi_values = report.get("pi", []) if road.get("show_pi", False) else []
    vehicle_length = float(road.get("vehicle_length", 5.0))
    vehicle_width = float(road.get("vehicle_width", 2.0))

    for agent_idx in range(history_xy.shape[1]):
        color = COLORS[agent_idx % len(COLORS)]
        dark = DARK_COLORS[agent_idx % len(DARK_COLORS)]
        traj = history_xy[:, agent_idx, :]
        if len(traj) >= 2:
            hist_xy = _smooth_xy(traj, samples_per_segment=6)
            ax.plot(hist_xy[:, 0], hist_xy[:, 1], color=HIST_CLR if agent_idx == 0 else color,
                    linewidth=2.0 if agent_idx == 0 else 1.2,
                    alpha=0.95 if agent_idx == 0 else 0.55,
                    zorder=7 if agent_idx == 0 else 6)
        if agent_idx < len(predicted):
            pred = _visible_xy(predicted[agent_idx], limits)
            if len(pred) >= 2:
                pred = _smooth_xy(pred, samples_per_segment=8)
                ax.plot(pred[:, 0], pred[:, 1], color=dark,
                        linestyle="-" if agent_idx == 0 else "--",
                        linewidth=2.1 if agent_idx == 0 else 1.25,
                        alpha=0.92 if agent_idx == 0 else 0.55,
                        zorder=5 if agent_idx == 0 else 4)
                sample_stride = max(8, len(pred) // 8)
                for sample in pred[sample_stride::sample_stride]:
                    if limits[0] - 2 <= sample[0] <= limits[1] + 2:
                        ax.add_patch(plt.Rectangle(
                            (sample[0] - 0.5 * vehicle_length * 0.72,
                             sample[1] - 0.5 * vehicle_width * 0.72),
                            vehicle_length * 0.72,
                            vehicle_width * 0.72,
                            facecolor=dark,
                            edgecolor="none",
                            alpha=0.08 if agent_idx == 0 else 0.11,
                            zorder=3,
                        ))
        center = traj[-1]
        heading = 0.0
        if len(traj) >= 2:
            delta = traj[-1] - traj[-2]
            if np.linalg.norm(delta) > 1e-6:
                heading = float(np.arctan2(delta[1], delta[0]))
        render_scene(
            ax,
            SceneRenderSpec(
                vehicles=(VehicleLayer(
                    center=(float(center[0]), float(center[1])),
                    heading=heading,
                    length=vehicle_length,
                    width=vehicle_width,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.2 if agent_idx == 0 else 0.9,
                    zorder=10 if agent_idx == 0 else 9,
                    gid=f"vehicle_{agent_idx}",
                ),),
            ),
        )
        ax.text(float(center[0]), float(center[1]) + 1.45,
                agent_names[agent_idx], color="white", fontsize=8,
                ha="center", va="bottom", zorder=12,
                bbox={"facecolor": "#020617", "edgecolor": color,
                      "alpha": 0.82, "pad": 1.8})
        if agent_idx < len(pi_values):
            pi_text = np.array2string(
                np.asarray(pi_values[agent_idx]), precision=2, separator=","
            )
            ax.text(0.02, 0.96 - agent_idx * 0.06,
                    f"{agent_names[agent_idx]} pi={pi_text}",
                    transform=ax.transAxes, color=color, fontsize=8, va="top",
                    bbox={"facecolor": "#020617", "edgecolor": "none",
                          "alpha": 0.65, "pad": 1.5})

    ax.text(0.02, 1.035,
            f"step={report.get('step', 0)}  solve={report.get('solve_ms', 0.0):.1f}ms",
            transform=ax.transAxes, fontsize=9, va="bottom", color="#e5e7eb",
            clip_on=False,
            bbox={"facecolor": "#020617", "edgecolor": "#475569",
                  "alpha": 0.90, "pad": 2.5})


def save_game_video(reports, output_path, road=None, fps=5):
    """Save a multi-agent rollout animation to an MP4 video via ffmpeg."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.44), dpi=150)
    fig.patch.set_facecolor("#0a1120")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.91)

    def update(frame_idx):
        limits = compute_game_limits(reports[:frame_idx + 1], road=road)
        render_game_frame(ax, reports[frame_idx], road=road, limits=limits)
        return []

    anim = FuncAnimation(fig, update, frames=len(reports),
                         interval=1000 / fps, blit=False)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


# Backward-compatible name for older imports. It now writes video, not GIF.
save_game_animation = save_game_video
