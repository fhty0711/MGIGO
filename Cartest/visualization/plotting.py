"""High-level plotting entry points for Cartest MPC visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np

from Cartest.visualization.frenet_scene import default_straight_scenario
from Cartest.visualization.frenet_renderer import render_frenet_panel


def setup_axes():
    """Create (trajectory, kinematics) twin axes for MPC animation."""
    plt.ion()
    fig, (ax_traj, ax_kin) = plt.subplots(1, 2, figsize=(16, 5))
    return fig, ax_traj, ax_kin


def render_frame(ax_traj, ax_kin, report, obstacles, obs_safe_dist, gen_dt, scenario=None):
    """Render one MPC frame onto the given axes."""
    ax_traj.cla()
    ax_kin.cla()

    render_frenet_panel(
        ax_traj,
        report,
        scenario=scenario or default_straight_scenario(),
        obstacles=obstacles,
        obs_safe_dist=obs_safe_dist,
    )

    t_arr = np.arange(len(report.sp)) * gen_dt
    ax_kin.set_facecolor("#10192b")
    ax_kin.plot(t_arr, report.a_long, label="a_long")
    ax_kin.plot(t_arr, report.a_lat, label="a_lat")
    ax_kin.plot(t_arr, report.jm, label="jerk")
    ax_kin.legend()
    ax_kin.grid(alpha=0.25)
    ax_kin.tick_params(colors="white")
    for spine in ax_kin.spines.values():
        spine.set_color("#40506b")
    ax_kin.set_title(
        f"max |a_long|={report.max_along:.1f}  "
        f"|a_lat|={report.max_alat:.1f}  jerk={report.max_jerk:.1f}",
        color="white",
    )


def save_animation(fig, frames, render_fn, output_path, fps=12):
    """Create and save an MPC animation as an MP4 video via ffmpeg."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, render_fn, frames=len(frames), interval=80)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"saved {output_path}")
