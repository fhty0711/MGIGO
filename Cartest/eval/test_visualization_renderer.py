"""Smoke tests for the Cartest visualization renderer."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_scene_renderer_draws_basic_layers():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Cartest.visualization.scene_renderer import (
        CircleLayer,
        LineLayer,
        RectLayer,
        SceneRenderSpec,
        VehicleLayer,
        render_scene,
    )

    spec = SceneRenderSpec(
        facecolor="#000000",
        xlim=(-1.0, 5.0),
        ylim=(-2.0, 4.0),
        aspect="equal",
        patches=(
            RectLayer(
                xy=(0.0, 0.0),
                width=2.0,
                height=1.0,
                facecolor="#ffffff",
                gid="rect_layer",
            ),
            CircleLayer(
                center=(3.0, 1.0),
                radius=0.4,
                facecolor="#00ff00",
                gid="circle_layer",
            ),
        ),
        lines=(
            LineLayer(
                x=(0.0, 4.0),
                y=(2.0, 2.0),
                color="#abcdef",
                gid="line_layer",
            ),
        ),
        vehicles=(
            VehicleLayer(
                center=(1.0, 2.0),
                heading=0.25,
                length=4.5,
                width=1.9,
                facecolor="#22cc88",
                gid="vehicle_layer",
            ),
        ),
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        render_scene(ax, spec)
        patch_gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
        line_gids = {line.get_gid() for line in ax.lines if line.get_gid()}
    finally:
        plt.close(fig)

    assert {"rect_layer", "circle_layer", "vehicle_layer"} <= patch_gids
    assert "line_layer" in line_gids


def test_frenet_renderer_draws_report_scene():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Cartest.eval.reporting import StepReport
    from Cartest.planning.scenarios import get_scenario
    from Cartest.visualization.frenet_renderer import render_frenet_panel

    report = StepReport(
        step=3,
        hx=np.array([0.0, 5.0, 10.0]),
        hy=np.array([0.0, 0.3, 0.6]),
        hv=np.array([10.0, 11.0, 12.0]),
        px=np.linspace(10.0, 35.0, 20),
        py=np.linspace(0.6, 2.5, 20),
        sp=np.linspace(12.0, 14.0, 20),
        a_long=np.zeros(20),
        a_lat=np.zeros(20),
        jm=np.zeros(20),
        solve_ms=23.0,
        min_obs=12.5,
        max_along=0.5,
        max_alat=0.4,
        max_jerk=0.2,
        cost=1.25,
        g_values={"lane": 0.0, "obs": 0.0, "jerk": 0.0, "acc": 0.0, "spd": 0.0},
    )
    obstacles = [{"x": 24.0, "y": 1.0, "r": 2.0}]

    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    try:
        render_frenet_panel(
            ax,
            report,
            scenario=get_scenario("lane_borrow_overtake"),
            obstacles=obstacles,
            obs_safe_dist=0.5,
        )
        patch_gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
        line_gids = {line.get_gid() for line in ax.lines if line.get_gid()}
    finally:
        plt.close(fig)

    assert "road_surface" in patch_gids
    assert "ego_vehicle" in patch_gids
    assert "obstacle" in patch_gids
    assert "safety_envelope" in patch_gids
    assert "executed_history" in line_gids
    assert "planned_path" in line_gids


def test_frenet_renderer_converts_history_through_reference_path():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Cartest.eval.reporting import StepReport
    from Cartest.planning.scenarios import get_scenario
    from Cartest.visualization.frenet_renderer import render_frenet_panel

    scenario = get_scenario("circle_track")
    report = StepReport(
        step=0,
        hx=np.array([0.0, 10.0]),
        hy=np.array([0.0, 0.0]),
        hv=np.array([12.0, 12.0]),
        px=np.array([0.0, 10.0]),
        py=np.array([100.0, 99.5]),
        sp=np.ones(4) * 12.0,
        a_long=np.zeros(4),
        a_lat=np.zeros(4),
        jm=np.zeros(4),
        solve_ms=1.0,
        min_obs=1e9,
        max_along=0.0,
        max_alat=0.0,
        max_jerk=0.0,
        cost=0.0,
        g_values={"lane": 0.0, "obs": 0.0, "jerk": 0.0, "acc": 0.0, "spd": 0.0},
    )

    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    try:
        render_frenet_panel(ax, report, scenario=scenario)
        history_line = next(line for line in ax.lines if line.get_gid() == "executed_history")
        road_line = next(line for line in ax.lines if line.get_gid() == "road_boundary")
        history_x = np.asarray(history_line.get_xdata(), dtype=float)
        history_y = np.asarray(history_line.get_ydata(), dtype=float)
        road_y = np.asarray(road_line.get_ydata(), dtype=float)
    finally:
        plt.close(fig)

    assert np.isclose(history_x[0], 0.0, atol=1e-4)
    assert np.isclose(history_y[0], 100.0, atol=1e-4)
    assert float(np.std(road_y)) > 0.01


def test_frenet_renderer_smoke_renders_all_registered_scenarios():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Cartest.eval.reporting import StepReport
    from Cartest.planning.scenarios import SCENARIOS, scenario_kind
    from Cartest.visualization.frenet_renderer import render_frenet_panel

    for name, scenario in SCENARIOS.items():
        if scenario_kind(scenario) != "single_agent":
            continue
        s0 = float(scenario["ego"]["s"])
        d0 = float(scenario["ego"]["d"])
        px, py = scenario["ref_path"].frenet_to_cartesian(
            np.linspace(s0, s0 + 30.0, 12),
            np.full(12, d0),
        )
        report = StepReport(
            step=0,
            hx=np.array([s0, s0 + 5.0]),
            hy=np.array([d0, d0]),
            hv=np.array([float(scenario["ego"]["s_dot"])] * 2),
            px=np.asarray(px, dtype=float),
            py=np.asarray(py, dtype=float),
            sp=np.ones(12) * float(scenario["ego"]["s_dot"]),
            a_long=np.zeros(12),
            a_lat=np.zeros(12),
            jm=np.zeros(12),
            solve_ms=1.0,
            min_obs=1e9,
            max_along=0.0,
            max_alat=0.0,
            max_jerk=0.0,
            cost=0.0,
            g_values={"lane": 0.0, "obs": 0.0, "jerk": 0.0, "acc": 0.0, "spd": 0.0},
        )
        fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
        try:
            render_frenet_panel(ax, report, scenario=scenario)
            patch_gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
            line_gids = {line.get_gid() for line in ax.lines if line.get_gid()}
        finally:
            plt.close(fig)

        assert "road_surface" in patch_gids, name
        assert "ego_vehicle" in patch_gids, name
        assert "road_boundary" in line_gids, name
        assert "executed_history" in line_gids, name
        assert "planned_path" in line_gids, name


def test_plotting_render_frame_keeps_default_straight_road_compatibility():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Cartest.eval.reporting import StepReport
    from Cartest.visualization.plotting import render_frame

    report = StepReport(
        step=0,
        hx=np.array([0.0, 5.0]),
        hy=np.array([0.0, 0.0]),
        hv=np.array([10.0, 10.5]),
        px=np.linspace(5.0, 20.0, 8),
        py=np.zeros(8),
        sp=np.ones(8) * 10.0,
        a_long=np.zeros(8),
        a_lat=np.zeros(8),
        jm=np.zeros(8),
        solve_ms=1.0,
        min_obs=1e9,
        max_along=0.0,
        max_alat=0.0,
        max_jerk=0.0,
        cost=0.0,
        g_values={"lane": 0.0, "obs": 0.0, "jerk": 0.0, "acc": 0.0, "spd": 0.0},
    )

    fig, (ax_traj, ax_kin) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)
    try:
        render_frame(ax_traj, ax_kin, report, [], 0.1, 0.1)
        patch_gids = {patch.get_gid() for patch in ax_traj.patches if patch.get_gid()}
        line_gids = {line.get_gid() for line in ax_traj.lines if line.get_gid()}
    finally:
        plt.close(fig)

    assert "road_surface" in patch_gids
    assert "ego_vehicle" in patch_gids
    assert "executed_history" in line_gids


def test_game_renderer_draws_nash_panel_ghosts_and_saves_frame(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Cartest.visualization.game_renderer import (
        _create_game_figure,
        render_game_frame,
        save_game_frame,
        save_game_video,
    )

    history = np.array([
        [[0.0, 0.0], [8.0, 3.5], [-7.0, 3.5]],
        [[2.0, 0.2], [10.0, 3.5], [-5.0, 3.5]],
    ])
    predicted = [
        np.column_stack([np.linspace(2.0, 25.0, 12), np.linspace(0.2, 3.5, 12)]),
        np.column_stack([np.linspace(10.0, 31.0, 12), np.full(12, 3.5)]),
        np.column_stack([np.linspace(-5.0, 18.0, 12), np.full(12, 3.5)]),
    ]
    report = {
        "step": 4,
        "solve_ms": 92.0,
        "history_xy": history,
        "predicted_xy": predicted,
        "agent_names": ["ego", "front", "rear"],
        "keyframe_name": "max_rss",
        "agent_status": [
            {
                "name": name,
                "speed": 15.0 - aid,
                "target_speed": 17.5,
                "mode": [1, aid],
                "epsilon_mode": 1e-4 * aid,
                "pi_s": [0.1, 0.8, 0.1],
                "pi_d": [0.7, 0.2, 0.1],
            }
            for aid, name in enumerate(("ego", "front", "rear"))
        ],
        "best_response_xy": predicted,
        "best_response_diag": {
            name: {
                "epsilon_br": 1e-3 * (aid + 1),
                "equilibrium_expected_cost": 0.126,
                "best_response_cost": 0.125,
            }
            for aid, name in enumerate(("ego", "front", "rear"))
        },
    }

    fig, (road_ax, info_ax) = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [4.8, 2.2]})
    try:
        render_game_frame(road_ax, report, info_ax=info_ax)
        text = "\n".join(item.get_text() for item in info_ax.texts)
        line_gids = {
            line.get_gid() for line in road_ax.lines if line.get_gid()
        }
    finally:
        plt.close(fig)

    assert "v/target" in text
    assert "epsilon_mode" in text
    assert "epsilon_br" in text
    assert "solid=joint plan" in text
    assert "best_response_ego" in line_gids

    output = tmp_path / "nash_frame.png"
    save_game_frame(report, output)
    assert output.exists()
    assert output.stat().st_size > 1000

    encoded_fig, _, _ = _create_game_figure(with_panel=True)
    try:
        pixel_width, pixel_height = (
            encoded_fig.get_size_inches() * encoded_fig.dpi
        ).astype(int)
    finally:
        plt.close(encoded_fig)
    assert pixel_width % 2 == 0
    assert pixel_height % 2 == 0

    from matplotlib.animation import writers
    if writers.is_available("ffmpeg"):
        video = tmp_path / "nash_frame.mp4"
        save_game_video([report], video, fps=1)
        assert video.exists()
        assert video.stat().st_size > 1000


if __name__ == "__main__":
    test_scene_renderer_draws_basic_layers()
    test_frenet_renderer_draws_report_scene()
    test_frenet_renderer_converts_history_through_reference_path()
    test_frenet_renderer_smoke_renders_all_registered_scenarios()
    test_plotting_render_frame_keeps_default_straight_road_compatibility()
    print("visualization renderer tests ok")
