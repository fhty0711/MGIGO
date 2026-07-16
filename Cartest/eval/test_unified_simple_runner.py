"""Smoke tests for the unified Cartest runner configuration."""

from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Cartest.planning.scenarios import SCENARIOS, get_scenario, scenario_kind


def test_game_scenarios_are_registered_with_multi_agent_kind():
    for name in ("game_2a_basic", "game_2b_constran", "game_2b_asymmetric", "three_agent_track"):
        scenario = get_scenario(name)
        assert name in SCENARIOS
        assert scenario_kind(scenario) == "multi_agent_game"
        assert "agents" in scenario
        assert "game" in scenario
        assert "block_to_agent" in scenario["game"]


def test_existing_scenarios_default_to_single_agent_kind():
    assert scenario_kind(get_scenario("empty")) == "single_agent"
    assert scenario_kind(get_scenario("lane_borrow_overtake")) == "single_agent"




def test_game_renderer_smoke(tmp_path=None):
    import os
    import numpy as np
    import shutil
    from Cartest.visualization.game_renderer import save_game_video

    if shutil.which("ffmpeg") is None:
        print("skipping game renderer smoke: ffmpeg is not installed")
        return

    reports = [
        {
            "step": 0,
            "history_xy": np.array([[[0.0, 0.0], [2.0, 3.5]]]),
            "predicted_xy": [np.array([[0.0, 0.0], [1.0, 0.2]]),
                              np.array([[2.0, 3.5], [3.0, 3.5]])],
            "agent_names": ["agent0", "agent1"],
            "pi": [np.array([0.6, 0.3, 0.1]), np.array([0.2, 0.5, 0.3])],
            "solve_ms": 12.0,
        }
    ]
    out_dir = tmp_path if tmp_path else "/tmp/mgigo-gametest"
    os.makedirs(out_dir, exist_ok=True)
    output = os.path.join(out_dir, "game.mp4")
    save_game_video(reports, output, road={"lane_centers_d": (0.0, 3.5)})
    assert os.path.exists(output) and os.path.getsize(output) > 0


def test_game_renderer_limits_include_all_current_agents():
    import numpy as np
    from Cartest.visualization.game_renderer import compute_game_limits

    reports = [
        {
            "history_xy": np.array([[[29.5, 0.0], [38.0, 3.5], [8.1, 3.5]]]),
            "predicted_xy": [np.array([[0.0, 0.0], [100.0, 0.0]]),
                              np.array([[2.0, 3.5], [3.0, 30.0]])],
        }
    ]
    x_min, x_max, y_min, y_max = compute_game_limits(
        reports, road={"lane_bounds_d": (-1.75, 5.25)}, x_window=44.0
    )
    assert x_min <= 8.1 - 3.0
    assert x_max >= 38.0 + 3.0
    assert x_max - x_min <= 52.0
    assert y_min == -2.55
    assert y_max == 6.05



def test_simple_module_exposes_single_and_game_runners():
    import Cartest.Simple as simple

    assert callable(simple.run_single_agent_mpc)
    assert callable(simple.run_multi_agent_game)
    assert callable(simple.block_until_ready)
    assert callable(simple.run)


def test_simple_runner_uses_scenario_default_steps():
    import Cartest.Simple as simple

    assert simple.default_steps_for_scenario("three_agent_track") == 25
    assert simple.default_steps_for_scenario("game_2a_basic") == 30
    assert simple.default_steps_for_scenario("empty") == 150


def test_simple_runner_builds_timestamped_output_video_path():
    import Cartest.Simple as simple

    output = simple.make_output_video_path(
        "three_agent_track", now=datetime(2026, 7, 16, 12, 3, 4)
    )

    assert output.parent == simple.OUTPUT
    assert output.name == "three_agent_track_20260716_120304.mp4"


def test_three_agent_track_executes_third_plan_sample():
    from Cartest.planning.scenarios import get_scenario

    scenario = get_scenario("three_agent_track")

    assert scenario["game"]["execute_index"] == 3


def test_three_agent_track_uses_validated_realtime_iteration_budget():
    from Cartest.planning.scenarios import get_scenario

    scenario = get_scenario("three_agent_track")

    assert scenario["game"]["T"] == 100


def test_three_agent_track_initial_spacing_is_safe():
    from Cartest.planning.scenarios import get_scenario

    scenario = get_scenario("three_agent_track")
    safe_clearance = scenario["safety"]["vehicle_length"] + scenario["safety"]["safe_gap"]

    ego, front, rear = scenario["agents"]
    assert front["s"] - rear["s"] >= safe_clearance
    assert ego["s"] - rear["s"] >= safe_clearance

if __name__ == "__main__":
    test_game_scenarios_are_registered_with_multi_agent_kind()
    test_existing_scenarios_default_to_single_agent_kind()
    test_game_renderer_smoke()
    test_game_renderer_limits_include_all_current_agents()
    test_simple_module_exposes_single_and_game_runners()
    test_simple_runner_uses_scenario_default_steps()
    test_simple_runner_builds_timestamped_output_video_path()
    test_three_agent_track_executes_third_plan_sample()
    test_three_agent_track_uses_validated_realtime_iteration_budget()
    test_three_agent_track_initial_spacing_is_safe()
    print("unified runner scenario tests ok")
