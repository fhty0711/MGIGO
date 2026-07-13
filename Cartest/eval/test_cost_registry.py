"""Smoke tests for scenario-selected cost and constraint config."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp

from Cartest.planning.costs import (
    available_costs,
    make_constraint_config_from_scenario,
)
from Cartest.planning.costs.registry import get_cost_factory, get_cost_spec
from Cartest.planning.scenarios import SCENARIOS, get_scenario


def test_scenario_registry():
    """All registered scenarios have required keys."""
    required = {"ref_path", "road", "obstacles", "safety", "behavior", "cost", "ego"}
    for name, scenario in SCENARIOS.items():
        missing = required - set(scenario)
        assert not missing, f"scenario {name!r} missing keys: {missing}"
        assert "lane_hw" in scenario["road"], f"scenario {name!r} road missing lane_hw"
        assert "obs_safe_dist" in scenario["safety"], f"scenario {name!r} safety missing obs_safe_dist"
        assert "v_target" in scenario["behavior"], f"scenario {name!r} behavior missing v_target"


def test_get_scenario_unknown():
    """Unknown scenario raises ValueError."""
    try:
        get_scenario("nonexistent")
        assert False, "should have raised"
    except ValueError as exc:
        assert "nonexistent" in str(exc)


def test_default_lyapunov_config():
    """Default constraint config has correct structure and isolation."""
    scenario = get_scenario("empty")
    cfg_a = make_constraint_config_from_scenario(scenario)
    cfg_b = make_constraint_config_from_scenario(scenario)

    assert cfg_a["enabled"] == ("obs", "lane", "speed", "acc", "jerk")
    assert cfg_a["constran"]["k_inner"] == 1.0
    assert cfg_a["specs"]["obs"]["priority"] == 1
    assert cfg_a["specs"]["obs"]["mode"] == "hard"

    # Deep-copy isolation
    cfg_a["specs"]["obs"]["priority"] = 99
    assert cfg_b["specs"]["obs"]["priority"] == 1


def test_cross_order_costs_registered():
    """All 5 cross-order templates are registered with safety_overrides."""
    costs = available_costs()
    for template in ("conservative", "standard", "active", "aggressive", "emergency"):
        name = f"cross_order_{template}"
        assert name in costs, f"{name} not registered"


def test_cross_order_safety_overrides():
    """Cross-order active cost provides acc_max/jerk_max overrides."""
    scenario = dict(get_scenario("empty"))
    scenario["cost"] = {"name": "cross_order_active", "params": {}}

    cfg = make_constraint_config_from_scenario(scenario)
    assert "safety_overrides" in cfg
    assert cfg["safety_overrides"]["acc_max"] == 7.0
    assert cfg["safety_overrides"]["jerk_max"] == 3.0


def test_lane_borrow_overtake_registered():
    """Lane-borrow overtake cost and scenario are registered."""
    costs = available_costs()
    assert "lane_borrow_overtake" in costs
    assert "lane_borrow_overtake" in SCENARIOS


def test_lane_borrow_overtake_params():
    """Lane-borrow scenario has acc_weight/jerk_weight > 0."""
    scenario = get_scenario("lane_borrow_overtake")
    params = scenario["cost"]["params"]
    assert params["acc_weight"] > 0.0
    assert params["jerk_weight"] > 0.0
    assert params["borrow_lane_d"] != params["start_lane_d"]


def test_lane_borrow_overtake_config():
    """Lane-borrow cost factory provides constraint config."""
    scenario = get_scenario("lane_borrow_overtake")
    cfg = make_constraint_config_from_scenario(scenario)
    assert cfg["enabled"] == ("obs", "lane", "speed", "acc", "jerk")
    # No safety_overrides for lane_borrow (uses default limits)
    assert "safety_overrides" not in cfg or cfg.get("safety_overrides") == {}


def test_lane_bounds_d_in_scenarios():
    """Scenarios with lane_bounds_d expose asymmetric bounds."""
    scenario = get_scenario("lane_borrow_overtake")
    assert "lane_bounds_d" in scenario["road"]
    bounds = scenario["road"]["lane_bounds_d"]
    assert bounds[0] != bounds[1]


def test_dynamic_obstacle_predictions():
    """build_obstacle_predictions handles dynamic obstacles."""
    from Cartest.planning.scenarios import build_obstacle_predictions

    class FakeGen:
        T = 100
        dt = 0.1

    scenario = {
        "obstacles": {
            "static": [{"x": 50.0, "y": 0.0, "r": 2.0}],
            "dynamic": [{"x": 10.0, "y": 3.0, "r": 1.5, "v": 5.0, "yaw": 0.0}],
        }
    }
    obs_pos, obs_rad = build_obstacle_predictions(scenario, FakeGen())
    assert obs_pos.shape == (100, 2, 2)  # [T, N=2, 2]
    assert obs_rad.shape == (100, 2)
    # Static obstacle doesn't move
    assert float(obs_pos[0, 0, 0]) == 50.0
    assert float(obs_pos[99, 0, 0]) == 50.0
    # Dynamic obstacle moves in x
    assert float(obs_pos[0, 1, 0]) == 10.0
    assert float(obs_pos[99, 1, 0]) > 10.0


def test_first_frame_obstacles():
    """first_frame_obstacles extracts plotting-friendly dict list."""
    from Cartest.planning.scenarios import first_frame_obstacles

    obs_pos = jnp.zeros((100, 2, 2))
    obs_pos = obs_pos.at[0, 0].set(jnp.array([50.0, 0.0]))
    obs_pos = obs_pos.at[0, 1].set(jnp.array([10.0, 3.0]))
    obs_rad = jnp.zeros((100, 2))
    obs_rad = obs_rad.at[0, 0].set(2.0)
    obs_rad = obs_rad.at[0, 1].set(1.5)

    result = first_frame_obstacles(obs_pos, obs_rad)
    assert len(result) == 2
    assert result[0]["x"] == 50.0
    assert result[0]["r"] == 2.0
    assert result[1]["x"] == 10.0


def test_factory_resolution():
    """make_objective_from_scenario resolves a factory for each scenario."""
    for name in SCENARIOS:
        scenario = get_scenario(name)
        cost_name, _, _ = get_cost_spec(scenario)
        factory = get_cost_factory(cost_name)
        assert callable(factory), f"factory for {name!r} is not callable"


def test_default_lyapunov_acc_jerk_weight():
    """default_lyapunov DEFAULT_PARAMS includes acc_weight/jerk_weight."""
    from Cartest.planning.costs.default_lyapunov import DEFAULT_PARAMS
    assert "acc_weight" in DEFAULT_PARAMS
    assert "jerk_weight" in DEFAULT_PARAMS
    assert DEFAULT_PARAMS["acc_weight"] == 0.0


def main():
    tests = [
        test_scenario_registry,
        test_get_scenario_unknown,
        test_default_lyapunov_config,
        test_cross_order_costs_registered,
        test_cross_order_safety_overrides,
        test_lane_borrow_overtake_registered,
        test_lane_borrow_overtake_params,
        test_lane_borrow_overtake_config,
        test_lane_bounds_d_in_scenarios,
        test_dynamic_obstacle_predictions,
        test_first_frame_obstacles,
        test_factory_resolution,
        test_default_lyapunov_acc_jerk_weight,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {test.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
    if passed != len(tests):
        sys.exit(1)


if __name__ == "__main__":
    main()
