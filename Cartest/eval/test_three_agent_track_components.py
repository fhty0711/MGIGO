"""Tests for the shared three-agent cost component layer."""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Cartest.planning.costs.three_agent_track_components import (
    acc_limit_violation,
    forward_motion_violation,
    jerk_limit_violation,
    kinematics_violation,
    lane_boundary_violation,
    speed_limit_violation,
    rss_cvar_risk,
    role_soft_objective,
)


def _plan(s, d, v=10.0):
    T = len(s)
    vehicle = jnp.zeros((T, 9), dtype=jnp.float32)
    vehicle = vehicle.at[:, 2].set(v)
    return {
        "s": jnp.asarray(s, dtype=jnp.float32),
        "d": jnp.asarray(d, dtype=jnp.float32),
        "s_dot": jnp.full((T,), v, dtype=jnp.float32),
        "d_dot": jnp.zeros((T,), dtype=jnp.float32),
        "s_ddot": jnp.zeros((T,), dtype=jnp.float32),
        "d_ddot": jnp.zeros((T,), dtype=jnp.float32),
        "s_dddot": jnp.zeros((T,), dtype=jnp.float32),
        "d_dddot": jnp.zeros((T,), dtype=jnp.float32),
        "vehicle": vehicle,
    }


def test_lane_boundary_uses_vehicle_body_tightened_bounds():
    scenario = {"road": {"lane_bounds_d": (-1.75, 5.25)}, "safety": {"vehicle_width": 2.0}}
    inside = _plan([0.0, 1.0], [0.0, 3.5])
    outside = _plan([0.0, 1.0], [-1.0, 5.0])
    assert float(lane_boundary_violation(inside, scenario)) == 0.0
    assert float(lane_boundary_violation(outside, scenario)) > 0.0


def test_forward_motion_penalizes_negative_s_dot():
    plan = _plan([1.0, 0.5], [0.0, 0.0], v=10.0)
    plan["s_dot"] = jnp.array([10.0, -0.2], dtype=jnp.float32)
    assert float(forward_motion_violation(plan, {})) > 0.0


def test_speed_limit_checks_lower_and_upper_bounds():
    scenario = {"safety": {"v_min": 2.0, "v_max": 12.0}}
    ok = _plan([0.0, 1.0], [0.0, 0.0], v=10.0)
    fast = _plan([0.0, 1.0], [0.0, 0.0], v=15.0)
    slow = _plan([0.0, 1.0], [0.0, 0.0], v=1.0)
    assert float(speed_limit_violation(ok, scenario)) == 0.0
    assert float(speed_limit_violation(fast, scenario)) > 0.0
    assert float(speed_limit_violation(slow, scenario)) > 0.0


def test_kinematics_layer_is_max_of_acc_and_jerk():
    scenario = {"safety": {"acc_max": 2.0, "jerk_max": 1.0}}
    plan = _plan([0.0, 1.0], [0.0, 0.0], v=10.0)
    plan["vehicle"] = plan["vehicle"].at[:, 4].set(3.0)
    plan["vehicle"] = plan["vehicle"].at[:, 6].set(1.25)
    acc = acc_limit_violation(plan, scenario)
    jerk = jerk_limit_violation(plan, scenario)
    kin = kinematics_violation(plan, scenario)
    assert float(kin) == float(jnp.maximum(acc, jerk))


def test_rss_cvar_risk_increases_when_gap_is_small():
    scenario = {"safety": {"vehicle_length": 5.0, "vehicle_width": 2.0}}
    ego = _plan([10.0, 12.0, 14.0], [3.5, 3.5, 3.5], v=15.0)
    far_front = _plan([50.0, 52.0, 54.0], [3.5, 3.5, 3.5], v=10.0)
    near_front = _plan([18.0, 19.0, 20.0], [3.5, 3.5, 3.5], v=10.0)
    far = rss_cvar_risk(ego, (far_front,), scenario, dt=0.15)
    near = rss_cvar_risk(ego, (near_front,), scenario, dt=0.15)
    assert float(near) > float(far)


def test_role_objectives_share_formula_but_not_role_targets():
    scenario = {
        "agents": [
            {"role": "ego", "v_target": 17.5},
            {"role": "front", "v_target": 20.0},
            {"role": "rear", "v_target": 17.5},
        ],
        "behavior": {"ego_target_d": 3.5, "upper_lane_d": 3.5},
        "safety": {"vehicle_length": 5.0, "vehicle_width": 2.0},
    }
    ctx = {}
    ego = _plan([10.0, 11.0, 12.0], [0.0, 0.2, 0.4], v=15.0)
    front = _plan([30.0, 31.0, 32.0], [3.5, 3.5, 3.5], v=18.0)
    rear = _plan([0.0, 1.0, 2.0], [3.5, 3.5, 3.5], v=15.0)
    plans = (ego, front, rear)
    ego_cost = role_soft_objective(plans, scenario, ctx, agent_idx=0, dt=0.15)
    front_cost = role_soft_objective(plans, scenario, ctx, agent_idx=1, dt=0.15)
    assert jnp.isfinite(ego_cost)
    assert jnp.isfinite(front_cost)
    assert float(ego_cost) != float(front_cost)


if __name__ == "__main__":
    test_lane_boundary_uses_vehicle_body_tightened_bounds()
    test_forward_motion_penalizes_negative_s_dot()
    test_speed_limit_checks_lower_and_upper_bounds()
    test_kinematics_layer_is_max_of_acc_and_jerk()
    test_rss_cvar_risk_increases_when_gap_is_small()
    test_role_objectives_share_formula_but_not_role_targets()
    print("three agent track component tests ok")
