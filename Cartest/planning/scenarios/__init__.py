"""Scenario registry for Cartest demos."""

from __future__ import annotations

import jax.numpy as jnp

from Cartest.execution.execute import FrenetState
from Cartest.planning.scenarios.empty import SCENARIO as EMPTY
from Cartest.planning.scenarios.single_offset import SCENARIO as SINGLE_OFFSET
from Cartest.planning.scenarios.three_blocking import SCENARIO as THREE_BLOCKING
from Cartest.planning.scenarios.circle_track import SCENARIO as CIRCLE_TRACK
from Cartest.planning.scenarios.lane_borrow_overtake import SCENARIO as LANE_BORROW_OVERTAKE
from Cartest.planning.scenarios.curved_cruise import SCENARIO as CURVED_CRUISE
from Cartest.planning.scenarios.game_2a_basic import SCENARIO as GAME_2A_BASIC
from Cartest.planning.scenarios.game_2b_constran import SCENARIO as GAME_2B_CONSTRAN
from Cartest.planning.scenarios.game_2b_asymmetric import SCENARIO as GAME_2B_ASYMMETRIC
from Cartest.planning.scenarios.three_agent_track import SCENARIO as THREE_AGENT_TRACK


SCENARIOS = {
    "empty": EMPTY,
    "single_offset": SINGLE_OFFSET,
    "three_blocking": THREE_BLOCKING,
    "circle_track": CIRCLE_TRACK,
    "lane_borrow_overtake": LANE_BORROW_OVERTAKE,
    "curved_cruise": CURVED_CRUISE,
    "game_2a_basic": GAME_2A_BASIC,
    "game_2b_constran": GAME_2B_CONSTRAN,
    "game_2b_asymmetric": GAME_2B_ASYMMETRIC,
    "three_agent_track": THREE_AGENT_TRACK,
}


def get_scenario(name: str):
    """Return a predefined scenario by name."""
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario {name!r}. Available: {available}") from exc


def scenario_kind(scenario):
    """Return the scenario execution kind."""
    return scenario.get("type", "single_agent")


def make_initial_state(scenario):
    """Build the initial FrenetState declared by a scenario."""
    ego = scenario["ego"]
    return FrenetState(
        s=ego["s"],
        s_dot=ego["s_dot"],
        s_ddot=ego["s_ddot"],
        d=ego["d"],
        d_dot=ego["d_dot"],
        d_ddot=ego["d_ddot"],
        psi=ego.get("psi", 0.0),
    )


def _parse_obstacles(scenario):
    """Return (static_list, dynamic_list) from scenario obstacles.

    Supports two formats:
      - list of {"x","y","r"}          -> all static (backward compat)
      - dict {"static": [...], "dynamic": [...]}
        dynamic items have extra "v" and "yaw" keys
    """
    obs = scenario["obstacles"]
    if isinstance(obs, dict):
        return list(obs.get("static", [])), list(obs.get("dynamic", []))
    return list(obs), []


def build_obstacle_predictions(scenario, gen, mpc_time=0.0):
    """Build per-sample time-varying circular obstacle predictions.

    Static obstacles are repeated for the whole horizon.  Dynamic obstacles
    use constant-velocity straight-line motion in Cartesian space.

    Returns:
        obs_pos: [T, N, 2] Cartesian positions per time step
        obs_rad: [T, N]    radii per time step
    """
    static_obs, dynamic_obs = _parse_obstacles(scenario)
    dt = gen.dt
    t = mpc_time + jnp.arange(gen.T, dtype=jnp.float32) * dt

    pos_series = []
    rad_series = []

    for obs in static_obs:
        x, y = float(obs["x"]), float(obs["y"])
        pos = jnp.stack([
            jnp.full_like(t, x),
            jnp.full_like(t, y),
        ], axis=-1)
        rad = jnp.full_like(t, float(obs["r"]))
        pos_series.append(pos)
        rad_series.append(rad)

    for obs in dynamic_obs:
        x0, y0 = float(obs["x"]), float(obs["y"])
        yaw = float(obs.get("yaw", 0.0))
        v = float(obs.get("v", 0.0))
        x = x0 + v * t * jnp.cos(yaw)
        y = y0 + v * t * jnp.sin(yaw)
        pos_series.append(jnp.stack([x, y], axis=-1))
        rad_series.append(jnp.full_like(t, float(obs["r"])))

    if not pos_series:
        return jnp.zeros((gen.T, 0, 2), dtype=jnp.float32), \
               jnp.zeros((gen.T, 0), dtype=jnp.float32)

    obs_pos = jnp.stack(pos_series, axis=1)   # [T, N, 2]
    obs_rad = jnp.stack(rad_series, axis=1)   # [T, N]
    return obs_pos, obs_rad


def build_obstacles(scenario, gen=None):
    """Extract static obstacle arrays from scenario (backward compat).

    Returns [T, N, 2] and [T, N] arrays (static only, repeated across T).
    Requires *gen* for horizon length; if None, falls back to [N, 2] / [N].
    """
    if gen is not None:
        return build_obstacle_predictions(scenario, gen)
    # Legacy: no gen -> return first-frame arrays
    static_obs, _ = _parse_obstacles(scenario)
    if not static_obs:
        return jnp.zeros((0, 2), dtype=jnp.float32), jnp.zeros(0, dtype=jnp.float32)
    obs_pos = jnp.array([[o["x"], o["y"]] for o in static_obs], dtype=jnp.float32)
    obs_rad = jnp.array([o["r"] for o in static_obs], dtype=jnp.float32)
    return obs_pos, obs_rad


def first_frame_obstacles(obs_pos, obs_rad):
    """Return plotting-friendly circular obstacles for the first horizon sample."""
    if obs_pos.ndim == 3:
        # [T, N, 2] -> first frame
        if obs_pos.shape[1] == 0:
            return []
        return [
            {"x": float(obs_pos[0, i, 0]), "y": float(obs_pos[0, i, 1]),
             "r": float(obs_rad[0, i])}
            for i in range(obs_pos.shape[1])
        ]
    # [N, 2] legacy
    if obs_pos.shape[0] == 0:
        return []
    return [
        {"x": float(obs_pos[i, 0]), "y": float(obs_pos[i, 1]),
         "r": float(obs_rad[i])}
        for i in range(obs_pos.shape[0])
    ]


__all__ = [
    "EMPTY", "SINGLE_OFFSET", "THREE_BLOCKING", "CIRCLE_TRACK",
    "LANE_BORROW_OVERTAKE",
    "CURVED_CRUISE",
    "GAME_2A_BASIC", "GAME_2B_CONSTRAN", "GAME_2B_ASYMMETRIC", "THREE_AGENT_TRACK",
    "SCENARIOS", "get_scenario", "make_initial_state",
    "scenario_kind",
    "build_obstacle_predictions", "build_obstacles", "first_frame_obstacles",
]
