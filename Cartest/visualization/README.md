# Cartest Visualization

This folder contains the shared visualization stack for Cartest MPC demos.
Keep drawing code here instead of adding Matplotlib logic inside experiment or
evaluation scripts.

## Module Layout

- `scene_renderer.py`
  - Low-level Matplotlib layer renderer.
  - Knows how to draw `RectLayer`, `CircleLayer`, `PolygonLayer`, `LineLayer`,
    `VehicleLayer`, and `TextLayer`.
  - Does not import planners, scenarios, costs, or reports.

- `frenet_scene.py`
  - Adapter from Cartest data to renderer-ready Cartesian scene data.
  - Converts executed history from Frenet `s/d` to Cartesian `x/y` through the
    scenario `ref_path`.
  - Samples road boundaries and lane centerlines from the same `ref_path`, so
    straight, circular, and piecewise-curvature scenarios use one code path.

- `frenet_renderer.py`
  - Draws a `FrenetScene` as a top-down road panel.
  - Responsible for visual styling: road surface, lane boundaries, ego vehicle,
    planned path, executed history, obstacle envelopes, and status text.

- `plotting.py`
  - High-level animation helpers used by demos: `setup_axes()`,
    `render_frame()`, and `save_animation()`.
  - Also renders the kinematics subplot next to the top-down trajectory panel.
  - Saves MP4 videos through ffmpeg.

New code should import plotting helpers from `Cartest.visualization.plotting`.
Do not add new Matplotlib entry points under `Cartest/eval`.

## Coordinate Contract

`StepReport` currently stores mixed coordinate data:

- `hx`, `hy`: executed history in Frenet coordinates (`s`, `d`).
- `px`, `py`: planned path in Cartesian coordinates (`x`, `y`).

Only `frenet_scene.py` should handle this conversion. Renderers should consume
Cartesian data only. This prevents curved-road bugs where a circular or clothoid
scenario is accidentally drawn as a straight road.

## Standard Usage

For demos, use the high-level helpers:

```python
from Cartest.visualization.plotting import setup_axes, render_frame, save_animation

fig, ax_t, ax_k = setup_axes()
save_animation(
    fig,
    reports,
    lambda i: render_frame(
        ax_t,
        ax_k,
        reports[i],
        obs_list,
        safe_dist,
        gen.dt,
        scenario=scenario,
    ),
    ROOT / "output" / "scenario_YYYYMMDD_HHMMSS.mp4",
)
```

For a single top-down panel:

```python
from Cartest.visualization.frenet_renderer import render_frenet_panel

render_frenet_panel(
    ax,
    report,
    scenario=scenario,
    obstacles=obs_list,
    obs_safe_dist=safe_dist,
)
```

## Adding A Cartest Scenario

A scenario works with this visualization stack when it provides:

- `scenario["ref_path"]` with `frenet_to_cartesian()` and `cartesian_to_frenet()`.
- `scenario["road"]` with either `lane_hw` or `lane_bounds_d`.
- `StepReport.hx/hy` as Frenet executed history and `StepReport.px/py` as
  Cartesian planned trajectory.

No renderer changes should be needed for new straight, circular, or curved
Cartest scenarios that follow this contract.

## Validation

Run these checks after visualization changes:

```bash
.venv/bin/python -m py_compile \
  Cartest/visualization/scene_renderer.py \
  Cartest/visualization/frenet_scene.py \
  Cartest/visualization/frenet_renderer.py \
  Cartest/visualization/game_renderer.py \
  Cartest/visualization/plotting.py \
  Cartest/Simple.py \
  Cartest/eval/test_visualization_renderer.py
.venv/bin/python Cartest/eval/test_visualization_renderer.py
.venv/bin/python Cartest/Simple.py curved_cruise --steps 1
```

The tests intentionally cover all registered `Cartest.planning.scenarios`
entries and verify that circular-road history is transformed through the
reference path instead of being drawn as raw `s/d`.

## Multi-Agent Game Rendering

Multi-agent game demos (`game_2a_basic`, `game_2b_constran`,
`three_agent_track`) render through `Cartest.visualization.game_renderer`.
The unified runner passes a list of report dicts with:

- `history_xy`: array shaped `[closed_loop_step, agent, xy]`.
- `predicted_xy`: list of per-agent predicted Cartesian trajectories.
- `agent_names`: display names taken from the scenario.
- `pi`: per-agent selected mixture probabilities.
- `solve_ms`: RNE solve time in milliseconds.

Game demos must not define local Matplotlib animation functions. Keep all
shared game plotting in `game_renderer.py` so every game scenario renders
the same way through `Simple.py <scenario>`.

Videos are written to `Cartest/output/` with timestamped filenames such as
`three_agent_track_20260716_120304.mp4`. The output directory is ignored by
git and can be cleared freely.

## Multi-Agent Coordinate Contract

Current Cartest game scenarios optimize every agent in B-spline Frenet
coordinates relative to the scenario's shared `ref_path`. Each agent owns its
own `(s, d)` control points and evaluates its own self cost from its own
trajectory, but `game_2a_basic`, `game_2b_constran`, and `three_agent_track`
all use the same reference path object for those Frenet coordinates.

This is intentional for straight-road multi-lane demos: ego, front, and rear
vehicles share one road-aligned longitudinal coordinate, so relative lane and
longitudinal relationships are well-defined in the common frame. Collision and
clearance terms should still convert each trajectory to Cartesian before
computing physical pairwise distances.

For future scenarios where agents follow different lanes, branches, ramps, or
intersection approaches, this contract must be upgraded. The planned structure
is per-agent reference frames: optimize agent `i` in `ref_path_i` Frenet,
compute self costs in that local Frenet frame, convert all plans to Cartesian
for physical collision constraints, and when an agent needs another vehicle in
its own perspective, project the other Cartesian trajectory through
`ref_path_i.cartesian_to_frenet(x, y)`. That is not required for the current
`three_agent_track` straight-road scene.

## Three-Agent Batched RNE Rendering

`three_agent_track` uses the Cartest-specific `cartest_batched_rne_blocks`
solver mode.  The mode keeps the same `T`, `B`, and `M_inner` scale parameters
as the generic RNE setup, but evaluates B-spline trajectories outside the
`B x M_inner` candidate/background pairing loop.  The pairwise game costs are
then computed by broadcasting cached ego/front/rear plans.

The current cost remains expressed in the shared straight-road Frenet frame.
For curved or agent-relative scenarios, each agent should eventually evaluate
its objective in its own reference/Frenet view; that coordinate-system upgrade
is documented as future work and is intentionally not mixed into the batching
change.
