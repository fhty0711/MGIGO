# Three-Agent Cost Design

## User Command

`python3 Cartest/Simple.py three_agent_track --steps 25`

## Call Chain

`Simple.py::run_multi_agent_game`
-> `build_cartest_nash_solver`
-> `_build_cartest_batched_solver`
-> `make_cartest_batched_rne_blocks_solver`
-> `batched_expected_costs_for_all_agents`
-> `three_agent_track_components`

## Cost Consistency

The three agents share component formulas and physical normalization scales.
They do **not** share identical final objectives, because ego is a
lane-change/overtake agent while front and rear are lane-keeping traffic
agents.

- ego: progress/speed tracking + target-lane merge + comfort + RSS risk
- front: upper-lane keeping + speed tracking + comfort + RSS risk against ego/rear
- rear: upper-lane keeping + speed tracking + comfort + RSS risk against ego/front

What is consistent across agents is the **measurement and safety language**
(lane boundary, speed/acceleration/jerk limits, forward motion, footprint
collision, RSS/CVaR risk, bridged-jerk comfort), not the final role preference.

## Components

Shared formulas live in
`Cartest/planning/costs/three_agent_track_components.py` and are consumed by
both the scalar Constran path (`three_agent_track.py`) and the batched path
(`three_agent_track_batched.py`), so the two cannot drift.

- objective tradeoffs: progress/speed target, lane target or lane keeping,
  bridged-jerk comfort, RSS CVaR
- raw feasibility diagnostics: speed, acceleration, jerk, forward, collision,
  lane boundary
- Constran layers, inner to outer: `objective -> speed -> kinematics -> forward -> safety_envelope`

```python
THREE_AGENT_CONSTRAINT_DEFS = (
    ("speed", "hard", 3, "max", "hard"),
    ("kinematics", "hard", 4, "max", "hard"),
    ("forward", "hard", 5, "max", "hard"),
    ("safety_envelope", "hard", 6, "max", "hard"),
)
```

`kinematics = max(acceleration, jerk)`.

`safety_envelope = max(collision, lane_boundary)`.

Lane centerline preference is an **objective** term. Road body-boundary is a
**feasibility** term. RSS, comfort, and progress are objective tradeoffs and are
never placed in outer Constran layers.

## Notes

- Per-timestep violation helpers (`*_violation_per_t`) are shape-agnostic over
  leading axes; the scalar `*_violation` aggregates reduce over time with `max`
  and are used directly as scalar Constran `g_fn` outputs (Constran's
  `_wrap_aggregate` applies `jnp.max`, which is the identity on an
  already-aggregated scalar). Both scalar and batched paths aggregate-then-transform,
  so scalar and batched nested costs are equivalent by construction.
- RSS/CVaR interaction risk is part of the objective. In the batched
  expected-cost (fast) path it is evaluated pairwise as `[B, M_inner]` against
  the background neighbour batches (mirroring the pairwise collision term), so
  the fast path matches the black-box scalar cost loop exactly. The full-plan
  validation path (`batched_nested_costs_from_plans`) evaluates RSS against the
  actual neighbour plans.
- `bridged_jerk_cost` bridges the horizon with the previous executed
  acceleration (`ctx["a_long_prev_a{idx}"]` / `["a_lat_prev_a{idx}"]`) when
  present and falls back to zero otherwise. The scalar path, batched fast path,
  and selected-plan postprocess all pass the same context through this term;
  the closed-loop runner does not yet fill those keys (shifted warmstart is a
  separate, later change).

## Validation

Exact commands used for this change:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/lenovo/mpcc/MGIGO/.venv/bin/python3 -m pytest \
  Cartest/eval/test_three_agent_track_components.py \
  Cartest/eval/test_batched_three_agent_eval.py \
  Cartest/eval/test_batched_rne_solver.py \
  Cartest/eval/test_game_solver_modes.py \
  Cartest/eval/test_unified_simple_runner.py -q

XLA_PYTHON_CLIENT_PREALLOCATE=false /home/lenovo/mpcc/MGIGO/.venv/bin/python3 Cartest/Simple.py three_agent_track --steps 20 --no-plot

XLA_PYTHON_CLIENT_PREALLOCATE=false /home/lenovo/mpcc/MGIGO/.venv/bin/python3 Cartest/Simple.py three_agent_track --steps 25
```

Expected results:

- pytest: all listed tests pass (58 tests).
- `--steps 20 --no-plot`: exits 0, prints 20 steps, no NaN/Inf; selected-plan
  collision and lane-boundary diagnostics stay near zero for executed states.
- `--steps 25`: exits 0 and writes a timestamped MP4 under `Cartest/output/`.
- Solve time stays in the same order as the prior batched solver (~150 ms/step).
