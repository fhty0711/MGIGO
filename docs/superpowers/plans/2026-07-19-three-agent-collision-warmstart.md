# Three-Agent Collision and Warm-Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the oversized three-agent hard collision envelope with explicit physical margins and give the three longitudinal GMM warm-start components target/current/yield semantics.

**Architecture:** `three_agent_track_components.py` owns one shared collision-clearance helper consumed by both scalar Constran and batched RNE evaluation. `build_multi_agent_warmstart` supports an opt-in semantic mode for this scenario while preserving legacy factor-based scenarios.

**Tech Stack:** Python 3, JAX, NumPy, pytest, Constran, Cartest B-spline/RNE, WSL2 CUDA.

## Global Constraints

- Hard clearances are exactly `vehicle_length + 0.5 m` longitudinally and `vehicle_width + 0.2 m` laterally.
- `safe_gap` is removed from the `three_agent_track` hard-collision configuration and is not routed into RSS.
- RSS equations and weights remain unchanged.
- The production solver remains `cartest_batched_rne_blocks`; scalar Constran is a parity oracle only.
- Three-agent semantic speeds are `max(current, target)`, `current`, and `max(v_min, current - 2.0)`.
- Other scenarios retaining `speed_factors` must preserve their current warm-start values.
- Every behavior change follows a failing-test, minimal-implementation, passing-test cycle.

---

### Task 1: Shared Hard-Collision Margins

**Files:**
- Modify: `Cartest/planning/scenarios/three_agent_track.py:35-40`
- Modify: `Cartest/planning/costs/three_agent_track_components.py:65-76,174-216`
- Modify: `Cartest/planning/costs/three_agent_track_batched.py:20-37,267-301`
- Modify: `Cartest/eval/test_three_agent_track_components.py`
- Modify: `Cartest/eval/test_batched_three_agent_eval.py:390-445`
- Modify: `Cartest/eval/test_unified_simple_runner.py:151-160`

**Interfaces:**
- Produces: `collision_clearances(scenario) -> tuple[float, float]` returning `(longitudinal_clearance, lateral_clearance)`.
- Consumes: `scenario["safety"]["vehicle_length"]`, `vehicle_width`, `collision_longitudinal_margin`, and `collision_lateral_margin`.

- [ ] **Step 1: Write failing shared-geometry tests**

Add imports and tests to `test_three_agent_track_components.py`:

```python
from Cartest.planning.costs.three_agent_track_components import (
    collision_clearances,
    collision_violation_per_t,
)

def test_collision_clearances_use_body_plus_axis_specific_margins():
    scenario = {"safety": {
        "vehicle_length": 5.0,
        "vehicle_width": 2.0,
        "collision_longitudinal_margin": 0.5,
        "collision_lateral_margin": 0.2,
    }}
    assert collision_clearances(scenario) == (5.5, 2.2)

def test_adjacent_lane_centers_do_not_trigger_hard_collision():
    scenario = get_scenario("three_agent_track")
    ego = _plan([0.0, 0.0], [0.0, 0.0])
    front = _plan([0.0, 0.0], [3.5, 3.5])
    far = _plan([100.0, 100.0], [3.5, 3.5])
    assert float(jnp.max(collision_violation_per_t(
        (ego, front, far), scenario, 0))) == 0.0

def test_partial_body_overlap_triggers_hard_collision():
    scenario = get_scenario("three_agent_track")
    ego = _plan([0.0, 0.0], [0.0, 0.0])
    overlap = _plan([5.4, 5.4], [2.1, 2.1])
    far = _plan([100.0, 100.0], [3.5, 3.5])
    assert float(jnp.max(collision_violation_per_t(
        (ego, overlap, far), scenario, 0))) > 0.0
```

Import `get_scenario` from `Cartest.planning.scenarios` in the same test module.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
python -m pytest Cartest/eval/test_three_agent_track_components.py -q
```

Expected: collection fails because `collision_clearances` does not exist, or geometry assertions fail under the old `safe_gap` envelope.

- [ ] **Step 3: Implement the shared clearance helper and scenario fields**

Replace `collision_lateral_clearance` with this helper in `three_agent_track_components.py`:

```python
def collision_clearances(scenario):
    """Expanded physical-body center clearances for hard collision checks."""
    safety = scenario["safety"]
    longitudinal = (
        float(safety.get("vehicle_length", 5.0))
        + float(safety.get("collision_longitudinal_margin", 0.5))
    )
    lateral = (
        float(safety.get("vehicle_width", 2.0))
        + float(safety.get("collision_lateral_margin", 0.2))
    )
    return longitudinal, lateral
```

In `collision_violation_per_t`, replace the local `vehicle_length/safe_gap` arithmetic with:

```python
longitudinal_clearance, lateral_clearance = collision_clearances(scenario)
```

In `three_agent_track.py`, replace `safe_gap` with:

```python
"collision_longitudinal_margin": 0.5,
"collision_lateral_margin": 0.2,
```

- [ ] **Step 4: Route the batched pairwise evaluator through the same helper**

In `three_agent_track_batched.py`, import `collision_clearances`, remove `collision_lateral_clearance`, and replace lines 269-272 with:

```python
longitudinal_clearance, lateral_clearance = collision_clearances(scenario)
```

No other collision arithmetic may remain in this module.

- [ ] **Step 5: Update boundary/parity regression tests**

Replace the old point-distance assertion in `test_three_agent_collision_uses_vehicle_footprint_not_point_distance` with direct threshold cases using `5.5` and `2.2`. Update `test_three_agent_track_initial_spacing_is_safe` to compute:

```python
from Cartest.planning.costs.three_agent_track_components import collision_clearances

longitudinal_clearance, _ = collision_clearances(scenario)
ego, front, rear = scenario["agents"]
assert ego["s"] - rear["s"] >= longitudinal_clearance
assert "safe_gap" not in scenario["safety"]
```

Keep the existing scalar-vs-batched nested-cost test unchanged so it verifies the shared semantics end to end.

- [ ] **Step 6: Run collision and parity tests**

Run:

```powershell
python -m pytest Cartest/eval/test_three_agent_track_components.py Cartest/eval/test_batched_three_agent_eval.py Cartest/eval/test_unified_simple_runner.py -q
```

Expected: all tests pass; no reference to `scenario["safety"]["safe_gap"]` remains in three-agent tests or implementation.

- [ ] **Step 7: Commit the collision change**

```powershell
git add Cartest/planning/scenarios/three_agent_track.py Cartest/planning/costs/three_agent_track_components.py Cartest/planning/costs/three_agent_track_batched.py Cartest/eval/test_three_agent_track_components.py Cartest/eval/test_batched_three_agent_eval.py Cartest/eval/test_unified_simple_runner.py
git commit -m "fix(cartest): separate collision margins from RSS gap"
```

---

### Task 2: Semantic Longitudinal GMM Warm-Start

**Files:**
- Modify: `Cartest/planning/scenarios/three_agent_track.py:38-40`
- Modify: `Cartest/planning/solver_modes.py:315-345`
- Modify: `Cartest/eval/test_game_solver_modes.py`

**Interfaces:**
- Produces: `_warmstart_longitudinal_speeds(scenario, state, k_count) -> jnp.ndarray`.
- Consumes: optional `scenario["warmstart"]`, per-agent `v_target`, and safety `v_min`.
- Preserves: legacy `scenario["behavior"]["speed_factors"]` behavior.

- [ ] **Step 1: Write failing semantic and compatibility tests**

Add to `test_game_solver_modes.py`:

```python
from Cartest.planning.solver_modes import _warmstart_longitudinal_speeds

def test_three_agent_semantic_warmstart_uses_target_current_yield():
    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    states = _states(scenario)
    speeds = _warmstart_longitudinal_speeds(scenario, states[0], 3, agent_idx=0)
    assert jnp.allclose(speeds, jnp.array([17.5, 15.0, 13.0]))

def test_semantic_warmstart_rejects_non_three_component_layout():
    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    with pytest.raises(ValueError, match="requires K=3"):
        _warmstart_longitudinal_speeds(scenario, _states(scenario)[0], 4, agent_idx=0)

def test_legacy_speed_factors_remain_unchanged():
    scenario = copy.deepcopy(get_scenario("game_2a_basic"))
    state = _states(scenario)[0]
    expected = state.s_dot * jnp.asarray(scenario["behavior"]["speed_factors"])
    actual = _warmstart_longitudinal_speeds(scenario, state, 3, agent_idx=0)
    assert jnp.allclose(actual, expected)
```

Use the existing scenario/state helpers in that test module; add `copy`, `pytest`, and `jax.numpy as jnp` imports only if absent.

- [ ] **Step 2: Run the focused warm-start tests and verify failure**

Run:

```powershell
python -m pytest Cartest/eval/test_game_solver_modes.py -k "semantic_warmstart or legacy_speed_factors" -q
```

Expected: collection fails because `_warmstart_longitudinal_speeds` does not exist.

- [ ] **Step 3: Add the opt-in scenario configuration**

Remove `speed_factors` from the three-agent `behavior` dictionary and add:

```python
"warmstart": {
    "longitudinal_modes": "target_current_yield",
    "yield_delta": 2.0,
},
```

Keep all three agent `v_target` values at `17.5`.

- [ ] **Step 4: Implement semantic and legacy speed selection**

Add before `build_multi_agent_warmstart`:

```python
def _warmstart_longitudinal_speeds(
        scenario, state, k_count, *, agent_idx):
    cfg = scenario.get("warmstart", {})
    if cfg.get("longitudinal_modes") == "target_current_yield":
        if k_count != 3:
            raise ValueError("target_current_yield warmstart requires K=3")
        current = float(state.s_dot)
        target = float(scenario["agents"][agent_idx]["v_target"])
        v_min = float(scenario.get("safety", {}).get("v_min", 0.0))
        yield_delta = float(cfg.get("yield_delta", 2.0))
        return jnp.asarray([
            max(current, target),
            current,
            max(v_min, current - yield_delta),
        ], dtype=jnp.float32)

    speed_factors = jnp.asarray(
        scenario.get("behavior", {}).get(
            "speed_factors", jnp.linspace(1.15, 0.85, k_count)))
    if speed_factors.shape[0] != k_count:
        speed_factors = jnp.linspace(1.15, 0.85, k_count)
    return float(state.s_dot) * speed_factors
```

In `build_multi_agent_warmstart`, compute this array once per block owner and replace:

```python
speed = float(state.s_dot) * speed_factors[comp_idx]
```

with:

```python
mode_speeds = _warmstart_longitudinal_speeds(
    scenario, state, k_count, agent_idx=agent_idx)
speed = mode_speeds[comp_idx]
```

Do not alter the random-key split order or the existing `0.1 * random.normal` addition, preserving the established random stream.

- [ ] **Step 5: Run warm-start and solver-mode tests**

Run:

```powershell
python -m pytest Cartest/eval/test_game_solver_modes.py Cartest/eval/test_batched_three_agent_eval.py -q
```

Expected: all tests pass, including exact legacy factor values and three-agent semantic values.

- [ ] **Step 6: Commit the warm-start change**

```powershell
git add Cartest/planning/scenarios/three_agent_track.py Cartest/planning/solver_modes.py Cartest/eval/test_game_solver_modes.py
git commit -m "feat(cartest): add semantic game warmstart modes"
```

---

### Task 3: Behavior Regression and CUDA Rollout

**Files:**
- No tracked source changes expected.
- Generated artifacts: `Cartest/output/t300_150_analysis/` (ignored by Git).

**Interfaces:**
- Consumes: updated scenario, collision helper, warm-start builder, and existing `D:/MGIGO/t300_150_full_analysis.py` diagnostic runner.
- Produces: fresh JSON/NPZ/MP4 evidence for the behavior change.

- [ ] **Step 1: Run the full related CPU suite**

Run:

```powershell
python -m pytest Cartest/eval/test_three_agent_track_components.py Cartest/eval/test_batched_three_agent_eval.py Cartest/eval/test_game_solver_modes.py Cartest/eval/test_three_agent_track_game.py Cartest/eval/test_unified_simple_runner.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run one CUDA first-step diagnostic**

Run in WSL2:

```bash
cd /mnt/d/MGIGO/MGIGO
MGIGO_ROOT=/mnt/d/MGIGO/MGIGO XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/.venvs/tcmgigo-jaxgpu/bin/python /mnt/d/MGIGO/diagnose_initial_behaviors.py
```

Expected for the original updated scenario: first executed ego `d >= -0.02`, zero hard-collision violation, and no hard-envelope-forced rear braking comparable to the old `14.42 m/s` first step.

- [ ] **Step 3: Run `T=300`, 150 MPC steps on CUDA**

Run:

```bash
cd /mnt/d/MGIGO/MGIGO
MGIGO_ROOT=/mnt/d/MGIGO/MGIGO XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/.venvs/tcmgigo-jaxgpu/bin/python /mnt/d/MGIGO/t300_150_full_analysis.py
```

Expected: lane change completes and remains stable; every hard constraint maximum is zero; timing summary reports only the solver call.

- [ ] **Step 4: Compare before/after behavior numerically**

Record from the new JSON alongside the previous `20260719_002337` baseline:

```text
first 10 ego d values
ego/front/rear speed minima
lane completion step
post-completion target error
executed jerk RMS/max
replan RMS/max
constraint maxima
median and P95 solve_ms
```

Expected: initial negative ego excursion is removed or stays within `-0.02 m`; collision remains zero; any rear yielding is attributable to nonzero RSS/objective trade-offs rather than the old 3.5 m lateral hard envelope.
