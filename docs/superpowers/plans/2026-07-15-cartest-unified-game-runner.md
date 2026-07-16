# Cartest Unified Game Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify Cartest single-agent MPC demos and multi-agent RNE game demos behind `Cartest/Simple.py`, while keeping scenarios, costs, solver construction, and visualization in the existing Cartest folders.

**Architecture:** `Cartest/Simple.py` becomes the only command runner. It dispatches by `scenario["type"]`: existing single-agent scenarios continue through the current MPC path, and game scenarios go through Cartest-level Nash/RNE helpers in `Cartest/planning/solver_modes.py`. RNE blocks are managed through `gmm_igo.solver_builder.build_nash_solver()` so Cartest demos no longer import `MPC_G_MS` directly.

**Tech Stack:** Python, JAX, NumPy, Matplotlib, Cartest Frenet B-spline utilities, `gmm_igo.solver_builder`, `Constraintdealer.Constran`.

---

## Scope Rules

- Do not create `Cartest/game/` or another parallel architecture.
- Put game scenarios in `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/`.
- Put game cost modules in `/home/lenovo/mpcc/MGIGO/Cartest/planning/costs/`.
- Put solver adaptation in `/home/lenovo/mpcc/MGIGO/Cartest/planning/solver_modes.py`.
- Put all game plotting in `/home/lenovo/mpcc/MGIGO/Cartest/visualization/`.
- Keep `Cartest/Simple.py` as the sole runner after migration.
- Delete `Cartest/Simple_game_2a.py` and `Cartest/Simple_game_2b_constran.py` only after the unified runner passes smoke tests.
- Ignore sandbox CUDA/JAX plugin warnings when commands exit zero; do not infer host GPU availability from sandbox logs.
- Do not revert unrelated worktree changes.

## Target Files

Create:

- `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/game_2a_basic.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/game_2b_constran.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/three_agent_track.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/planning/costs/game_2a_basic.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/planning/costs/game_2b_constran.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/planning/costs/three_agent_track.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/visualization/game_renderer.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_unified_simple_runner.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_game_solver_modes.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_three_agent_track_game.py`

Modify:

- `/home/lenovo/mpcc/MGIGO/gmm_igo/solver_builder.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/Simple.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/__init__.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/planning/costs/registry.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/planning/solver_modes.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/visualization/README.md`
- `/home/lenovo/mpcc/MGIGO/Cartest/carreadme.md`

Delete after replacement is verified:

- `/home/lenovo/mpcc/MGIGO/Cartest/Simple_game_2a.py`
- `/home/lenovo/mpcc/MGIGO/Cartest/Simple_game_2b_constran.py`

---

### Task 1: Expose Block-Level Nash Outputs

**Files:**
- Modify: `/home/lenovo/mpcc/MGIGO/gmm_igo/solver_builder.py`
- Create/modify test: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_game_solver_modes.py`

- [ ] **Step 1: Write failing diagnostic test**

Create `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_game_solver_modes.py` with a test that builds a tiny `build_nash_solver(..., solver="rne_blocks", block_to_agent=(0, 0, 1, 1), T=2, K=2, B=4, B0=2, M_inner=1)` problem and asserts:

```python
assert result.joint_x.shape == (4,)
assert result.per_agent_cost.shape == (2,)
assert result.diag["mu"].shape == (4, 2, 1)
assert result.diag["S_or_L"].shape == (4, 2, 1, 1)
assert result.diag["pi"].shape == (4, 2)
assert tuple(result.diag["block_to_agent"]) == (0, 0, 1, 1)
```

- [ ] **Step 2: Verify failure**

Run from `/home/lenovo/mpcc/MGIGO`:

```bash
.venv/bin/python Cartest/eval/test_game_solver_modes.py
```

Expected: failure for missing `diag` entries.

- [ ] **Step 3: Add diagnostics to `build_nash_solver()`**

In `/home/lenovo/mpcc/MGIGO/gmm_igo/solver_builder.py`, initialize `_metrics = None` before the solver branch inside `_solve()`. Change the final `return NashResult(...)` to include:

```python
diag={
    "mu": final_mu,
    "S_or_L": final_L,
    "pi": final_pi,
    "v": final_v,
    "block_to_agent": tuple(block_to_agent),
    "dims": tuple(dims),
    "metrics": _metrics,
}
```

- [ ] **Step 4: Verify pass**

Run:

```bash
.venv/bin/python Cartest/eval/test_game_solver_modes.py
```

Expected: the diagnostic test passes.

---

### Task 2: Register Game Scenarios In Existing Scenario Registry

**Files:**
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/game_2a_basic.py`
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/game_2b_constran.py`
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/three_agent_track.py`
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/__init__.py`
- Create/modify test: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_unified_simple_runner.py`

- [ ] **Step 1: Write failing registry test**

Create `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_unified_simple_runner.py` with tests that import `SCENARIOS`, `get_scenario`, and `scenario_kind`, then assert:

```python
for name in ("game_2a_basic", "game_2b_constran", "three_agent_track"):
    scenario = get_scenario(name)
    assert name in SCENARIOS
    assert scenario_kind(scenario) == "multi_agent_game"
    assert "agents" in scenario
    assert "game" in scenario
    assert "block_to_agent" in scenario["game"]

assert scenario_kind(get_scenario("empty")) == "single_agent"
assert scenario_kind(get_scenario("lane_borrow_overtake")) == "single_agent"
```

- [ ] **Step 2: Verify failure**

Run:

```bash
.venv/bin/python Cartest/eval/test_unified_simple_runner.py
```

Expected: missing `scenario_kind` or unknown scenarios.

- [ ] **Step 3: Create `game_2a_basic.py` scenario**

Use `type="multi_agent_game"`, `StraightReference()`, two agents, `block_layout=(agent0_s, agent0_d, agent1_s, agent1_d)`, `block_to_agent=(0, 0, 1, 1)`, and solver config copied from `Simple_game_2a.py`: `K=3`, `B=64`, `B0=20`, `T=200`, `dt=0.15`, `T_0=100`, `M_inner=30`. Use `cost.name="game_2a_basic"`.

- [ ] **Step 4: Create `game_2b_constran.py` scenario**

Use `type="multi_agent_game"`, two agents in the symmetric collision setup from `Simple_game_2b_constran.py`, the same 4-block layout, and `cost.name="game_2b_constran"`.

- [ ] **Step 5: Create `three_agent_track.py` scenario**

Use `type="multi_agent_game"`, `StraightReference()`, agents:

```python
ego:   s=15.0, d=0.0, s_dot=15.0, v_target=17.5
front: s=17.0, d=3.5, s_dot=10.0, v_target=20.0
rear:  s=13.0, d=3.5, s_dot=10.0, v_target=17.5
```

Use six blocks:

```python
block_layout=("ego_s", "ego_d", "front_s", "front_d", "rear_s", "rear_d")
block_to_agent=(0, 0, 1, 1, 2, 2)
```

Use Trackgame-like solver config: `K=3`, `B=60`, `B0=25`, `T=300`, `dt=0.15`, `T_0=300`, `M_inner=30`. Use `cost.name="three_agent_track"`.

- [ ] **Step 6: Register scenarios and scenario kind**

In `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/__init__.py`, import the three new scenario constants, add them to `SCENARIOS`, add:

```python
def scenario_kind(scenario):
    """Return the scenario execution kind."""
    return scenario.get("type", "single_agent")
```

Update `__all__`.

- [ ] **Step 7: Verify pass**

Run:

```bash
.venv/bin/python Cartest/eval/test_unified_simple_runner.py
```

Expected: registry tests pass.

---

### Task 3: Move Game Costs Into `planning/costs`

**Files:**
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/planning/costs/game_2a_basic.py`
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/planning/costs/game_2b_constran.py`
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/planning/costs/three_agent_track.py`
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/planning/costs/registry.py`
- Modify test: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_game_solver_modes.py`

- [ ] **Step 1: Write failing game-cost test**

Extend `test_game_solver_modes.py` with a test that loops over `game_2a_basic`, `game_2b_constran`, and `three_agent_track`, builds `FrenetBSplineTrajectory`, calls `make_agent_specs_from_scenario(gen, scenario)`, evaluates every objective on a zero joint vector, and asserts each value is finite.

- [ ] **Step 2: Verify failure**

Run:

```bash
.venv/bin/python Cartest/eval/test_game_solver_modes.py
```

Expected: missing `make_agent_specs_from_scenario` or missing cost modules.

- [ ] **Step 3: Implement shared local helpers in cost modules**

Each game cost module should locally provide helpers equivalent to:

```python
def _agent_ctx(ctx, agent_idx):
    return {
        "s0": ctx[f"s0_a{agent_idx}"],
        "s_dot0": ctx[f"s_dot0_a{agent_idx}"],
        "s_ddot0": ctx.get(f"s_ddot0_a{agent_idx}", 0.0),
        "d0": ctx[f"d0_a{agent_idx}"],
        "d_dot0": ctx.get(f"d_dot0_a{agent_idx}", 0.0),
        "d_ddot0": ctx.get(f"d_ddot0_a{agent_idx}", 0.0),
    }

def _theta_for_agent(joint_x, agent_idx, n_free):
    base = agent_idx * 2 * n_free
    return joint_x[base:base + 2 * n_free]

def _eval_agent_plan(gen, joint_x, ctx, agent_idx):
    theta = _theta_for_agent(joint_x, agent_idx, gen.n_free)
    return gen.evaluate_plan(theta[:gen.n_free], theta[gen.n_free:], _agent_ctx(ctx, agent_idx))
```

- [ ] **Step 4: Implement `game_2a_basic.make_agent_specs()`**

Move the Lyapunov-style per-agent objective from `Simple_game_2a.py` into `make_agent_specs(gen, scenario) -> {agent_id: (objective, [])}`. It should use scenario agent `v_target`, scenario behavior `target_d`, and B-spline Frenet errors.

- [ ] **Step 5: Implement `game_2b_constran.make_agent_specs()`**

Move the Constran collision-game logic from `Simple_game_2b_constran.py` into `make_agent_specs(gen, scenario)`. Return bare objectives plus `Deterministic` constraints for lane and collision, using `mode="hard"` for collision.

- [ ] **Step 6: Implement `three_agent_track.make_agent_specs()`**

Implement Trackgame-inspired B-spline objectives:

- Ego: speed tracking, target upper lane, lateral velocity/acceleration, jerk, full-horizon collision with front/rear.
- Front: speed tracking and short-horizon ego collision response.
- Rear: speed tracking, short-horizon ego collision response, front/rear longitudinal safety barrier, overtake punishment.

Use Cartesian trajectories from `gen.evaluate_plan(...)` for distance-based collision penalties.

- [ ] **Step 7: Add game-cost registry helper**

In `Cartest/planning/costs/registry.py`, add a mapping:

```python
GAME_AGENT_SPEC_FACTORIES = {
    "game_2a_basic": game_2a_basic_cost.make_agent_specs,
    "game_2b_constran": game_2b_constran_cost.make_agent_specs,
    "three_agent_track": three_agent_track_cost.make_agent_specs,
}
```

Add:

```python
def make_agent_specs_from_scenario(gen, scenario):
    """Build per-agent objective/constraint specs for a multi-agent game scenario."""
    cost_name, _params, _factory = get_cost_spec(scenario)
    try:
        return GAME_AGENT_SPEC_FACTORIES[cost_name](gen, scenario)
    except KeyError as exc:
        available = ", ".join(sorted(GAME_AGENT_SPEC_FACTORIES))
        raise ValueError(f"Unknown game cost {cost_name!r}. Available: {available}") from exc
```

- [ ] **Step 8: Verify pass**

Run:

```bash
.venv/bin/python Cartest/eval/test_game_solver_modes.py
```

Expected: game cost tests pass.

---

### Task 4: Add Cartest Nash Helpers To `solver_modes.py`

**Files:**
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/planning/solver_modes.py`
- Modify test: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_game_solver_modes.py`

- [ ] **Step 1: Write failing solver-mode test**

Extend `test_game_solver_modes.py` to import and test:

```python
build_cartest_nash_solver
build_multi_agent_context
build_multi_agent_warmstart
select_nash_plan
```

For `game_2a_basic`, assert:

```python
ctx["s0_a0"] == scenario["agents"][0]["s"]
mu.shape == (4, scenario["game"]["K"], gen.n_free)
L_inv.shape == (4, scenario["game"]["K"], gen.n_free, gen.n_free)
len(select_nash_plan(result, scenario)) == 2
```

Use a small copied scenario config with `T=2`, `B=4`, `B0=2`, `M_inner=1`.

- [ ] **Step 2: Verify failure**

Run:

```bash
.venv/bin/python Cartest/eval/test_game_solver_modes.py
```

Expected: missing helper imports.

- [ ] **Step 3: Implement `build_multi_agent_context(states)`**

Add to `solver_modes.py`: build `s0_a{i}`, `s_dot0_a{i}`, `s_ddot0_a{i}`, `d0_a{i}`, `d_dot0_a{i}`, `d_ddot0_a{i}` from a list of `FrenetState`.

- [ ] **Step 4: Implement `build_multi_agent_warmstart(gen, scenario, states, key)`**

Add block-wise warm-start using `scenario["game"]["block_layout"]`, `scenario["game"]["block_to_agent"]`, `scenario["game"]["K"]`, `gen.greville[2:gen.n_ctrl]`, and scenario `behavior.speed_factors`. Return `(mu, L_inv)` shaped `(N_blocks, K, gen.n_free)` and `(N_blocks, K, gen.n_free, gen.n_free)`.

- [ ] **Step 5: Implement `build_cartest_nash_solver(gen, scenario)`**

Inside `solver_modes.py`, call `make_agent_specs_from_scenario(gen, scenario)` and then `build_nash_solver(...)` with:

```python
dims=(gen.n_free,) * len(scenario["game"]["block_layout"])
solver=scenario["game"].get("solver", "rne_blocks")
block_to_agent=tuple(scenario["game"]["block_to_agent"])
T, dt, K, B, B0, T_0, M_inner from scenario["game"]
```

- [ ] **Step 6: Implement `select_nash_plan(result, scenario)`**

Decode `result.diag["mu"]` and `result.diag["pi"]` into per-agent dictionaries:

```python
{
    "agent_idx": agent_idx,
    "ctrl_s": selected_s_block,
    "ctrl_d": selected_d_block,
    "best_components": (best_s, best_d),
    "pi_s": pi[s_block],
    "pi_d": pi[d_block],
}
```

Require each agent to own exactly two blocks.

- [ ] **Step 7: Verify pass**

Run:

```bash
.venv/bin/python Cartest/eval/test_game_solver_modes.py
```

Expected: all solver-mode tests pass.

---

### Task 5: Add Shared Multi-Agent Renderer

**Files:**
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/visualization/game_renderer.py`
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/visualization/README.md`
- Modify test: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_unified_simple_runner.py`

- [ ] **Step 1: Write failing renderer smoke test**

Extend `test_unified_simple_runner.py` with a test that imports `save_game_animation`, builds one fake report with `history_xy`, `predicted_xy`, `agent_names`, `pi`, and `solve_ms`, writes to `tmp_path / "game.gif"`, and asserts the file exists and has nonzero size.

- [ ] **Step 2: Verify failure**

Run:

```bash
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/eval/test_unified_simple_runner.py
```

Expected: missing `Cartest.visualization.game_renderer`.

- [ ] **Step 3: Implement `game_renderer.py`**

Create renderer functions:

```python
def render_game_frame(ax, report, road=None):
    """Render one multi-agent game report."""

def save_game_animation(reports, output_path, road=None, fps=5):
    """Save a multi-agent game rollout animation."""
```

Use Matplotlib Agg, lane center lines from `road["lane_centers_d"]`, per-agent colors, history lines, current markers, predicted dashed paths, and compact `pi` labels.

- [ ] **Step 4: Document game rendering**

Add a `Multi-Agent Game Rendering` section to `/home/lenovo/mpcc/MGIGO/Cartest/visualization/README.md` explaining expected report keys and requiring all game demos to use `Cartest.visualization.game_renderer`.

- [ ] **Step 5: Verify pass**

Run:

```bash
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/eval/test_unified_simple_runner.py
```

Expected: renderer smoke test passes.

---

### Task 6: Refactor `Simple.py` Into Unified Dispatcher

**Files:**
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/Simple.py`
- Modify test: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_unified_simple_runner.py`

- [ ] **Step 1: Write failing dispatcher test**

Extend `test_unified_simple_runner.py`:

```python
import Cartest.Simple as simple

assert callable(simple.run_single_agent_mpc)
assert callable(simple.run_multi_agent_game)
assert callable(simple.run)
```

- [ ] **Step 2: Verify failure**

Run:

```bash
.venv/bin/python Cartest/eval/test_unified_simple_runner.py
```

Expected: missing `run_single_agent_mpc` or `run_multi_agent_game`.

- [ ] **Step 3: Rename existing `run()` body**

In `Cartest/Simple.py`, rename the current single-agent implementation from:

```python
def run(steps=150, seed=0, plot=True, scenario_name="empty"):
```

to:

```python
def run_single_agent_mpc(steps=150, seed=0, plot=True, scenario_name="empty"):
```

Do not change its internal behavior in this step.

- [ ] **Step 4: Add multi-agent runner imports**

Add imports for `scenario_kind`, `build_cartest_nash_solver`, `build_multi_agent_context`, `build_multi_agent_warmstart`, `select_nash_plan`, and `save_game_animation`.

- [ ] **Step 5: Add `_make_agent_states(scenario)`**

Build a list of `FrenetState` from `scenario["agents"]`.

- [ ] **Step 6: Add `run_multi_agent_game()`**

Implement closed-loop execution for game scenarios:

1. Load scenario and `FrenetBSplineTrajectory`.
2. Build `solver = build_cartest_nash_solver(gen, scenario)`.
3. For each step, build context and warm-start.
4. Call unified Nash solver.
5. Decode selected plans with `select_nash_plan()`.
6. Evaluate each plan with `gen.evaluate_plan()`.
7. Execute each agent with `execute_perfect_tracking()`.
8. Store reports for `game_renderer`.
9. Print compact per-step status.
10. Save animation when `plot=True`.

- [ ] **Step 7: Add unified `run()` dispatcher**

Implement:

```python
def run(steps=150, seed=0, plot=True, scenario_name="empty"):
    scenario = get_scenario(scenario_name)
    kind = scenario_kind(scenario)
    if kind == "single_agent":
        return run_single_agent_mpc(steps=steps, seed=seed, plot=plot, scenario_name=scenario_name)
    if kind == "multi_agent_game":
        return run_multi_agent_game(steps=steps, seed=seed, plot=plot, scenario_name=scenario_name)
    raise ValueError(f"Unknown scenario type {kind!r} for scenario {scenario_name!r}")
```

Keep CLI parsing pointed at `run()`.

- [ ] **Step 8: Verify pass**

Run:

```bash
.venv/bin/python Cartest/eval/test_unified_simple_runner.py
```

Expected: dispatcher tests pass.

---

### Task 7: Add Three-Agent One-Step Smoke Test

**Files:**
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_three_agent_track_game.py`

- [ ] **Step 1: Write one-step test**

Create a test that loads `three_agent_track`, builds a small copied scenario with `T=2`, `B=4`, `B0=2`, `M_inner=1`, constructs states from `scenario["agents"]`, builds context and warm-start, runs `build_cartest_nash_solver()`, decodes plans, and asserts:

```python
assert len(plans) == 3
assert result.per_agent_cost.shape == (3,)
assert jnp.all(jnp.isfinite(result.per_agent_cost))
```

- [ ] **Step 2: Verify pass**

Run:

```bash
.venv/bin/python Cartest/eval/test_three_agent_track_game.py
```

Expected: one small RNE solve passes.

---

### Task 8: Remove Old Game Demo Entrypoints

**Files:**
- Delete: `/home/lenovo/mpcc/MGIGO/Cartest/Simple_game_2a.py`
- Delete: `/home/lenovo/mpcc/MGIGO/Cartest/Simple_game_2b_constran.py`
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/carreadme.md`

- [ ] **Step 1: Verify replacement commands work**

Run:

```bash
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/Simple.py game_2a_basic --steps 1 --no-plot
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/Simple.py game_2b_constran --steps 1 --no-plot
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/Simple.py three_agent_track --steps 1 --no-plot
```

Expected: each exits zero and prints one step summary.

- [ ] **Step 2: Delete old files with `apply_patch`**

Delete the old script files only after Step 1 passes.

- [ ] **Step 3: Update docs**

In `/home/lenovo/mpcc/MGIGO/Cartest/carreadme.md`, replace current active game script examples with:

```bash
.venv/bin/python Cartest/Simple.py game_2a_basic --steps 30
.venv/bin/python Cartest/Simple.py game_2b_constran --steps 30
.venv/bin/python Cartest/Simple.py three_agent_track --steps 25
```

Do not rewrite historical experiment notes unless they present old scripts as current entrypoints.

- [ ] **Step 4: Confirm no active references remain**

Run:

```bash
rg -n "Simple_game_2a|Simple_game_2b_constran|MPC_G_MS import mmog_igo_rne_blocks_solver" Cartest
```

Expected: no active runner imports old scripts or directly imports `MPC_G_MS` from Cartest demo files.

---

### Task 9: Final Verification

**Files:**
- All changed files above.

- [ ] **Step 1: Compile changed Python files**

Run:

```bash
.venv/bin/python -m py_compile \
  gmm_igo/solver_builder.py \
  Cartest/Simple.py \
  Cartest/planning/scenarios/__init__.py \
  Cartest/planning/scenarios/game_2a_basic.py \
  Cartest/planning/scenarios/game_2b_constran.py \
  Cartest/planning/scenarios/three_agent_track.py \
  Cartest/planning/costs/registry.py \
  Cartest/planning/costs/game_2a_basic.py \
  Cartest/planning/costs/game_2b_constran.py \
  Cartest/planning/costs/three_agent_track.py \
  Cartest/planning/solver_modes.py \
  Cartest/visualization/game_renderer.py \
  Cartest/eval/test_unified_simple_runner.py \
  Cartest/eval/test_game_solver_modes.py \
  Cartest/eval/test_three_agent_track_game.py
```

Expected: exit code zero.

- [ ] **Step 2: Run focused tests**

Run:

```bash
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/eval/test_unified_simple_runner.py
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/eval/test_game_solver_modes.py
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/eval/test_three_agent_track_game.py
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/visualization/test_visualization_renderer.py
```

Expected: all exit zero.

- [ ] **Step 3: Run unified demo smoke commands**

Run:

```bash
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/Simple.py empty --steps 1 --no-plot
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/Simple.py lane_borrow_overtake --steps 1 --no-plot
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/Simple.py game_2a_basic --steps 1 --no-plot
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/Simple.py game_2b_constran --steps 1 --no-plot
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/Simple.py three_agent_track --steps 1 --no-plot
```

Expected: all exit zero.

- [ ] **Step 4: Verify plotted game output**

Run:

```bash
MPLCONFIGDIR=/tmp/mgigo-mplconfig .venv/bin/python Cartest/Simple.py three_agent_track --steps 2
```

Expected: command exits zero and writes `/home/lenovo/mpcc/MGIGO/Cartest/three_agent_track.gif`.

- [ ] **Step 5: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intended files are modified, created, or deleted. Do not revert unrelated user changes.

---

## Self-Review

- Spec coverage: The plan covers unified dispatch, scenario placement, cost placement, solver management, visualization, old demo removal, documentation, and tests.
- Placeholder scan: No task uses `TBD`, `TODO`, or undefined filenames.
- Type consistency: All multi-agent scenarios use `agents`, `game.block_layout`, and `game.block_to_agent`; solver helpers and costs consume those same fields.
- Main risk: `build_nash_solver()` currently forms `joint_x` by concatenating each agent's owned blocks. This plan keeps each agent's `(s, d)` blocks adjacent, so cost decoding by `agent_idx * 2 * n_free` remains consistent.

## Execution Choice

Plan complete and saved to `/home/lenovo/mpcc/MGIGO/docs/superpowers/plans/2026-07-15-cartest-unified-game-runner.md`. Two execution options:

1. Subagent-Driven (recommended): dispatch a fresh subagent per task, review between tasks, fast iteration.
2. Inline Execution: execute tasks in this session using executing-plans, with checkpoints after solver, runner, and visualization stages.

Choose one before implementation starts.
