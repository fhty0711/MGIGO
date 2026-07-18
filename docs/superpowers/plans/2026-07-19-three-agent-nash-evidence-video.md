# Three-Agent Nash Evidence and Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible per-step unilateral-deviation evidence, keyframe empirical distributional best responses, and a readable game video without changing executed controls or MPC solve timing.

**Architecture:** A focused evaluation module computes selected-mode and keyframe best-response diagnostics from stored RNE results. A reusable three-agent analysis runner performs the control rollout first, runs diagnostics in a separate phase, saves JSON/NPZ, and passes enriched reports to the renderer. The renderer uses a fixed information panel and optional ghost best-response paths.

**Tech Stack:** Python 3, JAX/JIT, NumPy, Matplotlib, FFmpeg, pytest, Cartest batched cost evaluator, MGIGO `build_solver`, WSL2 CUDA.

## Global Constraints

- Diagnostics never alter solver state, selected controls, executed states, random keys used by control, or `solve_ms`.
- `epsilon_mode` is labeled a restricted nine-component selected-plan residual.
- `epsilon_br` is labeled an empirical finite-sample distributional best-response residual, not a formal Nash proof.
- Opponent final GMM distributions are frozen during each best-response solve.
- Three recorded deterministic restarts are used per agent/keyframe; the lowest expected cost is retained.
- Keyframes are initial, maximum summed RSS, first lane-midpoint crossing, and first sustained target-lane completion.
- Four standalone frames and a contact sheet are visually inspected before the final MP4 is rendered.

---

### Task 1: Per-Step Selected-Mode Residual and Keyframe Selection

**Files:**
- Create: `Cartest/eval/nash_game_diagnostics.py`
- Create: `Cartest/eval/test_nash_game_diagnostics.py`

**Interfaces:**
- Produces: `build_component_mean_diagnostic(gen, scenario) -> Callable`.
- Produces: `mode_residual(selected_cost, best_cost) -> jnp.ndarray`.
- Produces: `select_nash_keyframes(state_history, rss_history, target_d=3.5) -> dict[str, int]`.

- [ ] **Step 1: Write failing pure diagnostic tests**

Create `test_nash_game_diagnostics.py`:

```python
import numpy as np
import jax.numpy as jnp

from Cartest.eval.nash_game_diagnostics import (
    mode_residual,
    select_nash_keyframes,
)

def test_mode_residual_clips_negative_roundoff():
    selected = jnp.array([1.0, 2.0, 3.0])
    best = jnp.array([1.0 + 1e-7, 1.5, 3.0])
    assert jnp.allclose(mode_residual(selected, best), [0.0, 0.5, 0.0])

def test_keyframes_are_unique_and_follow_rollout_events():
    states = np.zeros((9, 3, 6), dtype=float)
    states[:, 0, 1] = [0.0, 0.2, 0.8, 1.8, 2.8, 3.1, 3.4, 3.5, 3.5]
    rss = np.zeros((8, 3), dtype=float)
    rss[2] = [0.2, 0.4, 0.3]
    frames = select_nash_keyframes(states, rss, target_d=3.5,
                                   completion_tolerance=0.5,
                                   completion_hold=3)
    assert frames["initial"] == 0
    assert frames["max_rss"] == 2
    assert frames["lane_midpoint"] == 3
    assert frames["lane_complete"] == 5
    assert len(set(frames.values())) == len(frames.values())
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest Cartest/eval/test_nash_game_diagnostics.py -q
```

Expected: import fails because `nash_game_diagnostics.py` does not exist.

- [ ] **Step 3: Implement pure residual and keyframe functions**

Create the module with:

```python
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from Cartest.planning.costs.three_agent_track_batched import (
    batched_nested_costs_from_plans,
    evaluate_joint_plan_batch,
)

def mode_residual(selected_cost, best_cost):
    return jnp.maximum(jnp.asarray(selected_cost) - jnp.asarray(best_cost), 0.0)

def select_nash_keyframes(state_history, rss_history, *, target_d=3.5,
                          completion_tolerance=0.5, completion_hold=5):
    d = np.asarray(state_history)[:, 0, 1]
    rss = np.asarray(rss_history)
    candidates = [
        ("initial", 0),
        ("max_rss", int(np.argmax(np.sum(rss, axis=1)))),
        ("lane_midpoint", int(np.argmax(d >= 0.5 * target_d))),
    ]
    inside = np.abs(d - target_d) <= completion_tolerance
    complete = len(d) - 1
    for index in range(len(inside) - completion_hold + 1):
        if np.all(inside[index:index + completion_hold]):
            complete = index
            break
    candidates.append(("lane_complete", complete))
    unique = {}
    used = set()
    for name, index in candidates:
        if index not in used:
            unique[name] = index
            used.add(index)
    return unique
```

- [ ] **Step 4: Implement the compiled nine-mode evaluator**

Add `build_component_mean_diagnostic` using the existing shared evaluator:

```python
def build_component_mean_diagnostic(gen, scenario):
    combinations = jnp.asarray(
        [(ks, kd) for ks in range(3) for kd in range(3)], dtype=jnp.int32)

    @jax.jit
    def diagnostic(final_mu, selected_blocks, context):
        best_costs, best_indices = [], []
        for aid in range(3):
            sb, db = 2 * aid, 2 * aid + 1
            variants = jnp.broadcast_to(
                selected_blocks[None],
                (len(combinations),) + selected_blocks.shape)
            variants = variants.at[:, sb].set(final_mu[sb, combinations[:, 0]])
            variants = variants.at[:, db].set(final_mu[db, combinations[:, 1]])
            plans = evaluate_joint_plan_batch(
                gen, variants.reshape((len(combinations), -1)), context,
                agent_count=3)
            costs = batched_nested_costs_from_plans(
                plans, scenario, gen.dt, k_inner=0.1,
                obj_transform="standard", ctx=context)[:, aid]
            index = jnp.argmin(costs)
            best_costs.append(costs[index])
            best_indices.append(index)
        return jnp.stack(best_costs), jnp.stack(best_indices)
    return diagnostic
```

- [ ] **Step 5: Add a constructed component-deviation integration test**

Build a small `T=2, B=4, M_inner=2` three-agent scenario, generate a solver result, call the compiled evaluator, and assert:

```python
best_cost, best_index = diagnostic(
    result.diag["mu"], result.diag["selected_blocks"], context)
residual = mode_residual(result.per_agent_cost, best_cost)
assert residual.shape == (3,)
assert best_index.shape == (3,)
assert jnp.all(jnp.isfinite(residual))
assert jnp.all(residual >= 0.0)
```

- [ ] **Step 6: Run and commit selected-mode diagnostics**

Run:

```powershell
python -m pytest Cartest/eval/test_nash_game_diagnostics.py -q
```

Expected: all tests pass.

Commit:

```powershell
git add Cartest/eval/nash_game_diagnostics.py Cartest/eval/test_nash_game_diagnostics.py
git commit -m "feat(cartest): add Nash mode residual diagnostics"
```

---

### Task 2: Empirical Distributional Best Response

**Files:**
- Modify: `Cartest/planning/costs/three_agent_track_batched.py`
- Modify: `Cartest/eval/nash_game_diagnostics.py`
- Modify: `Cartest/eval/test_nash_game_diagnostics.py`

**Interfaces:**
- Produces: `sample_frozen_opponent_blocks(mu, precision_cholesky, pi, count, key, agent_idx)`.
- Produces: `make_empirical_best_response_solver(gen, scenario, agent_idx, opponent_samples)`.
- Produces: `run_empirical_best_response(gen, scenario, agent_idx, equilibrium_cost, mu, precision_cholesky, pi, selected_blocks, context, background_key, restart_keys) -> dict` with costs, residuals, controls, trajectory, seeds, and runtime.

- [ ] **Step 1: Write failing frozen-opponent and restart tests**

Add tests that use small synthetic arrays:

```python
def test_frozen_opponent_samples_do_not_change_deviating_blocks():
    samples = sample_frozen_opponent_blocks(
        mu, L, pi, count=5, key=jax.random.PRNGKey(4), agent_idx=1)
    assert samples.shape == (5, 6, n_free)
    assert jnp.all(jnp.isnan(samples[:, 2:4]))
    assert jnp.all(jnp.isfinite(samples[:, jnp.array([0, 1, 4, 5])]))

def test_best_response_summary_uses_lowest_restart_cost():
    summary = summarize_best_response_restarts(
        equilibrium_cost=0.8,
        restart_costs=jnp.array([0.7, 0.75, 0.6]))
    assert summary["best_restart"] == 2
    assert np.isclose(summary["epsilon_br"], 0.2)
    assert np.isclose(summary["epsilon_br_relative"], 0.25)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest Cartest/eval/test_nash_game_diagnostics.py -k "frozen_opponent or best_response_summary" -q
```

Expected: imports fail because the functions do not exist.

- [ ] **Step 3: Expose fixed-background expected cost**

Add to `three_agent_track_batched.py`:

```python
def expected_cost_for_agent_controls(
        gen, own_controls, frozen_joint_controls, ctx, scenario, aid,
        *, k_inner=0.1, obj_transform="standard"):
    """Expected nested cost for own [B,2,D] controls vs frozen [M,6,D]."""
    B, M = own_controls.shape[0], frozen_joint_controls.shape[0]
    joint = jnp.broadcast_to(
        frozen_joint_controls[None], (B, M) + frozen_joint_controls.shape[1:])
    joint = joint.at[:, :, 2 * aid:2 * aid + 2].set(
        own_controls[:, None])
    flat = joint.reshape((B * M, -1))
    plans = evaluate_joint_plan_batch(gen, flat, ctx, agent_count=3)
    costs = batched_nested_costs_from_plans(
        plans, scenario, gen.dt, k_inner=k_inner,
        obj_transform=obj_transform, ctx=ctx)
    return costs[:, aid].reshape((B, M)).mean(axis=1)
```

Add a parity test comparing one candidate against an explicit Python loop over the same frozen joint controls.

- [ ] **Step 4: Implement deterministic frozen-opponent sampling**

Reuse `_sample_all_blocks` from `Cartest.planning.solvers.batched_rne_solver`, transpose its `[blocks, samples, D]` result to `[samples, blocks, D]`, and mark the deviating agent’s two blocks with `NaN` so accidental use is test-visible:

```python
def sample_frozen_opponent_blocks(mu, precision_cholesky, pi, count, key,
                                  agent_idx):
    precision = jax.vmap(jax.vmap(lambda L: L @ L.T))(precision_cholesky)
    sampled = _sample_all_blocks(mu, precision, pi, count, key)
    sampled = jnp.swapaxes(sampled, 0, 1)
    return sampled.at[:, 2 * agent_idx:2 * agent_idx + 2].set(jnp.nan)
```

Before cost evaluation, fill the two `NaN` own blocks from the candidate controls as shown in Step 3.

- [ ] **Step 5: Build the two-block empirical best-response solver**

Construct a pure expected-nested-cost objective and `build_solver` once per agent/keyframe signature:

```python
def make_empirical_best_response_solver(gen, scenario, agent_idx,
                                        frozen_joint_controls):
    def objective(x, ctx):
        own = x.reshape(1, 2, gen.n_free)
        return expected_cost_for_agent_controls(
            gen, own, frozen_joint_controls, ctx, scenario, agent_idx)[0]
    return build_solver(
        objective,
        dims=(gen.n_free, gen.n_free),
        solver="m22",
        T=int(scenario["game"]["T"]),
        dt=float(scenario["game"]["dt"]),
        K=3,
        B=int(scenario["game"]["B"]),
        B0=int(scenario["game"]["B0"]),
        T_0=int(scenario["game"]["T_0"]),
    )
```

Initialize each restart from the equilibrium agent blocks for one restart and the semantic warm-start plus deterministic perturbations for the other two. Record the exact restart/background keys. Block until ready before measuring diagnostic runtime.

- [ ] **Step 6: Summarize and validate best-response output**

Implement:

```python
def summarize_best_response_restarts(equilibrium_cost, restart_costs):
    costs = np.asarray(restart_costs, dtype=float)
    best = int(np.argmin(costs))
    epsilon = max(0.0, float(equilibrium_cost) - float(costs[best]))
    return {
        "best_restart": best,
        "best_response_cost": float(costs[best]),
        "epsilon_br": epsilon,
        "epsilon_br_relative": epsilon / max(abs(float(equilibrium_cost)), 1e-6),
    }
```

Assert the returned plan is finite, opponent samples are unchanged across restarts, and replaying stored keys produces identical costs.

- [ ] **Step 7: Run and commit best-response diagnostics**

Run:

```powershell
python -m pytest Cartest/eval/test_nash_game_diagnostics.py Cartest/eval/test_batched_three_agent_eval.py -q
```

Expected: all tests pass, including explicit-loop parity.

Commit:

```powershell
git add Cartest/planning/costs/three_agent_track_batched.py Cartest/eval/nash_game_diagnostics.py Cartest/eval/test_nash_game_diagnostics.py
git commit -m "feat(cartest): add empirical Nash best responses"
```

---

### Task 3: Reusable Three-Agent Analysis Runner

**Files:**
- Create: `Cartest/eval/three_agent_analysis.py`
- Modify: `Cartest/eval/test_nash_game_diagnostics.py`

**Interfaces:**
- Produces: `run_three_agent_analysis(steps=150, T=300, seed=0, output_dir=None) -> dict`.
- Produces: JSON/NPZ paths and enriched renderer reports.
- Consumes: `build_cartest_nash_solver`, selected-mode diagnostic, keyframe selector, and best-response runner.

- [ ] **Step 1: Write a small-run output-contract test**

Use a patched solver/result to avoid compiling RNE:

```python
def test_analysis_summary_separates_solver_and_diagnostic_timing(monkeypatch, tmp_path):
    summary = run_three_agent_analysis(
        steps=2, T=2, seed=0, output_dir=tmp_path,
        render_video=False, run_best_response=False)
    assert summary["configuration"]["steps"] == 2
    assert len(summary["timing"]["solve_ms"]) == 2
    assert "diagnostic_ms" in summary["timing"]
    assert "diagnostic_ms" not in summary["timing"]["solve_ms"]
    assert Path(summary["artifacts"]["json"]).exists()
    assert Path(summary["artifacts"]["npz"]).exists()
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
python -m pytest Cartest/eval/test_nash_game_diagnostics.py -k analysis_summary -q
```

Expected: import fails because `three_agent_analysis.py` does not exist.

- [ ] **Step 3: Move the external rollout logic into the reusable module**

Port the proven data collection from `D:/MGIGO/t300_150_full_analysis.py` into functions rather than importing the external script. The timed region must contain only:

```python
started = time.perf_counter_ns()
result = solver(
    solve_key, context=context,
    initial_mu=initial_mu, initial_S_or_L=initial_l)
block_until_ready(result)
solve_ms = (time.perf_counter_ns() - started) / 1e6
```

Use a distinct diagnostic key stream derived from `jax.random.fold_in(PRNGKey(seed), 0x4E415348)` so diagnostics cannot perturb control keys.

- [ ] **Step 4: Add post-rollout keyframe best responses and serialization**

Store each step’s context and final `mu/S_or_L/pi/selected_blocks` needed for replay. After all executed states are fixed:

```python
keyframes = select_nash_keyframes(state_history, rss_history)
for frame_name, step in keyframes.items():
    for aid in range(3):
        stored = solver_snapshots[step]
        br_results[frame_name][aid] = run_empirical_best_response(
            gen,
            scenario,
            aid,
            float(stored["per_agent_cost"][aid]),
            stored["mu"],
            stored["S_or_L"],
            stored["pi"],
            stored["selected_blocks"],
            stored["context"],
            diagnostic_keys[step]["background"],
            diagnostic_keys[step]["restarts"][aid],
        )
```

Serialize scalar lists/dicts to JSON and dense histories to NPZ. Include method labels `restricted_component_mean` and `empirical_distributional_best_response_3_restart`.

- [ ] **Step 5: Add CLI and run small CPU smoke test**

Support:

```text
python -m Cartest.eval.three_agent_analysis --steps 2 --T 2 --seed 0 --no-video --no-best-response
```

Expected: exits zero and prints JSON/NPZ paths.

- [ ] **Step 6: Commit the analysis runner**

```powershell
git add Cartest/eval/three_agent_analysis.py Cartest/eval/test_nash_game_diagnostics.py
git commit -m "feat(cartest): add reproducible three-agent analysis runner"
```

---

### Task 4: Nash Information Panel and Ghost Trajectories

**Files:**
- Modify: `Cartest/visualization/game_renderer.py`
- Modify: `Cartest/eval/test_visualization_renderer.py`
- Modify: `Cartest/eval/three_agent_analysis.py`

**Interfaces:**
- Consumes optional report fields: `agent_status`, `best_response_xy`, `best_response_diag`, and `keyframe_name`.
- Produces: `save_game_frame(report, output_path, road=None, limits=None)` and enriched MP4 frames.

- [ ] **Step 1: Write failing renderer layer tests**

Create a three-agent report containing status and ghost trajectories, render one frame, and assert:

```python
text = "\n".join(item.get_text() for item in ax.texts)
line_gids = {line.get_gid() for line in ax.lines if line.get_gid()}
assert "v/target" in text
assert "epsilon_mode" in text
assert "epsilon_br" in text
assert "best_response_ego" in line_gids
```

Also call `save_game_frame` into `tmp_path` and assert the PNG is nonempty.

- [ ] **Step 2: Run renderer tests and verify failure**

Run:

```powershell
python -m pytest Cartest/eval/test_visualization_renderer.py -k "game.*nash or save_game_frame" -q
```

Expected: missing function/layer assertions fail.

- [ ] **Step 3: Add a fixed diagnostics panel**

Change video layout to a road axis plus a fixed-width information axis. Render one row per agent using report values:

```text
ego   v/target=15.0/17.5  mode=(1,0)  epsilon_mode=0.000e+00
      pi_s=[0.10,0.80,0.10]  pi_d=[0.70,0.20,0.10]
```

On keyframes append:

```text
epsilon_br=1.2e-03  J_eq=0.126 -> J_br=0.125
```

Use monospaced 8–9 pt text, stable axis bounds, and no autosizing based on string length.

- [ ] **Step 4: Draw optional ghost best-response paths**

For each available path:

```python
line, = ax.plot(
    ghost[:, 0], ghost[:, 1],
    color=color, linestyle=(0, (3, 3)), linewidth=1.8,
    alpha=0.7, zorder=6)
line.set_gid(f"best_response_{agent_name}")
```

Do not smooth the ghost path differently from equilibrium paths.

- [ ] **Step 5: Add standalone frame saving**

Implement `save_game_frame` using the same layout/rendering function as the video so PNG review and MP4 output cannot drift.

- [ ] **Step 6: Run renderer tests and commit**

Run:

```powershell
python -m pytest Cartest/eval/test_visualization_renderer.py Cartest/eval/test_unified_simple_runner.py -q
```

Expected: all tests pass and existing reports without Nash fields render unchanged.

Commit:

```powershell
git add Cartest/visualization/game_renderer.py Cartest/eval/test_visualization_renderer.py Cartest/eval/three_agent_analysis.py
git commit -m "feat(cartest): visualize Nash deviation evidence"
```

---

### Task 5: CUDA Analysis, Visual Review, and Final Artifacts

**Files:**
- Generated artifacts only: `Cartest/output/t300_150_nash_analysis/`.

**Interfaces:**
- Consumes: complete analysis CLI and renderer.
- Produces: four PNG keyframes, contact sheet, JSON, NPZ, and final MP4.

- [ ] **Step 1: Run all related CPU tests**

Run:

```powershell
python -m pytest Cartest/eval/test_nash_game_diagnostics.py Cartest/eval/test_visualization_renderer.py Cartest/eval/test_batched_three_agent_eval.py Cartest/eval/test_three_agent_track_components.py Cartest/eval/test_game_solver_modes.py Cartest/eval/test_three_agent_track_game.py Cartest/eval/test_unified_simple_runner.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run the full CUDA analysis without final video**

Run:

```bash
cd /mnt/d/MGIGO/MGIGO
XLA_PYTHON_CLIENT_PREALLOCATE=false /root/.venvs/tcmgigo-jaxgpu/bin/python \
  -m Cartest.eval.three_agent_analysis --steps 150 --T 300 --seed 0 --keyframes-only
```

Expected: JSON, NPZ, and four standalone PNGs are saved; solver timing excludes diagnostics.

- [ ] **Step 3: Inspect and revise the four keyframes**

Open every PNG at original resolution with the image viewer. Check text/vehicle overlap, roadway height, panel readability, equilibrium-vs-ghost distinction, stable camera bounds, and value clipping. If any check fails, adjust only renderer layout/style, rerun renderer tests, and regenerate the four PNGs.

- [ ] **Step 4: Build and inspect the contact sheet**

Use FFmpeg tile or Matplotlib to combine the four approved PNGs into `keyframes_contact_sheet.png`. Inspect it at original resolution; repeat Step 3 if any panel is unreadable at the sheet’s normal zoom.

- [ ] **Step 5: Render and inspect the final MP4**

Run:

```bash
cd /mnt/d/MGIGO/MGIGO
XLA_PYTHON_CLIENT_PREALLOCATE=false /root/.venvs/tcmgigo-jaxgpu/bin/python \
  -m Cartest.eval.three_agent_analysis --steps 150 --T 300 --seed 0
```

Use `ffprobe` to confirm H.264, 150 frames, and expected duration. Extract early/conflict/crossing/settled frames and inspect them again for flicker or clipping.

- [ ] **Step 6: Report evidence without overclaiming**

Report:

```text
lane completion and smoothness
hard constraint maxima
per-agent epsilon_mode distribution
per-keyframe per-agent epsilon_br and relative residual
three restart costs and diagnostic runtime
median/P95 solver time excluding diagnostics
absolute paths to MP4/JSON/NPZ/keyframes/contact sheet
```

State explicitly that the video demonstrates strategic responses and empirical unilateral-deviation checks, while finite-sample best responses are not a formal proof of exact Nash equilibrium.
