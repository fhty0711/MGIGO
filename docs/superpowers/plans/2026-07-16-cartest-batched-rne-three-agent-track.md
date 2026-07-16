# Cartest Batched RNE Three-Agent Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Cartest-specific batched RNE path for `three_agent_track` that precomputes B-spline trajectories outside the `B x M_inner` cost loop, without modifying `/home/lenovo/mpcc/MGIGO/gmm_igo/MPC_G_MS.py`.

**Architecture:** Keep the existing generic `rne_blocks` solver untouched as the baseline. Add a Cartest-local solver mode, `cartest_batched_rne_blocks`, that reuses the same block-level IGO update math but replaces black-box scalar fitness evaluation with a structured three-agent B-spline batch evaluator. The expensive B-spline/Frenet/vehicle/Cartesian trajectory evaluation is performed once for candidate samples and once for background samples, then pairwise game costs are computed by broadcasting over cached trajectories.

**Tech Stack:** Python, JAX, JAX NumPy, Cartest Frenet B-spline utilities, existing `gmm_igo` update conventions, existing Cartest scenario/cost registry, unittest-style test scripts.

---

## Non-Negotiable Constraints

- Do not modify `/home/lenovo/mpcc/MGIGO/gmm_igo/MPC_G_MS.py`.
- Do not reduce `T=300`, `B=60`, or `M_inner=30` to hide runtime cost.
- Use host GPU for runtime verification.
- Preserve the current command shape: `.venv/bin/python Cartest/Simple.py three_agent_track --steps 25`.
- Keep the existing generic `rne_blocks` path available for comparison.
- First optimize the `three_agent_track` path only; do not generalize prematurely to arbitrary games.

## Current Root Cause Hypothesis

The current generic black-box RNE path calls:

```python
fitness_fn_j(agent_idx, joint_m_modified.flatten(), context)
```

inside the `B x M_inner` loop. For `three_agent_track`, each fitness call evaluates every agent plan from B-spline controls. That means expensive trajectory evaluation is repeated per pairwise candidate/background combination.

The target design keeps the same Monte Carlo pairing semantics, but changes cost evaluation into:

```text
expensive trajectory evaluation: O(M_agent * (B + M_inner))
cheap pairwise combination:      O(M_agent * B * M_inner)
```

instead of doing expensive trajectory evaluation inside the full Cartesian product.

## File Structure

- Create `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_game_eval.py`
  - Three-agent B-spline trajectory batch evaluation helpers.
  - Cached plan pytree shape definitions by convention.
  - Batched cost functions for ego/front/rear matching `Cartest.planning.costs.three_agent_track` semantics.

- Create `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_rne_solver.py`
  - Cartest-local RNE block solver using the same block sampling/update structure as `MPC_G_MS.py`.
  - Structured evaluator hook for `three_agent_track`.
  - No imports from private `_step_fn_rne_blocks` except for optional comments/comparison; duplicate only the required update math locally if needed.

- Modify `/home/lenovo/mpcc/MGIGO/Cartest/planning/solver_modes.py`
  - Route `game["solver"] == "cartest_batched_rne_blocks"` to the new solver wrapper.
  - Leave existing `rne_blocks` behavior unchanged.

- Modify `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/three_agent_track.py`
  - Switch only `three_agent_track` to `"solver": "cartest_batched_rne_blocks"` after tests pass.
  - Keep all scale parameters unchanged: `T=300`, `B=60`, `M_inner=30`.

- Create `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_three_agent_eval.py`
  - Unit tests for trajectory batch shapes.
  - Unit tests that batched costs match scalar `three_agent_track` costs on fixed joint samples.

- Create `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_rne_solver.py`
  - Small fixed-size solver tests.
  - Golden comparison against generic `rne_blocks` for `T=1`, `B=4`, `M_inner=3`, `K=2` on fixed seeds.

- Create `/home/lenovo/mpcc/MGIGO/Cartest/eval/benchmark_three_agent_batched.py`
  - GPU benchmark for generic vs batched path with explicit `block_until_ready()`.
  - Reports compile time separately from steady-state solve time.

- Modify `/home/lenovo/mpcc/MGIGO/Cartest/visualization/README.md`
  - Add a short note that `three_agent_track` uses a batched Cartest RNE adapter and why Frenet-coordinate cost perspective remains a documented future item.

---

### Task 1: Add Scalar/Batched Cost Equivalence Tests

**Files:**
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_three_agent_eval.py`
- Read-only reference: `/home/lenovo/mpcc/MGIGO/Cartest/planning/costs/three_agent_track.py`

- [ ] **Step 1: Write the failing test file**

Create `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_three_agent_eval.py`:

```python
"""Tests for three-agent batched B-spline game evaluation."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.execution.execute import FrenetState
from Cartest.planning.scenarios import get_scenario
from Cartest.planning.solver_modes import build_multi_agent_context, build_multi_agent_warmstart
from Cartest.planning.costs.three_agent_track import make_agent_specs


BASIS = ROOT / "Cartest" / "basis" / "bspline_basis.npz"


def _states(scenario):
    return [
        FrenetState(
            s=agent["s"], s_dot=agent["s_dot"], s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"], d_dot=agent.get("d_dot", 0.0), d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]


def _joint_from_mu(mu):
    parts = []
    for agent_idx in range(3):
        s_block = agent_idx * 2
        d_block = s_block + 1
        parts.append(mu[s_block, 0])
        parts.append(mu[d_block, 0])
    return jnp.concatenate(parts)


def test_batched_plan_eval_shapes_match_three_agents():
    from Cartest.planning.batched_game_eval import evaluate_joint_plan_batch

    scenario = get_scenario("three_agent_track")
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    ctx = build_multi_agent_context(_states(scenario))
    mu, _ = build_multi_agent_warmstart(gen, scenario, _states(scenario), jax.random.PRNGKey(0))

    samples = jnp.stack([_joint_from_mu(mu), _joint_from_mu(mu) + 0.05], axis=0)
    plans = evaluate_joint_plan_batch(gen, samples, ctx, agent_count=3)

    assert len(plans) == 3
    assert plans[0]["s"].shape == (2, gen.T)
    assert plans[0]["x"].shape == (2, gen.T)
    assert plans[0]["vehicle"].shape == (2, gen.T, 9)


def test_batched_agent_cost_matches_scalar_cost_for_fixed_joint_samples():
    from Cartest.planning.batched_game_eval import evaluate_joint_plan_batch, batched_agent_costs_from_plans

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, _ = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(1))
    joint = _joint_from_mu(mu)
    joint_batch = jnp.stack([joint, joint + 0.01], axis=0)

    specs = make_agent_specs(gen, scenario)
    scalar = jnp.stack([
        jnp.stack([specs[aid][0](sample, ctx) for aid in range(3)])
        for sample in joint_batch
    ], axis=0)

    plans = evaluate_joint_plan_batch(gen, joint_batch, ctx, agent_count=3)
    batched = batched_agent_costs_from_plans(plans, scenario)

    assert batched.shape == (2, 3)
    assert jnp.all(jnp.isfinite(batched))
    assert jnp.allclose(batched, scalar, rtol=2e-4, atol=2e-3)


if __name__ == "__main__":
    test_batched_plan_eval_shapes_match_three_agents()
    test_batched_agent_cost_matches_scalar_cost_for_fixed_joint_samples()
    print("batched three-agent eval tests ok")
```

- [ ] **Step 2: Run the test and confirm it fails because the module is missing**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_batched_three_agent_eval.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'Cartest.planning.batched_game_eval'`.

- [ ] **Step 3: Commit the failing tests**

```bash
cd /home/lenovo/mpcc/MGIGO
git add Cartest/eval/test_batched_three_agent_eval.py
git commit -m "test: specify batched three-agent game evaluation"
```

---

### Task 2: Implement Batched Three-Agent Plan Evaluation

**Files:**
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_game_eval.py`
- Test: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_three_agent_eval.py`

- [ ] **Step 1: Add the batched evaluator module**

Create `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_game_eval.py`:

```python
"""Batched three-agent B-spline game evaluation for Cartest.

This module is intentionally Cartest-specific.  It avoids the generic
black-box fitness interface so B-spline trajectories can be evaluated once
and then reused across B x M_inner game-cost combinations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import vmap


def agent_ctx(ctx, agent_idx):
    return {
        "s0": ctx[f"s0_a{agent_idx}"],
        "s_dot0": ctx[f"s_dot0_a{agent_idx}"],
        "s_ddot0": ctx.get(f"s_ddot0_a{agent_idx}", 0.0),
        "d0": ctx[f"d0_a{agent_idx}"],
        "d_dot0": ctx.get(f"d_dot0_a{agent_idx}", 0.0),
        "d_ddot0": ctx.get(f"d_ddot0_a{agent_idx}", 0.0),
    }


def theta_for_agent(joint_x, agent_idx, n_free):
    base = agent_idx * 2 * n_free
    return joint_x[base:base + 2 * n_free]


def evaluate_agent_plan_batch(gen, joint_batch, ctx, agent_idx):
    """Evaluate one agent's plan for a batch of joint vectors."""
    n_free = gen.n_free
    a_ctx = agent_ctx(ctx, agent_idx)

    def one(joint_x):
        theta = theta_for_agent(joint_x, agent_idx, n_free)
        frenet, vehicle, (x, y) = gen.evaluate_plan(theta[:n_free], theta[n_free:], a_ctx)
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = frenet
        return {
            "s": s,
            "d": d,
            "s_dot": s_dot,
            "d_dot": d_dot,
            "s_ddot": s_ddot,
            "d_ddot": d_ddot,
            "s_dddot": s_dddot,
            "d_dddot": d_dddot,
            "vehicle": vehicle,
            "x": x,
            "y": y,
        }

    return vmap(one)(joint_batch)


def evaluate_joint_plan_batch(gen, joint_batch, ctx, agent_count=3):
    """Evaluate all agent plans for joint vectors shaped [batch, joint_dim]."""
    return tuple(
        evaluate_agent_plan_batch(gen, joint_batch, ctx, agent_idx)
        for agent_idx in range(agent_count)
    )


def pair_distance_violation(x0, y0, x1, y1, safe_dist):
    dist = jnp.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + 1e-6)
    return jnp.maximum(0.0, safe_dist - dist)


def batched_agent_costs_from_plans(plans, scenario):
    """Return scalar objective costs shaped [batch, 3].

    This intentionally matches the objective portion of
    Cartest.planning.costs.three_agent_track.  Constran constraints are
    handled by separate batched violation functions in later tasks.
    """
    ego = plans[0]
    front = plans[1]
    rear = plans[2]
    ego_target_d = float(scenario["behavior"].get("ego_target_d", 3.5))

    ego_v_ref = float(scenario["agents"][0]["v_target"])
    front_v_ref = float(scenario["agents"][1]["v_target"])
    rear_v_ref = float(scenario["agents"][2]["v_target"])

    ego_cost = (
        3.0 * jnp.sum((ego["s_dot"] - ego_v_ref) ** 2, axis=-1)
        + 10.0 * jnp.sum((ego["d"] - ego_target_d) ** 2, axis=-1)
        + 5.0 * jnp.sum(ego["d_dot"] ** 2, axis=-1)
        + 0.5 * jnp.sum(ego["d_ddot"] ** 2, axis=-1)
        + jnp.sum(ego["s_dddot"] ** 2 + ego["d_dddot"] ** 2, axis=-1)
    )
    front_cost = (
        3.0 * jnp.sum((front["s_dot"] - front_v_ref) ** 2, axis=-1)
        + 2.0 * jnp.sum(front["s_ddot"] ** 2, axis=-1)
    )
    rear_cost = (
        3.0 * jnp.sum((rear["s_dot"] - rear_v_ref) ** 2, axis=-1)
        + 2.0 * jnp.sum(rear["s_ddot"] ** 2, axis=-1)
    )
    return jnp.stack([ego_cost, front_cost, rear_cost], axis=-1)
```

- [ ] **Step 2: Run the batched evaluator tests**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_batched_three_agent_eval.py
```

Expected: PASS and prints `batched three-agent eval tests ok`.

- [ ] **Step 3: Commit the evaluator**

```bash
cd /home/lenovo/mpcc/MGIGO
git add Cartest/planning/batched_game_eval.py Cartest/eval/test_batched_three_agent_eval.py
git commit -m "feat: add batched three-agent plan evaluation"
```

---

### Task 3: Add Batched Constraint and Nested Cost Evaluation

**Files:**
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_game_eval.py`
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_three_agent_eval.py`

- [ ] **Step 1: Extend tests for nested cost equivalence**

Append this test to `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_three_agent_eval.py`:

```python
def test_batched_nested_cost_matches_constran_scalar_specs():
    from Cartest.planning.batched_game_eval import evaluate_joint_plan_batch, batched_nested_costs_from_plans
    from Constraintdealer.Constran import build_multi_agent

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, _ = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(2))
    joint = _joint_from_mu(mu)
    joint_batch = jnp.stack([joint, joint + 0.02], axis=0)

    specs = make_agent_specs(gen, scenario)
    scalar_fns = build_multi_agent(specs, k_inner=1.0, obj_transform="standard")
    scalar = jnp.stack([
        jnp.stack([scalar_fns[aid](aid, sample, ctx) for aid in range(3)])
        for sample in joint_batch
    ], axis=0)

    plans = evaluate_joint_plan_batch(gen, joint_batch, ctx, agent_count=3)
    batched = batched_nested_costs_from_plans(plans, scenario, k_inner=1.0, obj_transform="standard")

    assert batched.shape == (2, 3)
    assert jnp.all(jnp.isfinite(batched))
    assert jnp.allclose(batched, scalar, rtol=2e-4, atol=2e-3)
```

- [ ] **Step 2: Run the test and confirm it fails because `batched_nested_costs_from_plans` is missing**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_batched_three_agent_eval.py
```

Expected: FAIL with `ImportError` or `AttributeError` for `batched_nested_costs_from_plans`.

- [ ] **Step 3: Implement batched constraints and nested transform**

Append these functions to `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_game_eval.py`:

```python
def _aggregate(values, mode):
    if mode == "max":
        return jnp.max(values, axis=-1)
    if mode == "q95":
        return jnp.quantile(values, 0.95, axis=-1)
    raise ValueError(f"unsupported aggregate {mode!r}")


def _softplus_violation(g, k_inner=1.0):
    return jnp.logaddexp(0.0, k_inner * g) / k_inner


def _nested_add(cost, violation, priority_scale):
    return cost + priority_scale * violation


def batched_constraint_violations_from_plans(plans, scenario):
    safe_gap = float(scenario["safety"].get("safe_gap", 3.0))
    vehicle_length = float(scenario["safety"].get("vehicle_length", 5.0))
    v_min = float(scenario["safety"].get("v_min", 2.0))
    v_max = float(scenario["safety"].get("v_max", 35.0))
    acc_max = float(scenario["safety"].get("acc_max", 5.0))
    jerk_max = float(scenario["safety"].get("jerk_max", 2.0))
    lane_min, lane_max = scenario["road"].get("lane_bounds_d", (-1.75, 5.25))

    out = []
    for aid in range(3):
        plan = plans[aid]
        vehicle = plan["vehicle"]
        lane = jnp.maximum(jnp.maximum(0.0, lane_min - plan["d"]), jnp.maximum(0.0, plan["d"] - lane_max))
        v = vehicle[..., 2]
        speed = jnp.maximum(jnp.maximum(0.0, v_min - v), jnp.maximum(0.0, v - v_max))
        a_long, a_lat = vehicle[..., 4], vehicle[..., 5]
        a_mag = jnp.sqrt(a_long ** 2 + a_lat ** 2)
        acc = jnp.maximum(
            jnp.maximum(0.0, jnp.abs(a_long) - acc_max),
            jnp.maximum(jnp.maximum(0.0, jnp.abs(a_lat) - acc_max), jnp.maximum(0.0, a_mag - acc_max)),
        )
        j_long, j_lat = vehicle[..., 6], vehicle[..., 7]
        j_mag = jnp.sqrt(j_long ** 2 + j_lat ** 2)
        jerk = jnp.maximum(
            jnp.maximum(0.0, jnp.abs(j_long) - jerk_max),
            jnp.maximum(jnp.maximum(0.0, jnp.abs(j_lat) - jerk_max), jnp.maximum(0.0, j_mag - jerk_max)),
        )

        if aid == 0:
            col = jnp.maximum(
                pair_distance_violation(plan["x"], plan["y"], plans[1]["x"], plans[1]["y"], safe_gap),
                pair_distance_violation(plan["x"], plan["y"], plans[2]["x"], plans[2]["y"], safe_gap),
            )
        elif aid == 1:
            short = slice(0, 2)
            col = jnp.zeros_like(plan["x"])
            ego_short = pair_distance_violation(
                plan["x"][..., short], plan["y"][..., short],
                plans[0]["x"][..., short], plans[0]["y"][..., short], safe_gap,
            )
            rear_short = pair_distance_violation(
                plan["x"][..., short], plan["y"][..., short],
                plans[2]["x"][..., short], plans[2]["y"][..., short], safe_gap,
            )
            col = col.at[..., short].set(jnp.maximum(ego_short, rear_short))
        else:
            short = slice(0, 2)
            col = jnp.zeros_like(plan["x"])
            ego_short = pair_distance_violation(
                plan["x"][..., short], plan["y"][..., short],
                plans[0]["x"][..., short], plans[0]["y"][..., short], safe_gap,
            )
            col = col.at[..., short].set(ego_short)
            clearance = vehicle_length + safe_gap
            clearance_violation = jnp.maximum(0.0, clearance - (plans[1]["s"] - plan["s"]))
            col = jnp.maximum(col, clearance_violation)

        out.append({"lane": lane, "speed": speed, "acc": acc, "jerk": jerk, "collision": col})
    return tuple(out)


def batched_nested_costs_from_plans(plans, scenario, k_inner=1.0, obj_transform="standard"):
    del obj_transform
    costs = batched_agent_costs_from_plans(plans, scenario)
    violations = batched_constraint_violations_from_plans(plans, scenario)
    priority_scales = {
        "lane": 10.0,
        "speed": 100.0,
        "acc": 1000.0,
        "jerk": 10000.0,
        "collision": 100000.0,
    }
    aggs = {"lane": "q95", "speed": "max", "acc": "max", "jerk": "max", "collision": "max"}

    result = []
    for aid in range(3):
        c = costs[:, aid]
        for name in ("lane", "speed", "acc", "jerk", "collision"):
            g = _aggregate(violations[aid][name], aggs[name])
            c = _nested_add(c, _softplus_violation(g, k_inner), priority_scales[name])
        result.append(c)
    return jnp.stack(result, axis=-1)
```

- [ ] **Step 4: Run the nested equivalence test**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_batched_three_agent_eval.py
```

Expected: PASS. If it fails only on numeric tolerance, inspect `Constraintdealer.Constran` transform constants and align the priority/transform implementation exactly before proceeding.

- [ ] **Step 5: Commit nested evaluator**

```bash
cd /home/lenovo/mpcc/MGIGO
git add Cartest/planning/batched_game_eval.py Cartest/eval/test_batched_three_agent_eval.py
git commit -m "feat: add batched three-agent nested cost evaluation"
```

---

### Task 4: Implement a Fixed-Sample Batched F-Hat Evaluator

**Files:**
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_game_eval.py`
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_rne_solver.py`

- [ ] **Step 1: Write fixed-sample f-hat equivalence test**

Create `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_rne_solver.py`:

```python
"""Tests for Cartest batched RNE helper logic."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.execution.execute import FrenetState
from Cartest.planning.scenarios import get_scenario
from Cartest.planning.solver_modes import build_multi_agent_context, build_multi_agent_warmstart
from Cartest.planning.costs.three_agent_track import make_agent_specs
from Constraintdealer.Constran import build_multi_agent


BASIS = ROOT / "Cartest" / "basis" / "bspline_basis.npz"


def _states(scenario):
    return [
        FrenetState(
            s=agent["s"], s_dot=agent["s_dot"], s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"], d_dot=agent.get("d_dot", 0.0), d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]


def _make_samples(gen, scenario):
    states = _states(scenario)
    mu, _ = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(4))
    base = mu[:, 0]
    B = 4
    M = 3
    samples_b = jnp.stack([base + 0.01 * i for i in range(B)], axis=1)
    samples_m = jnp.stack([base - 0.02 * i for i in range(M)], axis=1)
    return samples_b, samples_m


def test_batched_f_hat_matches_black_box_scalar_loop():
    from Cartest.planning.batched_game_eval import batched_expected_cost_for_agent

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    ctx = build_multi_agent_context(_states(scenario))
    samples_b, samples_m = _make_samples(gen, scenario)

    specs = make_agent_specs(gen, scenario)
    scalar_fns = build_multi_agent(specs, k_inner=1.0, obj_transform="standard")

    for aid in range(3):
        block_mask = jnp.asarray(scenario["game"]["block_to_agent"]) == aid
        expected = []
        for b in range(samples_b.shape[1]):
            vals = []
            for m in range(samples_m.shape[1]):
                joint = jnp.where(block_mask[:, None], samples_b[:, b, :], samples_m[:, m, :]).reshape(-1)
                vals.append(scalar_fns[aid](aid, joint, ctx))
            expected.append(jnp.mean(jnp.stack(vals)))
        expected = jnp.stack(expected)

        actual = batched_expected_cost_for_agent(gen, samples_b, samples_m, ctx, scenario, aid)
        assert actual.shape == expected.shape
        assert jnp.allclose(actual, expected, rtol=2e-4, atol=2e-3)


if __name__ == "__main__":
    test_batched_f_hat_matches_black_box_scalar_loop()
    print("batched rne helper tests ok")
```

- [ ] **Step 2: Run the test and confirm it fails because the helper is missing**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_batched_rne_solver.py
```

Expected: FAIL with missing `batched_expected_cost_for_agent`.

- [ ] **Step 3: Implement fixed-sample f-hat helper**

Append to `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_game_eval.py`:

```python
def _flatten_blocks(blocks):
    return blocks.reshape((blocks.shape[0], -1))


def batched_expected_cost_for_agent(gen, samples_b, samples_m, ctx, scenario, agent_idx):
    """Compute f_hat[B] for one agent from fixed block samples.

    samples_b: [N_blocks, B, D]
    samples_m: [N_blocks, M_inner, D]
    """
    block_to_agent = jnp.asarray(scenario["game"]["block_to_agent"])
    block_mask = block_to_agent == agent_idx
    B = samples_b.shape[1]
    M_inner = samples_m.shape[1]

    def joint_for_bm(b_idx, m_idx):
        joint_blocks = jnp.where(block_mask[:, None], samples_b[:, b_idx, :], samples_m[:, m_idx, :])
        return joint_blocks.reshape(-1)

    joints = vmap(
        lambda b: vmap(lambda m: joint_for_bm(b, m))(jnp.arange(M_inner))
    )(jnp.arange(B))
    flat_joints = joints.reshape((B * M_inner, -1))
    plans = evaluate_joint_plan_batch(gen, flat_joints, ctx, agent_count=3)
    costs = batched_nested_costs_from_plans(plans, scenario)
    return costs[:, agent_idx].reshape((B, M_inner)).mean(axis=1)
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_batched_rne_solver.py
```

Expected: PASS.

- [ ] **Step 5: Commit f-hat helper**

```bash
cd /home/lenovo/mpcc/MGIGO
git add Cartest/planning/batched_game_eval.py Cartest/eval/test_batched_rne_solver.py
git commit -m "test: compare batched rne expected costs"
```

---

### Task 5: Replace Flattened Joint Batch with True Cached Candidate/Background Plans

**Files:**
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_game_eval.py`
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_rne_solver.py`

- [ ] **Step 1: Add an instrumentation test proving trajectory evaluation count is reduced**

Append to `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_rne_solver.py`:

```python
def test_batched_f_hat_uses_b_plus_m_trajectory_evaluations(monkeypatch=None):
    from Cartest.planning import batched_game_eval

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    ctx = build_multi_agent_context(_states(scenario))
    samples_b, samples_m = _make_samples(gen, scenario)

    counts = {"calls": 0}
    original = batched_game_eval.evaluate_agent_control_batch

    def counted(*args, **kwargs):
        counts["calls"] += 1
        return original(*args, **kwargs)

    batched_game_eval.evaluate_agent_control_batch = counted
    try:
        _ = batched_game_eval.batched_expected_cost_for_agent(gen, samples_b, samples_m, ctx, scenario, 0)
    finally:
        batched_game_eval.evaluate_agent_control_batch = original

    # Agent 0 needs: ego B candidates + front M backgrounds + rear M backgrounds.
    assert counts["calls"] == 3
```

- [ ] **Step 2: Run and confirm the instrumentation test fails**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_batched_rne_solver.py
```

Expected: FAIL because `evaluate_agent_control_batch` is missing or because the flattened implementation evaluates all `B*M_inner` joints.

- [ ] **Step 3: Implement true cached plan batches**

Add this function to `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_game_eval.py`:

```python
def evaluate_agent_control_batch(gen, ctrl_s_batch, ctrl_d_batch, ctx, agent_idx):
    """Evaluate controls shaped [batch, n_free] for one agent only."""
    a_ctx = agent_ctx(ctx, agent_idx)

    def one(ctrl_s, ctrl_d):
        frenet, vehicle, (x, y) = gen.evaluate_plan(ctrl_s, ctrl_d, a_ctx)
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = frenet
        return {
            "s": s,
            "d": d,
            "s_dot": s_dot,
            "d_dot": d_dot,
            "s_ddot": s_ddot,
            "d_ddot": d_ddot,
            "s_dddot": s_dddot,
            "d_dddot": d_dddot,
            "vehicle": vehicle,
            "x": x,
            "y": y,
        }

    return vmap(one)(ctrl_s_batch, ctrl_d_batch)
```

Then replace `batched_expected_cost_for_agent()` with an implementation that:

```python
def _plans_for_agent_source(gen, source, ctx, agent_idx):
    s_block = agent_idx * 2
    d_block = s_block + 1
    return evaluate_agent_control_batch(gen, source[s_block], source[d_block], ctx, agent_idx)


def _take_plan(plan, idx):
    return {key: value[idx] for key, value in plan.items()}


def _broadcast_pair_plan(candidate_plan, background_plan, B, M_inner, use_candidate):
    if use_candidate:
        return {key: jnp.repeat(value[:, None, ...], M_inner, axis=1).reshape((B * M_inner,) + value.shape[1:])
                for key, value in candidate_plan.items()}
    return {key: jnp.repeat(value[None, :, ...], B, axis=0).reshape((B * M_inner,) + value.shape[1:])
            for key, value in background_plan.items()}


def batched_expected_cost_for_agent(gen, samples_b, samples_m, ctx, scenario, agent_idx):
    B = samples_b.shape[1]
    M_inner = samples_m.shape[1]
    candidate_plans = tuple(_plans_for_agent_source(gen, samples_b, ctx, aid) for aid in range(3))
    background_plans = tuple(_plans_for_agent_source(gen, samples_m, ctx, aid) for aid in range(3))
    plans = tuple(
        _broadcast_pair_plan(candidate_plans[aid], background_plans[aid], B, M_inner, aid == agent_idx)
        for aid in range(3)
    )
    costs = batched_nested_costs_from_plans(plans, scenario)
    return costs[:, agent_idx].reshape((B, M_inner)).mean(axis=1)
```

- [ ] **Step 4: Run all batched evaluator tests**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_batched_three_agent_eval.py
.venv/bin/python Cartest/eval/test_batched_rne_solver.py
```

Expected: both PASS.

- [ ] **Step 5: Commit true caching implementation**

```bash
cd /home/lenovo/mpcc/MGIGO
git add Cartest/planning/batched_game_eval.py Cartest/eval/test_batched_rne_solver.py
git commit -m "feat: cache three-agent candidate and background trajectories"
```

---

### Task 6: Add Cartest Batched RNE Solver Wrapper

**Files:**
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_rne_solver.py`
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_rne_solver.py`

- [ ] **Step 1: Add small solver API test**

Append to `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_batched_rne_solver.py`:

```python
def test_cartest_batched_rne_solver_runs_small_problem():
    from Cartest.planning.batched_rne_solver import cartest_batched_rne_blocks_solver

    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"] = dict(scenario["game"], T=2, B=4, B0=2, M_inner=3, K=2)
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, L_inv = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(5))
    mu = mu[:, :2]
    L_inv = L_inv[:, :2]

    result = cartest_batched_rne_blocks_solver(
        jax.random.PRNGKey(6), gen, scenario, context=ctx,
        initial_mu=mu, initial_L_inv=L_inv,
    )

    assert result["mu"].shape == mu.shape
    assert result["L_inv"].shape == L_inv.shape
    assert result["pi"].shape == (6, 2)
    assert jnp.all(jnp.isfinite(result["mu"]))
```

- [ ] **Step 2: Run and confirm it fails because solver module is missing**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_batched_rne_solver.py
```

Expected: FAIL with missing `Cartest.planning.batched_rne_solver`.

- [ ] **Step 3: Implement the solver wrapper**

Create `/home/lenovo/mpcc/MGIGO/Cartest/planning/batched_rne_solver.py` with a local implementation that mirrors the non-public math in `MPC_G_MS.py` but calls `batched_expected_cost_for_agent()` for `f_hat`.

Use these public signatures:

```python
def cartest_batched_rne_blocks_solver(
    key,
    gen,
    scenario,
    *,
    context,
    initial_mu,
    initial_L_inv,
    initial_v=None,
):
    """Run Cartest-specific batched RNE blocks and return dict diagnostics.

    Return keys:
      mu: [N_blocks, K, D]
      L_inv: [N_blocks, K, D, D]
      pi: [N_blocks, K]
      v: [N_blocks, K - 1]
      metrics: dict with mean_fitness history
    """
```

Implementation requirements:

```text
1. Read T, dt, K, B, B0, T_0, M_inner, block_to_agent from scenario["game"].
2. Convert initial_L_inv to precision S = L @ L.T.
3. In each scan iteration:
   a. sample samples_B [N_blocks, B, D]
   b. sample samples_M [N_blocks, M_inner, D]
   c. for aid in 0..2 call batched_expected_cost_for_agent(...)
   d. compute elite weights from f_hat
   e. broadcast agent weights back to blocks
   f. update block Gaussian components using the same equations as MPC_G_MS.py
4. Return final Cholesky precision factors and pi.
```

Do not import or call `_step_fn_rne_blocks`.

- [ ] **Step 4: Run small solver test**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_batched_rne_solver.py
```

Expected: PASS.

- [ ] **Step 5: Commit solver wrapper**

```bash
cd /home/lenovo/mpcc/MGIGO
git add Cartest/planning/batched_rne_solver.py Cartest/eval/test_batched_rne_solver.py
git commit -m "feat: add Cartest batched RNE blocks solver"
```

---

### Task 7: Route the New Solver Mode Through Cartest

**Files:**
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/planning/solver_modes.py`
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/three_agent_track.py`
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_game_solver_modes.py`

- [ ] **Step 1: Add routing test**

Append to `/home/lenovo/mpcc/MGIGO/Cartest/eval/test_game_solver_modes.py`:

```python
def test_three_agent_track_uses_cartest_batched_rne_solver_mode():
    scenario = get_scenario("three_agent_track")
    assert scenario["game"]["solver"] == "cartest_batched_rne_blocks"
```

- [ ] **Step 2: Run and confirm it fails before switching scenario**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_game_solver_modes.py
```

Expected: FAIL because scenario still says `rne_blocks`.

- [ ] **Step 3: Add solver mode branch**

Modify `/home/lenovo/mpcc/MGIGO/Cartest/planning/solver_modes.py` in `build_cartest_nash_solver()`:

```python
    if game.get("solver") == "cartest_batched_rne_blocks":
        from Cartest.planning.batched_rne_solver import cartest_batched_rne_blocks_solver

        def _solve(key, context=None, initial_mu=None, initial_S_or_L=None, initial_pi=None, warm_start=None):
            del initial_pi
            mu = initial_mu
            L_inv = initial_S_or_L
            v = None
            if warm_start is not None:
                ws = warm_start if isinstance(warm_start, dict) else warm_start.__dict__
                if mu is None:
                    mu = ws.get("mu", ws.get("initial_mu"))
                if L_inv is None:
                    L_inv = ws.get("L_inv", ws.get("S_or_L"))
                v = ws.get("v")
            raw = cartest_batched_rne_blocks_solver(
                key, gen, scenario, context=context,
                initial_mu=mu, initial_L_inv=L_inv, initial_v=v,
            )
            from gmm_igo.solver_builder import NashResult, SolverResult
            solutions = {}
            joint_parts = []
            block_to_agent = tuple(game["block_to_agent"])
            for aid in range(len(scenario["agents"])):
                my_blocks = [b for b, owner in enumerate(block_to_agent) if owner == aid]
                best_k = jnp.argmax(raw["pi"][jnp.array(my_blocks)], axis=1)
                x_agent = jnp.concatenate([
                    raw["mu"][block, best_k[j], :gen.n_free]
                    for j, block in enumerate(my_blocks)
                ])
                joint_parts.append(x_agent)
                solutions[aid] = SolverResult(
                    x=x_agent, cost=0.0,
                    mu=raw["mu"][jnp.array(my_blocks)],
                    S_or_L=raw["L_inv"][jnp.array(my_blocks)],
                    pi=raw["pi"][jnp.array(my_blocks)],
                    solver_name=f"nash_cartest_batched_rne_blocks_agent{aid}",
                )
            return NashResult(solutions=solutions, diag=raw)

        return _solve
```

If `NashResult` or `SolverResult` constructors differ, inspect `/home/lenovo/mpcc/MGIGO/gmm_igo/solver_builder.py` and use the exact current dataclass fields.

- [ ] **Step 4: Switch only the three-agent scenario**

Modify `/home/lenovo/mpcc/MGIGO/Cartest/planning/scenarios/three_agent_track.py`:

```python
"solver": "cartest_batched_rne_blocks",
```

Keep `T`, `B`, and `M_inner` unchanged.

- [ ] **Step 5: Run routing and small scenario tests**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/test_game_solver_modes.py
.venv/bin/python Cartest/eval/test_three_agent_track_game.py
.venv/bin/python Cartest/eval/test_unified_simple_runner.py
```

Expected: all PASS.

- [ ] **Step 6: Commit routing**

```bash
cd /home/lenovo/mpcc/MGIGO
git add Cartest/planning/solver_modes.py Cartest/planning/scenarios/three_agent_track.py Cartest/eval/test_game_solver_modes.py
git commit -m "feat: route three-agent track through batched rne solver"
```

---

### Task 8: Add GPU Benchmark and Compare Against Generic RNE

**Files:**
- Create: `/home/lenovo/mpcc/MGIGO/Cartest/eval/benchmark_three_agent_batched.py`

- [ ] **Step 1: Add benchmark script**

Create `/home/lenovo/mpcc/MGIGO/Cartest/eval/benchmark_three_agent_batched.py`:

```python
"""Benchmark generic vs Cartest batched three-agent RNE on GPU."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import time

import jax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Cartest.core.frenet_traj import FrenetBSplineTrajectory
from Cartest.execution.execute import FrenetState
from Cartest.planning.scenarios import get_scenario
from Cartest.planning.solver_modes import build_cartest_nash_solver, build_multi_agent_context, build_multi_agent_warmstart


BASIS = ROOT / "Cartest" / "basis" / "bspline_basis.npz"


def _states(scenario):
    return [
        FrenetState(
            s=agent["s"], s_dot=agent["s_dot"], s_ddot=agent.get("s_ddot", 0.0),
            d=agent["d"], d_dot=agent.get("d_dot", 0.0), d_ddot=agent.get("d_ddot", 0.0),
            psi=agent.get("psi", 0.0),
        )
        for agent in scenario["agents"]
    ]


def _block(result):
    leaves = jax.tree_util.tree_leaves(result)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def run_one(label, solver_name):
    scenario = copy.deepcopy(get_scenario("three_agent_track"))
    scenario["game"] = dict(scenario["game"], solver=solver_name)
    gen = FrenetBSplineTrajectory(BASIS, scenario["ref_path"])
    states = _states(scenario)
    ctx = build_multi_agent_context(states)
    mu, L_inv = build_multi_agent_warmstart(gen, scenario, states, jax.random.PRNGKey(10))
    solver = build_cartest_nash_solver(gen, scenario)

    t0 = time.perf_counter()
    r0 = solver(jax.random.PRNGKey(11), context=ctx, initial_mu=mu, initial_S_or_L=L_inv)
    _block(r0)
    compile_plus_first = time.perf_counter() - t0

    t1 = time.perf_counter()
    r1 = solver(jax.random.PRNGKey(12), context=ctx, initial_mu=mu, initial_S_or_L=L_inv)
    _block(r1)
    steady = time.perf_counter() - t1
    print(f"{label}: compile+first={compile_plus_first:.3f}s steady={steady:.3f}s backend={jax.default_backend()}")


def main():
    print("devices:", jax.devices())
    run_one("generic_rne_blocks", "rne_blocks")
    run_one("cartest_batched_rne_blocks", "cartest_batched_rne_blocks")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark on host GPU**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/eval/benchmark_three_agent_batched.py
```

Expected:

```text
devices: [CudaDevice(...)]
generic_rne_blocks: compile+first=...s steady=...s backend=gpu
cartest_batched_rne_blocks: compile+first=...s steady=...s backend=gpu
```

Record both steady-state times in the final report.

- [ ] **Step 3: Commit benchmark**

```bash
cd /home/lenovo/mpcc/MGIGO
git add Cartest/eval/benchmark_three_agent_batched.py
git commit -m "bench: compare generic and batched three-agent rne"
```

---

### Task 9: Closed-Loop Smoke Test and Video Verification

**Files:**
- Runtime output: `/home/lenovo/mpcc/MGIGO/Cartest/output/*.mp4`

- [ ] **Step 1: Run one no-plot step on GPU**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/Simple.py three_agent_track --steps 1 --no-plot
```

Expected: exits zero, prints one `step 00` line, no Python exceptions.

- [ ] **Step 2: Run a short plotted video**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
.venv/bin/python Cartest/Simple.py three_agent_track --steps 5
```

Expected: exits zero and writes a timestamped MP4 under `/home/lenovo/mpcc/MGIGO/Cartest/output/`.

- [ ] **Step 3: Inspect generated output path**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
ls -t Cartest/output/three_agent_track_*.mp4 | head -1
```

Expected: newest MP4 path is printed.

- [ ] **Step 4: Commit only if code changed during smoke fixes**

If a small smoke-test fix was needed:

```bash
cd /home/lenovo/mpcc/MGIGO
git add <changed-files>
git commit -m "fix: stabilize batched three-agent smoke run"
```

If no code changed, skip this commit.

---

### Task 10: Documentation

**Files:**
- Modify: `/home/lenovo/mpcc/MGIGO/Cartest/visualization/README.md`
- Optionally modify: `/home/lenovo/mpcc/MGIGO/README.md`

- [ ] **Step 1: Add implementation note to visualization README**

Append this section to `/home/lenovo/mpcc/MGIGO/Cartest/visualization/README.md`:

```markdown
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
```

- [ ] **Step 2: Run docs grep sanity check**

Run:

```bash
cd /home/lenovo/mpcc/MGIGO
rg -n "cartest_batched_rne_blocks|B x M_inner|Frenet" Cartest/visualization/README.md
```

Expected: the new section lines are printed.

- [ ] **Step 3: Commit docs**

```bash
cd /home/lenovo/mpcc/MGIGO
git add Cartest/visualization/README.md
git commit -m "docs: explain batched three-agent rne visualization path"
```

---

## Final Verification Commands

Run these from `/home/lenovo/mpcc/MGIGO`:

```bash
.venv/bin/python Cartest/eval/test_batched_three_agent_eval.py
.venv/bin/python Cartest/eval/test_batched_rne_solver.py
.venv/bin/python Cartest/eval/test_game_solver_modes.py
.venv/bin/python Cartest/eval/test_three_agent_track_game.py
.venv/bin/python Cartest/eval/test_unified_simple_runner.py
.venv/bin/python Cartest/eval/benchmark_three_agent_batched.py
.venv/bin/python Cartest/Simple.py three_agent_track --steps 1 --no-plot
.venv/bin/python Cartest/Simple.py three_agent_track --steps 5
```

Success criteria:

- All tests pass.
- Benchmark reports `backend=gpu` and shows batched steady-state faster than generic steady-state.
- `three_agent_track` still uses `T=300`, `B=60`, `M_inner=30`.
- A timestamped MP4 is generated in `/home/lenovo/mpcc/MGIGO/Cartest/output/`.
- `/home/lenovo/mpcc/MGIGO/gmm_igo/MPC_G_MS.py` has no diff.

## Self-Review Notes

- Spec coverage: the plan addresses the user's hypothesis by moving expensive trajectory evaluation out of the `B x M_inner` loop, keeps solver parameters unchanged, preserves the existing command path, and explicitly avoids editing `MPC_G_MS.py`.
- Risk: exact Constran nested transform constants may differ from the simplified implementation shown in Task 3. The equivalence test is intentionally placed before solver integration so this mismatch is caught early and aligned against `/home/lenovo/mpcc/MGIGO/Constraintdealer/Constran.py`.
- Risk: a fully JIT-compatible monkeypatch instrumentation test may need to run outside JIT. It is only a structural guard for Python helper decomposition; numeric equivalence tests are the primary correctness checks.
