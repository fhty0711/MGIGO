# Batched RNE Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct tie handling, GMM weight initialization/reset, asymmetric collision horizon, and duplicated constraint metadata in the Cartest batched RNE path while preserving its current PRNG stream.

**Architecture:** Add small pure helpers for elite utilities and mixture-weight conversion/reset so their edge cases can be tested independently. Centralize the three-agent constraint layer definitions in the scalar cost module and let the batched evaluator compile its static nesting tuple from those definitions. Both collision implementations read `scenario["game"]["execute_index"]` and check through that sample.

**Tech Stack:** Python 3, JAX/JAX NumPy, existing script-style Cartest tests, WSL2 CUDA smoke test.

## Global Constraints

- Keep `_sample_all_blocks` as the production sampler.
- Preserve the existing PRNG split and sample stream.
- Tie handling uses exact equality and preserves total utility mass `B0 / B`.
- Ego collision checking remains full horizon; front and rear check through `execute_index`.
- Explicit solver arguments take precedence over `warm_start` values.
- Preserve public solver call signatures.

---

### Task 1: Tie-aware elite utilities

**Files:**
- Modify: `Cartest/planning/batched_rne_solver.py`
- Test: `Cartest/eval/test_batched_rne_solver.py`

**Interfaces:**
- Produces: `_tie_aware_elite_weights(f_hats, elite_count)` returning an array with the same shape as `f_hats`.
- Consumes: two-dimensional agent-by-sample costs in the solver and one-dimensional arrays in unit tests.

- [ ] **Step 1: Write failing utility tests**

Add tests that import `_tie_aware_elite_weights` and assert:

```python
assert jnp.allclose(
    _tie_aware_elite_weights(jnp.array([0., 1., 2., 3.]), 2),
    jnp.array([0.25, 0.25, 0., 0.]))
assert jnp.allclose(
    _tie_aware_elite_weights(jnp.array([0., 1., 1., 3.]), 2),
    jnp.array([0.25, 0.125, 0.125, 0.]))
assert jnp.allclose(
    _tie_aware_elite_weights(jnp.ones(4), 2),
    jnp.full(4, 0.125))
```

Also test a `[2, 4]` batch and assert each row sums to `2 / 4`.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 bash -lc 'cd /mnt/d/MGIGO/MGIGO && JAX_PLATFORMS=cpu /root/.venvs/tcmgigo-jaxgpu/bin/python -m pytest Cartest/eval/test_batched_rne_solver.py -k tie_aware -q'
```

Expected: collection/import failure because `_tie_aware_elite_weights` does not exist.

- [ ] **Step 3: Implement and connect the helper**

Implement the threshold rule using sorted costs, strict/tied masks, and remaining mass. Replace the two `argsort` calls and `jnp.where` with:

```python
agent_w_hat = _tie_aware_elite_weights(f_hats, B0)
```

Do not touch either sampling function or key splitting.

- [ ] **Step 4: Run targeted and existing solver tests**

Run the tie tests and then the complete `test_batched_rne_solver.py`. Expected: tie tests pass; the exact generic-vs-batched test remains green for its non-tied fixture.

- [ ] **Step 5: Commit**

```powershell
git add Cartest/planning/batched_rne_solver.py Cartest/eval/test_batched_rne_solver.py
git commit -m "fix: handle tied batched RNE elites"
```

### Task 2: Preserve initial mixture weights

**Files:**
- Modify: `Cartest/planning/batched_rne_solver.py`
- Modify: `Cartest/planning/solver_modes.py`
- Test: `Cartest/eval/test_batched_rne_solver.py`
- Test: `Cartest/eval/test_game_solver_modes.py`

**Interfaces:**
- Produces: `_should_reset_mixture_weights(t, period)` and `_pi_to_v(initial_pi)`.
- `_pi_to_v` accepts positive finite probabilities whose last axis sums to one and returns `log(pi[..., :-1] / pi[..., -1:])`.

- [ ] **Step 1: Write failing tests**

Test that reset is false at `t=0`, true at `t=T_0`, and false otherwise. Test that `_pi_to_v([[0.2, 0.3, 0.5]])` maps back through `_v_to_pi` to the original probabilities. Add invalid zero, negative, non-finite, and non-unit-sum cases expecting `ValueError`. Add a wrapper test showing explicit `initial_pi` overrides `warm_start["v"]`.

- [ ] **Step 2: Run the new tests and verify RED**

Run both test files filtered to `initial_pi or reset_mixture`. Expected: missing helper failures and the wrapper still ignoring `initial_pi`.

- [ ] **Step 3: Implement minimal conversion and reset behavior**

Use:

```python
def _should_reset_mixture_weights(t, period):
    return jnp.logical_and(t > 0, (t % period) == 0)
```

Validate `initial_pi` outside JIT with NumPy, convert it to `v`, and apply explicit-argument precedence. Change the scan reset condition to call the helper.

- [ ] **Step 4: Run targeted tests and the two complete files**

Expected: all tests pass without changes to sampled values when no `initial_pi` is supplied.

- [ ] **Step 5: Commit**

```powershell
git add Cartest/planning/batched_rne_solver.py Cartest/planning/solver_modes.py Cartest/eval/test_batched_rne_solver.py Cartest/eval/test_game_solver_modes.py
git commit -m "fix: preserve batched RNE mixture initialization"
```

### Task 3: Align collision horizon with MPC execution

**Files:**
- Modify: `Cartest/planning/costs/three_agent_track.py`
- Modify: `Cartest/planning/batched_game_eval.py`
- Test: `Cartest/eval/test_batched_three_agent_eval.py`

**Interfaces:**
- Produces: `_collision_prefix(scenario)` returning `slice(0, execute_index + 1)`.
- Both scalar and batched collision implementations consume the same helper.

- [ ] **Step 1: Write failing scalar and batched collision tests**

Construct five-sample fake plans whose only geometric collision is at index 3. With `execute_index=3`, assert front and rear collision violations at index 3 are positive in both `_violations_for_agent` and the scalar `Deterministic.g_fn`. Assert index 4 remains zero for the short-horizon geometric term.

- [ ] **Step 2: Run tests and verify RED**

Run `test_batched_three_agent_eval.py -k collision_prefix -q`. Expected: index 3 remains zero because the implementation uses `slice(0, 2)`.

- [ ] **Step 3: Implement shared prefix selection**

Define `_collision_prefix` in `costs/three_agent_track.py`, use it in scalar constraints, import it in `batched_game_eval.py`, and replace all hard-coded `:2` slices. Preserve rear/front longitudinal clearance across the full horizon.

- [ ] **Step 4: Run the complete batched evaluator tests**

Expected: new tests pass and scalar/batched nested costs remain within existing tolerances.

- [ ] **Step 5: Commit**

```powershell
git add Cartest/planning/costs/three_agent_track.py Cartest/planning/batched_game_eval.py Cartest/eval/test_batched_three_agent_eval.py
git commit -m "fix: cover executed state in collision checks"
```

### Task 4: Centralize three-agent constraint metadata

**Files:**
- Modify: `Cartest/planning/costs/three_agent_track.py`
- Modify: `Cartest/planning/batched_game_eval.py`
- Test: `Cartest/eval/test_batched_three_agent_eval.py`

**Interfaces:**
- Produces: `THREE_AGENT_CONSTRAINT_DEFS`, ordered tuples containing `name`, `mode`, `priority`, `aggregate`, and `transform`.
- Produces: `three_agent_batched_layers()` returning `(name, aggregate, table, baseline, resolution)` tuples compiled through real `Deterministic` specs.

- [ ] **Step 1: Write failing metadata tests**

Assert that the shared definitions contain exactly lane, speed, acc, jerk, and collision in priority order. For each scalar spec returned by `make_agent_specs`, assert mode, priority, aggregate, transform, baseline, and transform table match the corresponding batched layer.

- [ ] **Step 2: Run the metadata test and verify RED**

Expected: shared definitions and compiler function do not exist.

- [ ] **Step 3: Implement shared definitions and consume them**

Build scalar `Deterministic` objects by pairing the shared definitions with the five violation functions. Compile `_THREE_AGENT_LAYERS` by calling `three_agent_batched_layers()` instead of restating constants in `batched_game_eval.py`.

- [ ] **Step 4: Run all four regression files**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 bash -lc 'cd /mnt/d/MGIGO/MGIGO && export JAX_PLATFORMS=cpu; for f in Cartest/eval/test_batched_three_agent_eval.py Cartest/eval/test_batched_rne_solver.py Cartest/eval/test_game_solver_modes.py Cartest/eval/test_three_agent_track_game.py; do /root/.venvs/tcmgigo-jaxgpu/bin/python "$f" || exit 1; done'
```

Expected: all scripts print their success messages and exit zero.

- [ ] **Step 5: Commit**

```powershell
git add Cartest/planning/costs/three_agent_track.py Cartest/planning/batched_game_eval.py Cartest/eval/test_batched_three_agent_eval.py
git commit -m "refactor: share three-agent constraint metadata"
```

### Task 5: Final verification

**Files:**
- Verify only; no production changes expected.

**Interfaces:**
- Consumes all preceding changes.

- [ ] **Step 1: Verify no PRNG-path change**

Run `git diff origin/llss -- Cartest/planning/batched_rne_solver.py` and confirm `_sample_all_blocks`, calls at candidate/background sampling, and `random.split` structure are unchanged.

- [ ] **Step 2: Run CPU regression suite**

Run the four scripts from Task 4 and `git diff --check`. Expected: zero exits and no whitespace errors.

- [ ] **Step 3: Run WSL/CUDA smoke test**

```powershell
wsl.exe -d Ubuntu-22.04 bash -lc 'cd /mnt/d/MGIGO/MGIGO && unset JAX_PLATFORMS; XLA_PYTHON_CLIENT_PREALLOCATE=false /root/.venvs/tcmgigo-jaxgpu/bin/python Cartest/Simple.py three_agent_track --steps 2 --no-plot'
```

Expected: two finite three-agent reports with solve times and no JAX/CUDA errors.

- [ ] **Step 4: Review repository status**

Run `git status --short --branch` and `git log --oneline origin/llss..HEAD`. Expected: only intentional commits and a clean worktree.
