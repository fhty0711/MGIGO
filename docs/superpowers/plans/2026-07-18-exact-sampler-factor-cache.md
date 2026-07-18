# Exact Sampler Factor Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache the 18 Cartest RNE component covariance factors once per optimizer iteration and share them between the B and M sample pools while preserving every sample and complete solver output bit-for-bit.

**Architecture:** Add pure factorization and factor-based sampling helpers to `batched_rne_solver.py`, plus one iteration-pool helper that owns the single factorization and the unchanged B/M key split. Keep `_sample_all_blocks` as an exact compatibility wrapper. Route the JIT scan through the iteration-pool helper without changing costs, selection, updates, or returned diagnostics.

**Tech Stack:** Python 3.10+, JAX/JAX NumPy, pytest, Windows CPU JAX, WSL2 CUDA JAX.

## Global Constraints

- Direct samples must satisfy `jnp.array_equal`; tolerance-based equivalence is insufficient.
- Complete solver `mu`, `L_inv`, `pi`, `v`, and `metrics` must be elementwise identical.
- Preserve block keys, component-choice keys, sample keys, normal draw shape, and dtype.
- Do not change `gmm_igo.MPC_G_MS`, cost definitions, scenario parameters, or iteration budgets.
- Do not enable or reintroduce `_sample_all_blocks_fast`.
- Preserve the user's existing `Cartest/basis/spline.py` worktree change.

---

### Task 1: Lock the exact sampler contract with failing tests

**Files:**
- Modify: `Cartest/eval/test_batched_rne_solver.py`
- Test: `Cartest/eval/test_batched_rne_solver.py`

**Interfaces:**
- Consumes: existing `_sample_all_blocks(mu, S, pi_all, count, key)` behavior.
- Produces: required helpers `_precision_covariance_factors(S)`, `_sample_all_blocks_from_factors(mu, factors, pi_all, count, key)`, and `_sample_iteration_pools(mu, S, pi_all, B, M_inner, key)`.

- [ ] **Step 1: Add a test-local legacy reference and a failing direct-equivalence test**

Add a `_legacy_sample_all_blocks` function that copies the current sampler exactly. Add `test_factor_cached_sampler_matches_legacy_stream_exactly`, which first asserts that the three required production helpers exist, then compares B and M pools for multiple keys using `jnp.array_equal` and a nonuniform mixture.

```python
def _legacy_sample_all_blocks(mu, S, pi_all, count, key):
    block_count, component_count = mu.shape[:2]

    def sample_block(block_index, block_key):
        components = jax.random.choice(
            block_key, component_count, p=pi_all[block_index], shape=(count,))

        def sample_one(component, sample_key):
            cov = jnp.linalg.inv(
                S[block_index, component]
                + jnp.eye(S.shape[-1], dtype=S.dtype) * 1e-7)
            return jax.random.multivariate_normal(
                sample_key, mu[block_index, component], cov)

        return jax.vmap(sample_one)(
            components, jax.random.split(block_key, count))

    return jax.vmap(sample_block)(
        jnp.arange(block_count), jax.random.split(key, block_count))
```

- [ ] **Step 2: Run the direct test and verify RED**

Run:

```powershell
python -m pytest Cartest/eval/test_batched_rne_solver.py -k factor_cached_sampler_matches_legacy_stream_exactly -q
```

Expected: FAIL because `_precision_covariance_factors`, `_sample_all_blocks_from_factors`, and `_sample_iteration_pools` do not yet exist.

- [ ] **Step 3: Add a failing complete-solver equivalence test**

Build the optimized small solver normally. Then temporarily patch `_sample_iteration_pools` with a test-local legacy B/M implementation, build and trace a reference solver with the same key and inputs, and assert `jnp.array_equal` for `mu`, `L_inv`, `pi`, `v`, and every metrics leaf.

- [ ] **Step 4: Run the complete-solver test and verify RED**

Run:

```powershell
python -m pytest Cartest/eval/test_batched_rne_solver.py -k complete_solver_matches_legacy_sampler_exactly -q
```

Expected: FAIL because the iteration-pool seam is missing.

### Task 2: Implement factor caching and route the scan through it

**Files:**
- Modify: `Cartest/planning/solvers/batched_rne_solver.py:55-76`
- Modify: `Cartest/planning/solvers/batched_rne_solver.py:122-126`
- Test: `Cartest/eval/test_batched_rne_solver.py`

**Interfaces:**
- Consumes: the helper contracts established by Task 1.
- Produces: an exact compatibility `_sample_all_blocks` and a scan that factors once per iteration for both sample pools.

- [ ] **Step 1: Implement `_precision_covariance_factors`**

```python
def _precision_covariance_factors(S):
    eye = jnp.eye(S.shape[-1], dtype=S.dtype) * 1e-7
    covariance = vmap(vmap(jnp.linalg.inv))(S + eye)
    return vmap(vmap(jnp.linalg.cholesky))(covariance)
```

- [ ] **Step 2: Implement factor-based sampling without changing keys**

```python
def _sample_all_blocks_from_factors(mu, factors, pi_all, count, key):
    block_count, component_count = mu.shape[:2]

    def sample_block(block_index, block_key):
        components = random.choice(
            block_key, component_count, p=pi_all[block_index], shape=(count,))

        def sample_one(component, sample_key):
            noise = random.normal(sample_key, (mu.shape[-1],), dtype=mu.dtype)
            return mu[block_index, component] + jnp.einsum(
                "ij,j->i", factors[block_index, component], noise)

        return vmap(sample_one)(components, random.split(block_key, count))

    return vmap(sample_block)(
        jnp.arange(block_count), random.split(key, block_count))
```

- [ ] **Step 3: Implement the compatibility wrapper and shared iteration pools**

```python
def _sample_all_blocks(mu, S, pi_all, count, key):
    return _sample_all_blocks_from_factors(
        mu, _precision_covariance_factors(S), pi_all, count, key)


def _sample_iteration_pools(mu, S, pi_all, B, M_inner, key):
    factors = _precision_covariance_factors(S)
    key_B, key_M = random.split(key)
    return (
        _sample_all_blocks_from_factors(mu, factors, pi_all, B, key_B),
        _sample_all_blocks_from_factors(mu, factors, pi_all, M_inner, key_M),
    )
```

- [ ] **Step 4: Route the solver step through `_sample_iteration_pools`**

Replace the two independent `_sample_all_blocks` calls with:

```python
samples_B, samples_M = _sample_iteration_pools(
    mu_t, S_t, pi_all, B, M_inner, step_key)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest Cartest/eval/test_batched_rne_solver.py -k "factor_cached_sampler or complete_solver_matches_legacy_sampler" -q
```

Expected: both tests PASS with exact equality.

- [ ] **Step 6: Run the complete related CPU test suite**

Run:

```powershell
python -m pytest Cartest/eval/test_batched_rne_solver.py Cartest/eval/test_game_solver_modes.py Cartest/eval/test_three_agent_track_cost.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 7: Verify CUDA equality and performance**

Run the direct multi-seed sampler comparison and T=100/T=300 synchronized solver comparison with `/root/.venvs/tcmgigo-jaxgpu/bin/python` in WSL2. Required correctness output:

```text
array_equal=True
max_abs=0
```

Record compile time separately and compare steady-state medians only.

- [ ] **Step 8: Review diff and commit implementation**

Run:

```powershell
git diff --check
git status --short
```

Stage only the solver, tests, and this plan. Do not stage `Cartest/basis/spline.py`.

```powershell
git add -- Cartest/planning/solvers/batched_rne_solver.py Cartest/eval/test_batched_rne_solver.py docs/superpowers/plans/2026-07-18-exact-sampler-factor-cache.md
git commit -m "perf(cartest): cache exact sampler factors"
```
