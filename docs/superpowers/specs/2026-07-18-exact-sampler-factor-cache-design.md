# Exact sampler factor-cache design

## Goal

Reduce the three-agent batched RNE sampling cost without changing any sample,
PRNG stream, solver update, nested cost, or selected plan.

The optimized implementation must be elementwise identical to the current
sampler on both CPU and CUDA for the same inputs and key.

## Current behavior

For every optimization iteration, the Cartest batched solver creates a B
candidate pool and an M background pool. Each pool independently performs the
following operation for every sampled component:

1. choose `count` component indices from the block key;
2. split the same block key into `count` sample keys;
3. invert the selected component precision matrix;
4. Cholesky-factor the covariance inside `random.multivariate_normal`;
5. draw one normal vector from the corresponding sample key.

With six blocks, three components, B=60, and M=30, this repeats small matrix
factorizations hundreds of times even though only 18 distinct component
matrices exist in an iteration.

## Chosen design

Split sampling into two pure helpers:

- `_precision_covariance_factors(S)` computes
  `cholesky(inv(S + 1e-7 I))` once for every block/component.
- `_sample_all_blocks_from_factors(mu, factors, pi, count, key)` performs the
  existing component choice, block-key split, sample-key split, normal draw,
  and affine transform.

The optimizer step computes the factor tensor once and passes it to both the B
and M sampling calls. `_sample_all_blocks(...)` remains as a compatibility
wrapper that computes factors and delegates to the new sampling helper.

The implementation deliberately preserves:

- `random.split(key, N_blocks)` for block keys;
- `random.choice(block_key, ...)` for component indices;
- `random.split(block_key, count)` for sample keys;
- one `random.normal(sample_key, (D,), dtype=mu.dtype)` per sample;
- the same `mean + einsum("ij,j->i", factor, noise)` computation used by
  JAX's Cholesky `random.multivariate_normal` implementation.

## Rejected alternatives

### Cache independently inside each pool

This preserves samples but still computes the same 18 factors twice per
iteration. It is useful as an intermediate reference, not as the final solver
path.

### Sample directly from the precision Cholesky

Triangular solves generate the correct distribution but do not produce the
same sample values for a fixed normal vector. This violates strict equivalence.

### Group samples by component

Grouping improves batching but changes sample-key assignment and therefore the
random stream. This is the behavior rejected for `_sample_all_blocks_fast`.

## Tests

1. A test-local copy of the legacy sampler is the reference implementation.
2. For nontrivial means, SPD precision matrices, mixture probabilities, pool
   sizes, and multiple keys, assert `array_equal` between the reference and:
   - the compatibility `_sample_all_blocks` wrapper;
   - factor-once `_sample_all_blocks_from_factors` calls for B and M keys.
3. Assert B/M factor sharing does not change the complete small solver outputs:
   `mu`, `L_inv`, `pi`, `v`, and `metrics`.
4. Run the existing Cartest batched/scalar equivalence tests.
5. Repeat direct sample and complete-solver equality checks in WSL/CUDA.
6. Benchmark synchronized steady-state T=100 and T=300 calls after compilation.

No tolerance is accepted for the direct sample or complete-solver checks:
required difference is exactly zero.

## Scope

Only the Cartest-specific batched RNE solver and its tests change. The generic
`gmm_igo.MPC_G_MS` solver, cost definitions, scenario parameters, iteration
budget, and experimental profiling scripts remain unchanged.
