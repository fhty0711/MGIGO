# Batched RNE correctness fixes

## Scope

Fix correctness and maintenance gaps in the `three_agent_track` batched RNE
path without changing its sampling algorithm or PRNG stream. CUDA work and the
unused `_sample_all_blocks_fast` path are explicitly out of scope.

## Required behavior

### Tie-aware elite weights

Replace double-`argsort` rank selection with a threshold-based utility. Samples
strictly below the `B0`-th cost receive weight `1 / B`. Samples exactly equal
to the threshold share the remaining elite mass uniformly. The total weight is
therefore always `B0 / B`, including all-equal and partially tied inputs.
Non-tied inputs select the same elite set and weights as the current code.

### Mixture-weight initialization and reset

The Cartest batched solver wrapper converts an explicit `initial_pi` to natural
parameters `v = log(pi[:-1] / pi[-1])`. Explicit arguments take precedence over
`warm_start`, matching the generic solver builder.

Periodic mixture-weight reset must not run at iteration zero. A supplied
`initial_v` affects the first iteration; reset occurs only when `t > 0` and
`t % T_0 == 0`.

### Collision horizon

Ego retains full-horizon collision checking. Front and rear retain their
short-horizon asymmetric behavior, but the checked prefix must include the
state executed by the MPC runner. For `execute_index = i`, the prefix is
`slice(0, i + 1)`, clipped to the trajectory length by normal slicing.

Both scalar and batched evaluators derive this horizon from the same scenario
field so their results remain equivalent.

### Constraint-layer consistency

The hot JIT path may continue to consume a static tuple, but that tuple must be
derived from shared three-agent constraint metadata rather than independently
restating layer order, aggregation, mode, baseline, and resolution in the
batched evaluator. Scalar `ConstraintSpec` construction and batched nesting use
the same metadata.

## Compatibility

- Keep `_sample_all_blocks` as the production sampler.
- Preserve the existing PRNG split and sample stream.
- Preserve objective transforms, nesting order, baselines, and aggregation.
- Preserve public solver call signatures.
- Reject invalid `initial_pi` values clearly rather than silently ignoring them.

## Tests

Each behavior is introduced test-first and observed failing before production
changes:

1. Elite utilities: no ties, boundary ties, all equal, and total mass.
2. `initial_pi`: conversion, precedence over warm start, and invalid values.
3. Initial `v`: first-iteration distribution is not reset; later periodic reset
   remains covered.
4. Collision prefix includes `execute_index` in scalar and batched evaluators.
5. Constraint metadata produces the existing five ordered layers and scalar /
   batched cost equivalence remains green.
6. Existing batched evaluator, batched solver, solver-mode, and scenario tests
   continue to pass on CPU; the three-agent smoke run is repeated on WSL/CUDA.

## Success criteria

All targeted and existing tests pass, the repository sampler remains unchanged,
and the WSL/CUDA smoke run completes without recompilation or shape errors.
