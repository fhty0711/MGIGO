# Batched RNE 100 ms design

## Goal

Move the default three-agent batched RNE path into the 100 ms class on the
target RTX 4060 without changing its sampling stream or nested-cost values.

## Design

1. Use JAX `searchsorted(method="compare_all")` for the 11--13 entry
   `T_alpha` tables. This changes only the GPU lookup implementation; outputs
   must match the previous `scan` method within floating-point equality.
2. JIT the complete Nash postprocess: component selection, selected joint
   trajectory evaluation, and final per-agent nested cost. The Python wrapper
   must not call `float()` or `int(jnp.argmax(...))` on the timed path.
3. Return selected block controls and component indices in diagnostics so
   `select_nash_plan` reuses device results instead of synchronizing six times.
4. Set the default `three_agent_track` iteration budget to `T=100`. The
   existing `T=300` value remains available as an explicit scenario override.

## Non-goals

- Do not enable `_sample_all_blocks_fast`.
- Do not change PRNG splitting or sample values.
- Do not replace SPD projection or component update mathematics.
- Do not merge experimental profiling scripts.

## Verification

- Test `compare_all` against a local `scan` reference on zero, knot boundaries,
  very small values, and large positive/negative values.
- Test JIT postprocess outputs and the no-resynchronization selection path.
- Run existing scalar/batched equivalence tests.
- Run the two-step WSL/CUDA scenario after warm-up and report synchronized
  solve times.
