# Three-Agent Batched RNE Acceleration

This note documents the acceleration path for `Cartest`'s `three_agent_track`
scenario. It focuses on performance-oriented rewrites and the current call
chain.

## Entry Point

Run the three-agent B-spline game scenario with:

```bash
python Cartest/Simple.py three_agent_track --steps 25
```

Use `python3` if the virtual environment does not provide `python`:

```bash
python3 Cartest/Simple.py three_agent_track --steps 25
```

By default the runner performs a JAX warmup before the first timed MPC step, so
`solve=...ms` reports steady-state solver time. To include first-call JIT
compile time in step 0, run:

```bash
python Cartest/Simple.py three_agent_track --steps 1 --no-plot --include-compile-time
```

## Current Call Chain

```text
Cartest/Simple.py
  run(...)
    -> run_multi_agent_game(...)
      -> build_cartest_nash_solver(gen, scenario)
        -> _build_cartest_batched_solver(gen, scenario, dims)
          -> make_cartest_batched_rne_blocks_solver(gen, scenario)
          -> _build_cartest_batched_postprocess(gen, scenario)

MPC step
  build_multi_agent_context(states)
  build_multi_agent_warmstart(gen, scenario, states, key)
  solver(solve_key, context=ctx, initial_mu=mu, initial_S_or_L=L_inv)
    -> batched RNE scan
      -> _sample_all_blocks(..., B)
      -> _sample_all_blocks(..., M_inner)
      -> batched_expected_costs_for_all_agents(...)
        -> evaluate_agent_control_batch(...) for candidate plans
        -> evaluate_agent_control_batch(...) for background plans
        -> _fast_expected_cost_for_agent(...) for each agent
      -> _update_block_component_core(...) from gmm_igo.MPC_G_MS
    -> JIT postprocess
      -> select best component per block
      -> build joint_x
      -> evaluate selected-plan per-agent cost
  select_nash_plan(result, scenario)
  execute_perfect_tracking_at(...)
```

## Acceleration Steps

### 1. Cache B-Spline Trajectory Evaluation

The original generic RNE path treated each agent cost as a black-box scalar
fitness. For each acting agent candidate, it paired that candidate with every
background sample and evaluated B-spline trajectories inside the full
`B x M_inner` loop.

The batched evaluator separates expensive trajectory evaluation from cheap
pairwise cost composition:

```text
Before: O(agent_count * B * M_inner) expensive trajectory evaluations
After:  O(agent_count * (B + M_inner)) expensive trajectory evaluations
```

For the current three-agent setup this means candidate plans and background
plans are evaluated once per agent and reused across all pairwise combinations.

Relevant code:

- `Cartest/planning/costs/three_agent_track_batched.py`
- `evaluate_agent_control_batch(...)`
- `batched_expected_costs_for_all_agents(...)`

### 2. Compute Pairwise Cost From Cached Aggregates

The non-interaction constraints for an acting agent are functions of that
agent's own trajectory only:

```text
lane, speed, acceleration, jerk
```

Those terms are now aggregated once per candidate and broadcast across the
background sample axis. Collision remains pairwise because it depends on both
the acting candidate and the sampled opponent trajectories.

The objective uses the same context as the scalar path. This matters for
`bridged_jerk_cost`, which can read the previously executed acceleration from
`ctx`. Candidate-only objective terms broadcast over the background axis, while
RSS stays pairwise.

The nested saturated cost is still evaluated over the candidate/background
pairing. The optimization removes repeated trajectory and repeated aggregate
work; it does not average before the nonlinear nesting step.

Relevant code:

- `_own_constraint_aggregates(...)`
- `_collision_aggregate_bm(...)`
- `_nest_one_agent_from_aggregates(...)`
- `_fast_expected_cost_for_agent(...)`

### 3. Reuse Scalar Joint Plan Evaluation With `prepare_fn`

The scalar `Constran` path can also avoid evaluating the same three-agent joint
plan once for the objective and again for every constraint. `prepare_fn` lets a
cost build shared intermediate data once and pass it to the objective and
constraint functions.

For `three_agent_track`, the prepared value is the tuple of evaluated agent
plans:

```text
(ego_plan, front_plan, rear_plan)
```

This keeps the scalar path useful as a reference while removing repeated work
inside one scalar cost call.

Relevant code:

- `Constraintdealer/Constran.py`
- `Cartest/planning/costs/three_agent_track.py`
- `_attach_joint_plan_prepare(...)`
- `_prepared_plans(...)`

### 4. Add a Cartest-Specific Batched RNE Adapter

The generic `gmm_igo.MPC_G_MS` implementation remains unchanged. Cartest uses a
local adapter for `three_agent_track` so it can call the structured batched
fitness evaluator while preserving the block-level RNE update math.

The adapter reuses:

```python
gmm_igo.MPC_G_MS._update_block_component_core
```

It keeps the production sampler compatible with the generic RNE path. The
unused alternate fast sampler was removed to avoid maintaining a second PRNG
stream that is not part of the validated path.

Relevant code:

- `Cartest/planning/solvers/batched_rne_solver.py`
- `make_cartest_batched_rne_blocks_solver(...)`
- `_sample_all_blocks(...)`

### 5. Build the JIT Solver Once Per Scenario

The batched solver is constructed as a reusable factory. The JIT-compiled scan
core is created when `build_cartest_nash_solver(...)` builds the solver, rather
than being recreated inside every MPC step.

This reduces repeated JIT setup overhead in the closed-loop runner while keeping
the per-step inputs explicit:

```text
key, context, initial_mu, initial_L_inv, optional initial_v
```

Relevant code:

- `make_cartest_batched_rne_blocks_solver(...)`
- `_build_cartest_batched_solver(...)`

### 6. JIT the Nash Postprocess

After the RNE scan, the solver needs to select the most likely component for
each block, build `joint_x`, and evaluate the selected plan's per-agent cost.
This is now done in one JIT-compiled postprocess function.

The selected blocks are stored in `NashResult.diag`, so `select_nash_plan(...)`
can reuse them without triggering extra host/device synchronization through
another `argmax`.

Relevant code:

- `_build_cartest_batched_postprocess(...)`
- `select_nash_plan(...)`

### 7. Use `compare_all` for Small Transform Tables

`T_alpha(...)` uses small fixed transform tables. The lookup now requests
JAX's `compare_all` search method:

```python
jnp.searchsorted(..., method="compare_all")
```

This keeps the same table-lookup semantics while using a better implementation
strategy for small tables.

Relevant code:

- `Constraintdealer/Constran.py`
- `T_alpha(...)`

## Equivalence Checks

The acceleration path is checked with script-style tests:

```bash
python Cartest/eval/test_batched_three_agent_eval.py
python Cartest/eval/test_batched_rne_solver.py
python Cartest/eval/test_game_solver_modes.py
python Cartest/eval/test_unified_simple_runner.py
python Cartest/eval/test_three_agent_track_game.py
```

The most important consistency test compares the Cartest batched solver against
the generic `rne_blocks` solver on a small fixed seed/warmstart problem:

```text
final mu: allclose
final pi: allclose
per-agent selected-plan cost: allclose
```

This is the practical equivalence check for the accelerated fitness path. The
full production setting remains `T=300`, `B=60`, `M_inner=30` for the default
`three_agent_track` scenario.

## Timing Interpretation

Two timing modes are available:

```text
default runner timing: warmup first, then report steady-state solve time
--include-compile-time: skip warmup, so step 0 includes JAX compilation
```

Use the default mode for MPC runtime comparisons and `--include-compile-time`
when measuring first-call latency.
