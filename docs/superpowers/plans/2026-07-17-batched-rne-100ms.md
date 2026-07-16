# Batched RNE 100 ms Implementation Plan

> **For agentic workers:** Execute inline with test-driven development; no subagent delegation is required.

**Goal:** Land the two experimentally validated exact optimizations and use the validated `T=100` real-time budget.

**Architecture:** Keep the existing JIT solver core and add a second reusable JIT callable for all postprocessing. Change only the lookup method inside `T_alpha`; retain all transform tables, nesting, sampling, and GMM update mathematics.

**Tech Stack:** Python, JAX 0.6.2, WSL2 CUDA.

### Task 1: Small-table transform lookup

- Add a failing method/equivalence test in `Cartest/eval/test_batched_three_agent_eval.py`.
- Change `Constraintdealer/Constran.py:T_alpha` to `method="compare_all"`.
- Run the evaluator regression script and commit.

### Task 2: JIT Nash postprocess

- Add failing tests for the postprocess factory and selected-block reuse.
- Add a reusable JIT postprocess in `Cartest/planning/solver_modes.py`.
- Remove Python scalar conversions from `_build_cartest_batched_solver`.
- Make `select_nash_plan` consume the selected diagnostic arrays.
- Run solver-mode and batched-solver regressions and commit.

### Task 3: Real-time budget and merge verification

- Add a failing scenario assertion for `T=100`.
- Change `Cartest/planning/scenarios/three_agent_track.py` from `T=300` to `T=100`.
- Run all four CPU regression scripts and a synchronized two-step CUDA smoke run.
- Merge the feature branch into local `llss` and rerun verification there.
