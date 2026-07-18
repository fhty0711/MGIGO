# Three-Agent Collision Envelope, Warm-Start, and Nash Evidence Design

## Purpose

Refine the `three_agent_track` B-spline/RNE experiment so that:

1. hard collision constraints prevent physical overlap without forbidding normal adjacent-lane approach;
2. longitudinal GMM warm-start modes have explicit physical meaning;
3. saved analysis and video make the game-theoretic interaction and empirical unilateral-deviation evidence visible;
4. diagnostics do not contaminate the reported MPC solve time.

This design keeps the current `cartest_batched_rne_blocks` production solver and current lane-change scenario. A separate, more dramatic demonstration scenario may be added later, but it is not part of this change.

## Current Behavior and Terminology

### `safe_gap`

In the current three-agent implementation, `safe_gap=3.0` is not an RSS parameter. It is reused by the hard rectangular collision envelope:

- longitudinal center clearance: `vehicle_length + safe_gap = 5.0 + 3.0 = 8.0 m`;
- lateral center clearance: `min(vehicle_width + safe_gap, lane_width) = min(2.0 + 3.0, 3.5) = 3.5 m`.

The lateral result covers an entire lane width. At the initial state, ego and front are exactly on that lateral boundary, so any small move toward the target lane can activate the outer hard constraint while their longitudinal gap is below 8 m. This produces the observed initial motion away from the target lane and forces front/rear gap-opening behavior.

RSS already has separate dynamic-distance parameters, including braking assumptions, reaction time, `RSS_MIN_GAP`, a lateral margin, and a smooth lateral-overlap gate. It does not need `safe_gap` from the scenario.

### Scalar and batched paths

“Scalar” means the current Constran-based, per-joint-sample reference evaluator assembled through `make_agent_specs`. It is not the historical pre-RSS cost version. The production rollout uses the batched CUDA-oriented evaluator through `cartest_batched_rne_blocks`.

The scalar path remains a semantic oracle for tests. Both paths must consume the same component formulas and collision-clearance helper so that a future change cannot silently alter only one path.

### Existing speed factors

`speed_factors=(1.15, 1.0, 0.75)` was introduced in commit `001d25fa0551ba18383c071e14352ce5268ae987` on 2026-07-16 as part of routing the new three-agent scenario through batched RNE. It initializes the three longitudinal GMM components of every agent; it does not assign different target speeds to ego/front/rear.

Each agent owns independent longitudinal and lateral blocks, each with `K=3`. The speed factors diversify only the longitudinal block. The lateral components currently start at the same current lateral position and separate during RNE updates. Consequently, each agent has nine possible selected component-mean pairs `(k_s, k_d)`, not three pre-bound complete behavior policies.

## Hard Collision Envelope

### Configuration

Replace the ambiguous three-agent `safe_gap` collision configuration with explicit fields:

```python
"safety": {
    "vehicle_length": 5.0,
    "vehicle_width": 2.0,
    "collision_longitudinal_margin": 0.5,
    "collision_lateral_margin": 0.2,
    ...
}
```

The three-agent scenario will no longer define or consume `safe_gap` for collision checking.

### Geometry

For equal-sized vehicles, use rectangular Frenet center clearances:

```text
longitudinal_clearance = vehicle_length + collision_longitudinal_margin
lateral_clearance      = vehicle_width  + collision_lateral_margin
```

With the approved values, the thresholds are `5.5 m` longitudinally and `2.2 m` laterally. A collision violation is positive only when both axes overlap their corresponding expanded vehicle envelope.

Introduce one shared helper, `collision_clearances(scenario)`, in the shared three-agent component module. The scalar collision functions and the batched `B x M_inner` collision aggregation must call this helper. No duplicate clearance arithmetic remains in the batched evaluator.

### Separation from RSS

The hard envelope answers only “would the expanded physical bodies overlap?” RSS remains a soft objective term that answers “is this longitudinal interaction dynamically risky?” Its current smooth lateral gate remains unchanged:

```text
sigmoid((vehicle_width + RSS_LATERAL_MARGIN - lateral_gap)
        / RSS_LATERAL_SOFTNESS)
```

At adjacent lane centers (`3.5 m` apart), RSS weight is approximately `0.00034`; it increases smoothly during the merge and becomes significant near lateral overlap.

## Semantic Longitudinal Warm-Start Modes

### Three-agent opt-in

Remove `behavior["speed_factors"]` from `three_agent_track` and add:

```python
"warmstart": {
    "longitudinal_modes": "target_current_yield",
    "yield_delta": 2.0,
}
```

For each agent, initialize the three longitudinal component speeds as:

```text
target/aggressive = max(current_speed, v_target)
current           = current_speed
yield             = max(v_min, current_speed - yield_delta)
```

These values seed the GMM means only. They do not change the objective’s per-agent `v_target`, and RNE remains free to move all component means, covariances, and probabilities.

### Compatibility

`build_multi_agent_warmstart` will support the new semantic mode when explicitly requested by a scenario. Existing game scenarios that still define legacy `speed_factors` retain their current initialization behavior. This limits behavioral change to `three_agent_track`.

For the semantic mode, `K` must be exactly three. A clear `ValueError` is raised during setup if the scenario selects `target_current_yield` with another `K`, rather than silently inventing additional modes.

## Nash Evidence

The RNE solver produces blockwise GMM distributions, while the video renders one selected component-mean joint trajectory. The diagnostics therefore distinguish selected-plan evidence from distributional evidence.

### Per-step selected-mode residual

For every recorded MPC step:

1. hold the other agents’ selected blocks fixed;
2. for agent `i`, enumerate its nine final component-mean combinations `(k_s, k_d)`;
3. evaluate each combination with the same nested cost definition;
4. record:

```text
epsilon_mode_i = max(0, J_selected_i - min(J_i over 9 combinations))
```

This is cheap and useful for the animation, but is explicitly labeled a restricted selected-mode residual, not a continuous or mixed-strategy Nash certificate.

### Keyframe distributional best response

After the rollout, select four diagnostic frames:

1. initial frame;
2. frame with maximum summed RSS cost;
3. first frame where ego crosses the lane midpoint;
4. first frame where ego remains within `0.5 m` of the target center for five consecutive executed states.

At each unique keyframe and for each agent:

1. freeze the other two agents’ final GMM distributions from that MPC solve;
2. use fixed, recorded background sample keys to estimate expected cost against those distributions;
3. re-optimize the deviating agent’s longitudinal and lateral blocks jointly;
4. perform three deterministic recorded restarts and retain the lowest expected cost;
5. record absolute and relative empirical exploitability:

```text
epsilon_br_i = max(0, J_equilibrium_i - J_best_response_i)
epsilon_br_relative_i = epsilon_br_i / max(abs(J_equilibrium_i), 1e-6)
```

The result is called an empirical distributional best-response residual. Because the optimizer and expectation use finite stochastic samples, it is not described as a formal proof of exact Nash equilibrium.

The best-response analysis runs after the control rollout. Its time is stored separately and never included in `solve_ms` or performance claims.

## Reporting and Visualization

### Stored data

JSON and NPZ analysis artifacts store:

- solver name and iteration count;
- per-agent current speed and target speed;
- selected `(k_s, k_d)`;
- `pi_s` and `pi_d`;
- `epsilon_mode` at every step;
- keyframe `epsilon_br` and relative residual;
- equilibrium and best-response costs;
- best-response trajectory for visual comparison;
- diagnostic seeds, restart count, and diagnostic runtime;
- collision clearances and all constraint maxima.

### Video overlay

Each agent receives a compact status row containing:

```text
name  v/v_target  mode=(k_s,k_d)  epsilon_mode
```

The full `pi_s` and `pi_d` values are displayed in a compact side or lower panel rather than over the roadway. On keyframes, the panel additionally shows `epsilon_br` and `J_eq -> J_br`. The agent’s best-response trajectory is rendered as a semi-transparent dashed/ghost line using the same agent color, visually distinct from the equilibrium prediction.

### Visual iteration workflow

Do not render the final 150-frame video immediately after implementing the overlay. First render standalone frames at:

- initial state;
- maximum-RSS conflict;
- lane-midpoint crossing;
- settled target-lane state.

Inspect these images at original resolution. Adjust figure aspect ratio, panel placement, font size, line styles, camera limits, and label density until:

- no text overlaps vehicles or critical trajectories;
- equilibrium and best-response trajectories are distinguishable;
- values remain readable at normal desktop zoom;
- the roadway retains enough vertical space;
- the panel does not cause camera jitter between frames.

Create a four-frame contact sheet for final visual review, then render and inspect the full MP4. If the full animation exposes flicker, clipping, or unreadable transitions, revise the renderer and repeat the keyframe/full-video check.

## Tests

### Collision geometry

- same-lane vehicles below `5.5 m` center distance violate the hard envelope;
- same-lane vehicles at or above `5.5 m` do not violate it;
- vehicles at adjacent lane centers do not violate it even with longitudinal overlap;
- partial lateral overlap below `2.2 m` plus longitudinal overlap below `5.5 m` violates it;
- scalar and batched collision/nested costs agree on boundary and random cases;
- the batched evaluator obtains clearances from the shared helper.

### Warm-start

- all three agents retain `v_target=17.5 m/s` in the current scenario;
- longitudinal initial means correspond to target/current/yield speeds;
- lateral initial means remain at the current lateral state;
- semantic mode rejects `K != 3` with an explanatory error;
- legacy factor-based scenarios retain their existing values.

### Nash diagnostics

- `epsilon_mode` is zero when the selected combination is the best enumerated combination;
- a constructed profitable component deviation produces the expected positive residual;
- best-response optimization freezes opponent distributions;
- the best of three restarts is reported;
- diagnostic work does not modify the executed controls or solver timing;
- repeated analysis with the same stored seeds is reproducible.

### Closed-loop acceptance

Run `T=300`, 150 MPC steps, seed 0 and verify:

- ego does not initially move below `d=-0.02 m`;
- all hard collision violations remain zero;
- ego completes and holds the lane-change target;
- rear behavior is governed by RSS/objective trade-offs rather than the old lane-width hard envelope;
- scalar/batched parity tests pass;
- video, JSON, NPZ, four keyframes, and contact sheet are produced;
- median/P95 solver timing excludes all diagnostic and rendering work.

## Out of Scope

- changing RSS equations or their weights;
- replacing RNE with another game solver;
- CUDA kernel redesign or performance optimization;
- claiming a formal exact Nash proof;
- replacing the current experiment with a hand-tuned showcase scenario.
