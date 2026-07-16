"""Two‑phase solver wrapper — extends ``gmm_igo.solver_builder``.

Wraps two ``build_solver`` instances (Phase 1 + Phase 2) into a single
callable that follows the MGIGO convention::

    solver = build_two_phase_solver(gen, ...)
    result = solver(key, context=ctx, initial_mu=mu_init)

Internally:  P1 → z_ref → P2 → ctrl.

Constraint nesting (aligned with the Constran σ‑nesting methodology):
  Phase 1:  obs outer → jerk inner  (geometry first)
  Phase 2:  jerk outer → obs inner  (physics first)

Map warmstart:  K‑modal GMM seeded from lane data (``map_warmstart()``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import random

from Cartest.planning.constraints import make_constraints as _make_constraints
from Constraintdealer.Constran import Deterministic
from Cartest.planning.costs.default_lyapunov import DEFAULT_CONSTRAINTS
from gmm_igo.solver_builder import build_solver, build_nash_solver, NashResult, SolverResult
from Cartest.planning.costs.registry import make_agent_specs_from_scenario


# ═══════════════════════════════════════════════════════════════════════
# Two‑phase constraint builders
# ═══════════════════════════════════════════════════════════════════════

def build_p2_constraints(gen, lane_hw, safe_dist, acc_max=5.0, jerk_max=2.0):
    """Phase 2 constraints: jerk outer → obs inner (physics first).

    Reversed priority from the default — jerk/acc wrap everything.
    """
    from Cartest.planning.constraints import _eval_frenet, _eval_vehicle_states

    def _obs_g(theta, ctx):
        n_obs = ctx['obs_pos'].shape[1]
        if n_obs == 0:
            return jnp.zeros(gen.T)
        st = _eval_vehicle_states(theta, ctx, gen)
        x, y, v = st[:, 0], st[:, 1], st[:, 2]
        d_rss = v * safe_dist + v ** 2 / 16.0
        dx = x[:, None] - ctx['obs_pos'][:, :, 0]
        dy = y[:, None] - ctx['obs_pos'][:, :, 1]
        r_obs = ctx['obs_rad']
        pen_x = jnp.maximum(0., d_rss[:, None] + r_obs - jnp.abs(dx))
        pen_y = jnp.maximum(0., r_obs - jnp.abs(dy))
        return jnp.maximum(pen_x, pen_y).max(axis=-1)

    def _lane_g(theta, ctx):
        _, d, _, _, _, _, _, _ = _eval_frenet(theta, ctx, gen)
        return jnp.maximum(0., jnp.abs(d) - lane_hw)

    def _speed_g(theta, ctx):
        st = _eval_vehicle_states(theta, ctx, gen)
        v = st[:, 2]
        return jnp.maximum(jnp.maximum(0., 2.0 - v), jnp.maximum(0., v - 35.0))

    def _acc_g(theta, ctx):
        st = _eval_vehicle_states(theta, ctx, gen)
        a_long, a_lat = st[:, 4], st[:, 5]
        am = jnp.sqrt(a_long ** 2 + a_lat ** 2)
        return jnp.maximum(
            jnp.maximum(0., jnp.abs(a_long) - acc_max),
            jnp.maximum(jnp.maximum(0., jnp.abs(a_lat) - acc_max),
                        jnp.maximum(0., am - acc_max)))

    def _jerk_g(theta, ctx):
        st = _eval_vehicle_states(theta, ctx, gen)
        j_long, j_lat = st[:, 6], st[:, 7]
        jm = jnp.sqrt(j_long ** 2 + j_lat ** 2)
        return jnp.maximum(
            jnp.maximum(0., jnp.abs(j_long) - jerk_max),
            jnp.maximum(jnp.maximum(0., jnp.abs(j_lat) - jerk_max),
                        jnp.maximum(0., jm - jerk_max)))

    # Reversed: jerk(1) outermost → obs(5) innermost
    return [
        Deterministic(_jerk_g,  mode='soft', priority=1, aggregate='max',
                      transform='soft'),
        Deterministic(_acc_g,   mode='soft', priority=2, aggregate='max',
                      transform='soft'),
        Deterministic(_speed_g, mode='soft', priority=3, aggregate='max',
                      transform='soft'),
        Deterministic(_lane_g,  mode='soft', priority=4, aggregate='q95',
                      transform='soft'),
        Deterministic(_obs_g,   mode='hard', priority=5, aggregate='max',
                      transform='hard'),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TwoPhaseResult:
    x: jnp.ndarray          # final control points [2 * n_free]
    cost: float             # Phase 2 cost
    z_ref: tuple            # 8 Frenet reference arrays from Phase 1
    cost_p1: float
    cost_p2: float


def build_two_phase_solver(
    gen,
    lane_hw: float = 2.0,
    safe_dist: float = 0.1,
    v_target: float = 15.0,
    # Phase 1 kwargs
    p1_T: int = 200, p1_dt: float = 0.15, p1_B: int = 96, p1_B0: int = 40,
    p1_acc_max: float = 10.0, p1_jerk_max: float = 6.0,
    # Phase 2 kwargs
    p2_T: int = 150, p2_dt: float = 0.30, p2_B: int = 64, p2_B0: int = 30,
    p2_acc_max: float = 5.0, p2_jerk_max: float = 2.0,
    omega_d: float = 4.0,
    # Common
    K: int = 3,
):
    """Build a two‑phase solver that follows the MGIGO solver convention.

    Returns a callable ``solver(key, context=ctx, initial_mu=mu)`` that
    internally runs Phase 1 (geometry exploration) → Phase 2 (Lyapunov
    tracking) and returns a ``TwoPhaseResult``.

    Args:
        gen:          FrenetBSplineTrajectory
        lane_hw:      lane half‑width [m]
        safe_dist:    RSS reaction time [s]
        v_target:     desired speed [m/s]

    Phase 1 (geometry):
        p1_T, p1_dt, p1_B, p1_B0:  IGO parameters
        p1_acc_max, p1_jerk_max:   loose constraint limits

    Phase 2 (tracking):
        p2_T, p2_dt, p2_B, p2_B0:  IGO parameters
        p2_acc_max, p2_jerk_max:   tight constraint limits
        omega_d:                    lateral Lyapunov gain

    Returns:
        callable: solver(key, context=ctx, initial_mu=mu) → TwoPhaseResult
    """
    n_free = gen.n_free

    # ── Phase 1 objective: speed guidance + weak lane-centre preference ──
    def p1_obj(theta, ctx):
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
            theta[:n_free], theta[n_free:2 * n_free],
            ctx['s0'], ctx['s_dot0'], ctx['s_ddot0'],
            ctx['d0'], ctx['d_dot0'], ctx['d_ddot0'])
        return jnp.sum((s_dot - v_target) ** 2)

    p1_cons = _make_constraints(gen, {"lane_hw": lane_hw},
                                {"obs_safe_dist": safe_dist,
                                 "acc_max": p1_acc_max, "jerk_max": p1_jerk_max},
                                DEFAULT_CONSTRAINTS)
    sp1 = build_solver(
        p1_obj, dims=(n_free, n_free), constraints=p1_cons,
        solver='m22', T=p1_T, dt=p1_dt, K=K,
        B=p1_B, B0=p1_B0, T_0=p1_T,
        k_inner=1.0, obj_transform='standard',
    )

    # ── Phase 2 objective: Lyapunov tracking (z_ref from ctx) ──
    w = omega_d

    def p2_obj(theta, ctx):
        s, d, s_dot, d_dot, s_ddot, d_ddot, s_dddot, d_dddot = gen.evaluate(
            theta[:n_free], theta[n_free:2 * n_free],
            ctx['s0'], ctx['s_dot0'], ctx['s_ddot0'],
            ctx['d0'], ctx['d_dot0'], ctx['d_ddot0'])
        sr = ctx['zr_s'];    sdr  = ctx['zr_s_dot']; sddr = ctx['zr_s_ddot']
        dr = ctx['zr_d'];    ddr  = ctx['zr_d_dot']; dddr = ctx['zr_d_ddot']
        es = s - sr;          ed  = d - dr
        esd = s_dot - sdr;    edd = d_dot - ddr
        esdd = s_ddot - sddr; eddd = d_ddot - dddr
        t1 = es**2 + ed**2
        t2 = (esd + w*es)**2 + (edd + w*ed)**2
        t3 = ((esdd + 2*w*esd + w**2*es)**2
              + (eddd + 2*w*edd + w**2*ed)**2)
        return jnp.sum(t1) + jnp.sum(t2) + jnp.sum(t3)

    p2_cons = build_p2_constraints(gen, lane_hw, safe_dist,
                                   acc_max=p2_acc_max, jerk_max=p2_jerk_max)
    sp2 = build_solver(
        p2_obj, dims=(n_free, n_free), constraints=p2_cons,
        solver='m22', T=p2_T, dt=p2_dt, K=K,
        B=p2_B, B0=p2_B0, T_0=1000,
        k_inner=1.0, obj_transform='standard',
    )

    # ── Wrapped solver callable ──
    def _solve(key, context=None, initial_mu=None, warm_start=None):
        k1, k2 = random.split(key)

        r1 = sp1(k1, context=context, initial_mu=initial_mu,
                 warm_start=warm_start)
        cs1 = r1.x[:n_free]
        cd1 = r1.x[n_free:]

        # P1 ctrl → z_ref
        frenet1, st1, _ = gen.evaluate_plan(cs1, cd1, context)
        z_ref = gen.from_vehicle_states(
            st1[:, 0], st1[:, 1], st1[:, 2], st1[:, 3],
            st1[:, 4], st1[:, 5], st1[:, 6], st1[:, 7])

        # Inject z_ref into ctx
        ctx_keys = ['zr_s', 'zr_s_dot', 'zr_s_ddot',
                    'zr_d', 'zr_d_dot', 'zr_d_ddot']
        zr_vals = [z_ref[0], z_ref[2], z_ref[4],
                   z_ref[1], z_ref[3], z_ref[5]]
        for k, v in zip(ctx_keys, zr_vals):
            context[k] = v

        r2 = sp2(k2, context=context, initial_mu=initial_mu)

        return TwoPhaseResult(
            x=r2.x,
            cost=float(r2.cost),
            z_ref=z_ref,
            cost_p1=float(r1.cost),
            cost_p2=float(r2.cost),
        )

    # Warmup method
    def _warmup(key, ctx, mu):
        ctx_p2 = dict(ctx)
        for k in ['zr_s', 'zr_s_dot', 'zr_s_ddot',
                  'zr_d', 'zr_d_dot', 'zr_d_ddot']:
            ctx_p2[k] = jnp.zeros(gen.T)
        _ = sp1(random.PRNGKey(999), context=ctx, initial_mu=mu)
        _ = sp2(random.PRNGKey(999), context=ctx_p2, initial_mu=mu)

    _solve.warmup = _warmup
    return _solve


# ═══════════════════════════════════════════════════════════════════════
# Map warmstart
# ═══════════════════════════════════════════════════════════════════════

def map_warmstart(gen, s0: float, v0: float,
                  d_lanes: list[float]) -> jnp.ndarray:
    """K‑modal GMM warmstart from map lane data.

    Each lane → one GMM component with free control points set to that
    lane's constant d offset and the corresponding constant‑speed
    longitudinal profile.  The B‑spline C0/C1 clamping naturally
    handles the transition from the current vehicle state.

    Args:
        gen:     FrenetBSplineTrajectory
        s0, v0:  current longitudinal position / speed
        d_lanes: lane offsets, one per GMM component (e.g. [-3.5, 0.0, 3.5])

    Returns:
        initial_mu array of shape (M=2, K, D=n_free)
    """
    n_free = gen.n_free
    g = gen.greville[2:gen.n_ctrl]

    s_comp, d_comp = [], []
    for d_lane in d_lanes:
        s_comp.append(s0 + v0 * g)
        d_comp.append(jnp.full((n_free,), d_lane, dtype=jnp.float32))

    # (M=2, K, D)
    return jnp.stack([jnp.stack(s_comp, axis=0),
                      jnp.stack(d_comp, axis=0)], axis=0)


# ═══════════════════════════════════════════════════════════════════════
# Multi-agent Nash/RNE helpers (unified through build_nash_solver)
# ═══════════════════════════════════════════════════════════════════════

def build_multi_agent_context(states):
    """Build JAX-friendly context keys for multi-agent Cartest game costs."""
    ctx = {}
    for idx, state in enumerate(states):
        ctx[f"s0_a{idx}"] = float(state.s)
        ctx[f"s_dot0_a{idx}"] = float(state.s_dot)
        ctx[f"s_ddot0_a{idx}"] = float(state.s_ddot)
        ctx[f"d0_a{idx}"] = float(state.d)
        ctx[f"d_dot0_a{idx}"] = float(state.d_dot)
        ctx[f"d_ddot0_a{idx}"] = float(state.d_ddot)
    return ctx


def build_multi_agent_warmstart(gen, scenario, states, key):
    """Build block-wise GMM means and precision factors for a game scenario."""
    n_free = gen.n_free
    game = scenario["game"]
    k_count = int(game.get("K", 3))
    n_blocks = len(game["block_layout"])
    speed_factors = jnp.asarray(
        scenario.get("behavior", {}).get("speed_factors", (1.15, 1.0, 0.85))
    )
    if speed_factors.shape[0] != k_count:
        speed_factors = jnp.linspace(1.15, 0.85, k_count)

    mu = jnp.zeros((n_blocks, k_count, n_free), dtype=jnp.float32)
    greville = gen.greville[2:gen.n_ctrl]

    for block_idx, label in enumerate(game["block_layout"]):
        agent_idx = int(game["block_to_agent"][block_idx])
        state = states[agent_idx]
        is_d_block = label.endswith("_d")
        for comp_idx in range(k_count):
            key, subkey = random.split(key)
            if is_d_block:
                values = jnp.full((n_free,), float(state.d), dtype=jnp.float32)
            else:
                speed = float(state.s_dot) * speed_factors[comp_idx]
                values = float(state.s) + speed * greville
                values = values + 0.1 * random.normal(subkey, (n_free,))
            mu = mu.at[block_idx, comp_idx].set(values)

    L_inv = jnp.tile(jnp.eye(n_free, dtype=jnp.float32), (n_blocks, k_count, 1, 1))
    return mu, L_inv


def build_cartest_nash_solver(gen, scenario):
    """Build the unified Nash/RNE solver for a Cartest multi-agent scenario."""
    game = scenario["game"]
    dims = tuple([gen.n_free] * len(game["block_layout"]))
    if game.get("solver") == "cartest_batched_rne_blocks":
        return _build_cartest_batched_solver(gen, scenario, dims)
    return build_nash_solver(
        agent_specs=make_agent_specs_from_scenario(gen, scenario),
        dims=dims,
        solver=game.get("solver", "rne_blocks"),
        block_to_agent=tuple(game["block_to_agent"]),
        T=int(game.get("T", 300)),
        dt=float(game.get("dt", 0.15)),
        K=int(game.get("K", 3)),
        B=int(game.get("B", 64)),
        B0=int(game.get("B0", 20)),
        T_0=int(game.get("T_0", 100)),
        M_inner=int(game.get("M_inner", 30)),
    )


def _build_cartest_batched_solver(gen, scenario, dims):
    """Build a solver callable for the Cartest batched RNE blocks mode.

    This wraps ``make_cartest_batched_rne_blocks_solver`` (which evaluates B-spline
    trajectories outside the ``B x M_inner`` pairing loop) and packs its raw
    diagnostics into the same ``NashResult`` shape the generic
    ``rne_blocks`` path returns, so ``select_nash_plan`` and the closed-loop
    runner work unchanged.
    """
    from Cartest.planning.batched_rne_solver import make_cartest_batched_rne_blocks_solver
    from Cartest.planning.batched_game_eval import (
        evaluate_joint_plan_batch, batched_nested_costs_from_plans,
    )

    game = scenario["game"]
    block_to_agent = tuple(game["block_to_agent"])
    M_agent = len(scenario["agents"])
    n_free = gen.n_free
    # Match the generic build_nash_solver defaults so this is a faithful
    # (faster) drop-in for rne_blocks on three_agent_track.
    k_inner = 0.1
    obj_transform = "standard"
    batched_solver = make_cartest_batched_rne_blocks_solver(
        gen, scenario, k_inner=k_inner, obj_transform=obj_transform)

    def _solve(key, *, context=None, initial_mu=None, initial_S_or_L=None,
               initial_pi=None, warm_start=None):
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
        if mu is None or L_inv is None:
            raise ValueError(
                "cartest_batched_rne_blocks requires initial_mu and initial_S_or_L"
            )

        raw = batched_solver(
            key, context=context,
            initial_mu=mu, initial_L_inv=L_inv, initial_v=v,
        )

        final_mu = raw["mu"]
        final_L = raw["L_inv"]
        final_pi = raw["pi"]

        solutions = {}
        joint_parts = []
        for aid in range(M_agent):
            my_blocks = [b for b in range(len(block_to_agent))
                         if block_to_agent[b] == aid]
            blk_arr = jnp.array(my_blocks)
            mu_agent = final_mu[blk_arr]
            L_agent = final_L[blk_arr]
            pi_agent = final_pi[blk_arr]
            best_k = jnp.argmax(pi_agent, axis=1)
            x_parts = [mu_agent[j, best_k[j], :n_free] for j in range(len(my_blocks))]
            x_agent = jnp.concatenate(x_parts) if x_parts else jnp.zeros(0)
            joint_parts.append(x_agent)
            solutions[aid] = SolverResult(
                x=x_agent, cost=0.0, mu=mu_agent, S_or_L=L_agent, pi=pi_agent,
                solver_name=f"nash_cartest_batched_rne_blocks_agent{aid}",
            )
        joint_x = jnp.concatenate(joint_parts)

        # Real per-agent nested cost on the selected equilibrium joint_x,
        # using the same batched cost definition as the solver internals.
        plans = evaluate_joint_plan_batch(gen, joint_x[None], context, agent_count=M_agent)
        per_agent_cost = batched_nested_costs_from_plans(
            plans, scenario, k_inner=k_inner, obj_transform=obj_transform)[0]
        for aid in range(M_agent):
            solutions[aid] = SolverResult(
                x=solutions[aid].x, cost=float(per_agent_cost[aid]),
                mu=solutions[aid].mu, S_or_L=solutions[aid].S_or_L,
                pi=solutions[aid].pi, solver_name=solutions[aid].solver_name,
            )

        return NashResult(
            solutions=solutions,
            joint_x=joint_x,
            per_agent_cost=per_agent_cost,
            solver_name="cartest_batched_rne_blocks",
            diag={
                "mu": final_mu,
                "S_or_L": final_L,
                "pi": final_pi,
                "v": raw["v"],
                "block_to_agent": block_to_agent,
                "dims": dims,
                "metrics": raw["metrics"],
            },
        )

    return _solve


def select_nash_plan(result, scenario):
    """Decode block-level NashResult diagnostics into per-agent B-spline controls."""
    game = scenario["game"]
    mu = result.diag["mu"]
    pi = result.diag["pi"]
    block_to_agent = tuple(game["block_to_agent"])

    plans = []
    for agent_idx in range(len(scenario["agents"])):
        block_indices = [i for i, owner in enumerate(block_to_agent) if owner == agent_idx]
        if len(block_indices) != 2:
            raise ValueError(
                f"Agent {agent_idx} must own exactly two blocks, got {block_indices}"
            )
        s_block, d_block = block_indices
        best_s = int(jnp.argmax(pi[s_block]))
        best_d = int(jnp.argmax(pi[d_block]))
        plans.append({
            "agent_idx": agent_idx,
            "ctrl_s": mu[s_block, best_s],
            "ctrl_d": mu[d_block, best_d],
            "best_components": (best_s, best_d),
            "pi_s": pi[s_block],
            "pi_d": pi[d_block],
        })
    return plans
