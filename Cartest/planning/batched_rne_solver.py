"""Cartest-local batched RNE blocks solver for ``three_agent_track``.

Mirrors the block-level IGO update math of ``gmm_igo.MPC_G_MS`` (same
sampling, elite weighting, and geometric component update) but replaces the
black-box scalar fitness loop with the structured three-agent B-spline batch
evaluator in ``Cartest.planning.batched_game_eval``.  Expensive trajectory
evaluation is performed O(B + M_inner) times per iteration instead of
O(B * M_inner), and the pairwise game costs are computed by broadcasting
cached ego/front/rear plans.

Only the leaf update math (``_update_block_component_core``) is reused from
``MPC_G_MS``; the scan/step driver is duplicated locally so it can call the
batched evaluator instead of the black-box ``fitness_fn_j``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random, lax, vmap

from gmm_igo.MPC_G_MS import _update_block_component_core
from Cartest.planning.batched_game_eval import batched_expected_costs_for_all_agents


def _v_to_pi(v_m):
    """Map natural parameters v[K-1] -> simplex probabilities pi[K]."""
    exps = jnp.exp(jnp.clip(v_m, -70, 70))
    sum_e = 1.0 + jnp.sum(exps)
    return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])


def _sample_all_blocks(mu, S, pi_all, count, key):
    """Sample ``count`` points per block from the block GMMs.

    ``S`` is the precision (information) matrix, so the per-component
    covariance is ``inv(S + eps*I)`` - matching ``MPC_G_MS``.
    """
    N_blocks = mu.shape[0]
    K = mu.shape[1]

    def sample_single_block(b_idx, b_key):
        comps = random.choice(b_key, K, p=pi_all[b_idx], shape=(count,))

        def gen_sample(c_idx, s_key):
            cov = jnp.linalg.inv(S[b_idx, c_idx] + jnp.eye(S.shape[-1]) * 1e-7)
            return random.multivariate_normal(s_key, mu[b_idx, c_idx], cov)

        return vmap(gen_sample)(comps, random.split(b_key, count))

    return vmap(sample_single_block)(jnp.arange(N_blocks), random.split(key, N_blocks))


def cartest_batched_rne_blocks_solver(
    key,
    gen,
    scenario,
    *,
    context,
    initial_mu,
    initial_L_inv,
    initial_v=None,
    k_inner=0.1,
    obj_transform="standard",
):
    """Run Cartest-specific batched RNE blocks and return dict diagnostics.

    ``k_inner`` / ``obj_transform`` default to the same values the generic
    ``build_nash_solver`` path uses (0.1 / "standard") so the batched path is
    a faithful drop-in for ``rne_blocks`` on ``three_agent_track``.

    Returns keys:
      mu:       [N_blocks, K, D]
      L_inv:    [N_blocks, K, D, D]   (Cholesky factor of the precision matrix)
      pi:       [N_blocks, K]
      v:        [N_blocks, K - 1]
      metrics:  dict with mean_fitness history
    """
    game = scenario["game"]
    T = int(game.get("T", 300))
    dt = float(game.get("dt", 0.15))
    K = int(game.get("K", 3))
    B = int(game.get("B", 60))
    B0 = int(game.get("B0", 25))
    T_0 = int(game.get("T_0", 300))
    M_inner = int(game.get("M_inner", 30))
    block_to_agent = tuple(game["block_to_agent"])
    block_to_agent_arr = jnp.array(block_to_agent)
    N_blocks = len(game["block_layout"])
    M_agent = len(scenario["agents"])
    n_free = gen.n_free
    dims = tuple([n_free] * N_blocks)
    dims_arr = jnp.array(dims)

    ctx = {k: jnp.asarray(v) for k, v in context.items()}
    v_reset = jnp.zeros((N_blocks, K - 1))
    v_init = jnp.zeros((N_blocks, K - 1)) if initial_v is None else jnp.asarray(initial_v)[:, :K - 1]

    @jax.jit
    def _core(key, mu_init, L_inv_init, v_init, ctx):
        # L_inv is the Cholesky factor of the precision matrix; S = L @ L.T.
        S_init = vmap(vmap(lambda L: L @ L.T))(L_inv_init[:, :K, :, :])
        state = (mu_init[:, :K, :], S_init, v_init, 0)

        def step_fn(state, step_key):
            mu_t, S_t, v_t, t = state
            v_t = jnp.where((t % T_0) == 0, v_reset, v_t)
            pi_all = vmap(_v_to_pi)(v_t)  # [N_blocks, K]

            key_B, key_M = random.split(step_key)
            samples_B = _sample_all_blocks(mu_t, S_t, pi_all, B, key_B)     # [N_blocks, B, D]
            samples_M = _sample_all_blocks(mu_t, S_t, pi_all, M_inner, key_M)  # [N_blocks, M_inner, D]

            # Shared plan cache: 6 trajectory-eval calls total (3 candidates
            # from samples_B + 3 backgrounds from samples_M), reused across
            # all agents. Only each agent's own nested cost is computed.
            f_hats = batched_expected_costs_for_all_agents(
                gen, samples_B, samples_M, ctx, scenario,
                k_inner=k_inner, obj_transform=obj_transform)              # [M_agent, B]
            ranks = jnp.argsort(jnp.argsort(f_hats, axis=-1), axis=-1)     # [M_agent, B]
            agent_w_hat = jnp.where(ranks < B0, 1.0 / B, 0.0)              # [M_agent, B]
            mean_fitness = jnp.mean(f_hats, axis=-1)                       # [M_agent]
            block_w_hat = agent_w_hat[block_to_agent_arr]                  # [N_blocks, B]

            def update_block(b_idx):
                D_m = dims_arr[b_idx]
                mu_base, S_base = mu_t[b_idx, K - 1], S_t[b_idx, K - 1]
                new_mu_b, new_S_b, v_deltas = vmap(
                    _update_block_component_core,
                    in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None),
                )(jnp.arange(K), mu_t[b_idx], S_t[b_idx], samples_B[b_idx], block_w_hat[b_idx],
                  pi_all[b_idx], mu_t[b_idx], S_t[b_idx], dt, mu_base, S_base, D_m)
                return new_mu_b, new_S_b, v_deltas[:K - 1]

            next_mu, next_S, next_v_deltas = vmap(update_block)(jnp.arange(N_blocks))
            next_v = jnp.clip(v_t + dt * next_v_deltas, -70.0, 70.0)
            return (next_mu, next_S, next_v, t + 1), {"mean_fitness": mean_fitness}

        keys = random.split(key, T)
        final_state, metrics_history = lax.scan(step_fn, state, keys)

        final_mu = final_state[0]
        final_L = vmap(vmap(jnp.linalg.cholesky))(final_state[1])
        final_pi = vmap(_v_to_pi)(final_state[2])
        final_v = final_state[2]
        return final_mu, final_L, final_pi, final_v, metrics_history

    final_mu, final_L, final_pi, final_v, metrics_history = _core(
        key, initial_mu, initial_L_inv, v_init, ctx
    )
    return {
        "mu": final_mu,
        "L_inv": final_L,
        "S_or_L": final_L,
        "pi": final_pi,
        "v": final_v,
        "block_to_agent": block_to_agent,
        "dims": dims,
        "metrics": metrics_history,
    }
