import jax
import jax.numpy as jnp
from jax import vmap, random, lax, jit
import functools

# I. Core Helpers
MIN_EIG = 1e-2
MAX_EIG = 1e3

def _safe_spd_projection(S):
    eigvals, eigvecs = jnp.linalg.eigh(S)
    eigvals = jnp.clip(eigvals, MIN_EIG, MAX_EIG)
    return eigvecs @ (eigvals[:, None] * eigvecs.T)

@jit
def _logsumexp(a, axis=None):
    return jnp.logaddexp.reduce(a, axis=axis)

@jit
def _gaussian_log_pdf_l_masked(xi, mu, S, D_m):
    diff = (xi - mu)
    mask = jnp.arange(xi.shape[0]) < D_m
    diff = diff * mask
    mahalanobis_sq = jnp.dot(diff, jnp.dot(S, diff))
    sign, logdet_S = jnp.linalg.slogdet(S)
    log_pdf = -0.5 * (D_m * jnp.log(2 * jnp.pi) - logdet_S + mahalanobis_sq)
    return log_pdf

# II. Single Component Update (Alg 4)
def _update_component_core(k_idx, mu_k, S_k, samples, elite_weights, pi_all, mu_all, S_all, delta_t, mu_base, S_base, D_m):
    log_pdf_k = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_k, S_k, D_m))(samples)
    log_pdf_base = vmap(lambda x: _gaussian_log_pdf_l_masked(x, mu_base, S_base, D_m))(samples)
    def mog_pdf_fn(xi):
        l_pdfs = vmap(lambda m, s: _gaussian_log_pdf_l_masked(xi, m, s, D_m))(mu_all, S_all)
        return _logsumexp(jnp.log(pi_all + 1e-15) + l_pdfs)
    log_mog = vmap(mog_pdf_fn)(samples)
    a_i = jnp.exp(jnp.clip(log_pdf_k - log_mog, -70.0, 70.0))
    b_i = jnp.exp(jnp.clip(log_pdf_base - log_mog, -70.0, 70.0))
    diff = (samples - mu_k)
    def s_grad_fn(d):
        return S_k @ jnp.outer(d, d) @ S_k - S_k
    sum_S_grad = jnp.sum((elite_weights * a_i)[:, None, None] * vmap(s_grad_fn)(diff), axis=0)
    S_new = _safe_spd_projection(S_k - delta_t * sum_S_grad)
    S_diff = (S_k @ diff.T).T
    sum_mu_grad = jnp.sum((elite_weights * a_i)[:, None] * S_diff, axis=0)
    mu_new = mu_k + delta_t * jnp.linalg.solve(S_new, sum_mu_grad)
    v_delta = jnp.sum(elite_weights * (a_i - b_i))
    return mu_new, S_new, v_delta

# III. Alg 6: RNE Step with Sample Reuse
def _step_fn_rne(state, iter_data, M_agent, K, B, B0, dt, dims_arr, T_0, fitness_fns, v_reset, context, M_inner):
    mu, S, v, t = state
    key, _ = iter_data
    v = jnp.where((t % T_0) == 0, v_reset, v)
    def v_to_pi(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    pi_all = vmap(v_to_pi)(v)

    def sample_batch(count, sub_key):
        def sample_single_agent(agent_idx, a_key):
            comps = random.choice(a_key, K, p=pi_all[agent_idx], shape=(count,))
            def gen_sample(c_idx, s_key):
                cov = jnp.linalg.inv(S[agent_idx, c_idx] + jnp.eye(S.shape[-1]) * 1e-7)
                return random.multivariate_normal(s_key, mu[agent_idx, c_idx], cov)
            return vmap(gen_sample)(comps, random.split(a_key, count))
        return vmap(sample_single_agent)(jnp.arange(M_agent), random.split(sub_key, M_agent))

    key_B, key_M = random.split(key)
    samples_B = sample_batch(B, key_B)       
    samples_M = sample_batch(M_inner, key_M) 

    def evaluate_expected_cost(agent_idx):
        zb_candidates = samples_B[agent_idx]
        fitness_fn = fitness_fns[agent_idx]          # ← 取出该 agent 专属的函数
        
        def compute_f_hat_for_one_zb(zb_i):
            def eval_with_context_m(m_idx):
                joint_m = samples_M[:, m_idx, :].at[agent_idx].set(zb_i)
                return fitness_fn(agent_idx, joint_m.flatten(), context)
            f_vals = vmap(eval_with_context_m)(jnp.arange(M_inner))
            return jnp.mean(f_vals)
        
        f_hat = vmap(compute_f_hat_for_one_zb)(zb_candidates)
        ranks = jnp.argsort(jnp.argsort(f_hat))
        return jnp.where(ranks < B0, 1.0 / B, 0.0)

    w_hat_m = jnp.stack([evaluate_expected_cost(agent_idx) for agent_idx in range(M_agent)])


    def update_block(m_idx):
        D_m = dims_arr[m_idx]
        mu_base, S_base = mu[m_idx, K-1], S[m_idx, K-1]
        new_mu_m, new_S_m, v_deltas = vmap(
            _update_component_core,
            in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None)
        )(jnp.arange(K), mu[m_idx], S[m_idx], samples_B[m_idx], w_hat_m[m_idx], 
          pi_all[m_idx], mu[m_idx], S[m_idx], dt, mu_base, S_base, D_m)
        return new_mu_m, new_S_m, v_deltas[:K-1]

    next_mu, next_S, next_v_deltas = vmap(update_block)(jnp.arange(M_agent))
    next_v = jnp.clip(v + dt * next_v_deltas, -70.0, 70.0)
    return (next_mu, next_S, next_v, t + 1), None


# IV. Top-level Solver


@functools.partial(jit, static_argnums=(1, 3, 4, 5, 7, 9, 13))
def mmog_igo_rne_solver(key, T, dt, M_agent, K, B, B0, dims, T_0, 
                        fitness_fns, initial_mu_k, initial_L_inv_k, context, M_inner):
    """
    fitness_fns: List of callables, length == M_agent
                 每个 agent 可以传入完全独立的 fitness 函数
    """
    dims_array = jnp.array(dims)
    v_reset = jnp.zeros((M_agent, K-1))
    
    S_init = vmap(vmap(lambda L: L @ L.T))(initial_L_inv_k[:, :K, :, :])
    mu_init = initial_mu_k[:, :K, :]
    v_init = jnp.zeros((M_agent, K-1)) 

    state = (mu_init, S_init, v_init, 0)
    
    loop_fn = functools.partial(
        _step_fn_rne, M_agent=M_agent, K=K, B=B, B0=B0, dt=dt, 
        dims_arr=dims_array, T_0=T_0, 
        fitness_fns=fitness_fns,
        v_reset=v_reset, context=context, M_inner=M_inner
    )
    
    final_state, _ = lax.scan(loop_fn, state, (random.split(key, T), jnp.arange(T)))
    
    final_mu = final_state[0]
    final_L = vmap(vmap(jnp.linalg.cholesky))(final_state[1])
    
    def v_to_pi_final(v_m):
        exps = jnp.exp(jnp.clip(v_m, -70, 70))
        sum_e = 1.0 + jnp.sum(exps)
        return jnp.concatenate([exps / sum_e, jnp.array([1.0 / sum_e])])
    final_pi = vmap(v_to_pi_final)(final_state[2])
    
    return final_mu, final_L, final_pi