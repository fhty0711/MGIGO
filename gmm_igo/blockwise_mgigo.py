import jax
import jax.numpy as jnp
from jax import lax, random
import functools


Array = jax.Array


def initialize_blockwise_mgigo_params(
    key: Array,
    dims: tuple[int, ...],
    k_components: int,
    mean_std: float = 2.0,
    precision_scale: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Initialize (mu, S, pi) for blockwise MGIGO.

    Args:
        key: PRNG key.
        dims: Per-block dimensions, e.g. (3, 2, 4).
        k_components: Number of Gaussian components per block.
        mean_std: Std for mean initialization.
        precision_scale: Diagonal precision scale.

    Returns:
        mu0: (N, K, D_max)
        s0: (N, K, D_max, D_max) precision matrices
        pi0: (N, K) mixture weights
    """
    n_blocks = len(dims)
    d_max = max(dims)

    key_mu, _ = random.split(key)
    mu0 = mean_std * random.normal(key_mu, (n_blocks, k_components, d_max))

    dims_arr = jnp.array(dims)
    mask = (jnp.arange(d_max)[None, :] < dims_arr[:, None]).astype(mu0.dtype)
    mu0 = mu0 * mask[:, None, :]

    eye = jnp.eye(d_max, dtype=mu0.dtype)
    s0 = jnp.tile(eye[None, None, :, :], (n_blocks, k_components, 1, 1))
    s0 = s0 * precision_scale

    pi0 = jnp.full((n_blocks, k_components), 1.0 / k_components, dtype=mu0.dtype)
    return mu0, s0, pi0


def _pack_sample_from_blocks(x_b: Array, dims: tuple[int, ...]) -> Array:
    parts = [x_b[j, : dims[j]] for j in range(len(dims))]
    return jnp.concatenate(parts, axis=0)


def _masked_precision(
    s: Array,
    dim_j: Array,
    d_max: int,
    eps: Array,
) -> Array:
    mask = (jnp.arange(d_max) < dim_j).astype(s.dtype)
    m2 = jnp.outer(mask, mask)
    s_proj = s * m2
    s_proj = s_proj + jnp.eye(d_max, dtype=s.dtype) * (1.0 - mask)
    s_proj = 0.5 * (s_proj + s_proj.T)
    s_proj = s_proj + eps * jnp.eye(d_max, dtype=s.dtype)
    return s_proj


def _masked_vector(v: Array, dim_j: Array) -> Array:
    mask = (jnp.arange(v.shape[0]) < dim_j).astype(v.dtype)
    return v * mask


def _gaussian_logpdf_precision_masked(x: Array, mu: Array, s: Array, dim_j: Array) -> Array:
    diff = _masked_vector(x - mu, dim_j)
    q = diff @ (s @ diff)
    l = jnp.linalg.cholesky(s)
    logdet_s = 2.0 * jnp.sum(jnp.log(jnp.diag(l) + 1e-20))
    d = dim_j.astype(x.dtype)
    return 0.5 * (logdet_s - d * jnp.log(2.0 * jnp.pi) - q)


def _sample_from_precision(
    key: Array,
    mu: Array,
    s: Array,
    dim_j: Array,
) -> Array:
    d_max = mu.shape[0]
    z = random.normal(key, (d_max,), dtype=mu.dtype)
    l = jnp.linalg.cholesky(s)
    y = lax.linalg.triangular_solve(
        l,
        z[:, None],
        left_side=True,
        lower=True,
        transpose_a=True,
        conjugate_a=False,
        unit_diagonal=False,
    )[:, 0]
    x = mu + y
    return _masked_vector(x, dim_j)


def _update_block(
    mu_j: Array,
    s_j: Array,
    pi_j: Array,
    x_elite_j: Array,
    b_samples: Array,
    dim_j: Array,
    alpha_t: Array,
    t: Array,
    t_reset: Array,
    eps: Array,
) -> tuple[Array, Array, Array]:
    k_components = mu_j.shape[0]
    # b0 = x_elite_j.shape[0]
    d_max = mu_j.shape[1]
    w = 1.0 / b_samples

    def logpdf_one_k(mu_k: Array, s_k: Array) -> Array:
        return jax.vmap(lambda xb: _gaussian_logpdf_precision_masked(xb, mu_k, s_k, dim_j))(x_elite_j)

    logpdf_kb = jax.vmap(logpdf_one_k)(mu_j, s_j)
    log_denom = jax.nn.logsumexp(jnp.log(pi_j + 1e-20)[:, None] + logpdf_kb, axis=0)
    a_kb = jnp.exp(logpdf_kb - log_denom[None, :])

    def update_one_component(mu_k: Array, s_k: Array, a_k: Array) -> tuple[Array, Array]:
        diff = x_elite_j - mu_k[None, :]

        def one_term(d: Array) -> Array:
            return (s_k @ jnp.outer(d, d) @ s_k) - s_k

        nat_terms = jax.vmap(one_term)(diff)
        delta_s = jnp.sum((w * a_k)[:, None, None] * nat_terms, axis=0)

        s_new = s_k - alpha_t * delta_s
        s_new = _masked_precision(s_new, dim_j, d_max, eps)

        grad_mu_terms = jax.vmap(lambda d: s_k @ d)(diff)
        sum_mu_grad = jnp.sum((w * a_k)[:, None] * grad_mu_terms, axis=0)
        delta_mu = jnp.linalg.solve(s_new, sum_mu_grad)
        mu_new = mu_k + alpha_t * delta_mu
        mu_new = _masked_vector(mu_new, dim_j)
        return mu_new, s_new

    mu_new, s_new = jax.vmap(update_one_component)(mu_j, s_j, a_kb)

    def update_pi(_: None) -> Array:
        log_ratio_old = jnp.log((pi_j[:-1] + 1e-20) / (pi_j[-1] + 1e-20))
        grad_ratio = jnp.sum(w * (a_kb[:-1, :] - a_kb[-1:, :]), axis=1)
        log_ratio_new = log_ratio_old + alpha_t * grad_ratio
        logits = jnp.concatenate([log_ratio_new, jnp.zeros((1,), dtype=pi_j.dtype)])
        return jax.nn.softmax(logits)

    # def reset_pi(_: None) -> Array:
        # return jnp.full_like(pi_j, 1.0 / k_components)

    pi_new = update_pi(None)
    return mu_new, s_new, pi_new


def _one_iteration(
    state: tuple[Array, Array, Array],
    inputs: tuple[Array, Array],
    objective_fn,
    context,
    dims: tuple[int, ...],
    b_samples: int,
    b0_elite: int,
    alpha_t: float,
    t_reset: int,
    eps: float,
):
    mu, s, pi = state
    key_t, t = inputs

    n_blocks, k_components, d_max = mu.shape
    dims_arr = jnp.array(dims)

    key_blocks = random.split(key_t, n_blocks)

    def sample_one_block(block_key: Array, mu_j: Array, s_j: Array, pi_j: Array, dim_j: Array) -> Array:
        key_comp, key_noise = random.split(block_key)
        comp_idx = random.choice(key_comp, k_components, shape=(b_samples,), p=pi_j)
        keys_b = random.split(key_noise, b_samples)
        mu_sel = mu_j[comp_idx]
        s_sel = s_j[comp_idx]

        x_block = jax.vmap(_sample_from_precision)(keys_b, mu_sel, s_sel, jnp.full((b_samples,), dim_j))
        return x_block

    x_blocks = jax.vmap(sample_one_block)(key_blocks, mu, s, pi, dims_arr)

    x_bn = jnp.transpose(x_blocks, (1, 0, 2))
    x_full = jax.vmap(lambda xb: _pack_sample_from_blocks(xb, dims))(x_bn)

    f_vals = jax.vmap(lambda x: objective_fn(x, context))(x_full)
    elite_idx = jnp.argsort(f_vals)[:b0_elite]
    x_elite = x_blocks[:, elite_idx, :]

    mu_new, s_new, pi_new = jax.vmap(_update_block)(
        mu,
        s,
        pi,
        x_elite,
        jnp.full((n_blocks,), b_samples),
        dims_arr,
        jnp.full((n_blocks,), alpha_t),
        jnp.full((n_blocks,), t),
        jnp.full((n_blocks,), t_reset),
        jnp.full((n_blocks,), eps),
    )

    return (mu_new, s_new, pi_new), {
        "best_f": jnp.min(f_vals),
        "mean_f": jnp.mean(f_vals),
    }


def blockwise_mgigo(
    key: Array,
    objective_fn,
    context,
    dims: tuple[int, ...],
    t_iters: int,
    b_samples: int,
    b0_elite: int,
    k_components: int,
    alpha_t: float,
    t_reset: int,
    mu0: Array | None = None,
    s0: Array | None = None,
    pi0: Array | None = None,
    eps: float = 1e-8,
    use_jit: bool = True,
) -> dict[str, Array]:
    """Blockwise MGIGO (Algorithm 3) with JAX GPU acceleration.

    objective_fn must be JAX-compatible and accept (x, context).
    """
    if isinstance(b0_elite, int) and isinstance(b_samples, int) and (b0_elite > b_samples):
        raise ValueError("b0_elite must be <= b_samples")

    if mu0 is None or s0 is None or pi0 is None:
        mu_init, s_init, pi_init = initialize_blockwise_mgigo_params(key, dims, k_components)
    else:
        mu_init, s_init, pi_init = mu0, s0, pi0

    keys = random.split(key, t_iters)
    ts = jnp.arange(t_iters)
    init_state = (mu_init, s_init, pi_init)

    scan_fn = lambda st, inp: _one_iteration(
        st,
        inp,
        objective_fn=objective_fn,
        context=context,
        dims=dims,
        b_samples=b_samples,
        b0_elite=b0_elite,
        alpha_t=alpha_t,
        t_reset=t_reset,
        eps=eps,
    )

    if use_jit:
        scan_fn = jax.jit(scan_fn)

    final_state, logs = lax.scan(scan_fn, init_state, (keys, ts))
    mu_t, s_t, pi_t = final_state

    return {
        "mu": mu_t,
        "S": s_t,
        "pi": pi_t,
        "best_f_history": logs["best_f"],
        "mean_f_history": logs["mean_f"],
    }


def _pi_from_v(v: Array) -> Array:
    exps = jnp.exp(jnp.clip(v, -70.0, 70.0))
    denom = 1.0 + jnp.sum(exps, axis=1, keepdims=True)
    pi_head = exps / denom
    pi_tail = 1.0 / denom
    return jnp.concatenate([pi_head, pi_tail], axis=1)


def _v_from_pi(pi: Array) -> Array:
    return jnp.log((pi[:, :-1] + 1e-20) / (pi[:, -1:] + 1e-20))


@functools.partial(jax.jit, static_argnums=(1, 3, 4, 5, 6, 7, 8, 9))
def mmog_igo_optimizer_mpc(
    key: Array,
    T: int,
    dt: float,
    M: int,
    K: int,
    B: int,
    B0: int,
    dims: tuple[int, ...],
    T_0: int,
    cost_function,
    initial_mu_k: Array,
    initial_L_inv_k: Array,
    initial_v_k: Array,
    context,
) -> tuple[Array, Array, Array, Array]:
    """MPCresetweight-compatible MGIGO solver interface.

    Signature and return format are aligned with igo/MPCresetweight.py:
    return final_mu, final_L_inv, final_pi, final_v
    """
    if len(dims) != M:
        raise ValueError("len(dims) must equal M")

    mu0 = initial_mu_k[:, :K, :]
    l_inv0 = initial_L_inv_k[:, :K, :, :]
    s0 = l_inv0 @ jnp.swapaxes(l_inv0, -1, -2)

    v0 = initial_v_k[:, : K - 1]
    pi0 = _pi_from_v(v0)

    out = blockwise_mgigo(
        key=key,
        objective_fn=cost_function,
        context=context,
        dims=dims,
        t_iters=T,
        b_samples=B,
        b0_elite=B0,
        k_components=K,
        alpha_t=dt,
        t_reset=T_0,
        mu0=mu0,
        s0=s0,
        pi0=pi0,
        eps=1e-8,
        use_jit=False,
    )

    final_mu = out["mu"]
    final_s = out["S"]
    final_pi = out["pi"]
    final_v = jnp.clip(_v_from_pi(final_pi), -70.0, 70.0)
    final_l_inv = jax.vmap(jax.vmap(jnp.linalg.cholesky))(final_s)
    return final_mu, final_l_inv, final_pi, final_v