import jax
import jax.numpy as jnp
import numpy as np
from jax import random, vmap

MIN_CHOLESKY_DIAG = 1e-6
MIN_EIGVAL = 1e-10
ELLIPSE_ABS_LIMIT = 100.0

def _sanitize_linv(l_inv):
    l_lower = jnp.tril(l_inv)
    diag = jnp.clip(jnp.diag(l_lower), min=MIN_CHOLESKY_DIAG)
    return l_lower - jnp.diag(jnp.diag(l_lower)) + jnp.diag(diag)

def _safe_cov_from_linv(l_inv):
    l_stable = _sanitize_linv(l_inv)
    dim = l_stable.shape[0]
    eye = jnp.eye(dim, dtype=l_stable.dtype)
    tmp = jnp.linalg.solve(l_stable, eye)
    cov = jnp.linalg.solve(l_stable.T, tmp)
    cov = (cov + cov.T) * 0.5
    eigvals, eigvecs = jnp.linalg.eigh(cov)
    eigvals = jnp.clip(eigvals, min=MIN_EIGVAL, max=1e8)
    return eigvecs @ (eigvals[:, None] * eigvecs.T)

def covariance_from_linv(l_inv):
    return _safe_cov_from_linv(l_inv)

def _safe_eigh_numpy(cov):
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.all(np.isfinite(cov)):
        dim = cov.shape[0] if cov.ndim == 2 else 2
        cov = np.eye(dim, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, MIN_EIGVAL, 1e8)
    return eigvals, eigvecs

def _corr_from_cov(cov):
    v1 = cov[0, 0]
    v2 = cov[1, 1]
    denom = jnp.sqrt(jnp.maximum(v1 * v2, 1e-12))
    return cov[0, 1] / denom

def build_joint_component_candidates(mu, l_inv):
    m = mu.shape[0]
    k = mu.shape[1]
    d_max = mu.shape[2]

    if m == 1 and d_max == 2:
        means = mu[0]
        covs = vmap(covariance_from_linv)(l_inv[0])
        return means, covs

    if m == 2 and d_max == 1:
        means_list = []
        covs_list = []
        for i in range(k):
            for j in range(k):
                mean_ij = jnp.array([mu[0, i, 0], mu[1, j, 0]])
                var1 = covariance_from_linv(l_inv[0, i])[0, 0]
                var2 = covariance_from_linv(l_inv[1, j])[0, 0]
                cov_ij = jnp.array([[var1, 0.0], [0.0, var2]])
                means_list.append(mean_ij)
                covs_list.append(cov_ij)
        return jnp.stack(means_list, axis=0), jnp.stack(covs_list, axis=0)
    raise ValueError(f"Unsupported shape: M={m}, D_max={d_max}")

def get_best_component_stats(mu, l_inv, fitness_fn):
    means, covs = build_joint_component_candidates(mu, l_inv)
    mu_vals = vmap(lambda m_val: fitness_fn(m_val, None))(means)
    best_comp = int(jnp.argmin(mu_vals))
    best_mu = np.asarray(means[best_comp])
    best_cov = np.asarray(covs[best_comp])
    best_sigma = np.sqrt(np.maximum(np.diag(best_cov), 1e-12))
    return best_comp, best_mu, best_cov, best_sigma

def sample_from_mog(key, mu, l_inv, pi, sample_count):
    mu = jnp.asarray(mu)
    l_inv = jnp.asarray(l_inv)
    pi = jnp.asarray(pi)
    dim = mu.shape[1]
    key_comp, key_noise = random.split(key)

    comp_ids = random.choice(key_comp, mu.shape[0], shape=(sample_count,), p=pi)
    noise = random.normal(key_noise, shape=(sample_count, dim))

    def _sample_one(c_idx, eps):
        l_stable = _sanitize_linv(l_inv[c_idx])
        noise_cov = jnp.linalg.solve(l_stable.T, eps)
        return mu[c_idx] + noise_cov

    samples = vmap(_sample_one)(comp_ids, noise)
    return samples, comp_ids

def sample_joint_from_solver(key, mu, l_inv, pi, sample_count):
    m = mu.shape[0]
    d_max = mu.shape[2]

    if m == 1 and d_max == 2:
        samples, _ = sample_from_mog(key, mu[0], l_inv[0], pi[0], sample_count)
        return samples

    if m == 2 and d_max == 1:
        key1, key2 = random.split(key)
        samples1, _ = sample_from_mog(key1, mu[0], l_inv[0], pi[0], sample_count)
        samples2, _ = sample_from_mog(key2, mu[1], l_inv[1], pi[1], sample_count)
        return jnp.concatenate([samples1, samples2], axis=1)

    raise ValueError(f"Unsupported shape for sampling: M={m}, D_max={d_max}")