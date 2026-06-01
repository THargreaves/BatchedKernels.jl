import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import time
import math
import sys


def gauss_likelihood_single(x, mu, Sigma):
    """log p(x | mu, Sigma) for one D-dim Gaussian (single batch element)."""
    D = x.shape[0]
    delta = x - mu
    # cholesky returns lower-triangular L with Sigma = L L^T
    L = jnp.linalg.cholesky(Sigma)
    # log|Sigma| = 2 sum log diag(L)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    # y = L \ delta, then mahalanobis = ||y||^2
    y = jsl.solve_triangular(L, delta, lower=True)
    mahal = jnp.sum(y * y)
    return -0.5 * (D * jnp.log(2.0 * jnp.pi) + log_det + mahal)


def gauss_likelihood_timing(
    D,
    seed=0,
    min_warmup_samples=5,
    min_warmup_time_s=1.0,
    max_samples=10_000,
    max_time_s=5.0,
    min_samples=50,
):
    dtype = jnp.float32

    N = int(math.ceil(1e9 / (4 * 1 * D * D)))
    key = jax.random.PRNGKey(seed)

    key, sub = jax.random.split(key)
    x = jax.random.uniform(sub, (N, D), minval=0.0, maxval=1.0, dtype=dtype)

    key, sub = jax.random.split(key)
    mu = jax.random.uniform(sub, (N, D), minval=0.0, maxval=1.0, dtype=dtype)

    # SPD Sigma per batch: Σ = M·Mᵀ/D + 0.1·I
    def make_spd_batch(k, eps=0.1):
        M = jax.random.uniform(k, (N, D, D), minval=0.0, maxval=1.0, dtype=dtype) / dtype(D)
        return M @ jnp.swapaxes(M, -1, -2) + eps * jnp.eye(D, dtype=dtype)

    key, sub = jax.random.split(key)
    Sigma = make_spd_batch(sub)

    x     = jax.device_put(x)
    mu    = jax.device_put(mu)
    Sigma = jax.device_put(Sigma)

    gauss_batched = jax.jit(jax.vmap(gauss_likelihood_single, in_axes=(0, 0, 0)))

    def run():
        p = gauss_batched(x, mu, Sigma)
        jax.block_until_ready(p)

    # Warmup
    warmup_start = time.perf_counter()
    num_warmup = 0
    while num_warmup < min_warmup_samples or time.perf_counter() - warmup_start < min_warmup_time_s:
        run()
        num_warmup += 1

    # Sampling
    times = []
    sample_start = time.perf_counter()
    for _ in range(max_samples):
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if len(times) >= min_samples and t1 - sample_start >= max_time_s:
            break

    median_time = float(jnp.median(jnp.array(times)))
    return median_time / N


if __name__ == "__main__":
    D = int(sys.argv[1])
    print(f"{gauss_likelihood_timing(D)}")
