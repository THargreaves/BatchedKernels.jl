import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import time
import math
import sys


def backsolve(U, B):
    # Solve U * X = B  with U upper triangular  →  X = U^{-1} B
    return jsl.solve_triangular(U, B, lower=False)


def get_upper(key, D, N, dtype=jnp.float32):
    # Matches `UpperTriangular(U_temp) + 0.5*I` from run_script.jl
    U = jax.random.uniform(key, (N, D, D), minval=0.0, maxval=1.0, dtype=dtype)
    U = jnp.triu(U)
    I = jnp.eye(D, dtype=dtype)
    return U + 0.5 * I


def backsolve_timing(
    D,
    seed=0,
    min_warmup_samples=5,
    min_warmup_time_s=1.0,
    max_samples=10_000,
    max_time_s=5.0,
    min_samples=50,
):
    dtype = jnp.float32

    N = int(math.ceil(1e9 / (4 * 3 * D * D)))
    key = jax.random.PRNGKey(seed)

    key, sub = jax.random.split(key)
    U = get_upper(sub, D, N, dtype=dtype)

    key, sub = jax.random.split(key)
    B = jax.random.uniform(sub, (N, D, D), minval=0.0, maxval=1.0, dtype=dtype)

    U = jax.device_put(U)
    B = jax.device_put(B)

    backsolve_batched = jax.jit(
        jax.vmap(backsolve, in_axes=(0, 0))
    )

    def run():
        C = backsolve_batched(U, B)
        jax.block_until_ready(C)

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
    print(f"{backsolve_timing(D)}")
