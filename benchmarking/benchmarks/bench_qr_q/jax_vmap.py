import jax
import jax.numpy as jnp
import time
import math
import sys


def qr_q(A):
    # jnp.linalg.qr defaults to 'reduced' mode and returns (Q, R).
    # We just take Q.
    Q, _ = jnp.linalg.qr(A)
    return Q


def qr_q_timing(
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
    A = jax.random.uniform(sub, (N, D, D), minval=0.0, maxval=1.0, dtype=dtype)

    A = jax.device_put(A)

    qr_q_batched = jax.jit(jax.vmap(qr_q, in_axes=0))

    def run():
        Q = qr_q_batched(A)
        jax.block_until_ready(Q)

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
    print(f"{qr_q_timing(D)}")
