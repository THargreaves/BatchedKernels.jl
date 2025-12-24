import jax
import jax.numpy as jnp
import time
import math


def matmul(A, B):
    return A @ B

def matmul_timing(D, reps=20, seed=0):
    dtype = jnp.float32

    N = int(math.ceil(1e9 / (4 * 2 * D * D)))
    key = jax.random.PRNGKey(seed)

    key, sub = jax.random.split(key)
    A = jax.random.uniform(sub, (N, D, D), minval=0.0, maxval=1.0, dtype=dtype) / dtype(D)

    key, sub = jax.random.split(key)
    B = jax.random.uniform(sub, (N, D, D), minval=0.0, maxval=1.0, dtype=dtype) / dtype(D)

    A = jax.device_put(A)
    B = jax.device_put(B)

    matmul_vmap = jax.jit(jax.vmap(matmul))

    def run():
        C = matmul_vmap(A, B)
        jax.block_until_ready(C)

    # Warp-up
    run()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    median_time = float(jnp.median(jnp.array(times)))
    time_per_matrix = median_time / N

    return time_per_matrix
