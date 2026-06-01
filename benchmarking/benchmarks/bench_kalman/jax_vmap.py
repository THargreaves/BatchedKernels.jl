import jax
import jax.numpy as jnp
import time
import math
import sys


def kalman_cov(P, A, Q, H, R):
    P_pred = A @ P @ A.T + Q

    P_pred_Ht = P_pred @ H.T
    S = H @ P_pred_Ht + R

    L = jnp.linalg.cholesky(S)
    K = jax.scipy.linalg.cho_solve((L, True), P_pred_Ht.T).T

    I_KH = jnp.eye(P.shape[0]) - K @ H
    P_new = I_KH @ P_pred

    return P_new


def get_mat(key, D, N, eps=0.1, dtype=jnp.float32):
    shape = (D, D) if N == 1 else (N, D, D)
    M = jax.random.uniform(key, shape, minval=0.0, maxval=1.0, dtype=dtype) / dtype(D)
    I = jnp.eye(D, dtype=dtype)
    return M @ jnp.swapaxes(M, -1, -2) + eps * I


def kalman_timing(
    D,
    seed=0,
    min_warmup_samples=5,
    min_warmup_time_s=1.0,
    max_samples=10_000,
    max_time_s=5.0,
    min_samples=50,
):
    dtype = jnp.float32

    N = int(math.ceil(1e9 / (4 * 2 * D * D)))
    key = jax.random.PRNGKey(seed)

    key, sub = jax.random.split(key)
    P_in = get_mat(sub, D, N, dtype=dtype)

    key, sub = jax.random.split(key)
    A = jax.random.uniform(sub, (D, D), minval=0.0, maxval=1.0, dtype=dtype) / dtype(D)

    key, sub = jax.random.split(key)
    Q = get_mat(sub, D, 1, dtype=dtype)

    key, sub = jax.random.split(key)
    H = jax.random.uniform(sub, (D, D), minval=0.0, maxval=1.0, dtype=dtype) / dtype(D)

    key, sub = jax.random.split(key)
    R = get_mat(sub, D, 1, dtype=dtype)

    P_in = jax.device_put(P_in)
    A = jax.device_put(A)
    Q = jax.device_put(Q)
    H = jax.device_put(H)
    R = jax.device_put(R)

    kalman_batched = jax.jit(
        jax.vmap(kalman_cov, in_axes=(0, None, None, None, None))
    )

    def run():
        P_new = kalman_batched(P_in, A, Q, H, R)
        jax.block_until_ready(P_new)

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
    print(f"{kalman_timing(D)}")