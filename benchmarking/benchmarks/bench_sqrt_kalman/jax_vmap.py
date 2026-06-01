import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import time
import math
import sys


def sqrt_kalman_step(S, A, S_Q, H, S_R):
    """One square-root Kalman covariance update.

    Inputs (all 2D, single batch element):
        S:   lower-triangular sqrt of P (D × D)
        A:   state transition (D × D)
        S_Q: lower-triangular sqrt of Q (D × D)
        H:   observation matrix (D × D)
        S_R: lower-triangular sqrt of R (D × D)

    Output: new lower-triangular sqrt of P (D × D).
    """
    # Predict-step pre-array:   [Xᵀ ; S_Qᵀ]   where X = A · S
    X = A @ S
    M_pred = jnp.concatenate([X.T, S_Q.T], axis=0)            # (2D, D)
    R_pred = jnp.linalg.qr(M_pred, mode='r')                  # (D, D), upper-tri

    # Update-step pre-array:    [S_Rᵀ  0     ]
    #                           [Yᵀ    R_pred]   where Y = H · S_pred = H · R_predᵀ
    S_pred = R_pred.T                                         # lower-tri
    Y = H @ S_pred
    D = S.shape[0]
    zero_block = jnp.zeros((D, D), dtype=S.dtype)
    top    = jnp.concatenate([S_R.T,   zero_block], axis=1)   # (D, 2D)
    bottom = jnp.concatenate([Y.T,     R_pred    ], axis=1)   # (D, 2D)
    M_upd  = jnp.concatenate([top, bottom], axis=0)           # (2D, 2D)

    R_upd = jnp.linalg.qr(M_upd, mode='r')                    # (2D, 2D), upper-tri
    R_22  = R_upd[D:, D:]                                     # (D, D), upper-tri
    return R_22.T                                             # lower-tri new S


def sqrt_kalman_timing(
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

    # A, H — D × D random, scaled by 1/D
    key, sub = jax.random.split(key)
    A = jax.random.uniform(sub, (D, D), minval=0.0, maxval=1.0, dtype=dtype) / dtype(D)

    key, sub = jax.random.split(key)
    H = jax.random.uniform(sub, (D, D), minval=0.0, maxval=1.0, dtype=dtype) / dtype(D)

    # Q, R — random SPD; sqrt them (lower-triangular Cholesky factors)
    def make_spd(k, eps=0.01):
        M = jax.random.uniform(k, (D, D), minval=0.0, maxval=1.0, dtype=dtype) / dtype(D) ** 2
        return M @ M.T + eps * jnp.eye(D, dtype=dtype)

    key, sub = jax.random.split(key)
    Q = make_spd(sub)
    S_Q = jnp.linalg.cholesky(Q)   # lower-triangular

    key, sub = jax.random.split(key)
    R = make_spd(sub)
    S_R = jnp.linalg.cholesky(R)

    # Per-batch S = chol(P).L for random SPD P_i
    def make_spd_batch(k, eps=0.1):
        M = jax.random.uniform(k, (N, D, D), minval=0.0, maxval=1.0, dtype=dtype) / dtype(D)
        return M @ jnp.swapaxes(M, -1, -2) + eps * jnp.eye(D, dtype=dtype)

    key, sub = jax.random.split(key)
    P_in = make_spd_batch(sub)
    S_in = jnp.linalg.cholesky(P_in)   # (N, D, D), lower-tri per batch

    S_in = jax.device_put(S_in)
    A    = jax.device_put(A)
    S_Q  = jax.device_put(S_Q)
    H    = jax.device_put(H)
    S_R  = jax.device_put(S_R)

    # vmap over the batched S; broadcast A, S_Q, H, S_R
    sqrt_kalman_batched = jax.jit(
        jax.vmap(sqrt_kalman_step, in_axes=(0, None, None, None, None))
    )

    def run():
        S_out = sqrt_kalman_batched(S_in, A, S_Q, H, S_R)
        jax.block_until_ready(S_out)

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
    print(f"{sqrt_kalman_timing(D)}")
