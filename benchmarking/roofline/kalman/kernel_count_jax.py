import jax
import jax.numpy as jnp
from numba import cuda
import math
import sys

@cuda.jit
def roofline_marker():
    return


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


def launch_one_jax(
    D,
    loop_count,
    seed=0,
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

    for i in range(loop_count):
        if i == loop_count - 1:
            roofline_marker[1, 1]()
            cuda.synchronize()
        run()


if __name__ == "__main__":
    D = int(sys.argv[1])
    loop_count = int(sys.argv[2])
    launch_one_jax(D, loop_count)
