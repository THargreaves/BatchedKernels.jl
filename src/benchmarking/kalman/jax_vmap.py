import jax
import jax.numpy as jnp
import time
import math


# @jax.jit
# def kalman_cov_batch(P, A, Q, H, R):
#     def single(Pi, Ai, Qi, Hi, Ri):
#         P_pred = Ai @ Pi @ Ai.T + Qi
#         S = Hi @ P_pred @ Hi.T + Ri
#         Ki = jnp.linalg.solve(S, (P_pred @ Hi.T).T).T
#         P_new_i = P_pred - Ki @ S @ Ki.T
#         return P_new_i
#     return jax.vmap(single)(P, A, Q, H, R)


def make_kalman_cov_batch(A, Q, H, R):
    @jax.jit
    def kalman(P):
        def single(Pi):
            P_pred = A @ Pi @ A.T + Q
            S      = H @ P_pred @ H.T + R
            PHt    = P_pred @ H.T
            K      = jnp.linalg.solve(S, PHt.T).T
            P_new  = P_pred - K @ S @ K.T
            return P_new
        return jax.vmap(single)(P)

    return kalman


def get_mat(key, D, N, eps=0.1, dtype=jnp.float32):
    shape = (D, D) if N == 1 else (N, D, D)
    M = jax.random.uniform(key, shape, minval=0.0, maxval=1.0, dtype=dtype) / dtype(D)
    I = jnp.eye(D, dtype=dtype)
    return M @ jnp.swapaxes(M, -1, -2) + eps * I


def kalman_timing(D, reps=20, seed=0):
    result = {}
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

    kalman_cov_batch = make_kalman_cov_batch(A, Q, H, R)

    def run():
        P_new = kalman_cov_batch(P_in) #, A, Q, H, R)
        jax.block_until_ready(P_new)

    # Warp-up
    run()

    # avg_time = timeit.timeit(run, number=reps) / reps

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    median_time = float(jnp.median(jnp.array(times)))
    time_per_matrix = median_time / N

    return time_per_matrix

# result = kalman_timing()

# print("[", end="")
# for k, v in result.items():
#     print(f"{v},", end="")
# print("]")