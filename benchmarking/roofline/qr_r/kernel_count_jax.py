import jax
import jax.numpy as jnp
from numba import cuda
import math
import sys

@cuda.jit
def roofline_marker():
    return


def qr_r(A):
    # mode='r' skips materialising Q — returns only the upper-triangular R.
    return jnp.linalg.qr(A, mode='r')


def launch_one_jax(
    D,
    loop_count,
    seed=0,
):
    dtype = jnp.float32

    N = int(math.ceil(1e9 / (4 * 2 * D * D)))
    key = jax.random.PRNGKey(seed)

    key, sub = jax.random.split(key)
    A = jax.random.uniform(sub, (N, D, D), minval=0.0, maxval=1.0, dtype=dtype)

    A = jax.device_put(A)

    qr_r_batched = jax.jit(jax.vmap(qr_r, in_axes=0))

    def run():
        R = qr_r_batched(A)
        jax.block_until_ready(R)

    for i in range(loop_count):
        if i == loop_count - 1:
            roofline_marker[1, 1]()
            cuda.synchronize()
        run()


if __name__ == "__main__":
    D = int(sys.argv[1])
    loop_count = int(sys.argv[2])
    launch_one_jax(D, loop_count)

# nsys profile --force-overwrite true --stats=true \
#   -o qr_r/tmp/qr_jax_trace \
#   ../myenv/bin/python qr_r/kernel_count_jax.py 8 100

# nsys profile --force-overwrite true --stats=true \
#   -o qr_r/profile_results/qr_jax_trace \
#   ../myenv/bin/python qr_r/kernel_count_jax.py 8 100
