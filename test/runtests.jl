using TestItems
using TestItemRunner

@run_package_tests

# Sub-kernel tests — exercise src/memory.jl + src/operations.jl directly
# via hand-written CUDA kernels.
include("sub_kernels/test_cholesky.jl")
include("sub_kernels/test_matmul.jl")
include("sub_kernels/test_solve.jl")
include("sub_kernels/test_mat_subtraction.jl")
include("sub_kernels/test_vector_ops.jl")
include("sub_kernels/test_qr.jl")
include("sub_kernels/test_transpose.jl")
include("sub_kernels/test_gauss_likelihood.jl")

# Fuser unit tests — CPU-only tests over tape / planner / scheduler.
include("fusion/test_schedule.jl")

# Fused-kernel end-to-end tests — exercise the broadcast API and the full
# trace → plan → codegen pipeline.