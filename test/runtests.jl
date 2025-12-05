using TestItems
using TestItemRunner

@run_package_tests

include("cholesky/test_cholesky.jl")
include("kalman/test_kalman.jl")
include("matmul/test_matmul.jl")
include("solve/test_solve.jl")
include("mat_subtraction/test_mat_subtraction.jl")
include("vector_ops/test_vector_ops.jl")