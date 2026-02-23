# Steps:
# export PYTHON=/scratch/sy440/BatchedKernels.jl/benchmarking/kalman/myenv/bin/python
# export PYCALL_BUILD_CONDA=false

using PyCall
sys = pyimport("sys")

const MATMUL_DIR = @__DIR__

sys = pyimport("sys")
py_paths = String.(sys."path")
if MATMUL_DIR ∉ py_paths
    pushfirst!(sys."path", MATMUL_DIR)
end

jax_mod = pyimport("jax_vmap")

function matmul_timing(C_cpu, _, _, _, ::Val{:jax_vmap})
    D = size(C_cpu, 1)

    return Float64(jax_mod.matmul_timing(D; reps=20))
end