# Steps:
# export PYTHON=/scratch/sy440/BatchedKernels.jl/benchmarking/kalman/myenv/bin/python
# export PYCALL_BUILD_CONDA=false

using PyCall
sys = pyimport("sys")

const KALMAN_DIR = @__DIR__

sys = pyimport("sys")
py_paths = String.(sys."path")
if KALMAN_DIR ∉ py_paths
    pushfirst!(sys."path", KALMAN_DIR)
end

# jax_mod = pyimport("jax_vmap")

function kalman_timing(_, P_in_cpu, _, _, _, _, _, ::Val{:jax_vmap})
    D = size(P_in_cpu, 1)
    script = joinpath(@__DIR__, "jax_vmap.py")
    python = joinpath(@__DIR__, "myenv", "bin", "python3")
    output = readchomp(`$python $script $D`)
    return parse(Float64, output)
end