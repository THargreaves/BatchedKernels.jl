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

# jax_mod = pyimport("jax_vmap")

function matmul_timing(C_out_cpu, _, _, _, _, ::Val{:jax_vmap})
    D = size(C_out_cpu, 1)
    script = normpath(joinpath(@__DIR__, "jax_vmap.py"))
    python = normpath(joinpath(@__DIR__, "..", "..", "myenv", "bin", "python3"))

    isfile(python) || error("Python venv not found at $python")
    isfile(script) || error("Script not found at $script")

    output = readchomp(`$python $script $D`)
    return parse(Float64, output)
end
