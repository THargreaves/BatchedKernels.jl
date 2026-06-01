# Steps:
# export PYTHON=/scratch/sy440/BatchedKernels.jl/benchmarking/kalman/myenv/bin/python
# export PYCALL_BUILD_CONDA=false

using PyCall
sys = pyimport("sys")

const GAUSS_LIKELIHOOD_DIR = @__DIR__

sys = pyimport("sys")
py_paths = String.(sys."path")
if GAUSS_LIKELIHOOD_DIR ∉ py_paths
    pushfirst!(sys."path", GAUSS_LIKELIHOOD_DIR)
end

# jax_mod = pyimport("jax_vmap")

function gauss_likelihood_timing(_, xs_in_cpu, _, _, _, _, ::Val{:jax_vmap})
    D = size(xs_in_cpu, 1)
    script = normpath(joinpath(@__DIR__, "jax_vmap.py"))
    python = normpath(joinpath(@__DIR__, "..", "..", "myenv", "bin", "python3"))

    isfile(python) || error("Python venv not found at $python")
    isfile(script) || error("Script not found at $script")

    output = readchomp(`$python $script $D`)
    return parse(Float64, output)
end
