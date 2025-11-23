using PyCall

@show pyimport("sys")."path"

jax_mod = pyimport("jax_vmap")

function kalman_timing(_, P_in_cpu, _, _, _, _, _, ::Val{:jax_vmap})
    D = size(P_in_cpu, 1)

    return Float64(jax_mod.kalman_timing(D; reps=20))
end