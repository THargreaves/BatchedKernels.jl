using StaticArrays
using BenchmarkTools
using LinearAlgebra

function forwardsolve_cpu_mt!(
    L::Array{T,3}, B::Array{T,3},
) where {T}
    D, _, N = size(L)

    L_squashed = reshape(L, D^2, N)
    B_squashed = reshape(B, D^2, N)

    L_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, L_squashed)
    B_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, B_squashed)

    Threads.@threads for i in 1:N
        @inbounds begin
            B[:, :, i] .= LowerTriangular(L_reinterp[i]) \ B_reinterp[i]
        end
    end
end

function forward_solve_timing(L, B, _, ::Val{:cpu_mt})
    N = size(B, 3)

    bench_results = @benchmark begin
        forwardsolve_cpu_mt!($L, $B)
    end

    return median(bench_results.times) / 1e9 / N
end