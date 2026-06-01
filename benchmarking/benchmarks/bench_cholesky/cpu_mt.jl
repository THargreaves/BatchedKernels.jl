using StaticArrays
using BenchmarkTools
using LinearAlgebra

function cholesky_cpu_mt!(
    U_out::Array{T,3}, A_in::Array{T,3},
) where {T}
    D, _, N = size(A_in)

    A_squashed = reshape(A_in, D^2, N)
    A_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, A_squashed)

    Threads.@threads for i in 1:N
        @inbounds begin
            Ai = A_reinterp[i]
            U_out[:, :, i] = cholesky(Ai).U
        end
    end
end

function cholesky_timing(U_out, A_in, _, _, ::Val{:cpu_mt})
    N = size(A_in, 3)

    bench_results = @benchmark begin
        cholesky_cpu_mt!($U_out, $A_in)
    end

    return median(bench_results.times) / 1e9 / N
end
