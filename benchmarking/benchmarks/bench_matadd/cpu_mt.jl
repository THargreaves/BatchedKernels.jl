using StaticArrays
using BenchmarkTools
using LinearAlgebra

function matadd_cpu_mt!(
    C_out::Array{T,3}, A_in::Array{T,3}, B_in::Array{T,3},
) where {T}
    D, _, N = size(A_in)

    A_squashed = reshape(A_in, D^2, N)
    A_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, A_squashed)

    B_squashed = reshape(B_in, D^2, N)
    B_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, B_squashed)

    Threads.@threads for i in 1:N
        @inbounds begin
            Ai = A_reinterp[i]
            Bi = B_reinterp[i]

            C_out[:, :, i] = Ai + Bi
        end
    end
end

function matadd_timing(C_out, A_in, B_in, _, _, ::Val{:cpu_mt})
    N = size(A_in, 3)

    bench_results = @benchmark begin
        matadd_cpu_mt!($C_out, $A_in, $B_in)
    end

    return median(bench_results.times) / 1e9 / N
end
