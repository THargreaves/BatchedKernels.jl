using StaticArrays
using BenchmarkTools

function matmul_cpu_mt!(
    C::Array{T,3}, A::Array{T,3}, B::Array{T,3},
) where {T}
    D, _, N = size(A)
    A_squashed = reshape(A, D^2, N)
    B_squashed = reshape(B, D^2, N)
    A_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, A_squashed)
    B_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, B_squashed)
    Threads.@threads for i in 1:size(A, 3)
        @inbounds C[:, :, i] .= A_reinterp[i] * B_reinterp[i]
    end
end

function matmul_timing(C, A, B, _, ::Val{:cpu_mt})
    N = size(C, 3)

    bench_results = @benchmark begin
        matmul_cpu_mt!($C, $A, $B)
    end

    return median(bench_results.times) / 1e9 / N
end