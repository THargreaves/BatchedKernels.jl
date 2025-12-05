using StaticArrays
using BenchmarkTools
using LinearAlgebra

function cholesky_cpu_mt!(A::Array{T,3}) where {T}
    D, _, N = size(A)

    A_squashed = reshape(A, D^2, N)
    A_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, A_squashed)

    Threads.@threads for i in 1:N
        @inbounds A[:, :, i] .= cholesky(Symmetric(A_reinterp[i])).L
    end
end

function cholesky_timing(A, _, ::Val{:cpu_mt})
    N = size(A, 3)

    bench_results = @benchmark begin
        cholesky_cpu_mt!($A)
    end

    return median(bench_results.times) / 1e9 / N
end