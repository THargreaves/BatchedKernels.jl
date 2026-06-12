using StaticArrays
using BenchmarkTools
using LinearAlgebra

function backsolve_cpu_mt!(
    C_out::Array{T,3}, U_in::Array{T,3}, B_in::Array{T,3},
) where {T}
    D, _, N = size(U_in)

    U_squashed = reshape(U_in, D^2, N)
    U_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, U_squashed)

    B_squashed = reshape(B_in, D^2, N)
    B_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, B_squashed)

    Threads.@threads for i in 1:N
        @inbounds begin
            Ui = U_reinterp[i]
            Bi = B_reinterp[i]

            C_out[:, :, i] = UpperTriangular(Ui) \ Bi
        end
    end
end

function backsolve_timing(C_out, U_in, B_in, _, _, ::Val{:cpu_mt})
    N = size(U_in, 3)

    bench_results = @benchmark begin
        backsolve_cpu_mt!($C_out, $U_in, $B_in)
    end

    return median(bench_results.times) / 1e9 / N
end
