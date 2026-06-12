using StaticArrays
using BenchmarkTools
using LinearAlgebra

function qr_q_cpu_mt!(
    Q_out::Array{T,3}, A_in::Array{T,3},
) where {T}
    D, _, N = size(A_in)

    A_squashed = reshape(A_in, D^2, N)
    A_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, A_squashed)

    Threads.@threads for i in 1:N
        @inbounds begin
            Ai = A_reinterp[i]
            Q_out[:, :, i] = Matrix(qr(Ai).Q)
        end
    end
end

function qr_q_timing(Q_out, A_in, _, _, ::Val{:cpu_mt})
    N = size(A_in, 3)

    bench_results = @benchmark begin
        qr_q_cpu_mt!($Q_out, $A_in)
    end

    return median(bench_results.times) / 1e9 / N
end
