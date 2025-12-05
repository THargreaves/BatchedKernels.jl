using StaticArrays
using BenchmarkTools
using LinearAlgebra

function kalman_cpu_mt!(
    P_out::Array{T,3}, P_in::Array{T,3}, A::Array{T,2}, Q::Array{T,2}, H::Array{T,2}, R::Array{T,2},
) where {T}
    D, _, N = size(P_in)

    P_in_squashed = reshape(P_in, D^2, N)
    P_in_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, P_in_squashed)

    A_r = SMatrix{D,D,T,D^2}(A)
    Q_r = SMatrix{D,D,T,D^2}(Q)
    H_r = SMatrix{D,D,T,D^2}(H)
    R_r = SMatrix{D,D,T,D^2}(R)

    Threads.@threads for i in 1:N
        @inbounds begin
            Pi = P_in_reinterp[i]

            P_pred = A_r * Pi * A_r' + Q_r
            S = H_r * P_pred * H_r' + R_r
            K = P_pred * H_r' / S
            P_out[:, :, i] = P_pred - K * S * K'
        end
    end
end

function kalman_timing(P_out, P_in, A, Q, H, R, _, ::Val{:cpu_mt})
    N = size(P_in, 3)

    bench_results = @benchmark begin
        kalman_cpu_mt!($P_out, $P_in, $A, $Q, $H, $R)
    end

    return median(bench_results.times) / 1e9 / N
end