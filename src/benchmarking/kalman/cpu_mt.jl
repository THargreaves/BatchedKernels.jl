using StaticArrays
using BenchmarkTools
using LinearAlgebra

function kalman_cpu_mt!(
    P_out::Array{T,3}, P_in::Array{T,3}, A::Array{T,3}, Q::Array{T,3}, H::Array{T,3}, R::Array{T,3},
) where {T}
    D, _, N = size(A)
    I_D = SMatrix{D,D,T}(I)

    P_in_squashed = reshape(P_in, D^2, N)
    A_squashed = reshape(A, D^2, N)
    Q_squashed = reshape(Q, D^2, N)
    H_squashed = reshape(H, D^2, N)
    R_squashed = reshape(R, D^2, N)

    P_in_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, P_in_squashed)
    A_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, A_squashed)
    Q_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, Q_squashed)
    H_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, H_squashed)
    R_reinterp = reinterpret(reshape, SMatrix{D,D,T,D^2}, R_squashed)

    Threads.@threads for i in 1:size(A, 3)
        @inbounds begin
            Pi = P_in_reinterp[i]
            Ai  = A_reinterp[i]
            Qi  = Q_reinterp[i]
            Hi  = H_reinterp[i]
            Ri  = R_reinterp[i]

            P_pred = Ai * Pi * Ai' + Qi
            S = Hi * P_pred * Hi' + Ri
            K = P_pred * Hi' / S
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