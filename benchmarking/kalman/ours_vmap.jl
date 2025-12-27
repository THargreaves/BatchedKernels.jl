using BatchedKernels
using BenchmarkTools
using LinearAlgebra
using CUDA
using CUDA: i32

function kalman_cov(P, A, Q, H, R)
    # Predict step
    P_pred = A * P * A' + Q

    # Kalman gain
    P_pred_H_trans = P_pred * H'
    S = H * P_pred_H_trans + R
    K = P_pred_H_trans / Symmetric(S)

    # Update step
    I_KH = I - K * H

    P_new = I_KH * P_pred

    return P_new
end

function kalman_timing(_, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, _, ::Val{:ours_vmap})
    _, _, N = size(P_in_cpu)
    
    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    kalman_cov_vmap = BatchedKernels.vmap(
        kalman_cov,
        in_type = (:batched, :shared, :shared, :shared, :shared),
    )

    bench_results = @benchmark begin
        CUDA.@sync $kalman_cov_vmap($P_in, $A, $Q, $H, $R)
    end

    return median(bench_results.times) / 1e9 / N
end
