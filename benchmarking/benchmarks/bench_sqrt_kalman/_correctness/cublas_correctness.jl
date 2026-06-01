using CUDA
using CUDA: i32
using LinearAlgebra
using Random

include("../cublas.jl")

function cpu_kalman_cov(P, A, Q, H, R)
    P_pred = A * P * A' + Q
    S = H * P_pred * H' + R
    K = P_pred * H' / S
    return P_pred - K * S * K'
end

function test_sqrt_kalman_correctness(; D=4, N=513, THRESH=10, T=Float32, atol=1e-3, rtol=1e-3)
    Random.seed!(1234)

    A_cpu = rand(T, D, D) / T(D)

    Q_elem = rand(T, D, D) / T(D)^2
    Q_cpu = Q_elem * Q_elem' + 0.01f0 * I

    H_cpu = rand(T, D, D) / T(D)

    R_elem = rand(T, D, D) / T(D)^2
    R_cpu = R_elem * R_elem' + 0.01f0 * I

    S_in_cpu = Array{T}(undef, D, D, N)
    P_in_cpu = Array{T}(undef, D, D, N)
    for i in 1:N
        P_i = rand(T, D, D) / T(D)
        P_i = P_i * P_i' + 0.1f0 * I
        P_in_cpu[:, :, i] = P_i
        S_in_cpu[:, :, i] = T.(Matrix(cholesky(P_i).L))
    end

    Ss_out_cpu = zeros(T, D, D, N)

    twoD = 2 * D
    S_Q_cpu = T.(Matrix(cholesky(Q_cpu).L))
    S_R_cpu = T.(Matrix(cholesky(R_cpu).L))

    Ss_in_d = cu(S_in_cpu)
    A_d     = cu(A_cpu)
    S_Q_d   = cu(S_Q_cpu)
    H_d     = cu(H_cpu)
    S_R_d   = cu(S_R_cpu)

    X       = CUDA.zeros(T, D,    D,    N)
    M_pred  = CUDA.zeros(T, twoD, D,    N)
    Y       = CUDA.zeros(T, D,    D,    N)
    M_upd   = CUDA.zeros(T, twoD, twoD, N)
    Ss_out_d = CUDA.zeros(T, D, D, N)

    tau_pred = CUDA.zeros(T, D,    N)
    tau_upd  = CUDA.zeros(T, twoD, N)

    dM_pred_ptrs  = _cublas_batch_ptrs(M_pred,  N, twoD * D)
    dM_upd_ptrs   = _cublas_batch_ptrs(M_upd,   N, twoD * twoD)
    dtau_pred_ptrs = _cublas_batch_ptrs(tau_pred, N, D)
    dtau_upd_ptrs  = _cublas_batch_ptrs(tau_upd,  N, twoD)

    info_pred = Ref{Cint}(0)
    info_upd  = Ref{Cint}(0)

    sqrt_kalman_cublas!(
        dM_pred_ptrs, dM_upd_ptrs,
        dtau_pred_ptrs, dtau_upd_ptrs,
        Ss_in_d, X, M_pred, Y, M_upd, Ss_out_d,
        tau_pred, tau_upd,
        A_d, H_d, S_Q_d, S_R_d,
        info_pred, info_upd,
        D, N,
    )
    CUDA.synchronize()
    S_out_cublas = Array(Ss_out_d)

    max_err = 0.0f0
    n_bad = 0
    for i in 1:N
        P_ref     = cpu_kalman_cov(P_in_cpu[:, :, i], A_cpu, Q_cpu, H_cpu, R_cpu)
        S_i       = S_out_cublas[:, :, i]
        P_cublas  = S_i * S_i'
        err = maximum(abs.(P_ref .- P_cublas))
        max_err = max(max_err, err)
        if !isapprox(P_ref, P_cublas; atol, rtol)
            n_bad += 1
            if n_bad <= 3
                println("Mismatch at batch $i:")
                println("  P_ref:    ", P_ref)
                println("  P_cublas: ", P_cublas)
                println("  max elem diff: $err")
            end
        end
    end

    println("\nD=$D, N=$N")
    println("Max element-wise error (P): $max_err")
    println("Mismatched batches: $n_bad / $N (atol=$atol, rtol=$rtol)")
    if n_bad == 0
        println("✓ PASS")
    else
        println("✗ FAIL")
    end

    return n_bad == 0
end

for D in 2:16
    test_sqrt_kalman_correctness(D=D, N=256)
    println()
end
