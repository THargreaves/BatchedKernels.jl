using StaticArrays
using BenchmarkTools
using LinearAlgebra

include(joinpath("..", "forward_solve", "magma_non_strided.jl"))
include(joinpath("..", "cholesky", "magma_non_strided.jl"))
include(joinpath("..", "matmul", "magma_non_strided.jl"))

function magma_non_strided_kalman!(
    dPo, dPi, dA, dQ, dH, dR, dB, info_d, D, N, queue_ptr,
)# where {T}
    L = Magma.LibMagma.MagmaLeft
    R = Magma.LibMagma.MagmaRight
    LO = Magma.LibMagma.MagmaLower
    NT = Magma.LibMagma.MagmaNoTrans
    TR = Magma.LibMagma.MagmaTrans
    NUNIT = Magma.LibMagma.MagmaNonUnit

    zero = 0.0f0
    one = 1.0f0

    # Temp = A * Pin, store in dB
    magmablas_sgemm_batched!(
        NT, NT,
        D, D, D,
        one, dA, D,
        dPi, D,
        zero, dB, D,
        N,
        queue_ptr[],
    )

    # P_pred = temp * A' => Q <- dB * A' + Q
    magmablas_sgemm_batched!(
        NT, TR,
        D, D, D,
        one, dB, D,  # temp = A * Pin
        dA,  D,  # A'
        one, dQ, D,   # P_pred
        N,
        queue_ptr[],
    )
    # Now dQ contains P_pred

    # dB <- H * P_pred
    magmablas_sgemm_batched!(
        NT, NT,
        D, D, D,
        one, dH, D,
        dQ, D,  # P_pred
        zero, dB, D,
        N,
        queue_ptr[],
    )

    # dR <- H * P_pred * H' + R
    magmablas_sgemm_batched!(
        NT, TR,
        D, D, D,
        one, dB, D,  # H * P_pred
        dH, D,
        one, dR, D,
        N,
        queue_ptr[],
    )
    # Now dR contains S

    # In-place cholesky for S = dR = H * P_pred * H' + R
    magma_spotrf_batched!(
        LO,
        D,
        dR, D,
        info_d,
        N,
        queue_ptr[],
    )
    # dR now contains L, where S = LL^T

    # Temp: dPi <- P_pred * H'
    magmablas_sgemm_batched!(
        NT, TR,
        D, D, D,
        one, dQ, D,  # P_pred
        dH, D,
        zero, dPi, D,
        N,
        queue_ptr[],
    )

    # Temp: dPi <- (P_pred * H') \ S (S=dR)
    magma_lower_solve_wrapper!(
        R, LO, TR, NUNIT,
        D, D,
        one,
        dR, D,  # L
        dPi, D,  # P_pred * H'
        N,
        queue_ptr[],
    )

    # Solve K = <- (P_pred * H') \ S (S=dR), stored in dPi
    magma_lower_solve_wrapper!(
        R, LO, NT, NUNIT,
        D, D,
        one, dR, D,  # L
        dPi, D,
        N,
        queue_ptr[],
    )
    # Now dPi contains K

    # dB <- K * L
    magmablas_sgemm_batched!(
        NT, NT,
        D, D, D,
        one, dPi, D,  # K
        dR, D,  # S
        zero, dB, D,
        N,
        queue_ptr[],
    )

    # dPo <- (K*L) * (K*L)^T = K LL^T K^T = K * S * K'
    magmablas_sgemm_batched!(
        NT, TR,
        D, D, D,
        one, dB, D,
        dB, D,
        zero, dPo, D,
        N,
        queue_ptr[],
    )

    # Answer is P_pred - P_new, i.e dQ - dPo
    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
end

function kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, queue_ptr, ::Val{:magma_non_strided})
    D, _, N = size(P_in_cpu)

    P_out = cu(P_out_cpu)
    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)
    B = CUDA.zeros(D, D, N)

    dPo = CUDA.CUBLAS.unsafe_strided_batch(P_out)
    dPi = CUDA.CUBLAS.unsafe_strided_batch(P_in)
    dA = CUDA.CUBLAS.unsafe_strided_batch(A)
    dQ = CUDA.CUBLAS.unsafe_strided_batch(Q)
    dH = CUDA.CUBLAS.unsafe_strided_batch(H)
    dR = CUDA.CUBLAS.unsafe_strided_batch(R)
    dB = CUDA.CUBLAS.unsafe_strided_batch(B)

    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    bench_results = @benchmark begin
        magma_non_strided_kalman!($dPo, $dPi, $dA, $dQ, $dH, $dR, $dB, $info_d, $D, $N, $queue_ptr)
    end

    return median(bench_results.times) / 1e9 / N
end



# Magma.LibMagma.magma_init()
# queue_ptr = Ref{Magma.LibMagma.magma_queue_t}()
# device = 0
# Magma.LibMagma.magma_queue_create_internal(
#     device,
#     queue_ptr,
#     C_NULL,  # func
#     C_NULL,  # file
#     0,       # line
# )

# T = Float32
# D = 3
# # N = 10
# N = Int(ceil(1e9 / (4 * 2 * D^2)))

# P_out_cpu = zeros(T, D, D, N)
# P_in_cpu = zeros(T, D, D, N)
# for i in 1:N
#     P_i = rand(T, D, D) / T(D)
#     P_i = P_i * P_i' + 0.1f0 * I
#     P_in_cpu[:, :, i] = P_i
# end

# A_cpu = rand(T, D, D, N)

# Q_cpu = zeros(T, D, D, N)
# for i in 1:N
#     Q_i = rand(T, D, D) / T(D)
#     Q_i = Q_i * Q_i' + 0.1f0 * I
#     Q_cpu[:, :, i] = Q_i
# end

# H_cpu = rand(T, D, D, N)

# R_cpu = zeros(T, D, D, N)
# for i in 1:N
#     R_i = rand(T, D, D) / T(D)
#     R_i = R_i * R_i' + 0.1f0 * I
#     R_cpu[:, :, i] = R_i
# end

# println(kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, queue_ptr, Val(:magma_non_strided)))


# P_out = cu(P_out_cpu)
# P_in = cu(P_in_cpu)
# A = cu(A_cpu)
# Q = cu(Q_cpu)
# H = cu(H_cpu)
# R = cu(R_cpu)
# B = CUDA.zeros(D, D, N)

# dPo = CUDA.CUBLAS.unsafe_strided_batch(P_out)
# dPi = CUDA.CUBLAS.unsafe_strided_batch(P_in)
# dA = CUDA.CUBLAS.unsafe_strided_batch(A)
# dQ = CUDA.CUBLAS.unsafe_strided_batch(Q)
# dH = CUDA.CUBLAS.unsafe_strided_batch(H)
# dR = CUDA.CUBLAS.unsafe_strided_batch(R)
# dB = CUDA.CUBLAS.unsafe_strided_batch(B)

# info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

# magma_non_strided_kalman!(dPo, dPi, dA, dQ, dH, dR, dB, info_d, D, N, queue_ptr)

# println(Array(P_out[:, :, 1]))