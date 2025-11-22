using CUDA
using Magma
using BenchmarkTools

function magmablas_sgemm_batched!(
    transA,
    transB,
    m,
    n,
    k,
    alpha,
    dA,
    ldda,
    dB,
    lddb,
    beta,
    dC,
    lddc,
    batchCount,
    queue,
)
    return ccall(
        (:magmablas_sgemm_batched, Magma.LibMagma.libmagma),
        Cvoid,
        (
            Magma.LibMagma.magma_trans_t,
            Magma.LibMagma.magma_trans_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Cfloat,
            CuPtr{Cfloat},
            Magma.LibMagma.magma_int_t,
            CuPtr{Cfloat},
            Magma.LibMagma.magma_int_t,
            Cfloat,
            CuPtr{Cfloat},
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_queue_t,
        ),
        transA,
        transB,
        m,
        n,
        k,
        alpha,
        dA,
        ldda,
        dB,
        lddb,
        beta,
        dC,
        lddc,
        batchCount,
        queue,
    )
end

function magma_matmul_non_strided!(
    C, A, B, D, N, queue_ptr,
)
    magmablas_sgemm_batched!(
        Magma.MagmaNoTrans,
        Magma.MagmaNoTrans,
        D,
        D,
        D,
        1.0f0,
        A,
        D,
        B,
        D,
        0.0f0,
        C,
        D,
        N,
        queue_ptr[],
    )
    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
end

function matmul_timing(C_cpu, A_cpu, B_cpu, queue_ptr, ::Val{:magma_non_strided})
    D, _, N = size(C_cpu)
    
    A = cu(A_cpu)
    B = cu(B_cpu)
    C = cu(C_cpu)

    dA = CUDA.CUBLAS.unsafe_strided_batch(A)
    dB = CUDA.CUBLAS.unsafe_strided_batch(B)
    dC = CUDA.CUBLAS.unsafe_strided_batch(C)

    bench_results = @benchmark begin
        magma_matmul_non_strided!($dC, $dA, $dB, $D, $N, $queue_ptr)
    end

    return median(bench_results.times) / 1e9 / N
end


# T = Float32
# D = 2
# N = Int(ceil(1e9 / (4 * 2 * D^2)))*5
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

# A_cpu = rand(T, D, D, N)
# B_cpu = rand(T, D, D, N)
# C_cpu = zeros(T, D, D, N)

# A = cu(A_cpu)
# B = cu(B_cpu)
# C = cu(C_cpu)

# dA = CUDA.CUBLAS.unsafe_strided_batch(A)
# dB = CUDA.CUBLAS.unsafe_strided_batch(B)
# dC = CUDA.CUBLAS.unsafe_strided_batch(C)

# # time = matmul_timing(C, A, B, queue_ptr, Val(:magma_non_strided))
# # println(time)

# magma_matmul_non_strided!(dC, dA, dB, queue_ptr)
# println(Array(C[:, :, N]))