using CUDA
using CUDA: i32
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
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            Cfloat,
            CuPtr{CuPtr{Cfloat}},
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

function matmul_magma!(
    dC, dA, dB,          # batched pointer arrays
    D, N, queue_ptr,
)
    NT = Magma.LibMagma.MagmaNoTrans

    zero = 0.0f0
    one = 1.0f0

    # C = A * B
    magmablas_sgemm_batched!(
        NT, NT, D, D, D,
        one, dA, D, dB, D,
        zero, dC, D, N, queue_ptr[],
    )

    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    CUDA.synchronize()
end

function matmul_timing(C_out_cpu, A_in_cpu, B_in_cpu, queue_ptr, _, ::Val{:magma})
    D, _, N = size(C_out_cpu)

    C_out = cu(C_out_cpu)
    A_in = cu(A_in_cpu)
    B_in = cu(B_in_cpu)

    dC = CUDA.CUBLAS.unsafe_strided_batch(C_out)
    dA = CUDA.CUBLAS.unsafe_strided_batch(A_in)
    dB = CUDA.CUBLAS.unsafe_strided_batch(B_in)

    bench_results = @benchmark begin
        matmul_magma!(
            $dC,
            $dA,
            $dB,
            $D,
            $N,
            $queue_ptr,
        )
    end setup=begin
        fill!($C_out, 0f0)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
