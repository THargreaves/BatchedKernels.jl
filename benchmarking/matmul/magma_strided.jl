using CUDA
using Magma
using BenchmarkTools

function magmablas_sgemm_batched_strided(
    transA,
    transB,
    m,
    n,
    k,
    alpha,
    dA,
    ldda,
    strideA,
    dB,
    lddb,
    strideB,
    beta,
    dC,
    lddc,
    strideC,
    batchCount,
    queue,
)
    return ccall(
        (:magmablas_sgemm_batched_strided, Magma.LibMagma.libmagma),
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
            Magma.LibMagma.magma_int_t,
            CuPtr{Cfloat},
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Cfloat,
            CuPtr{Cfloat},
            Magma.LibMagma.magma_int_t,
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
        strideA,
        dB,
        lddb,
        strideB,
        beta,
        dC,
        lddc,
        strideC,
        batchCount,
        queue,
    )
end

function magma_strided!(
    C::CuArray{T,3}, A::CuArray{T,3}, B::CuArray{T,3}, D, N, queue_ptr
) where {T}
    magmablas_sgemm_batched_strided(
        Magma.MagmaNoTrans,
        Magma.MagmaNoTrans,
        D,
        D,
        D,
        1.0f0,
        A,
        D,
        D * D,
        B,
        D,
        D * D,
        0.0f0,
        C,
        D,
        D * D,
        N,
        queue_ptr[],
    )
    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
end

function matmul_timing(C_cpu, A_cpu, B_cpu, queue_ptr, ::Val{:magma_strided})
    D, _, N = size(C_cpu)
    
    A = cu(A_cpu)
    B = cu(B_cpu)
    C = cu(C_cpu)

    bench_results = @benchmark begin
        magma_strided!($C, $A, $B, $D, $N, $queue_ptr)
    end

    return median(bench_results.times) / 1e9 / N
end