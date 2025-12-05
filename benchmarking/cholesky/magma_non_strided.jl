using CUDA
using Magma
using BenchmarkTools
using LinearAlgebra

function magma_spotrf_batched!(
    uplo::Magma.LibMagma.magma_uplo_t,
    n::Integer,
    dA,
    lda::Integer,
    info_array,
    batchCount::Integer,
    queue::Magma.LibMagma.magma_queue_t,
)
    return ccall(
        (:magma_spotrf_batched, Magma.LibMagma.libmagma),
        Cvoid,
        (
            Magma.LibMagma.magma_uplo_t,
            Magma.LibMagma.magma_int_t,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            CuPtr{Magma.LibMagma.magma_int_t},
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_queue_t,
        ),
        uplo,
        n,
        dA,
        lda,
        info_array,
        batchCount,
        queue
    )
end

function magma_chol_non_strided!(
    A,
    D,
    N,
    info_d,
    queue_ptr,
)
    magma_spotrf_batched!(
        Magma.MagmaLower,
        D,
        A,
        D,
        info_d,
        N,
        queue_ptr[],
    )

    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
end


function cholesky_timing(A_cpu, queue_ptr, ::Val{:magma_non_strided})
    D, _, N = size(A_cpu)

    A = cu(A_cpu)

    dA = CUDA.CUBLAS.unsafe_strided_batch(A)
    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    bench_results = @benchmark begin
        magma_chol_non_strided!($dA, $D, $N, $info_d, $queue_ptr)
    end

    return median(bench_results.times) / 1e9 / N
end
