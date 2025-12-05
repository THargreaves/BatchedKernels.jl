using CUDA
using Magma
using LinearAlgebra
using BenchmarkTools

function magma_lower_solve_wrapper!(
    side::Magma.LibMagma.magma_side_t,
    uplo::Magma.LibMagma.magma_uplo_t,
    transA::Magma.LibMagma.magma_trans_t,
    diag::Magma.LibMagma.magma_diag_t,
    m::Integer,
    n::Integer,
    alpha::Cfloat,
    dA_array,
    ldda::Integer,
    dB_array,
    lddb::Integer,
    batchCount::Integer,
    queue,
)
    ccall(
        (:magmablas_strsm_batched,Magma.LibMagma.libmagma),
        Cvoid,
        (
            Magma.LibMagma.magma_side_t,
            Magma.LibMagma.magma_uplo_t,
            Magma.LibMagma.magma_trans_t,
            Magma.LibMagma.magma_diag_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Cfloat,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_queue_t
        ),
        side,
        uplo,
        transA,
        diag,
        m,
        n,
        alpha,
        dA_array,
        ldda,
        dB_array,
        lddb,
        batchCount,
        queue,
    )
    return nothing
end

function magma_non_strided!(
    L,
    B,
    D,
    N,
    queue_ptr,
)

    side  = Magma.LibMagma.MagmaLeft
    uplo  = Magma.LibMagma.MagmaLower
    trans = Magma.LibMagma.MagmaNoTrans
    diag  = Magma.LibMagma.MagmaNonUnit

    m = D
    n = D
    alpha = 1.0f0

    magma_lower_solve_wrapper!(
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha,
        L,
        D,
        B,
        D,
        N,
        queue_ptr[],
    )
    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
end

function forward_solve_timing(L_cpu, B_cpu, queue_ptr, ::Val{:magma_non_strided})
    D, _, N = size(B_cpu)

    L = cu(L_cpu)
    B = cu(B_cpu)

    dL = CUDA.CUBLAS.unsafe_strided_batch(L)
    dB = CUDA.CUBLAS.unsafe_strided_batch(B)

    bench_results = @benchmark begin
        magma_non_strided!($dL, $dB, $D, $N, $queue_ptr)
    end

    return median(bench_results.times) / 1e9 / N
end
