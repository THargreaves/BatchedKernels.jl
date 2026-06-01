using CUDA
using CUDA: i32
using BenchmarkTools

# `magma_spotrf_batched` is in-place: it overwrites the input matrix with the
# Cholesky factor. We copy A → U in the @benchmark setup so every iteration
# starts fresh with U holding a copy of A, then factor in-place.
#
# uplo = MagmaUpper gives A = U' * U  (upper triangle of U holds the factor;
# lower triangle is untouched, matching the convention used by `ours.jl`).
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
        queue,
    )
end

function cholesky_magma!(dU, info_d, D, N, queue_ptr)
    LO = Magma.LibMagma.MagmaLower
    magma_spotrf_batched!(LO, D, dU, D, info_d, N, queue_ptr[])
    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    CUDA.synchronize()
end

function cholesky_timing(U_out_cpu, A_in_cpu, queue_ptr, _, ::Val{:magma})
    D, _, N = size(U_out_cpu)

    U_out = cu(U_out_cpu)
    As = cu(A_in_cpu)
    dU = CUDA.CUBLAS.unsafe_strided_batch(U_out)

    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    bench_results = @benchmark begin
        cholesky_magma!(
            $dU,
            $info_d,
            $D,
            $N,
            $queue_ptr,
        )
    end setup=begin
        copyto!($U_out, $As)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
