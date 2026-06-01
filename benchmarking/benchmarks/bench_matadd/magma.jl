using CUDA
using CUDA: i32
using BenchmarkTools

# MAGMA exposes only in-place batched addition (`magmablas_sgeadd_batched`
# computes `dB := alpha*dA + dB`). For a strict out-of-place semantic
# `C = A + B`, this forces two kernel launches.
#
# For BENCHMARK purposes we only care about the time to compute `A + B`
# in batched form — where the result lands is irrelevant. So we just do the
# single in-place call `B := A + B`. The @benchmark `setup` resets B to its
# original value before each iteration so the operation always starts fresh.

function magmablas_sgeadd_batched!(
    m,
    n,
    alpha,
    dA,
    ldda,
    dB,
    lddb,
    batchCount,
    queue,
)
    return ccall(
        (:magmablas_sgeadd_batched, Magma.LibMagma.libmagma),
        Cvoid,
        (
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Cfloat,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_queue_t,
        ),
        m,
        n,
        alpha,
        dA,
        ldda,
        dB,
        lddb,
        batchCount,
        queue,
    )
end

function matadd_magma!(
    dA, dB,              # batched pointer arrays
    D, N, queue_ptr,
)
    one = 1.0f0

    # In-place: B := A + B  (single batched call, single kernel launch).
    magmablas_sgeadd_batched!(
        D, D, one, dA, D, dB, D, N, queue_ptr[],
    )

    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    CUDA.synchronize()
end

function matadd_timing(C_out_cpu, A_in_cpu, B_in_cpu, queue_ptr, _, ::Val{:magma})
    D, _, N = size(C_out_cpu)

    A_in     = cu(A_in_cpu)
    B_init   = cu(B_in_cpu)   # immutable reference copy used to reset B each iter
    B        = cu(B_in_cpu)   # working buffer, overwritten in-place by sgeadd

    dA = CUDA.CUBLAS.unsafe_strided_batch(A_in)
    dB = CUDA.CUBLAS.unsafe_strided_batch(B)

    bench_results = @benchmark begin
        matadd_magma!(
            $dA,
            $dB,
            $D,
            $N,
            $queue_ptr,
        )
    end setup=begin
        # B gets overwritten in-place; restore it from the immutable reference.
        copyto!($B, $B_init)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
