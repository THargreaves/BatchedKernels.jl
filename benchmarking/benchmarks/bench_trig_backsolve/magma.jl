using CUDA
using CUDA: i32
using BenchmarkTools

# `magmablas_strsm_batched` solves op(A) * X = alpha * B  (or X * op(A) = ...)
# in-place: B is overwritten with X. We solve U * C = B  →  C = U^{-1} B,
# so we seed C with B in the @benchmark setup and let trsm do the work.
function magmablas_strsm_batched!(
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
        (:magmablas_strsm_batched, Magma.LibMagma.libmagma),
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
            Magma.LibMagma.magma_queue_t,
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

function backsolve_magma!(
    dU, dC,              # batched pointer arrays
    U_in, C_out,         # underlying storages (function args keep them alive
                         # in the BenchmarkTools-generated sample function's
                         # stack frame, independent of any outer GC.@preserve)
    D, N, queue_ptr,
)
    LE = Magma.LibMagma.MagmaLeft
    UP = Magma.LibMagma.MagmaUpper
    NT = Magma.LibMagma.MagmaNoTrans
    NUNIT = Magma.LibMagma.MagmaNonUnit

    one = 1.0f0

    # Without this, `U_in` is only reachable through `dU`'s raw device pointers
    # — Julia's GC doesn't follow those, so `gcscrub` between benchmark samples
    # would free it and the next strsm corrupts GPU memory. (`C_out`/`Bs` are
    # already pinned by the $-interpolation in `setup=…`, but include C_out
    # here defensively.)
    GC.@preserve U_in C_out begin
        # Solves U * C = 1.0 * C  (with C seeded to B) → C := U^{-1} B
        magmablas_strsm_batched!(
            LE, UP, NT, NUNIT,
            D, D, one,
            dU, D, dC, D,
            N, queue_ptr[],
        )

        Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
        CUDA.synchronize()
    end
end

function backsolve_timing(C_out_cpu, U_in_cpu, B_in_cpu, queue_ptr, _, ::Val{:magma})
    D, _, N = size(C_out_cpu)

    # MAGMA's `magmablas_strsm_batched` writes a larger region per batch than
    # the declared D × D for non-padded leading dims (its kernel block is
    # bigger than D, and the routine doesn't gate writes by D). For non-last
    # batches the overflow lands in adjacent batches' slots (corrupting
    # their result — irrelevant for timing). For the LAST batch the overflow
    # goes past the end of the allocation → unmapped memory → hardware fault.
    # Compute-sanitizer confirms this manifests as a hardware MMU exception.
    #
    # Fix: over-allocate U_in and C_out with `SLACK` extra batch slots so the
    # last-batch overflow lands in our padding instead of unmapped memory.
    # We still pass `batchCount = N` to MAGMA, so only the first N batches
    # are actually computed; the padding exists solely to absorb writes.
    SLACK = 32

    Bs = cu(B_in_cpu)

    U_in = CUDA.zeros(Float32, D, D, N + SLACK)
    copyto!(view(U_in, :, :, 1:N), cu(U_in_cpu))

    C_out = CUDA.zeros(Float32, D, D, N + SLACK)
    copyto!(view(C_out, :, :, 1:N), Bs)

    # `unsafe_strided_batch` produces N+SLACK pointers; MAGMA only reads the
    # first N (batchCount=N). That's fine — the extra pointers are unused.
    dU = CUDA.CUBLAS.unsafe_strided_batch(U_in)
    dC = CUDA.CUBLAS.unsafe_strided_batch(C_out)

    bench_results = @benchmark begin
        backsolve_magma!(
            $dU,
            $dC,
            $U_in,
            $C_out,
            $D,
            $N,
            $queue_ptr,
        )
    end setup=begin
        copyto!(view($C_out, :, :, 1:$N), $Bs)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
