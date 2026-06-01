using CUDA
using CUDA: i32
using BenchmarkTools

# `magma_sgeqrf_batched_smallsq` is MAGMA's small-square-batched QR — tuned for
# D in roughly 2..32, which is exactly our regime. It's a single dedicated CUDA
# kernel rather than the panel/block algorithm of the general
# `magma_sgeqrf_batched`, whose overhead dominates at these sizes.
#
# Output format is identical to the general routine (LAPACK convention):
#   - upper triangle of A is overwritten with R
#   - lower triangle of A holds the Householder vectors (we don't need them)
#   - dtau_array[i] receives the τ scalars for batch i  (length = n)
#
# Note: signature drops `m` since the routine is square-only.
#
# We seed R with A in the @benchmark setup so each iteration starts fresh,
# then run geqrf in-place on R. The upper triangle of R then holds the
# requested factor; the lower triangle is ignored.
function magma_sgeqrf_batched_smallsq!(
    n::Integer,
    dA_array,
    Ai::Integer,
    Aj::Integer,
    ldda::Integer,
    dtau_array,
    taui::Integer,
    info_array,
    batchCount::Integer,
    queue::Magma.LibMagma.magma_queue_t,
)
    return ccall(
        (:magma_sgeqrf_batched_smallsq, Magma.LibMagma.libmagma),
        Magma.LibMagma.magma_int_t,
        (
            Magma.LibMagma.magma_int_t,           # n
            CuPtr{CuPtr{Cfloat}},                 # dA_array
            Magma.LibMagma.magma_int_t,           # Ai (row offset into each dA[k])
            Magma.LibMagma.magma_int_t,           # Aj (col offset into each dA[k])
            Magma.LibMagma.magma_int_t,           # ldda
            CuPtr{CuPtr{Cfloat}},                 # dtau_array
            Magma.LibMagma.magma_int_t,           # taui (offset into each dtau[k])
            CuPtr{Magma.LibMagma.magma_int_t},    # info_array
            Magma.LibMagma.magma_int_t,           # batchCount
            Magma.LibMagma.magma_queue_t,         # queue
        ),
        n,
        dA_array,
        Ai,
        Aj,
        ldda,
        dtau_array,
        taui,
        info_array,
        batchCount,
        queue,
    )
end

function qr_r_magma!(dR, dtau, tau_storage, info_d, D, N, queue_ptr)
    # `tau_storage` is passed as a real argument (rather than just lived in the
    # caller's scope) so the BenchmarkTools-generated sample function keeps it
    # alive in its stack frame regardless of Julia 1.12's stricter world-age
    # semantics around `invokelatest` and `GC.@preserve`. The inner @preserve
    # then makes the lifetime explicit during the actual ccall.
    GC.@preserve tau_storage begin
        # Ai = Aj = taui = 0: factor the whole matrix (no sub-block offset).
        # These offsets exist so MAGMA can reuse smallsq as a panel kernel inside
        # blocked routines; for a standalone batched QR they're always zero.
        magma_sgeqrf_batched_smallsq!(
            D, dR, 0, 0, D, dtau, 0, info_d, N, queue_ptr[],
        )
        Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
        CUDA.synchronize()
    end
end

function qr_r_timing(Rs_cpu, As_cpu, queue_ptr, _, ::Val{:magma})
    D, _, N = size(Rs_cpu)

    Rs = cu(Rs_cpu)
    As = cu(As_cpu)
    copyto!(Rs, As)
    dR = CUDA.CUBLAS.unsafe_strided_batch(Rs)

    # τ buffer: D-vector per batch  →  D × N flat storage, pointer-array view
    tau_storage = CUDA.zeros(Float32, D, N)
    dtau = CUDA.CUBLAS.unsafe_strided_batch(tau_storage)

    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    # `tau_storage` is passed via $-interpolation as a real argument to
    # `qr_r_magma!`, which keeps it alive in the BenchmarkTools-generated
    # sample function's stack frame. `dtau` holds raw device pointers into
    # it, and Julia's GC doesn't follow raw pointers — without this, the
    # inter-iteration `gcscrub` would free tau_storage and corrupt the next
    # geqrf call. (An outer `GC.@preserve` doesn't suffice in Julia 1.12
    # because the sample function is invoked via `invokelatest` in a
    # different world, where the preserve's syntactic scope doesn't reach.)
    bench_results = @benchmark begin
        qr_r_magma!(
            $dR,
            $dtau,
            $tau_storage,
            $info_d,
            $D,
            $N,
            $queue_ptr,
        )
    end setup=begin
        copyto!($Rs, $As)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
