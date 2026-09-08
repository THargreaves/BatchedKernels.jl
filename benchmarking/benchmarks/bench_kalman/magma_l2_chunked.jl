include(joinpath(@__DIR__, "magma_non_strided.jl"))

# Target L2 cache budget (bytes) we want a single chunk's hot data to fit within.
const L2_BYTES = 72 * 2^20
# The batched GEMM/TRSM/POTRF kernels also use L2 for their own tiling, so we
# only let the persistent per-batch data claim a fraction of it and leave headroom.
const L2_TARGET_FRACTION = 0.66
# Per batch element, P_in, P_out and the scratch B stay hot across the ~10
# successive MAGMA calls of one Kalman step (A/Q/H/R alias a single matrix each,
# so they are trivially resident and don't scale with the chunk).
const N_RESIDENT_MATRICES = 3

# Largest chunk whose resident working set still fits the L2 budget. Scales as
# 1/D^2, which keeps every chunk near a fixed fraction of L2 across the D sweep.
function l2_chunk_size(D, T)
    bytes_per_elem = N_RESIDENT_MATRICES * D^2 * sizeof(T)
    return max(1, floor(Int, L2_TARGET_FRACTION * L2_BYTES / bytes_per_elem))
end

function magma_l2_chunked_kalman!(
    dPo_chunks, dPi_chunks, dA, dQ, dH, dR, dB, info_d, D, chunk_sizes, queue_ptr
)
    for (dPo_c, dPi_c, n_c) in zip(dPo_chunks, dPi_chunks, chunk_sizes)
        magma_non_strided_kalman!(
            dPo_c,
            dPi_c,
            view(dA, 1:n_c),
            view(dQ, 1:n_c),
            view(dH, 1:n_c),
            view(dR, 1:n_c),
            view(dB, 1:n_c),
            view(info_d, 1:n_c),
            D,
            n_c,
            queue_ptr,
        )
    end
end

function kalman_timing(
    P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, queue_ptr, ::Val{:magma_l2_chunked}
)
    D, _, N = size(P_in_cpu)
    T = eltype(P_in_cpu)

    P_out = cu(P_out_cpu)
    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    chunk = min(N, l2_chunk_size(D, T))

    # Per-chunk pointer arrays slice the full P_in/P_out, so each chunk streams a
    # distinct D×D×chunk slab. Built once here, outside the timed region.
    offsets = 0:chunk:(N - 1)
    chunk_sizes = [min(chunk, N - off) for off in offsets]
    dPo_chunks = [
        CUDA.CUBLAS.unsafe_strided_batch(view(P_out, :, :, (off + 1):(off + n))) for
        (off, n) in zip(offsets, chunk_sizes)
    ]
    dPi_chunks = [
        CUDA.CUBLAS.unsafe_strided_batch(view(P_in, :, :, (off + 1):(off + n))) for
        (off, n) in zip(offsets, chunk_sizes)
    ]

    # Scratch + aliased fixed matrices, sized to the (max) chunk and reused.
    B = CUDA.zeros(T, D, D, chunk)
    dB = CUDA.CUBLAS.unsafe_strided_batch(B)
    dA = unsafe_strided_batch_repeat(A, chunk)
    dQ = unsafe_strided_batch_repeat(Q, chunk)
    dH = unsafe_strided_batch_repeat(H, chunk)
    dR = unsafe_strided_batch_repeat(R, chunk)
    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, chunk)

    bench_results = @benchmark begin
        magma_l2_chunked_kalman!(
            $dPo_chunks,
            $dPi_chunks,
            $dA,
            $dQ,
            $dH,
            $dR,
            $dB,
            $info_d,
            $D,
            $chunk_sizes,
            $queue_ptr,
        )
    end

    return median(bench_results.times) / 1e9 / N
end
