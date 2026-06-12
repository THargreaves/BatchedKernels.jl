# cublas.jl
#
# cuBLAS batched QR (`cublasSgeqrfBatched`).
# Note: cuSOLVER does NOT have a batched QR — only the non-batched
# `cusolverDnSgeqrf`. cuBLAS does, as a Level-3 extension routine.
# Drop-in for the benchmark harness — same `qr_r_timing(...)` API.
#
# Implementation notes:
#   - `cublasSgeqrfBatched` performs Householder QR in-place:
#       * upper triangle of A is overwritten with R
#       * lower triangle of A holds the Householder vectors (we don't need them)
#       * Tau[i] receives the τ scalars for batch i (length min(m,n) = D)
#   - We seed R with A in the @benchmark setup, then run geqrf in-place.
#   - Q is never materialised.
#   - `info` is a single host int (scalar error code for the whole call,
#     not per-batch as the API parameter name `infoArray` might suggest).

using CUDA
using CUDA: i32
using BenchmarkTools
using Statistics

# ─── Device pointer-array builder ───────────────────────────────────

# ptrs[i] → start of batch slice i within a contiguous array stored in column-
# major order. For an M×N matrix slice, pass `slice_elems` = M*N. For a length-K
# vector slice, pass `slice_elems` = K.
function _fill_batch_ptrs_kernel!(ptrs, A, slice_elems)
    tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    gstride = gridDim().x * blockDim().x
    n = length(ptrs)
    i = tid
    while i <= n
        idx = (Int64(i) - 1) * Int64(slice_elems) + 1
        @inbounds ptrs[i] = reinterpret(Ptr{Float32}, pointer(A, idx))
        i += gstride
    end
    return
end

function _batch_ptrs(A::DenseCuArray{Float32}, N::Integer, slice_elems::Integer)
    ptrs = CuArray{Ptr{Float32}}(undef, N)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _fill_batch_ptrs_kernel!(ptrs, A, Int32(slice_elems))
    return ptrs
end

# ─── Main QR (R only) ───────────────────────────────────────────────

function qr_r_cublas!(
    dR, dtau,        # batched pointer arrays
    info_h, D, N,
)
    h = CUDA.CUBLAS.handle()           # cuBLAS handle, bound to current stream

    CUDA.CUBLAS.cublasSgeqrfBatched(h, D, D, dR, D, dtau, info_h, N)

    CUDA.synchronize()
    return nothing
end

function qr_r_timing(Rs_cpu, As_cpu, _, _, ::Val{:cublas})
    D, _, N = size(Rs_cpu)
    DD = D * D

    Rs = cu(Rs_cpu)
    As = cu(As_cpu)
    copyto!(Rs, As)

    dR = _batch_ptrs(Rs, N, DD)

    # τ buffer: D-vector per batch  →  D × N flat storage, pointer-array view
    tau_storage = CUDA.zeros(Float32, D, N)
    dtau = _batch_ptrs(tau_storage, N, D)

    # cuBLAS geqrf info: single host int (scalar error code for the whole call)
    info_h = Ref{Cint}(0)

    # GC.@preserve keeps `tau_storage` alive: `dtau` holds raw device pointers
    # into it, but Julia's GC doesn't follow raw pointers. Without this,
    # BenchmarkTools' inter-iteration `gcscrub` can free tau_storage, leaving
    # `dtau` dangling and the next geqrf call corrupts GPU memory.
    bench_results = GC.@preserve tau_storage begin
        @benchmark begin
            qr_r_cublas!(
                $dR,
                $dtau,
                $info_h,
                $D,
                $N,
            )
        end setup=begin
            copyto!($Rs, $As)
        end evals=1
    end

    return median(bench_results.times) / 1e9 / N
end
