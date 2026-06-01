# cusolver.jl
#
# cuSOLVER batched Cholesky (cuBLAS itself has no Cholesky routine, so we use
# the cuSOLVER dense handle — same pattern as the kalman benchmark).
# Drop-in for the benchmark harness — same `cholesky_timing(...)` API.
#
# Implementation notes:
#   - `cusolverDnSpotrfBatched` factorises in-place. We copy A → U in the
#     @benchmark setup, then factor in-place.
#   - uplo = UPPER gives A = U' * U (upper triangle of U holds the factor).

using CUDA
using CUDA: i32
using BenchmarkTools
using Statistics

# ─── Device pointer-array builder ───────────────────────────────────

# ptrs[i] → start of batch slice i within a D×D×N contiguous array.
function _fill_batch_ptrs_kernel!(ptrs, A, DD)
    tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    gstride = gridDim().x * blockDim().x
    n = length(ptrs)
    i = tid
    while i <= n
        idx = (Int64(i) - 1) * Int64(DD) + 1          # Int64 to avoid overflow
        @inbounds ptrs[i] = reinterpret(Ptr{Float32}, pointer(A, idx))
        i += gstride
    end
    return
end

function _batch_ptrs(A::DenseCuArray{Float32}, N::Integer, DD::Integer)
    ptrs = CuArray{Ptr{Float32}}(undef, N)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _fill_batch_ptrs_kernel!(ptrs, A, Int32(DD))
    return ptrs
end

# ─── Main Cholesky ──────────────────────────────────────────────────

function cholesky_cusolver!(
    dU,              # batched pointer array (CuVector{Ptr{Float32}})
    info_d, D, N,
)
    sh = CUDA.CUSOLVER.dense_handle()   # cuSOLVER dense handle, current stream

    UPPER = CUDA.CUBLAS.CUBLAS_FILL_MODE_UPPER

    CUDA.CUSOLVER.cusolverDnSpotrfBatched(sh, UPPER, D, dU, D, info_d, N)

    CUDA.synchronize()
    return nothing
end

function cholesky_timing(U_out_cpu, A_in_cpu, _, _, ::Val{:cusolver})
    D, _, N = size(U_out_cpu)
    DD = D * D

    U_out = cu(U_out_cpu)

    # Pointer array is built once; the @benchmark setup reuses the same
    # device buffer (refilled with A each iteration), so these pointers
    # stay valid.
    dU = _batch_ptrs(U_out, N, DD)

    info_d = CUDA.zeros(Cint, N)

    bench_results = @benchmark begin
        cholesky_cusolver!(
            $dU,
            $info_d,
            $D,
            $N,
        )
    end setup=begin
        copyto!($U_out, $A_in_cpu)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
