# cublas.jl
#
# cuBLAS batched equivalent of the upper-triangular back-solve.
# Drop-in for the benchmark harness — same `backsolve_timing(...)` API.
#
# Implementation notes:
#   - `cublasStrsmBatched` solves op(A) * X = alpha * B in-place, overwriting
#     B with X. We seed C with B in the @benchmark setup, then trsm puts
#     X = U^{-1} B into C.

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

# ─── Main back-solve ────────────────────────────────────────────────

function backsolve_cublas!(
    dU, dC,         # batched pointer arrays
    D, N,
)
    h = CUDA.CUBLAS.handle()           # cuBLAS handle, bound to current stream

    LEFT    = CUDA.CUBLAS.CUBLAS_SIDE_LEFT
    UPPER   = CUDA.CUBLAS.CUBLAS_FILL_MODE_UPPER
    OP_N    = CUDA.CUBLAS.CUBLAS_OP_N
    NONUNIT = CUDA.CUBLAS.CUBLAS_DIAG_NON_UNIT

    α = 1.0f0

    # Solves U * X = α * C  (C seeded to B)  →  C := U^{-1} B
    CUDA.CUBLAS.cublasStrsmBatched(
        h, LEFT, UPPER, OP_N, NONUNIT,
        D, D, α, dU, D, dC, D, N,
    )

    CUDA.synchronize()
    return nothing
end

function backsolve_timing(C_out_cpu, U_in_cpu, B_in_cpu, _, _, ::Val{:cublas})
    D, _, N = size(C_out_cpu)
    DD = D * D

    C_out = cu(C_out_cpu)
    U_in  = cu(U_in_cpu)
    Bs    = cu(B_in_cpu)
    copyto!(C_out, Bs)  # seed initial C with B so pointer arrays index valid data

    dU = _batch_ptrs(U_in, N, DD)
    dC = _batch_ptrs(C_out, N, DD)

    bench_results = @benchmark begin
        backsolve_cublas!(
            $dU,
            $dC,
            $D,
            $N,
        )
    end setup=begin
        copyto!($C_out, $Bs)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
