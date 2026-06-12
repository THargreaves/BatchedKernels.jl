# cublas.jl
#
# cuBLAS batched equivalent of the MAGMA matmul.
# Drop-in for the benchmark harness — same `matmul_timing(...)` API.
#
# Implementation notes:
#   - A single `cublasSgemmBatched` call: C = A * B.
#   - A and B differ per batch (no shared operands), so all three pointer
#     arrays are built with `_batch_ptrs`.
#   - cuBLAS runs on the current CUDA stream; we issue a final
#     `CUDA.synchronize()` so the benchmark captures completion.

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

# ─── Main matmul ────────────────────────────────────────────────────

function matmul_cublas!(
    dC, dA, dB,         # batched pointer arrays (CuVector{Ptr{Float32}})
    D, N,
)
    h = CUDA.CUBLAS.handle()           # cuBLAS handle, bound to current stream

    OP_N = CUDA.CUBLAS.CUBLAS_OP_N

    α = 1.0f0
    β = 0.0f0

    # C = A * B
    CUDA.CUBLAS.cublasSgemmBatched(h, OP_N, OP_N, D, D, D,
        α, dA, D, dB, D, β, dC, D, N)

    CUDA.synchronize()
    return nothing
end

function matmul_timing(C_out_cpu, A_in_cpu, B_in_cpu, _, _, ::Val{:cublas})
    D, _, N = size(C_out_cpu)
    DD = D * D

    C_out = cu(C_out_cpu)
    A_in  = cu(A_in_cpu)
    B_in  = cu(B_in_cpu)

    # Pointer arrays are built once; the benchmark setup reuses the same
    # device buffers (only `C_out` is reset), so these pointers stay valid.
    dC = _batch_ptrs(C_out, N, DD)
    dA = _batch_ptrs(A_in,  N, DD)
    dB = _batch_ptrs(B_in,  N, DD)

    bench_results = @benchmark begin
        matmul_cublas!(
            $dC,
            $dA,
            $dB,
            $D,
            $N,
        )
    end setup=begin
        fill!($C_out, 0f0)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
