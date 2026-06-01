# cublas.jl
#
# cuBLAS equivalent of the matadd benchmark.
# Drop-in for the benchmark harness — same `matadd_timing(...)` API.
#
# Implementation notes:
#   - cuBLAS exposes `cublasSgeam` (single, non-batched: C = α·op(A) + β·op(B)).
#     There is no batched variant in cuBLAS.
#   - Matrix addition is purely elementwise, so the batch dimension carries no
#     structural information: we flatten A, B, C to a single (D*D*N)×1 column
#     and issue ONE `cublasSgeam` call. This is the most cuBLAS-native way to
#     express batched matrix addition.

using CUDA
using CUDA: i32
using BenchmarkTools
using Statistics

function matadd_cublas!(
    C_flat, A_flat, B_flat,    # flat CuVector views (length = D*D*N)
    M::Integer,                # = D * D * N
)
    h = CUDA.CUBLAS.handle()           # cuBLAS handle, bound to current stream

    OP_N = CUDA.CUBLAS.CUBLAS_OP_N

    α = 1.0f0
    β = 1.0f0

    # C = α·A + β·B  with α = β = 1, treating the whole batch as one column.
    CUDA.CUBLAS.cublasSgeam(
        h, OP_N, OP_N, M, 1,
        α, A_flat, M,
        β, B_flat, M,
        C_flat, M,
    )

    CUDA.synchronize()
    return nothing
end

function matadd_timing(C_out_cpu, A_in_cpu, B_in_cpu, _, _, ::Val{:cublas})
    D, _, N = size(C_out_cpu)
    M = D * D * N

    C_out = cu(C_out_cpu)
    A_in  = cu(A_in_cpu)
    B_in  = cu(B_in_cpu)

    # Flat views into contiguous GPU memory
    C_flat = reshape(C_out, :)
    A_flat = reshape(A_in, :)
    B_flat = reshape(B_in, :)

    bench_results = @benchmark begin
        matadd_cublas!(
            $C_flat,
            $A_flat,
            $B_flat,
            $M,
        )
    end setup=begin
        fill!($C_out, 0f0)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
