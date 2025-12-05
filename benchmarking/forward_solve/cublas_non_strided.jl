using CUDA
using BenchmarkTools
using LinearAlgebra

function cublas_forward_batched!(
    L, B, T,
)
    side  = 'L'
    uplo  = 'L'
    trans = 'N'
    diag  = 'N'
    alpha = one(T)

    CUDA.CUBLAS.trsm_batched!(
        side,
        uplo,
        trans,
        diag,
        alpha,
        L,
        B,
    )
end

function forward_solve_timing(L_cpu, B_cpu, _, ::Val{:cublas_non_strided})
    N = size(B_cpu, 3)
    T = eltype(B_cpu)

    L = cu(L_cpu)
    B = cu(B_cpu)

    A_batch = Vector{CuArray{T,2}}(undef, N)
    B_batch = Vector{CuArray{T,2}}(undef, N)

    @inbounds for i in 1:N
        A_batch[i] = @view L[:, :, i]
        B_batch[i] = @view B[:, :, i]
    end

    bench_results = @benchmark begin
        CUDA.@sync cublas_forward_batched!($A_batch, $B_batch, $T)
    end

    return median(bench_results.times) / 1e9 / N
end
