using CUDA
using BenchmarkTools

function cublas_strided!(
    C::CuArray{T, 3}, A::CuArray{T, 3}, B::CuArray{T,3},
) where {T}
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', T(1.0), A, B, T(0.0), C)
end


function matmul_timing(C_cpu, A_cpu, B_cpu, _, ::Val{:cublas_strided})
    N = size(C_cpu, 3)
    
    A = cu(A_cpu)
    B = cu(B_cpu)
    C = cu(C_cpu)

    bench_results = @benchmark begin
        CUDA.@sync cublas_strided!($C, $A, $B)
    end

    return median(bench_results.times) / 1e9 / N
end