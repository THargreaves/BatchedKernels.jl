using CUDA
using BenchmarkTools

function cublas_non_strided!(
    C::Vector{CuArray{T,2,CUDA.DeviceMemory}},
    A::Vector{CuArray{T,2,CUDA.DeviceMemory}},
    B::Vector{CuArray{T,2,CUDA.DeviceMemory}},
) where {T}
    CUDA.CUBLAS.gemm_batched!('N', 'N', one(T), A, B, zero(T), C)
    return C
end

function matmul_timing(C_cpu, A_cpu, B_cpu, _, ::Val{:cublas_non_strided})
    N = size(C_cpu, 3)
    
    A = [cu(A_cpu[:, :, i]) for i in 1:N]
    B = [cu(B_cpu[:, :, i]) for i in 1:N]
    C = [cu(C_cpu[:, :, i]) for i in 1:N]

    bench_results = @benchmark begin
        CUDA.@sync cublas_non_strided!($C, $A, $B)
    end

    return median(bench_results.times) / 1e9 / N
end


# T = Float32
# D = 2
# N = 10
# A = rand(T, D, D, N)
# B = rand(T, D, D, N)
# C = similar(A)

# println(matmul_timing(C, A, B, 0, Val(:cublas_non_strided)))