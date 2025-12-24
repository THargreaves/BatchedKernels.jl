using BatchedKernels
using BenchmarkTools
using LinearAlgebra
using CUDA
using CUDA: i32

function kernel_matmul(A, B)
    return A * B
end

function matmul_timing(_, A_cpu, B_cpu, _, ::Val{:ours_vmap})
    _, _, N = size(A_cpu)
    
    A = cu(A_cpu)
    B = cu(B_cpu)

    kernel_matmul_vmap = vmap(kernel_matmul)

    bench_results = @benchmark begin
        CUDA.@sync $kernel_matmul_vmap($A, $B)
    end

    return median(bench_results.times) / 1e9 / N
end
