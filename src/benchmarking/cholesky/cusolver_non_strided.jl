using CUDA
using LinearAlgebra

function cusolver_strided!(mats)
    CUDA.CUSOLVER.potrfBatched!('L', mats)
end

function cholesky_timing(A_cpu, _, ::Val{:cusolver_non_strided})
    N = size(A_cpu, 3)
    
    A = cu(A_cpu)
    mats = [view(A, :, :, i) for i in 1:N]

    bench_results = @benchmark begin
        CUDA.@sync cusolver_strided!($mats)
    end

    return median(bench_results.times) / 1e9 / N
end

# T = Float32
# D = 2
# N = 10
# A = zeros(T, D, D, N)
# for i in 1:N
#     A_temp = rand(Float32, D, D)
#     A[:, :, i] = A_temp * A_temp' + 0.1f0 * I
# end
# println(matmul_timing(A, 0, Val(:cusolver_non_strided)))