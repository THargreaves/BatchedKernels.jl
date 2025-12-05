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
