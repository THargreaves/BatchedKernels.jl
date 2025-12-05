using BatchedKernels
using BenchmarkTools
using CUDA

function cholesky_timing(A_cpu, _, ::Val{:ours})
    D, _, N = size(A_cpu)
    T = eltype(A_cpu)
    
    A = cu(A_cpu)
    U = CUDA.zeros(T, D, D, N)

    nthreads = 2^8
    nblocks = cld(N, nthreads//32 * (32 ÷ D))

    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $nthreads blocks = $nblocks kernel_cholesky_inplace!(
            $U, $A, Val(Int32($D)), Val(Int32($nthreads)), Int32($N), $Val(:small), Val(:indep),
        )
    end

    return median(bench_results.times) / 1e9 / N
end