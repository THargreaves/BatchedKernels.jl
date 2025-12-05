using BatchedKernels
using BenchmarkTools
using CUDA

function forward_solve_timing(L_cpu, B_cpu, _, ::Val{:ours})
    D, _, N = size(B_cpu)
    
    L = cu(L_cpu)
    B = cu(B_cpu)

    nthreads = 2^8
    nblocks = cld(N, nthreads//32 * (32 ÷ D))

    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $nthreads blocks = $nblocks kernel_forward_solve!(
            $B, $L, $B, Val(Int32($D)), Val(Int32($nthreads)), Int32($N), $Val(:small), Val(:indep),
        )
    end

    return median(bench_results.times) / 1e9 / N
end