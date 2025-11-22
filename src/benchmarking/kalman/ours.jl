using BatchedKernels
using BenchmarkTools
using CUDA

function kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, _, ::Val{:ours})
    D, _, N = size(P_in_cpu)
    
    P_out = cu(P_out_cpu)
    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    nthreads = 2^8
    nblocks = cld(N, nthreads//32 * (32 ÷ D))

    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $nthreads blocks = $nblocks kernel_kalman!(
            $P_out,
            $P_in,
            $A,
            $Q,
            $H,
            $R,
            Val(Int32($D)),
            Val(Int32($nthreads)),
            Int32($N),
            $Val(:small),
            $Val(:indep),
        )
    end

    return median(bench_results.times) / 1e9 / N
end