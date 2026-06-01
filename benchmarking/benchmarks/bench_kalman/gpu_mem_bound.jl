function kalman_timing(_, P_in_cpu, _, _, _, _, _, nthreads, ::Val{:gpu_mem_bound})
    T = eltype(P_in_cpu)
    D, _, N = size(P_in_cpu)
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))
    GPU_BANDWIDTH = 1008 * 10^9
    return (2 * D^2 * N + 4 * D^2 * nblocks) / N * sizeof(T) / GPU_BANDWIDTH
end