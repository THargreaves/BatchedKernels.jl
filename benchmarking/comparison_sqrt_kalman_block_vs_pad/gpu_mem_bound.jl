function sqrt_kalman_timing(Ss_out_cpu, _, _, _, _, _, _, ::Val, nthreads, ::Val{:gpu_mem_bound})
    T = eltype(Ss_out_cpu)
    D = size(Ss_out_cpu, 1)
    GPU_BANDWIDTH = 1008 * 10^9
    return 2 * D^2 * sizeof(T) / GPU_BANDWIDTH
end