function kalman_timing(_, P_in_cpu, _, _, _, _, _, _, ::Val{:gpu_mem_bound})
    T = eltype(P_in_cpu)
    D = size(P_in_cpu, 1)
    GPU_BANDWIDTH = 1008 * 10^9
    # return 2 * D * (D + 1) / 2 * sizeof(T) / GPU_BANDWIDTH
    return 3 * D^2 * sizeof(T) / GPU_BANDWIDTH
end