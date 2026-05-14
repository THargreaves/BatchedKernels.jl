function backsolve_timing(C, _, _, ::Val{:gpu_mem_bound})
    T = eltype(C)
    D = size(C, 1)
    GPU_BANDWIDTH = 1008 * 10^9
    # return 2 * D * (D + 1) / 2 * sizeof(T) / GPU_BANDWIDTH
    return 3 * D^2 * sizeof(T) / GPU_BANDWIDTH
end