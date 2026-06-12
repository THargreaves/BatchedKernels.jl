function matadd_timing(C_out_cpu, _, _, _, _, ::Val{:gpu_mem_bound})
    T = eltype(C_out_cpu)
    D = size(C_out_cpu, 1)
    GPU_BANDWIDTH = 1008 * 10^9
    return 3 * D^2 * sizeof(T) / GPU_BANDWIDTH
end