function qr_r_timing(Rs, _, _, _, ::Val{:gpu_mem_bound})
    T = eltype(Rs)
    D = size(Rs, 1)
    GPU_BANDWIDTH = 1008 * 10^9
    return 2 * D^2 * sizeof(T) / GPU_BANDWIDTH
end