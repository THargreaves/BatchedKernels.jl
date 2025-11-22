function forward_solve_timing(_, B, _, ::Val{:gpu_mem_bound})
    T = eltype(B)
    D = size(B, 1)
    GPU_BANDWIDTH = 1008 * 10^9
    return 3 * D^2 * sizeof(T) / GPU_BANDWIDTH / 2
end