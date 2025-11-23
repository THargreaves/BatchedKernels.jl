function forward_solve_timing(_, B, _, ::Val{:gpu_mem_bound})
    T = eltype(B)
    D = size(B, 1)
    GPU_BANDWIDTH = 1008 * 10^9
    return (2 * D^2 + D * (D + 1) / 2) * sizeof(T) / GPU_BANDWIDTH
end