function cholesky_timing(A, _, ::Val{:gpu_mem_bound})
    T = eltype(A)
    D = size(A, 1)
    GPU_BANDWIDTH = 1008 * 10^9
    return 2 * D^2 * sizeof(T) / GPU_BANDWIDTH / 2
end