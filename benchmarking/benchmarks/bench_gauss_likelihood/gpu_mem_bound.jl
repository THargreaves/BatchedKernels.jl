function gauss_likelihood_timing(_, _, _, Σs_in_cpu, _, _, ::Val{:gpu_mem_bound})
    T = eltype(Σs_in_cpu)
    D = size(Σs_in_cpu, 1)
    GPU_BANDWIDTH = 1008 * 10^9
    return (1 * D^2 + 2 * D + 1) * sizeof(T) / GPU_BANDWIDTH
end