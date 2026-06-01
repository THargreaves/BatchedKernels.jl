function matmul_timing(M_out_cpu, _, _, _, ::Val{D}, ::Val{:gpu_mem_bound}) where {D}
    T = eltype(M_out_cpu)
    GPU_BANDWIDTH = 1008 * 10^9
    return 2 * D^2 * sizeof(T) / GPU_BANDWIDTH
end