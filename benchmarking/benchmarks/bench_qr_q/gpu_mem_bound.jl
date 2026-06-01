function qr_q_timing(Qs, _, _, _,::Val{:gpu_mem_bound})
    T = eltype(Qs)
    D = size(Qs, 1)
    GPU_BANDWIDTH = 1008 * 10^9
    return 2 * D^2 * sizeof(T) / GPU_BANDWIDTH
end