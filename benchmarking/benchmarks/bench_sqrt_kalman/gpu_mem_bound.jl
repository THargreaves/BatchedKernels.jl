function sqrt_kalman_timing(Ss_out, _, _, _, _, _, _, ::Val, nthreads, ::Val{:gpu_mem_bound})
    T = eltype(Ss_out)
    D, _, N = size(Ss_out)
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))
    GPU_BANDWIDTH = 1008 * 10^9
    return (2 * D^2 * N + 4 * D^2 * nblocks) / N * sizeof(T) / GPU_BANDWIDTH
end