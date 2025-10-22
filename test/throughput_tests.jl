# A performance test with trivial operation to test just the memory access throughput
@testitem "Memory Access Throughput Test" begin
    using BenchmarkTools
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8

    mem_clock = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    bus_width = attribute(device(), CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    mem_bandwidth = 2.0f0 * mem_clock * (bus_width / 8) / 1e6 * 1e9   # in B/s

    # Set full shared memory usage
    CUDA.cache_config!(CUDA.FUNC_CACHE_PREFER_SHARED)

    function kernel_matmul_n_mats_per_warp!(
        Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32
    ) where {D,nthreads}
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        wid = div(tid - 1i32, 32i32) + 1i32
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)

        shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load A
        intermediate_layout_load!(shmem_3, As, Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)

        # Load B
        intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N)

        # Create dual-access matrices
        A = DualAccessMatrix(shmem_1, Val(D), wid, warp_matrix_id)
        B = DualAccessMatrix(shmem_2, Val(D), wid, warp_matrix_id)
        C = DualAccessMatrix(shmem_3, Val(D), wid, warp_matrix_id)

        # Perform a trivial operation
        batch_op!(*, C, A, B, d, Val(D))

        # Store C
        dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)
        intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N)

        return nothing
    end

    function kernel_matmul_one_mats_per_warp!(
        Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32
    ) where {D,nthreads}
        n_mats_per_warp = 1i32
        n_warps = nthreads ÷ 32i32
        n_mats_per_block = n_warps * n_mats_per_warp
        interm_pad_freq = div(32i32, D & -D) * D

        padded_amount_per_warp = (D * D) ÷ interm_pad_freq
        warp_shmem_size = D * D + padded_amount_per_warp

        tid = threadIdx().x
        lid = mod1(tid, 32i32)

        block_matrix_id = div(tid - 1i32, 32i32) + 1i32
        d = mod1(lid, D)

        shmem_elems = warp_shmem_size * n_mats_per_block
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        intermediate_layout_load!(shmem_1, As, Val(D), Val(nthreads), N)
        intermediate_layout_load!(shmem_2, Bs, Val(D), Val(nthreads), N)

        A = DualAccessMatrix(shmem_1, Val(D), block_matrix_id)
        B = DualAccessMatrix(shmem_2, Val(D), block_matrix_id)
        C = DualAccessMatrix(shmem_3, Val(D), block_matrix_id)
        
        if block_matrix_id <= n_mats_per_block && lid <= D
            batch_op!(*, C, A, B, d, Val(D))
        end

        intermediate_layout_write!(Cs, shmem_3, Val(D), Val(nthreads), N)

        return nothing
    end

    @static if BatchedKernels.VERSION === :NMatsPerWarp
        kernel! = kernel_matmul_n_mats_per_warp!
    elseif BatchedKernels.VERSION === :OneMatPerWarp
        kernel! = kernel_matmul_one_mats_per_warp!
    end

    # Throughput tests
    for D in 2:12
        # Target 1GB of input data to minimise impact of L1 cache
        N_bench = Int32(ceil(1e9 / (4 * 2 * D^2)))
        if BatchedKernels.VERSION == :NMatsPerWarp
            nblocks = cld(N_bench, nthreads//32 * (32 ÷ D))
        elseif BatchedKernels.VERSION == :OneMatPerWarp
            n_mats_per_warp = 1
            n_warps = nthreads ÷ 32
            n_mats_per_block = n_warps * n_mats_per_warp
            nblocks = cld(N_bench, n_mats_per_block)
        end

        CUDA.seed!(1234)

        As = CUDA.rand(Float32, D, D, N_bench)
        Bs = CUDA.rand(Float32, D, D, N_bench)
        Cs = CuArray{Float32}(undef, D, D, N_bench)

        bench_results = @benchmark begin
            CUDA.@sync @cuda threads = $nthreads blocks = $nblocks kernel!(
                $Cs, $As, $Bs, Val(Int32($D)), Val(Int32($nthreads)), Int32($N_bench)
            )
        end

        # Verify at least 90% of theoretical speed
        bytes_total = 4 * 3 * D^2 * N_bench  # 3 matrices read/written
        achieved_bandwidth = bytes_total / (median(bench_results).time / 1e9)
        debug = true
        debug |= !((@test 0.9 < achieved_bandwidth / mem_bandwidth < 1.0) isa Test.Pass)
        if debug
            @info "Dimension: $D, result: $(achieved_bandwidth / mem_bandwidth), time: $(median(bench_results))"
        end
    end
end
