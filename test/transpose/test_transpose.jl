@testitem "Transpose (out-of-place)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    function kernel_transpose!(
        Bs, As, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{mode},
    ) where {D,nthreads,mode}
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        n_mats_per_block = n_warps * n_mats_per_warp
        padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        bid = blockIdx().x
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)
        grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block

        shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load A
        intermediate_layout_load!(shmem_3, As, Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            # Create dual-access matrices
            A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
            B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))

            # Perform out-of-place Cholesky
            batch_op!(transpose, B, A, d, Val(D), Val(:small))
        end

        # Store result
        dual_to_interm_transfer!(shmem_3, shmem_2, Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Bs, shmem_3, Val(D), Val(nthreads), N, Val(:small), Val(mode))

        return nothing
    end

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        CUDA.seed!(1234)

        As = CUDA.rand(Float32, D, D, N)
        Bs = CUDA.zeros(Float32, D, D, N)
        As_cpu = Array(As)
        Bs_cpu = permutedims(As_cpu, (2, 1, 3))

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_transpose!(
            Bs, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), Val(:indep),
        )

        # CPU comparison
        max_error = maximum(abs.(Array(Bs) .- Bs_cpu))
        @test max_error < 1e-5
    end
end

@testitem "Transpose (in-place)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    function kernel_transpose!(
        Bs, As, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{mode},
    ) where {D,nthreads,mode}
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        n_mats_per_block = n_warps * n_mats_per_warp
        padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        bid = blockIdx().x
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)
        grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block

        shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load A
        intermediate_layout_load!(shmem_2, As, Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_1, shmem_2, Val(D), Val(nthreads), N, Val(:small))

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            # Create dual-access matrices
            A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))

            # Perform out-of-place Cholesky
            batch_op!(transpose, A, A, d, Val(D), Val(:small))
        end

        # Store result
        dual_to_interm_transfer!(shmem_2, shmem_1, Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Bs, shmem_2, Val(D), Val(nthreads), N, Val(:small), Val(mode))

        return nothing
    end

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        CUDA.seed!(1234)

        As = CUDA.rand(Float32, D, D, N)
        Bs = CUDA.zeros(Float32, D, D, N)
        As_cpu = Array(As)
        Bs_cpu = permutedims(As_cpu, (2, 1, 3))

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_transpose!(
            Bs, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), Val(:indep),
        )

        # CPU comparison
        max_error = maximum(abs.(Array(Bs) .- Bs_cpu))
        @test max_error < 1e-5
    end
end