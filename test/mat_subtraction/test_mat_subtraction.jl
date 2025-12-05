@testitem "Matrix Subtraction" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    @inline function kernel_sub!(Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{mode}) where {D,nthreads,mode}
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

        # Load B
        intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N, Val(:small))
        
        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            # Create dual-access matrices
            A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
            B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
            C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

            # Perform operation
            batch_op!(-, C, A, B, d, Val(D), Val(:small))
        end

        # Store C
        dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N, Val(:small), Val(mode))

        return nothing
    end

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            As = CUDA.rand(Float32, D, D, N)
            Bs = CUDA.rand(Float32, D, D, N)
            Cs = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_sub!(
                Cs, As, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), mode,
            )

            # CPU comparison
            As_cpu = Array(As)
            Bs_cpu = Array(Bs)
            Cs_cpu = As_cpu .- Bs_cpu

            max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
            @test max_error < 1e-5
        end
    end
end