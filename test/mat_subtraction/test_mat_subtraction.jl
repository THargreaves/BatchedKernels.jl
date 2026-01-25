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
        dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        bid = blockIdx().x
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)
        grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_elems = warp_shmem_size * n_warps
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

@testitem "Matrix Subtraction (non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    @inline function kernel_sub!(
        Cs,
        As,
        Bs,
        ::Val{D1},
        ::Val{D2},
        ::Val{nthreads},
        N::Int32,
    ) where {D1,D2,nthreads}
        D = max(D1, D2)
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        n_mats_per_block = n_warps * n_mats_per_warp
        dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        bid = blockIdx().x
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)
        grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_elems = warp_shmem_size * n_warps
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load A
        intermediate_layout_load!(shmem_3, As, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_1, shmem_3, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

        # Load B
        intermediate_layout_load!(shmem_3, Bs, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_2, shmem_3, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
        
        # Create dual-access matrices
        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            # Perform operation
            batch_op!(-, C, A, B, d, Val(D1), Val(D2), Val(D), Val(:small))
        end

        # Store C
        dual_to_interm_transfer!(shmem_1, C, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Cs, shmem_1, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

        return nothing
    end

    for D1 in 2:15
        for D2 in 2:15
            D = max(D1, D2)
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            CUDA.seed!(1234)

            As = CUDA.rand(Float32, D1, D2, N)
            Bs = CUDA.rand(Float32, D1, D2, N)
            Cs = CUDA.zeros(Float32, D1, D2, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_sub!(
                Cs, As, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N),
            )

            # CPU comparison
            As_cpu = Array(As)
            Bs_cpu = Array(Bs)
            Cs_cpu = As_cpu .- Bs_cpu

            max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
            
            if max_error > 1e-5
                println("D1=$D1, D2=$D2")
            end
            @test max_error < 1e-5
        end
    end
end

@testitem "Matrix Subtraction (vmap non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    function sub(A, B)
        return A - B
    end

    # Test parameters
    N = 2^12 + 113

    for D1 in 2:15
        for D2 in 2:15
            CUDA.seed!(1234)

            As = CUDA.rand(Float32, D1, D2, N)
            Bs = CUDA.rand(Float32, D1, D2, N)

            sub_vmap = BatchedKernels.vmap(sub)
            Cs = sub_vmap(As, Bs)

            # CPU comparison
            As_cpu = Array(As)
            Bs_cpu = Array(Bs)
            Cs_cpu = As_cpu .- Bs_cpu

            max_error = maximum(abs.(Array(Cs) .- Cs_cpu))

            @test max_error < 1e-5
        end
    end
end

@testitem "Shared Matrix Subtraction (vmap non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    function sub(A, B, dummy)
        return A - B
    end

    # Test parameters
    N = 2^12 + 113

    for D1 in 2:15
        for D2 in 2:15
            Dmax = max(D1, D2)
            for D in Dmax:(Dmax + 1)
                dummy = CUDA.rand(Float32, D, D, N)
                
                ### Test 1 ###
                CUDA.seed!(1234)

                As = CUDA.rand(Float32, D1, D2, N)
                Bs = CUDA.rand(Float32, D1, D2)

                sub_vmap1 = BatchedKernels.vmap(
                    sub,
                    in_type = (:batched, :shared, :batched)
                )
                Cs = sub_vmap1(As, Bs, dummy)

                # CPU comparison
                As_cpu = Array(As)
                Bs_cpu = Array(Bs)
                Cs_cpu = similar(As_cpu)
                for i in 1:N
                    Cs_cpu[:, :, i] = sub(As_cpu[:, :, i], Bs_cpu, 0)
                end

                max_error = maximum(abs.(Array(Cs) .- Cs_cpu))

                @test max_error < 1e-5

                ### Test 2 ###
                CUDA.seed!(1234)

                As = CUDA.rand(Float32, D1, D2)
                Bs = CUDA.rand(Float32, D1, D2, N)

                sub_vmap2 = BatchedKernels.vmap(
                    sub,
                    in_type = (:shared, :batched, :batched)
                )

                Cs = sub_vmap2(As, Bs, dummy)

                # CPU comparison
                As_cpu = Array(As)
                Bs_cpu = Array(Bs)
                Cs_cpu = similar(Bs_cpu)
                for i in 1:N
                    Cs_cpu[:, :, i] = sub(As_cpu, Bs_cpu[:, :, i], 0)
                end

                max_error = maximum(abs.(Array(Cs) .- Cs_cpu))

                @test max_error < 1e-5
            end
        end
    end
end