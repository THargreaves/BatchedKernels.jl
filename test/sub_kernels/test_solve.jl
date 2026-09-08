@testitem "Upper Triangular Backward Solve (out-of-place)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            # Create upper triangular matrices
            Us = CUDA.zeros(Float32, D, D, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D, D)
                Us[:, :, i] = UpperTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D, D, N)
            Cs = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_backward_solve!(
                Cs, Us, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), mode
            )

            # CPU comparison
            Us_cpu = Array(Us)
            Bs_cpu = Array(Bs)
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = UpperTriangular(Us_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Upper Triangular Backward Solve (out-of-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    for D1 in 2:15
        for D2 in 2:15
            D = max(D1, D2)
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            CUDA.seed!(1234)

            # Create upper triangular matrices
            Us = CUDA.zeros(Float32, D1, D1, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D1, D1)
                Us[:, :, i] = UpperTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D1, D2, N)
            Cs = CUDA.zeros(Float32, D1, D2, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_backward_solve!(
                Cs, Us, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N)
            )

            # CPU comparison
            Us_cpu = Array(Us)
            Bs_cpu = Array(Bs)
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = UpperTriangular(Us_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Cs) .- Cs_cpu))

            if max_error >= 1e-3
                println("D1=$D1, D2=$D2, error=$max_error")
            end

            @test max_error < 1e-3
        end
    end
end

@testitem "Upper Triangular Backward Solve (in-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    for D1 in 2:15
        for D2 in 2:15
            D = max(D1, D2)
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            CUDA.seed!(1234)

            # Create upper triangular matrices
            Us = CUDA.zeros(Float32, D1, D1, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D1, D1)
                Us[:, :, i] = UpperTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D1, D2, N)
            Us_cpu = Array(Us)
            Bs_cpu = Array(Bs)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_backward_solve!(
                Bs, Us, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N)
            )

            # CPU comparison
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = UpperTriangular(Us_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Bs) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Upper Triangular Backward Solve (in-place, trig)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            # Create upper triangular matrices
            Us = CUDA.zeros(Float32, D, D, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D, D)
                Us[:, :, i] = UpperTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D, D, N)
            Us_cpu = Array(Us)
            Bs_cpu = Array(Bs)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_backward_solve!(
                Us, Us, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), mode
            )

            # CPU comparison
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = UpperTriangular(Us_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Us) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Lower Triangular Forward Solve (out-of-place)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            # Create lower triangular matrices
            Ls = CUDA.zeros(Float32, D, D, N)
            for i in 1:N
                L_temp = CUDA.rand(Float32, D, D)
                Ls[:, :, i] = LowerTriangular(L_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D, D, N)
            Cs = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_forward_solve!(
                Cs, Ls, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), mode
            )

            # CPU comparison
            Ls_cpu = Array(Ls)
            Bs_cpu = Array(Bs)
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = LowerTriangular(Ls_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Lower Triangular Forward Solve (out-of-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    for D1 in 2:15
        for D2 in 2:15
            D = max(D1, D2)
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            CUDA.seed!(1234)

            # Create upper triangular matrices
            Ls = CUDA.zeros(Float32, D1, D1, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D1, D1)
                Ls[:, :, i] = LowerTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D1, D2, N)
            Cs = CUDA.zeros(Float32, D1, D2, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_forward_solve!(
                Cs, Ls, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N)
            )

            # CPU comparison
            Ls_cpu = Array(Ls)
            Bs_cpu = Array(Bs)
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = LowerTriangular(Ls_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Lower Triangular Forward Solve (in-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    for D1 in 2:15
        for D2 in 2:15
            D = max(D1, D2)
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            CUDA.seed!(1234)

            # Create upper triangular matrices
            Ls = CUDA.zeros(Float32, D1, D1, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D1, D1)
                Ls[:, :, i] = LowerTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D1, D2, N)
            Ls_cpu = Array(Ls)
            Bs_cpu = Array(Bs)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_forward_solve!(
                Bs, Ls, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N)
            )

            # CPU comparison
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = LowerTriangular(Ls_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Bs) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Lower Triangular Forward Solve (in-place, triangular)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            # Create lower triangular matrices
            Ls = CUDA.zeros(Float32, D, D, N)
            for i in 1:N
                L_temp = CUDA.rand(Float32, D, D)
                Ls[:, :, i] = LowerTriangular(L_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D, D, N)
            Ls_cpu = Array(Ls)
            Bs_cpu = Array(Bs)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_forward_solve!(
                Ls, Ls, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), mode
            )

            # CPU comparison
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = LowerTriangular(Ls_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Ls) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Lower Triangular Backward Solve vector (out-of-place)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    function kernel_backsolve_vec!(
        cs, Ls, bs, ::Val{D1}, ::Val{D}, ::Val{nthreads}, N::Int32
    ) where {D1,D,nthreads}
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        n_mats_per_block = n_warps * n_mats_per_warp
        dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        bid = blockIdx().x
        wid = div(tid - 1i32, 32i32) + 1i32
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)
        grid_mtrx_id =
            warp_matrix_id +
            (wid - 1i32) * n_mats_per_warp +
            (bid - 1i32) * n_mats_per_block

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_elems = warp_shmem_size * n_warps
        shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
        shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))

        shmem_vec_elems = D * n_warps * n_mats_per_warp
        shmem_vec_1 = CuDynamicSharedArray(
            Float32, shmem_vec_elems, 2 * shmem_elems * sizeof(Float32)
        )

        intermediate_layout_load!(shmem_2, Ls, Val(D1), Val(D1), Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(
            shmem_1, shmem_2, Val(D1), Val(D1), Val(D), Val(nthreads), N
        )

        vector_load!(shmem_vec_1, bs, Val(D1), Val(D), Val(nthreads), N)

        M1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
        v1 = BatchedVector(shmem_vec_1, Val(D), warp_matrix_id)

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            batch_op!(
                \, v1, LowerTriangular(M1), v1, d, Val(D1), Val(D), Val(D), warp_matrix_id
            )
        end

        sync_warp()

        vector_write!(cs, shmem_vec_1, Val(D1), Val(D), Val(nthreads), N)

        return nothing
    end

    # Test parameters
    nthreads = 2^8
    N = 2^9 + 1

    for D1 in 2:8
        for extra in 0:1
            D = D1 + extra
            nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

            CUDA.seed!(1234)

            # Create upper triangular matrices
            Ls_cpu = zeros(Float32, D1, D1, N)
            for i in 1:N
                L_temp = rand(Float32, D1, D1)
                Ls_cpu[:, :, i] = LowerTriangular(L_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            bs_cpu = rand(Float32, D1, N)
            cs_cpu = zeros(Float32, D1, N)

            cs = cu(cs_cpu)
            Ls = cu(Ls_cpu)
            bs = cu(bs_cpu)

            n_warps = nthreads ÷ 32
            n_mats_per_warp = 32 ÷ D
            shmem_elems = let
                dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
                warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1)
                warp_shmem_size * n_warps
            end
            shmem_vec_elems = D * n_warps * n_mats_per_warp
            shmem_bytes = (2 * shmem_elems + shmem_vec_elems) * sizeof(Float32)

            kernel = @cuda launch = false kernel_backsolve_vec!(
                cs, Ls, bs, Val(Int32(D1)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
            )
            CUDA.cuFuncSetAttribute(
                kernel.fun,
                CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shmem_bytes,
            )

            CUDA.@sync kernel(
                cs,
                Ls,
                bs,
                Val(Int32(D1)),
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N);
                threads=nthreads,
                blocks=nblocks,
                shmem=shmem_bytes,
            )

            cs_cpu = Array(cs)
            cs_ref = zeros(Float32, D1, N)

            for i in 1:N
                cs_ref[:, i] = LowerTriangular(Ls_cpu[:, :, i]) \ bs_cpu[:, i]
            end

            max_error = maximum(abs.(cs_cpu .- cs_ref))
            println("D1=$D1, D=$D, error=$max_error")
            @test max_error < 1e-3
        end
    end
end
