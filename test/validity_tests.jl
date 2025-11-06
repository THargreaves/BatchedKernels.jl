@testitem "Matrix Multiplication (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^12
    nthreads = 2^8

    # Test for both independent and consequtive modes
    modes = (Val(:indep), Val(:conseq))

    # Test all four combinations
    test_cases = [(false, false), (true, false), (false, true), (true, true)]

    # Accuracy tests
    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            As = CUDA.rand(Float32, D, D, N)
            Bs = CUDA.rand(Float32, D, D, N)
            As_cpu = Array(As)
            Bs_cpu = Array(Bs)

            for (A_adj, B_adj) in test_cases
                Cs = CUDA.zeros(Float32, D, D, N)

                CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_matmul!(
                    Cs, As, Bs, Val(A_adj), Val(B_adj), Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
                )

                # CPU comparison
                Cs_cpu = similar(As_cpu)
                for i in 1:N
                    if A_adj && B_adj
                        Cs_cpu[:, :, i] = As_cpu[:, :, i]' * Bs_cpu[:, :, i]'
                    elseif A_adj
                        Cs_cpu[:, :, i] = As_cpu[:, :, i]' * Bs_cpu[:, :, i]
                    elseif B_adj
                        Cs_cpu[:, :, i] = As_cpu[:, :, i] * Bs_cpu[:, :, i]'
                    else
                        Cs_cpu[:, :, i] = As_cpu[:, :, i] * Bs_cpu[:, :, i]
                    end
                end

                max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
                @test max_error < 1e-5
            end
        end
    end
end

@testitem "Matrix Multiplication (large)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^12

    # Test for both independent and consequtive modes
    modes = (Val(:indep), Val(:conseq))

    # Test all four combinations
    test_cases = [(false, false), (true, false), (false, true), (true, true)]

    # Accuracy tests
    for D in 2:32
        for mode in modes

            # Calculating nthreads and nblocks
            if mode === Val(:conseq)
                nthreads = max(256, 1 << (ceil(Int, log2(D^2))))
                n_mats_per_block = nthreads ÷ (D * D)
                nblocks = cld(N, n_mats_per_block)
            elseif mode === Val(:indep)
                n_cols_per_warp = max(1, prevpow(2, 32 ÷ D))
                n_elems_per_mat = D ÷ n_cols_per_warp * 32 + (D % n_cols_per_warp) * D
                n_mats_per_block = 1
                nthreads = ((n_mats_per_block * n_elems_per_mat + 31) ÷ 32) * 32
                nblocks = cld(N, n_mats_per_block)
            end

            CUDA.seed!(1234)

            As = CUDA.rand(Float32, D, D, N)
            Bs = CUDA.rand(Float32, D, D, N)
            As_cpu = Array(As)
            Bs_cpu = Array(Bs)

            for (A_adj, B_adj) in test_cases
                Cs = CUDA.zeros(Float32, D, D, N)

                CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_matmul!(
                    Cs, As, Bs, Val(A_adj), Val(B_adj), Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:large), mode,
                )

                # CPU comparison
                Cs_cpu = similar(As_cpu)
                for i in 1:N
                    if A_adj && B_adj
                        Cs_cpu[:, :, i] = As_cpu[:, :, i]' * Bs_cpu[:, :, i]'
                    elseif A_adj
                        Cs_cpu[:, :, i] = As_cpu[:, :, i]' * Bs_cpu[:, :, i]
                    elseif B_adj
                        Cs_cpu[:, :, i] = As_cpu[:, :, i] * Bs_cpu[:, :, i]'
                    else
                        Cs_cpu[:, :, i] = As_cpu[:, :, i] * Bs_cpu[:, :, i]
                    end
                end

                max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
                @test max_error < 1e-5
            end
        end
    end
end

@testitem "Matrix Subtraction" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113
    nblocks = 2^8

    function kernel_sub!(Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32) where {D,nthreads}
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

        # Perform operation
        batch_op!(-, C, A, B, d, Val(D))

        # Store C
        dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)
        intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N)

        return nothing
    end

    for D in 2:10
        CUDA.seed!(1234)

        As = CUDA.rand(Float32, D, D, N)
        Bs = CUDA.rand(Float32, D, D, N)
        Cs = CUDA.zeros(Float32, D, D, N)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_sub!(
            Cs, As, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
        )

        # CPU comparison
        As_cpu = Array(As)
        Bs_cpu = Array(Bs)
        Cs_cpu = As_cpu .- Bs_cpu

        max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
        @test max_error < 1e-5
    end
end

@testitem "Cholesky Decomposition (in-place)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113
    nblocks = 2^8

    function kernel_cholesky_inplace!(
        Us, As, ::Val{D}, ::Val{nthreads}, N::Int32
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

        # Load A
        intermediate_layout_load!(shmem_2, As, Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_1, shmem_2, Val(D), Val(nthreads), N)

        # Create dual-access matrix
        A = DualAccessMatrix(shmem_1, Val(D), wid, warp_matrix_id)

        # Perform in-place Cholesky
        batch_op!(cholesky, A, d, Val(D), n_mats_per_warp, warp_matrix_id)

        # Store result
        dual_to_interm_transfer!(shmem_2, shmem_1, Val(D), Val(nthreads), N)
        intermediate_layout_write!(Us, shmem_2, Val(D), Val(nthreads), N)

        return nothing
    end

    for D in 2:10
        CUDA.seed!(1234)

        # Create symmetric positive definite matrices
        As = CUDA.zeros(Float32, D, D, N)
        for i in 1:N
            A_temp = CUDA.rand(Float32, D, D)
            As[:, :, i] = A_temp * A_temp' + 0.1f0 * I
        end
        Us = CUDA.zeros(Float32, D, D, N)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_cholesky_inplace!(
            Us, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
        )

        # CPU comparison
        As_cpu = Array(As)
        Us_cpu = similar(As_cpu)
        for i in 1:N
            Us_cpu[:, :, i] = cholesky(As_cpu[:, :, i]).U
        end

        # Replace with upper triangle part
        Us_result = Array(Us)
        for i in 1:N
            Us_result[:, :, i] = UpperTriangular(Us_result[:, :, i])
        end
        max_error = maximum(abs.(Us_result .- Us_cpu))
        @test max_error < 1e-4
    end
end

@testitem "Cholesky Decomposition (out-of-place)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113
    nblocks = 2^8

    function kernel_cholesky!(
        Us, As, ::Val{D}, ::Val{nthreads}, N::Int32
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

        # Create dual-access matrices
        A = DualAccessMatrix(shmem_1, Val(D), wid, warp_matrix_id)
        U = DualAccessMatrix(shmem_2, Val(D), wid, warp_matrix_id)

        # Perform out-of-place Cholesky
        batch_op!(cholesky, U, A, d, Val(D), n_mats_per_warp, warp_matrix_id)

        # Store result
        dual_to_interm_transfer!(shmem_3, shmem_2, Val(D), Val(nthreads), N)
        intermediate_layout_write!(Us, shmem_3, Val(D), Val(nthreads), N)

        return nothing
    end

    for D in 2:10
        CUDA.seed!(1234)

        # Create symmetric positive definite matrices
        As = CUDA.zeros(Float32, D, D, N)
        for i in 1:N
            A_temp = CUDA.rand(Float32, D, D)
            As[:, :, i] = A_temp * A_temp' + 0.1f0 * I
        end
        Us = CUDA.zeros(Float32, D, D, N)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_cholesky!(
            Us, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
        )

        # CPU comparison
        As_cpu = Array(As)
        Us_cpu = similar(As_cpu)
        for i in 1:N
            Us_cpu[:, :, i] = cholesky(As_cpu[:, :, i]).U
        end

        # Replace with upper triangle part
        Us_result = Array(Us)
        for i in 1:N
            Us_result[:, :, i] = UpperTriangular(Us_result[:, :, i])
        end
        max_error = maximum(abs.(Us_result .- Us_cpu))
        @test max_error < 1e-4
    end
end

@testitem "Upper Triangular Backward Solve" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113
    nblocks = 2^8

    function kernel_backward_solve!(
        Cs, Us, Bs, ::Val{D}, ::Val{nthreads}, N::Int32
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

        # Load U
        intermediate_layout_load!(shmem_3, Us, Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)

        # Load B
        intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N)

        # Create dual-access matrices
        U_mat = DualAccessMatrix(shmem_1, Val(D), wid, warp_matrix_id)
        B = DualAccessMatrix(shmem_2, Val(D), wid, warp_matrix_id)
        C = DualAccessMatrix(shmem_3, Val(D), wid, warp_matrix_id)

        # Perform backward solve: C = U \ B
        U = UpperTriangular(U_mat)
        batch_op!(\, C, U, B, d, Val(D))

        # Store C
        dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)
        intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N)

        return nothing
    end

    for D in 2:10
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
            Cs, Us, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
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

@testitem "Lower Triangular Forward Solve" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113
    nblocks = 2^8

    function kernel_forward_solve!(
        Cs, Ls, Bs, ::Val{D}, ::Val{nthreads}, N::Int32
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

        # Load L
        intermediate_layout_load!(shmem_3, Ls, Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)

        # Load B
        intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N)

        # Create dual-access matrices
        L_mat = DualAccessMatrix(shmem_1, Val(D), wid, warp_matrix_id)
        B = DualAccessMatrix(shmem_2, Val(D), wid, warp_matrix_id)
        C = DualAccessMatrix(shmem_3, Val(D), wid, warp_matrix_id)

        # Perform forward solve: C = L \ B
        L = LowerTriangular(L_mat)
        batch_op!(\, C, L, B, d, Val(D))

        # Store C
        dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)
        intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N)

        return nothing
    end

    for D in 2:10
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
            Cs, Ls, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
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

@testitem "Transpose" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113
    nblocks = 2^8

    function kernel_transpose!(
        Bs, As, ::Val{D}, ::Val{nthreads}, N::Int32
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

        # Create dual-access matrices
        A = DualAccessMatrix(shmem_1, Val(D), wid, warp_matrix_id)
        B = DualAccessMatrix(shmem_2, Val(D), wid, warp_matrix_id)

        # Perform transpose
        batch_op!(transpose, B, A, d, Val(D))

        # Store B
        dual_to_interm_transfer!(shmem_3, shmem_2, Val(D), Val(nthreads), N)
        intermediate_layout_write!(Bs, shmem_3, Val(D), Val(nthreads), N)

        return nothing
    end

    for D in 2:10
        CUDA.seed!(1234)

        As = CUDA.rand(Float32, D, D, N)
        Bs = CUDA.zeros(Float32, D, D, N)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_transpose!(
            Bs, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
        )

        # CPU comparison
        As_cpu = Array(As)
        Bs_cpu = permutedims(As_cpu, (2, 1, 3))

        max_error = maximum(abs.(Array(Bs) .- Bs_cpu))
        @test max_error < 1e-5
    end
end
