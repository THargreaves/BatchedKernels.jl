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
                Cs, Us, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
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
                Cs, Us, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N), Val(:small),
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
                Bs, Us, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N), Val(:small),
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
                Us, Us, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
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
                Cs, Ls, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
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
                Cs, Ls, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N), Val(:small),
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
                Bs, Ls, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N), Val(:small),
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
                Ls, Ls, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
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
        cs,
        Ls,
        bs,
        ::Val{D1},
        ::Val{D},
        ::Val{nthreads},
        N::Int32,
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
        grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_elems = warp_shmem_size * n_warps
        shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
        shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))

        shmem_vec_elems = D * n_warps * n_mats_per_warp
        shmem_vec_1 = CuDynamicSharedArray(Float32, shmem_vec_elems, 2 * shmem_elems * sizeof(Float32))

        intermediate_layout_load!(shmem_2, Ls, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_1, shmem_2, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))

        vector_load!(shmem_vec_1, bs, Val(D1), Val(D), Val(nthreads), N)

        M1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        v1 = BatchedVector(shmem_vec_1, Val(D), warp_matrix_id)

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            batch_op!(\, v1, LowerTriangular(M1), v1, d, Val(D1), Val(D), Val(D), warp_matrix_id, Val(:small))
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
                cs,
                Ls,
                bs,
                Val(Int32(D1)),
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N),
            )
            CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

            CUDA.@sync kernel(
                cs, Ls, bs,
                Val(Int32(D1)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N);
                threads = nthreads, blocks = nblocks, shmem = shmem_bytes,
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

@testitem "Symmetric solve (vmap)" begin
    using BatchedKernels
    using GeneralisedFilters
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^12 + 113

    function solve(A, B, dummy)
        return B / Symmetric(A)
    end

    for D1 in 2:13
        for D2 in 2:13
            Dmax = max(D1, D2)
            for extra in 0:2
                D = Dmax + extra
                dummy = BatchedCuMatrix(CUDA.zeros(Float32, D, D, N))

                CUDA.seed!(1234)

                As = CUDA.zeros(Float32, D1, D1, N)
                for i in 1:N
                    A_temp = CUDA.rand(Float32, D1, D1)
                    As[:, :, i] = A_temp * A_temp' + 0.1f0 * I
                end
                As = BatchedCuMatrix(As)

                Bs = BatchedCuMatrix(CUDA.rand(Float32, D2, D1, N))

                As_cpu = Array(As.data)
                Bs_cpu = Array(Bs.data)

                solve_vmap = BatchedKernels.vmap(solve)
                Cs = solve_vmap(As, Bs, dummy)
                Cs_result = Array(Cs.data)

                Cs_cpu = similar(Bs_cpu)
                for i in 1:N
                    Cs_cpu[:, :, i] = solve(As_cpu[:, :, i], Bs_cpu[:, :, i], 0)
                end

                max_error = maximum(abs.(Cs_result .- Cs_cpu))

                @test max_error < 1e-3
            end
        end
    end
end

@testitem "Symmetric solve (shared, vmap)" begin
    using BatchedKernels
    using GeneralisedFilters
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^12 + 113

    function solve(A, B, dummy)
        return B / Symmetric(A)
    end

    for D1 in 2:13
        for D2 in 2:13
            Dmax = max(D1, D2)
            for extra in 0:1
                D = Dmax + extra
                dummy = BatchedCuMatrix(CUDA.zeros(Float32, D, D, N))

                CUDA.seed!(1234)

                # Make A shared
                A_temp = CUDA.rand(Float32, D1, D1)
                A = SharedCuMatrix(A_temp * A_temp' + 0.1f0 * I, N)

                Bs = BatchedCuMatrix(CUDA.rand(Float32, D2, D1, N))

                A_cpu = Array(A.data)
                Bs_cpu = Array(Bs.data)

                solve_vmap = BatchedKernels.vmap(solve)
                Cs = solve_vmap(A, Bs, dummy)
                Cs_result = Array(Cs.data)

                Cs_cpu = similar(Bs_cpu)
                for i in 1:N
                    Cs_cpu[:, :, i] = solve(A_cpu, Bs_cpu[:, :, i], 0)
                end

                max_error = maximum(abs.(Cs_result .- Cs_cpu))

                @test max_error < 1e-3


                # Make B shared
                CUDA.seed!(1234)

                As = CUDA.zeros(Float32, D1, D1, N)
                for i in 1:N
                    A_temp = CUDA.rand(Float32, D1, D1)
                    As[:, :, i] = A_temp * A_temp' + 0.1f0 * I
                end
                As = BatchedCuMatrix(As)

                B = SharedCuMatrix(CUDA.rand(Float32, D2, D1), N)

                As_cpu = Array(As.data)
                B_cpu = Array(B.data)

                solve_vmap = BatchedKernels.vmap(solve)
                Cs = solve_vmap(As, B, dummy)
                Cs_result = Array(Cs.data)

                Cs_cpu = zeros(Float32, D2, D1, N)
                for i in 1:N
                    Cs_cpu[:, :, i] = solve(As_cpu[:, :, i], B_cpu, 0)
                end

                max_error = maximum(abs.(Cs_result .- Cs_cpu))

                @test max_error < 1e-3
            end
        end
    end
end

@testitem "Non-symmetric solve (vmap)" begin
    using BatchedKernels
    using GeneralisedFilters
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^9

    function solve(X, Y, dummy)
        return X / Y
    end

    count = 0
    for Y_D1 in 2:10
        for Y_D2 in 2:10
            for X_D1 in 2:10
                X_D2 = Y_D2
                for extra in 0:1
                    global count
                    count += 1

                    D = max(Y_D1, Y_D2, X_D1, X_D2) + extra

                    CUDA.seed!(1234)

                    Xs = BatchedCuMatrix(CUDA.rand(Float32, X_D1, X_D2, N))
                    Ys = CUDA.rand(Float32, Y_D1, Y_D2, N)
                    for j in 1:min(Y_D1, Y_D2)
                        @views Ys[j, j, :] .+= 2f0
                    end
                    Ys = BatchedCuMatrix(Ys)
                    dummy = BatchedCuMatrix(CUDA.zeros(Float32, D, D, N))

                    solve_vmap = BatchedKernels.vmap(solve)
                    Zs = solve_vmap(Xs, Ys, dummy)

                    Xs_cpu = Array(Xs.data)
                    Ys_cpu = Array(Ys.data)
                    Zs_cpu = Array(Zs.data)

                    Zs_real = zeros(Float32, X_D1, Y_D1, N)
                    for i in 1:N
                        Zs_real[:, :, i] = solve(Xs_cpu[:, :, i], Ys_cpu[:, :, i], 0)
                    end

                    max_error = maximum(abs.(Zs_real - Zs_cpu))

                    @test max_error < 1e-3 && !isnan(max_error)

                    println("[$count/1458]: Y=($Y_D1,$Y_D2), X=($X_D1,$X_D2), max_error=$max_error")
                end
            end
        end
    end
end