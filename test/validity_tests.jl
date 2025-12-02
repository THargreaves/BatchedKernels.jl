@testitem "Kalman (small)" begin
    using CUDA
    using CUDA: i32
    using LinearAlgebra
    using BatchedKernels
    import StaticArrays: @MVector

    # Test parameters
    nthreads = 2^8
    N = 2^20 + 113

    # Test for both independent and consequtive modes
    modes = (Val(:indep), Val(:conseq))

    function cpu_kalman_cov(P, A, Q, H, R)
        # Predict step
        P_pred = A * P * A' + Q
        # Update step
        S = H * P_pred * H' + R
        K = P_pred * H' / S
        P_new = P_pred - K * S * K'
        return P_new, P_pred, K
    end

    for D in 2:14
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        # Generate test data
        A_elem = rand(Float32, D, D) / Float32(D)
        Q_elem = rand(Float32, D, D) / Float32(D)^2
        Q_elem = Q_elem * Q_elem' + 0.01f0 * I

        H_elem = rand(Float32, D, D) / Float32(D)
        R_elem = rand(Float32, D, D) / Float32(D)^2
        R_elem = R_elem * R_elem' + 0.01f0 * I

        P_cpu = Array{Float32}(undef, D, D, N)
        for i in 1:N
            P_i = rand(Float32, D, D) / Float32(D)
            P_i = P_i * P_i' + 0.1f0 * I
            P_cpu[:, :, i] = P_i
        end

        P_in = CuArray(P_cpu)

        A_gpu = CuArray(A_elem)
        Q_gpu = CuArray(Q_elem)
        H_gpu = CuArray(H_elem)
        R_gpu = CuArray(R_elem)

        for mode in modes
            P_out = CuArray{Float32}(undef, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_kalman!(
                P_out,
                P_in,
                A_gpu,
                Q_gpu,
                H_gpu,
                R_gpu,
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N),
                Val(:small),
                mode,
            )

            # Validate P_new (complete Kalman filter output)
            P_out_cpu = Array(P_out)
            max_error_P = 0.0
            n_tested = min(10000, N)

            for i in 1:n_tested
                P_new_ref, P_pred_ref, K_ref = cpu_kalman_cov(
                    P_cpu[:, :, i], A_elem, Q_elem, H_elem, R_elem
                )
                P_new_gpu = P_out_cpu[:, :, i]
                error = maximum(abs.(P_new_ref - P_new_gpu))
                max_error_P = max(max_error_P, error)
            end

            @test max_error_P < 1e-5
        end
    end
end

@testitem "Kalman predict (small)" begin
    using CUDA
    using CUDA: i32
    using LinearAlgebra
    using BatchedKernels
    import StaticArrays: @MVector

    # Test parameters
    nthreads = 2^8
    N = 2^20 + 113

    # Test for both independent and consequtive modes
    modes = (Val(:indep), Val(:conseq))

    function cpu_kalman_predict(P, A, Q, µ, b)
        P_pred = A * P * A' + Q
        x = A * µ + b

        return P_pred, x
    end

    for D in 2:14
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        # Generate test data
        A_elem = rand(Float32, D, D) / Float32(D)
        Q_elem = rand(Float32, D, D) / Float32(D)^2
        Q_elem = Q_elem * Q_elem' + 0.01f0 * I

        P_cpu = Array{Float32}(undef, D, D, N)
        for i in 1:N
            P_i = rand(Float32, D, D) / Float32(D)
            P_i = P_i * P_i' + 0.1f0 * I
            P_cpu[:, :, i] = P_i
        end

        P_in = CuArray(P_cpu)

        A_gpu = CuArray(A_elem)
        Q_gpu = CuArray(Q_elem)

        µ_cpu = rand(Float32, D, N)
        b_cpu = rand(Float32, D)
        µ_gpu = cu(µ_cpu)
        b_gpu = cu(b_cpu)

        for mode in modes
            P_out = CuArray{Float32}(undef, D, D, N)
            x_out = CuArray{Float32}(undef, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_kalman_predict!(
                P_out,
                P_in,
                A_gpu,
                Q_gpu,
                µ_gpu,
                b_gpu,
                x_out,
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N),
                Val(:small),
                mode,
            )

            # Validate P_new (complete Kalman filter output)
            P_out_cpu = Array(P_out)
            x_out_cpu = Array(x_out)
            max_error_P = 0.0
            n_tested = min(10000, N)

            for i in 1:n_tested
                P_new_ref, x_ref = cpu_kalman_predict(
                    P_cpu[:, :, i], A_elem, Q_elem, µ_cpu[:, i], b_cpu,
                )
                P_new_cpu = P_out_cpu[:, :, i]
                error_mat = maximum(abs.(P_new_ref - P_new_cpu))

                x_new_cpu = x_out_cpu[:, i]
                error_vec = maximum(abs.(x_ref - x_new_cpu))

                max_error_P = max(max_error_P, error_mat, error_vec)
            end

            @test max_error_P < 1e-5
        end
    end
end

@testitem "Kalman update (small)" begin
    using CUDA
    using CUDA: i32
    using LinearAlgebra
    using BatchedKernels
    import StaticArrays: @MVector

    # Test parameters
    nthreads = 2^8
    N = 2^20 + 113

    # Test for both independent and consequtive modes
    modes = (Val(:indep), Val(:conseq))

    function cpu_kalman_update(P_pred, H, R, x, z)
        # Update step
        S = H * P_pred * H' + R
        K = P_pred * H' / S
        P_new = P_pred - K * S * K'

        x_kk = (I - K * H) * x + K * z
        y = z - H * x_kk

        return P_new, y
    end

    for D in 2:14
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        # Generate test data
        A_elem = rand(Float32, D, D) / Float32(D)
        Q_elem = rand(Float32, D, D) / Float32(D)^2
        Q_elem = Q_elem * Q_elem' + 0.01f0 * I

        P_cpu = Array{Float32}(undef, D, D, N)
        for i in 1:N
            P_i = rand(Float32, D, D) / Float32(D)
            P_i = P_i * P_i' + 0.1f0 * I
            P_cpu[:, :, i] = P_i
        end

        P_in = CuArray(P_cpu)

        H_elem = rand(Float32, D, D) / Float32(D)
        R_elem = rand(Float32, D, D) / Float32(D)^2
        R_elem = R_elem * R_elem' + 0.01f0 * I

        H_gpu = CuArray(H_elem)
        R_gpu = CuArray(R_elem)

        x_cpu = rand(Float32, D, N)
        z_cpu = rand(Float32, D, N)
        x_gpu = cu(x_cpu)
        z_gpu = cu(z_cpu)

        for mode in modes
            P_out = CuArray{Float32}(undef, D, D, N)
            y_out = CuArray{Float32}(undef, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_kalman_update!(
                P_out,
                P_in,
                H_gpu,
                R_gpu,
                y_out,
                x_gpu,
                z_gpu,
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N),
                Val(:small),
                mode,
            )

            # Validate P_new (complete Kalman filter output)
            P_out_cpu = Array(P_out)
            y_out_cpu = Array(y_out)
            max_error_P = 0.0
            n_tested = min(10000, N)

            for i in 1:n_tested
                P_new_ref, y_ref = cpu_kalman_update(
                    P_cpu[:, :, i], H_elem, R_elem, x_cpu[:, i], z_cpu[:, i],
                )
                P_new_cpu = P_out_cpu[:, :, i]
                error_mat = maximum(abs.(P_new_ref - P_new_cpu))

                y_new_cpu = y_out_cpu[:, i]
                error_vec = maximum(abs.(y_ref - y_new_cpu))
                if any(isnan, P_new_ref)
                    println("P_new_ref")
                elseif any(isnan, P_new_cpu)
                    println("P_new_cpu")
                elseif any(isnan, y_ref)
                    println("y_ref")
                elseif any(isnan, y_new_cpu)
                    println("y_new_cpu")
                end

                max_error_P = max(max_error_P, error_mat, error_vec)
            end

            @test max_error_P < 1e-5
        end
    end
end

@testitem "Matrix Multiplication (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^12 + 113
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
                    Cs,
                    As,
                    Bs,
                    Val(A_adj),
                    Val(B_adj),
                    Val(Int32(D)),
                    Val(Int32(nthreads)),
                    Int32(N),
                    Val(:small),
                    mode,
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
                    Cs,
                    As,
                    Bs,
                    Val(A_adj),
                    Val(B_adj),
                    Val(Int32(D)),
                    Val(Int32(nthreads)),
                    Int32(N),
                    Val(:large),
                    mode,
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

@testitem "Matrix Subtraction (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    function kernel_sub!(Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{mode}) where {D,nthreads,mode}
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

@testitem "Cholesky Decomposition (in-place) (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            # Create symmetric positive definite matrices
            As = CUDA.zeros(Float32, D, D, N)
            for i in 1:N
                A_temp = CUDA.rand(Float32, D, D)
                As[:, :, i] = A_temp * A_temp' + 0.1f0 * I
            end
            Us = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_cholesky_inplace!(
                Us, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
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
end

@testitem "Cholesky Decomposition (out-of-place) (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            # Create symmetric positive definite matrices
            As = CUDA.zeros(Float32, D, D, N)
            for i in 1:N
                A_temp = CUDA.rand(Float32, D, D)
                As[:, :, i] = A_temp * A_temp' + 0.1f0 * I
            end
            Us = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_cholesky_out_of_place!(
                Us, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
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
end

@testitem "Upper Triangular Backward Solve (out-of-place) (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

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

@testitem "Lower Triangular Forward Solve (out-of-place) (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

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

@testitem "Transpose (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    function kernel_transpose!(
        Bs, As, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{mode},
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

            # Perform transpose
            batch_op!(transpose, B, A, d, Val(D), Val(:small))
        end

        # Store B
        dual_to_interm_transfer!(shmem_3, shmem_2, Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Bs, shmem_3, Val(D), Val(nthreads), N, Val(:small), Val(mode))

        return nothing
    end

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            As = CUDA.rand(Float32, D, D, N)
            Bs = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_transpose!(
                Bs, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), mode,
            )

            # CPU comparison
            As_cpu = Array(As)
            Bs_cpu = permutedims(As_cpu, (2, 1, 3))

            max_error = maximum(abs.(Array(Bs) .- Bs_cpu))
            @test max_error < 1e-5
        end
    end
end

@testitem "Vector Addition" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113
    nblocks = 2^8

    function kernel_vec_add!(
        zs, xs, ys, ::Val{D}, ::Val{nthreads}, N::Int32
    ) where {D,nthreads}
        n_vecs_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32

        tid = threadIdx().x
        wid = div(tid - 1i32, 32i32) + 1i32
        lid = mod1(tid, 32i32)
        warp_vector_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)

        shmem_elems = D * n_warps * n_vecs_per_warp
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load x and y
        vector_load!(shmem_1, xs, Val(D), Val(nthreads), N)
        vector_load!(shmem_2, ys, Val(D), Val(nthreads), N)

        # Create batched vectors
        x = BatchedVector(shmem_1, Val(D), warp_vector_id)
        y = BatchedVector(shmem_2, Val(D), warp_vector_id)
        z = BatchedVector(shmem_3, Val(D), warp_vector_id)

        # Perform operation
        batch_op!(+, z, x, y, d, Val(D))

        # Store z
        vector_write!(zs, shmem_3, Val(D), Val(nthreads), N)

        return nothing
    end

    for D in 2:10
        CUDA.seed!(1234)

        xs = CUDA.rand(Float32, D, N)
        ys = CUDA.rand(Float32, D, N)
        zs = CUDA.zeros(Float32, D, N)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_vec_add!(
            zs, xs, ys, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
        )

        # CPU comparison
        xs_cpu = Array(xs)
        ys_cpu = Array(ys)
        zs_cpu = xs_cpu .+ ys_cpu

        max_error = maximum(abs.(Array(zs) .- zs_cpu))
        @test max_error < 1e-5
    end
end

@testitem "Vector Subtraction" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113
    nblocks = 2^8

    function kernel_vec_sub!(
        zs, xs, ys, ::Val{D}, ::Val{nthreads}, N::Int32
    ) where {D,nthreads}
        n_vecs_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32

        tid = threadIdx().x
        wid = div(tid - 1i32, 32i32) + 1i32
        lid = mod1(tid, 32i32)
        warp_vector_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)

        shmem_elems = D * n_warps * n_vecs_per_warp
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load x and y
        vector_load!(shmem_1, xs, Val(D), Val(nthreads), N)
        vector_load!(shmem_2, ys, Val(D), Val(nthreads), N)

        # Create batched vectors
        x = BatchedVector(shmem_1, Val(D), warp_vector_id)
        y = BatchedVector(shmem_2, Val(D), warp_vector_id)
        z = BatchedVector(shmem_3, Val(D), warp_vector_id)

        # Perform operation
        batch_op!(-, z, x, y, d, Val(D))

        # Store z
        vector_write!(zs, shmem_3, Val(D), Val(nthreads), N)

        return nothing
    end

    for D in 2:10
        CUDA.seed!(1234)

        xs = CUDA.rand(Float32, D, N)
        ys = CUDA.rand(Float32, D, N)
        zs = CUDA.zeros(Float32, D, N)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_vec_sub!(
            zs, xs, ys, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
        )

        # CPU comparison
        xs_cpu = Array(xs)
        ys_cpu = Array(ys)
        zs_cpu = xs_cpu .- ys_cpu

        max_error = maximum(abs.(Array(zs) .- zs_cpu))
        @test max_error < 1e-5
    end
end

@testitem "Matrix-Vector Multiplication" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113
    nblocks = 2^8

    function kernel_matvec!(
        ys, As, xs, A_adj::Bool, ::Val{D}, ::Val{nthreads}, N::Int32
    ) where {D,nthreads}
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)

        shmem_mat_elems = (n_mats_per_warp * D + padding) * D * n_warps
        shmem_vec_elems = D * n_warps * n_mats_per_warp
        shmem_1 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_vec_elems,))
        shmem_4 = CuStaticSharedArray(Float32, (shmem_vec_elems,))

        # Load A
        intermediate_layout_load!(shmem_2, As, Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_1, shmem_2, Val(D), Val(nthreads), N, Val(:small))

        # Load x
        vector_load!(shmem_3, xs, Val(D), Val(nthreads), N)

        # Create dual-access matrix and batched vectors
        if warp_matrix_id <= n_mats_per_warp
            A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
            x = BatchedVector(shmem_3, Val(D), warp_matrix_id)
            y = BatchedVector(shmem_4, Val(D), warp_matrix_id)

            # Perform operation: y = A * x or y = A' * x
            if A_adj
                batch_op!(*, y, A', x, d, Val(D))
            else
                batch_op!(*, y, A, x, d, Val(D))
            end
        end

        # Store y
        vector_write!(ys, shmem_4, Val(D), Val(nthreads), N)

        return nothing
    end

    # Test both A * x and A' * x
    test_cases = [false, true]

    for D in 2:10
        CUDA.seed!(1234)

        As = CUDA.rand(Float32, D, D, N)
        xs = CUDA.rand(Float32, D, N)
        As_cpu = Array(As)
        xs_cpu = Array(xs)

        for A_adj in test_cases
            ys = CUDA.zeros(Float32, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_matvec!(
                ys, As, xs, A_adj, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
            )

            # CPU comparison
            ys_cpu = similar(xs_cpu)
            for i in 1:N
                if A_adj
                    ys_cpu[:, i] = As_cpu[:, :, i]' * xs_cpu[:, i]
                else
                    ys_cpu[:, i] = As_cpu[:, :, i] * xs_cpu[:, i]
                end
            end

            max_error = maximum(abs.(Array(ys) .- ys_cpu))
            @test max_error < 1e-5
        end
    end
end
