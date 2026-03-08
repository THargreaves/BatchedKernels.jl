@testitem "Full Kalman" begin
    using CUDA
    using CUDA: i32
    using LinearAlgebra
    using BatchedKernels
    import StaticArrays: @MVector

    include("kalman_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^10 + 113

    # Test for both independent and consequtive modes
    modes = (Val(:indep), Val(:conseq))

    function cpu_kalman_cov(P, A, Q, H, R, µ, b, z)
        # Predict  step
        P_pred = A * P * A' + Q
        µ_interm = A * µ + b

        # Update step
        S = H * P_pred * H' + R
        K = P_pred * H' / S
        P_new = P_pred - K * S * K'
        # µ_interm2 = (I - K * H) * µ_interm + K * z
        µ_new = (I - K * H) * µ_interm + K * z

        return P_new, µ_new
    end

    for D in 2:13
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

        µ_cpu = rand(Float32, D, N)
        b_cpu = rand(Float32, D)
        z_cpu = rand(Float32, D, N)

        µ_gpu = cu(µ_cpu)
        b_gpu = cu(b_cpu)
        z_gpu = cu(z_cpu)

        for mode in modes
            P_out = CuArray{Float32}(undef, D, D, N)
            µ_out = CuArray{Float32}(undef, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_kalman!(
                P_out,
                P_in,
                A_gpu,
                Q_gpu,
                H_gpu,
                R_gpu,
                µ_out,
                µ_gpu,
                b_gpu,
                z_gpu,
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N),
                Val(:small),
                mode,
            )

            # Validate P_new (complete Kalman filter output)
            P_out_cpu = Array(P_out)
            µ_out_cpu = Array(µ_out)
            max_error_P = 0.0
            n_tested = min(10000, N)

            for i in 1:n_tested
                P_new_ref, µ_new_ref = cpu_kalman_cov(
                    P_cpu[:, :, i], A_elem, Q_elem, H_elem, R_elem, µ_cpu[:, i], b_cpu, z_cpu[:, i],
                )

                P_new_cpu = P_out_cpu[:, :, i]
                error_mat = maximum(abs.(P_new_ref - P_new_cpu))

                µ_new_cpu = µ_out_cpu[:, i]
                error_vec = maximum(abs.(µ_new_ref - µ_new_cpu))

                max_error_P = max(max_error_P, error_mat, error_vec)
            end

            @test max_error_P < 1e-5
        end
    end
end

@testitem "Kalman full (shared, vmap)" begin
    using BatchedKernels
    using GeneralisedFilters
    using LinearAlgebra
    using CUDA
    using CUDA: i32

    function kalman_filter(P, A, Q, H, R, µ, b, z)
        # Predict step
        P_pred = A * P * A' + Q
        µ_pred = A * µ + b

        # Kalman gain
        P_pred_H_trans = P_pred * H'
        S = H * P_pred_H_trans + R
        K = P_pred_H_trans / Symmetric(S)

        # Update step
        I_KH = I - K * H
        
        µ_new = I_KH * µ_pred + K * z

        P_new = I_KH * P_pred

        return P_new, µ_new
    end

    N = 2^10 + 113
    T = Float32

    for D in 2:13
        A_elem = rand(T, D, D) / T(D)
        Q_elem = rand(T, D, D) / T(D)^2
        Q_elem = Q_elem * Q_elem' + 0.01f0 * I

        H_elem = rand(T, D, D) / T(D)
        R_elem = rand(T, D, D) / T(D)^2
        R_elem = R_elem * R_elem' + 0.01f0 * I

        P_cpu = Array{T}(undef, D, D, N)
        for i in 1:N
            P_i = rand(T, D, D) / T(D)
            P_i = P_i * P_i' + 0.1f0 * I
            P_cpu[:, :, i] = P_i
        end

        P_in_gpu = CuArray(P_cpu)

        A_gpu = CuArray(A_elem)
        Q_gpu = CuArray(Q_elem)
        H_gpu = CuArray(H_elem)
        R_gpu = CuArray(R_elem)

        µ_cpu = rand(T, D, N)
        b_cpu = rand(T, D)
        z_cpu = rand(T, D, N)

        µ_gpu = cu(µ_cpu)
        b_gpu = cu(b_cpu)
        z_gpu = cu(z_cpu)

        P_in = BatchedCuMatrix(P_in_gpu)

        A = SharedCuMatrix(A_gpu, N)
        Q = SharedCuMatrix(Q_gpu, N)
        H = SharedCuMatrix(H_gpu, N)
        R = SharedCuMatrix(R_gpu, N)

        µ = BatchedCuVector(µ_gpu)
        b = SharedCuVector(b_gpu, N)
        z = BatchedCuVector(z_gpu)

        kalman_vmap = BatchedKernels.vmap(
            kalman_filter,
        )

        P_out, µ_out = kalman_vmap(P_in, A, Q, H, R, µ, b, z)
        P_out_cpu = Array(P_out.data)
        µ_out_cpu = Array(µ_out.data)

        max_error_P = 0.0
        for i in 1:N            
            P_new_ref, µ_new_ref = kalman_filter(
                P_cpu[:, :, i], A_elem, Q_elem, H_elem, R_elem, µ_cpu[:, i], b_cpu, z_cpu[:, i],
            )

            P_new_cpu = P_out_cpu[:, :, i]
            error_mat = maximum(abs.(P_new_ref - P_new_cpu))

            µ_new_cpu = µ_out_cpu[:, i]
            error_vec = maximum(abs.(µ_new_ref - µ_new_cpu))

            max_error_P = max(max_error_P, error_mat, error_vec)
        end

        @test max_error_P < 1e-5
    end
end

@testitem "Kalman full (all batched, vmap)" begin
    using BatchedKernels
    using GeneralisedFilters
    using LinearAlgebra
    using CUDA
    using CUDA: i32

    function kalman_filter(P, A, Q, H, R, µ, b, z)
        # Predict step
        P_pred = A * P * A' + Q
        µ_pred = A * µ + b

        # Kalman gain
        S = H * P_pred * H' + R
        K = P_pred * H' / Symmetric(S)

        # Update step
        I_KH = I - K * H
        
        µ_new = I_KH * µ_pred + K * z

        P_new = I_KH * P_pred

        return P_new, µ_new
    end

    N = 2^10 + 113
    T = Float32

    for D in 2:8
        A_cpu = rand(T, D, D, N) / T(D)
        Q_cpu = zeros(T, D, D, N)
        for i in 1:N
            Q_elem = rand(T, D, D) / T(D)^2
            Q_cpu[:, :, i] = Q_elem * Q_elem' + 0.01f0 * I
        end

        H_cpu = rand(T, D, D, N) / T(D)
        R_cpu = zeros(T, D, D, N)
        for i in 1:N
            R_elem = rand(T, D, D) / T(D)^2
            R_cpu[:, :, i] = R_elem * R_elem' + 0.01f0 * I
        end

        P_in_cpu = Array{T}(undef, D, D, N)
        for i in 1:N
            P_i = rand(T, D, D) / T(D)
            P_i = P_i * P_i' + 0.1f0 * I
            P_in_cpu[:, :, i] = P_i
        end

        P_in = BatchedCuMatrix(CuArray(P_in_cpu))

        A = BatchedCuMatrix(CuArray(A_cpu))
        Q = BatchedCuMatrix(CuArray(Q_cpu))
        H = BatchedCuMatrix(CuArray(H_cpu))
        R = BatchedCuMatrix(CuArray(R_cpu))

        µ_cpu = rand(T, D, N)
        b_cpu = rand(T, D, N)
        z_cpu = rand(T, D, N)

        µ = BatchedCuVector(cu(µ_cpu))
        b = BatchedCuVector(cu(b_cpu))
        z = BatchedCuVector(cu(z_cpu))

        kalman_vmap = BatchedKernels.vmap(kalman_filter)

        P_out, µ_out = kalman_vmap(P_in, A, Q, H, R, µ, b, z)
        P_out_cpu = Array(P_out.data)
        µ_out_cpu = Array(µ_out.data)

        max_error_P = 0.0
        for i in 1:N            
            P_new_ref, µ_new_ref = kalman_filter(
                P_in_cpu[:, :, i], A_cpu[:, :, i], Q_cpu[:, :, i], H_cpu[:, :, i], R_cpu[:, :, i], µ_cpu[:, i], b_cpu[:, i], z_cpu[:, i],
            )

            P_new_cpu = P_out_cpu[:, :, i]
            error_mat = maximum(abs.(P_new_ref - P_new_cpu))

            µ_new_cpu = µ_out_cpu[:, i]
            error_vec = maximum(abs.(µ_new_ref - µ_new_cpu))

            max_error_P = max(max_error_P, error_mat, error_vec)
        end

        @test max_error_P < 1e-5
    end
end

@testitem "Kalman predict" begin
    using CUDA
    using CUDA: i32
    using LinearAlgebra
    using BatchedKernels
    import StaticArrays: @MVector

    include("kalman_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^20 + 113

    # Test for both independent and consequtive modes
    modes = (Val(:indep), Val(:conseq))

    function cpu_kalman_predict(P, A, Q, µ, b)
        P_pred = A * P * A' + Q
        µ_new = A * µ + b

        return P_pred, µ_new
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
            µ_out = CuArray{Float32}(undef, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_kalman_predict!(
                P_out,
                P_in,
                A_gpu,
                Q_gpu,
                µ_out,
                µ_gpu,
                b_gpu,
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N),
                Val(:small),
                mode,
            )

            # Validate P_new (complete Kalman filter output)
            P_out_cpu = Array(P_out)
            µ_out_cpu = Array(µ_out)
            max_error_P = 0.0
            n_tested = min(10000, N)

            for i in 1:n_tested
                P_new_ref, µ_ref = cpu_kalman_predict(
                    P_cpu[:, :, i], A_elem, Q_elem, µ_cpu[:, i], b_cpu,
                )
                P_new_cpu = P_out_cpu[:, :, i]
                error_mat = maximum(abs.(P_new_ref - P_new_cpu))

                µ_new_cpu = µ_out_cpu[:, i]
                error_vec = maximum(abs.(µ_ref - µ_new_cpu))

                max_error_P = max(max_error_P, error_mat, error_vec)
            end

            @test max_error_P < 1e-5
        end
    end
end

@testitem "Kalman update" begin
    using CUDA
    using CUDA: i32
    using LinearAlgebra
    using BatchedKernels
    import StaticArrays: @MVector

    include("kalman_kernels.jl")

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

        µ_new = (I - K * H) * x + K * z

        return P_new, µ_new
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

        µ_cpu = rand(Float32, D, N)
        z_cpu = rand(Float32, D, N)
        µ_gpu = cu(µ_cpu)
        z_gpu = cu(z_cpu)

        for mode in modes
            P_out = CuArray{Float32}(undef, D, D, N)
            µ_out = CuArray{Float32}(undef, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_kalman_update!(
                P_out,
                P_in,
                H_gpu,
                R_gpu,
                µ_out,
                µ_gpu,
                z_gpu,
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N),
                Val(:small),
                mode,
            )

            # Validate P_new (complete Kalman filter output)
            P_out_cpu = Array(P_out)
            µ_out_cpu = Array(µ_out)
            max_error_P = 0.0
            n_tested = min(10000, N)

            for i in 1:n_tested
                P_new_ref, µ_new_ref = cpu_kalman_update(
                    P_cpu[:, :, i], H_elem, R_elem, µ_cpu[:, i], z_cpu[:, i],
                )
                P_new_cpu = P_out_cpu[:, :, i]
                error_mat = maximum(abs.(P_new_ref - P_new_cpu))

                µ_new_cpu = µ_out_cpu[:, i]
                error_vec = maximum(abs.(µ_new_ref - µ_new_cpu))

                max_error_P = max(max_error_P, error_mat, error_vec)
            end

            @test max_error_P < 1e-5
        end
    end
end

@testitem "Kalman predict (inside kernel destructuring)" begin
    using CUDA
    using CUDA: i32
    using LinearAlgebra
    using BatchedKernels
    using GeneralisedFilters
    using PDMats
    using Distributions
    using Magma
    
    Magma.magma_init()

    function kalman_predict!(
        out_state,
        state,
        dyn_params,
        ::Val{D_state},
        ::Val{D},
        ::Val{nthreads},
        N::Int32,
    ) where {D_state,D,nthreads}
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        n_mats_per_block = n_warps * n_mats_per_warp
        dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        bid = blockIdx().x
        lid = mod1(tid, 32i32)
        wid = div(tid - 1i32, 32i32) + 1i32
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)
        grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_mat_elems = warp_shmem_size * n_warps
        shmem_vec_elems = D * n_warps * n_mats_per_warp
        shmem_M1 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
        shmem_M2 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
        shmem_M3 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
        shmem_v1 = CuStaticSharedArray(Float32, (shmem_vec_elems,))
        shmem_v2 = CuStaticSharedArray(Float32, (shmem_vec_elems,))

        M1 = DualAccessMatrix(shmem_M1, Val(D), warp_matrix_id, Val(:small))
        M2 = DualAccessMatrix(shmem_M2, Val(D), warp_matrix_id, Val(:small))
        M3 = DualAccessMatrix(shmem_M3, Val(D), warp_matrix_id, Val(:small))
        v1 = BatchedVector(shmem_v1, Val(D), warp_matrix_id)
        v2 = BatchedVector(shmem_v2, Val(D), warp_matrix_id)

        # Extracting inputs' fields
        μ_glob, Σ_glob = state.µ, state.Σ
        A_glob, b_glob, Q_glob = dyn_params.components.x1, dyn_params.components.x2, dyn_params.components.x3

        # Load v1 <- µ
        vector_load!(shmem_v1, μ_glob.data, Val(D_state), Val(D), Val(nthreads), N)

        # Load M1 <- A
        intermediate_layout_load!(shmem_M2, A_glob.data, Val(D_state), Val(D_state), Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_M1, shmem_M2, Val(D_state), Val(D_state), Val(D), Val(nthreads), N, Val(:small))

        # v2 <- A * µ
        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            A = M1
            µ = v1

            batch_op!(*, v2, A, µ, d, Val(D_state), Val(D_state), Val(D), Val(:small))
        end

        # Load v1 <- b
        vector_load!(shmem_v1, b_glob.data, Val(D_state), Val(D), Val(nthreads), N)

        # v1 <- (A * µ) + b = v2 + v1
        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            b = v1
            batch_op!(+, v1, v2, b, d, Val(D_state), Val(:small))
        end

        # Load M2 <- L = chol(Σ).L
        intermediate_layout_load!(shmem_M3, Σ_glob.components.chol.factors.data.data, Val(D_state), Val(D_state), Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_M2, shmem_M3, Val(D_state), Val(D_state), Val(D), Val(nthreads), N, Val(:small))

        # M1 = A Σ A' = (A L) * (A L)'
        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            A = M1
            L = M2

            # M3 <- A * L
            batch_op!(*, M3, A, LowerTriangular(L), d, Val(D_state), Val(D_state), Val(D), Val(:small))

            # M1 <- (AL) * (AL)'
            batch_op!(BatchedKernels.gram, M1, M3', d, Val(D_state), Val(D_state), Val(D), Val(:small))
        end

        # Load M2 <- Q
        intermediate_layout_load!(shmem_M3, Q_glob.components.mat.data, Val(D_state), Val(D_state), Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_M2, shmem_M3, Val(D_state), Val(D_state), Val(D), Val(nthreads), N, Val(:small))

        # M2 <- (A Σ A') + Q = M1 + Q
        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            Q = M2
            batch_op!(+, M2, M1, Q, d, Val(D_state), Val(D_state), Val(D), Val(:small))
        end

        # M3 <- L_out = chol(M2) = chol((A Σ A') + Q))
        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            batch_op!(cholesky, M3', M2, d, Val(D_state), Val(D), warp_matrix_id, Val(:small))
        end
        
        # Storing Σs mats
        dual_to_interm_transfer!(shmem_M1, M2, Val(D_state), Val(D_state), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(out_state.components.Σ.components.mat.data, shmem_M1, Val(D_state), Val(D_state), Val(D), Val(nthreads), N, Val(:small))

        # Storing Σs choleskys
        dual_to_interm_transfer!(shmem_M1, LowerTriangular(M3), Val(D_state), Val(D_state), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(out_state.components.Σ.components.chol.factors.data.data, shmem_M1, Val(D_state), Val(D_state), Val(D), Val(nthreads), N, Val(:small))

        # Storing µs
        vector_write!(out_state.µ.data, shmem_v1, Val(D_state), Val(D), Val(nthreads), N)

        return nothing
    end

    function kalman_predict_batched(state, dyn_params)
        μ, Σ = state.µ, state.Σ.mat
        A, b, Q = dyn_params.components.x1, dyn_params.components.x2, dyn_params.components.x3.mat
        
        Σ = X_A_Xt.(Σ, A) .+ Q
        Σ_PDs = PDMat.(Σ)
        µ = A .* μ .+ b

        return MvNormal.(µ, Σ_PDs)
    end

    # Test parameters
    nthreads = 2^8
    N = 2^9 + 113

    for D_state in 2:13
        for extra in 0:1
            D = D_state + extra
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            μs = BatchedCuVector(CUDA.randn(Float32, D_state, N))
            Σs_root = BatchedCuMatrix(CUDA.randn(Float32, D_state, D_state, N))
            Σs = Σs_root .* adjoint.(Σs_root) .+ Ref(I)

            As = BatchedCuMatrix(CUDA.randn(Float32, D_state, D_state, N))
            bs = BatchedCuVector(CUDA.randn(Float32, D_state, N))
            Q_root = BatchedCuMatrix(CUDA.randn(Float32, D_state, D_state, N))
            Qs = Q_root .* adjoint.(Q_root) .+ Ref(I)
            Qs = PDMat.(Qs)

            Σ_PDs = PDMat.(Σs)
            Gs = MvNormal.(μs, Σ_PDs)

            dyn_params = tuple.(As, bs, Qs)

            µs_out = BatchedCuVector(CUDA.zeros(Float32, D_state, N))
            Σs_out = BatchedCuMatrix(CUDA.zeros(Float32, D_state, D_state, N))
            Σ_PDs_out = PDMat.(Σs_out)
            out_state = MvNormal.(μs_out, Σ_PDs_out)

            CUDA.@sync @cuda threads=nthreads blocks=nblocks kalman_predict!(
                out_state,
                Gs,
                dyn_params,
                Val(Int32(D_state)),
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N),
            )

            out_real = kalman_predict_batched(Gs, dyn_params)

            max_error = 0.0
            for i in 1:N
                µ_real = out_real.µ[i]
                Σ_real = out_real.Σ.mat[i]
                L_real = LowerTriangular(out_real.Σ.chol.factors.data[i])

                µ = out_state.µ[i]
                Σ = out_state.Σ.mat[i]
                L = out_state.Σ.chol.factors.data[i]

                µ_error = maximum(abs.(µ_real .- µ))
                Σ_error = maximum(abs.(Σ_real .- Σ))
                L_error = maximum(abs.(L_real .- L))

                max_error = max(µ_error, Σ_error, L_error)
            end

            @test max_error < 1e-3
        end
    end
end

@testitem "Kalman Throughput (indep)" begin
    using PerformanceTestTools

    PerformanceTestTools.@include("throughput_script.jl")
end