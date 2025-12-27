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

@testitem "Kalman full (vmap)" begin
    using BatchedKernels
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

        P_in = CuArray(P_cpu)

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

        kalman_vmap = BatchedKernels.vmap(
            kalman_filter,
            in_type = (:batched, :shared, :shared, :shared, :shared, :batched, :shared, :batched),
        )

        P_out, µ_out = kalman_vmap(P_in, A_gpu, Q_gpu, H_gpu, R_gpu, µ_gpu, b_gpu, z_gpu)
        P_out_cpu = Array(P_out)
        µ_out_cpu = Array(µ_out)

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

@testitem "Kalman Throughput (indep)" begin
    using PerformanceTestTools

    PerformanceTestTools.@include("throughput_script.jl")
end