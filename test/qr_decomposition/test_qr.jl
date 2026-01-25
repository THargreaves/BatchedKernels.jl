@testitem "QR Decomposition (vmap, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^12 + 113

    function qr_decomp(A)
        res = qr(A)
        Q = res.Q
        R = res.R
        return Q, R
    end

    N = 2^12 + 113

    for D1 in 2:13
        for D2 in 2:D1
            for extra in 0:0
                D = max(D1, D2) + extra

                CUDA.seed!(1234)

                # Create symmetric positive definite matrices
                # This is numerically unstable if A^T A is near-singular.
                # TODO: Implement a direct QR decomposition that is more stable
                As = CUDA.rand(Float32, D1, D2, N)
                As_cpu = Array(As)
                dummy = CUDA.rand(Float32, D, D, N)

                qr_vmap = BatchedKernels.vmap(qr_decomp)
                Q, R = qr_vmap(As)
                Qs_result = Array(Q)
                Rs_result = Array(R)

                # Reconstruction comparison
                max_Q_error = 0.0
                max_recon_error = 0.0
                for i in 1:N
                    Q_error = maximum(abs.(I - Qs_result[:, :, i]' * Qs_result[:, :, i]))
                    recon_error = maximum(abs.(As_cpu[:, :, i] - Qs_result[:, :, i] * Rs_result[:, :, i]))

                    max_Q_error = max(max_Q_error, Q_error)
                    max_recon_error = max(max_recon_error, recon_error)
                end
                max_error = max(max_Q_error, max_recon_error)
                @test max_error < 1e-3 && !isnan(max_error)
            end
        end
    end
end