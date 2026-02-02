@testitem "QR decomposition (out-of-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    function qr_kernel!(
        Rs,
        Qthins,
        Qfulls,
        As,
        ::Val{D1},
        ::Val{D2},
        ::Val{D},
        ::Val{nthreads},
        N::Int32
    ) where {D1,D2,D,nthreads}
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
        shmem_4 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load A
        intermediate_layout_load!(shmem_4, As, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_1, shmem_4, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        R = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        Qfull = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))
        Qthin = DualAccessMatrix(shmem_4, Val(D), warp_matrix_id, Val(:small))

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            tau = batch_op!(qr, R, A, d, Val(D1), Val(D2), Val(D), warp_matrix_id, Val(:small))
            batch_op!(Val(:qr_Q_thin), Qthin, R, d, tau, Val(D1), Val(D2), Val(D), warp_matrix_id, Val(:small))
            batch_op!(Val(:qr_Q_full), Qfull, R, d, tau, Val(D1), Val(D2), Val(D), warp_matrix_id, Val(:small))
        end

        # Store R
        dual_to_interm_transfer!(shmem_1, UpperTriangular(R), Val(min(D1, D2)), Val(D2), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Rs, shmem_1, Val(min(D1, D2)), Val(D2), Val(D), Val(nthreads), N, Val(:small))

        # Store Qthin
        dual_to_interm_transfer!(shmem_1, Qthin, Val(D1), Val(min(D1, D2)), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Qthins, shmem_1, Val(D1), Val(min(D1, D2)), Val(D), Val(nthreads), N, Val(:small))

        # Store Qfull
        dual_to_interm_transfer!(shmem_1, Qfull, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Qfulls, shmem_1, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))

        return nothing
    end

    # Test parameters
    nthreads = 2^8
    N = 2^12

    for D1 in 2:10
        for D2 in 2:10
            for extra in 0:1
                D = max(D1, D2) + extra
                nblocks = cld(N, nthreads//32 * (32 ÷ D))

                CUDA.seed!(1234)

                As = CUDA.rand(Float32, D1, D2, N)
                Rs = CUDA.zeros(Float32, min(D1, D2), D2, N)
                Qthins = CUDA.zeros(Float32, D1, min(D1, D2), N)
                Qfulls = CUDA.zeros(Float32, D1, D1, N)

                CUDA.@sync @cuda threads = nthreads blocks = nblocks qr_kernel!(
                    Rs,
                    Qthins,
                    Qfulls,
                    As,
                    Val(Int32(D1)),
                    Val(Int32(D2)),
                    Val(Int32(D)),
                    Val(nthreads),
                    Int32(N),
                )

                # CPU comparison
                As_cpu = Array(As)
                Rs_cpu = Array(Rs)
                Qthins_cpu = Array(Qthins)
                Qfulls_cpu = Array(Qfulls)
                
                max_error = 0.0
                for i in 1:N
                    # Check reconstruction
                    recon_error = maximum(abs.(As_cpu[:, :, i] - Qthins_cpu[:, :, i] * triu(Rs_cpu[:, :, i])))

                    # Check R triangularity
                    R_trig_error = maximum(abs.(tril(Rs_cpu[:, :, i], -1)))

                    # Check Qthin orthogonality
                    Qthin_error = maximum(abs.(I - Qthins_cpu[:, :, i]' * Qthins_cpu[:, :, i]))

                    # Check Qfull orthogonality
                    Qfull_error_1 = maximum(abs.(I - Qfulls_cpu[:, :, i]' * Qfulls_cpu[:, :, i]))
                    Qfull_error_2 = maximum(abs.(I - Qfulls_cpu[:, :, i] * Qfulls_cpu[:, :, i]'))

                    # Check Qthin and Qfull compatibility
                    Q_prefix_error = maximum(abs.(Qfulls_cpu[:, 1:min(D1,D2), i] - Qthins_cpu[:, :, i]))

                    max_error = max(max_error, recon_error, R_trig_error, Qthin_error, Qfull_error_1, Qfull_error_2, Q_prefix_error)
                end

                @test max_error < 1e-5
            end
        end
    end
end

@testitem "QR decomposition (in-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    function qr_kernel!(
        Rs,
        Qthins,
        Qfulls,
        As,
        ::Val{D1},
        ::Val{D2},
        ::Val{D},
        ::Val{nthreads},
        N::Int32
    ) where {D1,D2,D,nthreads}
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
        shmem_4 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load A
        intermediate_layout_load!(shmem_4, As, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_1, shmem_4, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        R = A  # in-place
        Qfull = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        Qthin = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            tau = batch_op!(qr, R, A, d, Val(D1), Val(D2), Val(D), warp_matrix_id, Val(:small))
            batch_op!(Val(:qr_Q_thin), Qthin, R, d, tau, Val(D1), Val(D2), Val(D), warp_matrix_id, Val(:small))
            batch_op!(Val(:qr_Q_full), Qfull, R, d, tau, Val(D1), Val(D2), Val(D), warp_matrix_id, Val(:small))
        end

        # Store R
        dual_to_interm_transfer!(shmem_4, UpperTriangular(R), Val(min(D1, D2)), Val(D2), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Rs, shmem_4, Val(min(D1, D2)), Val(D2), Val(D), Val(nthreads), N, Val(:small))

        # Store Qthin
        dual_to_interm_transfer!(shmem_4, Qthin, Val(D1), Val(min(D1, D2)), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Qthins, shmem_4, Val(D1), Val(min(D1, D2)), Val(D), Val(nthreads), N, Val(:small))

        # Store Qfull
        dual_to_interm_transfer!(shmem_4, Qfull, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Qfulls, shmem_4, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))

        return nothing
    end

    # Test parameters
    nthreads = 2^8
    N = 2^12

    for D1 in 2:10
        for D2 in 2:10
            for extra in 0:1
                D = max(D1, D2) + extra
                nblocks = cld(N, nthreads//32 * (32 ÷ D))

                CUDA.seed!(1234)

                As = CUDA.rand(Float32, D1, D2, N)
                Rs = CUDA.zeros(Float32, min(D1, D2), D2, N)
                Qthins = CUDA.zeros(Float32, D1, min(D1, D2), N)
                Qfulls = CUDA.zeros(Float32, D1, D1, N)

                CUDA.@sync @cuda threads = nthreads blocks = nblocks qr_kernel!(
                    Rs,
                    Qthins,
                    Qfulls,
                    As,
                    Val(Int32(D1)),
                    Val(Int32(D2)),
                    Val(Int32(D)),
                    Val(nthreads),
                    Int32(N),
                )

                # CPU comparison
                As_cpu = Array(As)
                Rs_cpu = Array(Rs)
                Qthins_cpu = Array(Qthins)
                Qfulls_cpu = Array(Qfulls)
                
                max_error = 0.0
                for i in 1:N
                    # Check reconstruction
                    recon_error = maximum(abs.(As_cpu[:, :, i] - Qthins_cpu[:, :, i] * triu(Rs_cpu[:, :, i])))

                    # Check R triangularity
                    R_trig_error = maximum(abs.(tril(Rs_cpu[:, :, i], -1)))

                    # Check Qthin orthogonality
                    Qthin_error = maximum(abs.(I - Qthins_cpu[:, :, i]' * Qthins_cpu[:, :, i]))

                    # Check Qfull orthogonality
                    Qfull_error_1 = maximum(abs.(I - Qfulls_cpu[:, :, i]' * Qfulls_cpu[:, :, i]))
                    Qfull_error_2 = maximum(abs.(I - Qfulls_cpu[:, :, i] * Qfulls_cpu[:, :, i]'))

                    # Check Qthin and Qfull compatibility
                    Q_prefix_error = maximum(abs.(Qfulls_cpu[:, 1:min(D1,D2), i] - Qthins_cpu[:, :, i]))

                    max_error = max(max_error, recon_error, R_trig_error, Qthin_error, Qfull_error_1, Qfull_error_2, Q_prefix_error)
                end

                @test max_error < 1e-5
            end
        end
    end
end

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