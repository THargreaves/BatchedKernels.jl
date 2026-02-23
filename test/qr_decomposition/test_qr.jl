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

@testitem "QR and multiply (out-of-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    function qr_and_multiply_kernel!(
        Cs,
        Bs,
        As,
        ::Val{D1},
        ::Val{D2},
        ::Val{B_D1},
        ::Val{B_D2},
        ::Val{D},
        ::Val{nthreads},
        N::Int32
    ) where {D1,D2,B_D1,B_D2,D,nthreads}
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
        intermediate_layout_load!(shmem_3, Bs, Val(B_D1), Val(B_D2), Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_2, shmem_3, Val(B_D1), Val(B_D2), Val(D), Val(nthreads), N, Val(:small))

        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        R = A
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            tau = batch_op!(qr, R, A, d, Val(D1), Val(D2), Val(D), warp_matrix_id, Val(:small))
            batch_op!(Val(:qr_Q_multiply), Val(false), C, R, B, d, tau, Val(D1), Val(D2), Val(B_D1), Val(B_D2), Val(D), warp_matrix_id, Val(:small))
        end

        # Store C
        dual_to_interm_transfer!(shmem_1, C, Val(D1), Val(B_D2), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Cs, shmem_1, Val(D1), Val(B_D2), Val(D), Val(nthreads), N, Val(:small))

        return nothing
    end

    # Test parameters
    nthreads = 2^8
    N = 2^9

    count = 0
    for D1 in 2:10
        for D2 in 2:10
            for B_D1 in unique((D1, min(D1, D2)))
                for B_D2 in 2:10
                    for extra in 0:1
                        global count
                        D = max(D1, D2, B_D1, B_D2) + extra
                        nblocks = cld(N, nthreads//32 * (32 ÷ D))

                        CUDA.seed!(1234)

                        As = CUDA.rand(Float32, D1, D2, N)
                        Bs = CUDA.rand(Float32, B_D1, B_D2, N)
                        Cs = CUDA.zeros(Float32, D1, B_D2, N)

                        CUDA.@sync @cuda threads = nthreads blocks = nblocks qr_and_multiply_kernel!(
                            Cs,
                            Bs,
                            As,
                            Val(Int32(D1)),
                            Val(Int32(D2)),
                            Val(Int32(B_D1)),
                            Val(Int32(B_D2)),
                            Val(Int32(D)),
                            Val(nthreads),
                            Int32(N),
                        )

                        # CPU comparison
                        As_cpu = Array(As)
                        Bs_cpu = Array(Bs)
                        Cs_cpu = Array(Cs)

                        Cs_real = zeros(Float32, D1, B_D2, N)

                        for i in 1:N
                            Cs_real[:, :, i] = qr(As_cpu[:, :, i]).Q * Bs_cpu[:, :, i]
                        end

                        max_error = maximum(abs.(Cs_real - Cs_cpu))
                        if max_error >= 1e-4
                            error("ERROR: [$count/2106]: D1=$D1, D2=$D2, B_D1=$B_D1, B_D2=$B_D2, extra=$extra, max_error=$max_error")
                            break
                        end

                        @test max_error < 1e-4

                        count += 1
                        println("[$count/2106]: D1=$D1, D2=$D2, B_D1=$B_D1, B_D2=$B_D2, extra=$extra, max_error=$max_error")
                    end
                end
            end
        end
    end
end

@testitem "QR transpose and multiply (out-of-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    function qr_trans_and_multiply_kernel!(
        Cs,
        Bs,
        As,
        ::Val{D1},
        ::Val{D2},
        ::Val{B_D1},
        ::Val{B_D2},
        ::Val{D},
        ::Val{nthreads},
        N::Int32
    ) where {D1,D2,B_D1,B_D2,D,nthreads}
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
        intermediate_layout_load!(shmem_3, Bs, Val(B_D1), Val(B_D2), Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_2, shmem_3, Val(B_D1), Val(B_D2), Val(D), Val(nthreads), N, Val(:small))

        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        R = A
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            tau = batch_op!(qr, R, A, d, Val(D1), Val(D2), Val(D), warp_matrix_id, Val(:small))
            batch_op!(Val(:qr_Q_multiply), Val(true), C, R, B, d, tau, Val(D1), Val(D2), Val(B_D1), Val(B_D2), Val(D), warp_matrix_id, Val(:small))
        end

        # Store C
        dual_to_interm_transfer!(shmem_1, C, Val(D1), Val(B_D2), Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Cs, shmem_1, Val(D1), Val(B_D2), Val(D), Val(nthreads), N, Val(:small))

        return nothing
    end

    # Test parameters
    nthreads = 2^8
    N = 2^9

    count = 0
    for D1 in 2:10
        for D2 in 2:10
            B_D1 = D1
            for B_D2 in 2:10
                for extra in 0:1
                    global count
                    D = max(D1, D2, B_D1, B_D2) + extra
                    nblocks = cld(N, nthreads//32 * (32 ÷ D))

                    CUDA.seed!(1234)

                    As = CUDA.rand(Float32, D1, D2, N)
                    Bs = CUDA.rand(Float32, B_D1, B_D2, N)
                    Cs = CUDA.zeros(Float32, D1, B_D2, N)

                    CUDA.@sync @cuda threads = nthreads blocks = nblocks qr_trans_and_multiply_kernel!(
                        Cs,
                        Bs,
                        As,
                        Val(Int32(D1)),
                        Val(Int32(D2)),
                        Val(Int32(B_D1)),
                        Val(Int32(B_D2)),
                        Val(Int32(D)),
                        Val(nthreads),
                        Int32(N),
                    )

                    # CPU comparison
                    As_cpu = Array(As)
                    Bs_cpu = Array(Bs)
                    Cs_cpu = Array(Cs)

                    Cs_real = zeros(Float32, D1, B_D2, N)

                    for i in 1:N
                        Cs_real[:, :, i] = qr(As_cpu[:, :, i]).Q' * Bs_cpu[:, :, i]
                    end

                    max_error = maximum(abs.(Cs_real - Cs_cpu))
                    if max_error >= 1e-4
                        error("ERROR: [$count/1458]: D1=$D1, D2=$D2, B_D1=$B_D1, B_D2=$B_D2, extra=$extra, max_error=$max_error")
                        break
                    end

                    @test max_error < 1e-4

                    count += 1
                    println("[$count/1458]: D1=$D1, D2=$D2, B_D1=$B_D1, B_D2=$B_D2, extra=$extra, max_error=$max_error")
                end
            end
        end
    end
end

@testitem "QR and multiply (vmap)" begin
    using BatchedKernels
    using LinearAlgebra
    using CUDA

    function qr_mul(A, B, dummy)
        res = qr(A)
        return res.Q * B
    end

    # Test parameters
    N = 2^9

    count = 0
    for D1 in 2:10
        for D2 in 2:10
            for B_D1 in unique((D1, min(D1, D2)))
                for B_D2 in 2:10
                    for extra in 0:1
                        global count
                        count += 1
                        D = max(D1, D2, B_D1, B_D2) + extra

                        CUDA.seed!(1234)

                        As = CUDA.rand(Float32, D1, D2, N)
                        Bs = CUDA.rand(Float32, B_D1, B_D2, N)
                        dummy = CUDA.zeros(Float32, D, D, N)

                        qr_mul_vmap = BatchedKernels.vmap(qr_mul)
                        Cs = qr_mul_vmap(As, Bs, dummy)

                        # CPU comparison
                        As_cpu = Array(As)
                        Bs_cpu = Array(Bs)
                        Cs_cpu = Array(Cs)

                        Cs_real = zeros(Float32, D1, B_D2, N)

                        for i in 1:N
                            Cs_real[:, :, i] = qr_mul(As_cpu[:, :, i], Bs_cpu[:, :, i], 0)
                        end

                        max_error = maximum(abs.(Cs_real - Cs_cpu))

                        @test max_error < 1e-4 && !isnan(max_error)

                        println("[$count/2106]: D1=$D1, D2=$D2, B_D1=$B_D1, B_D2=$B_D2, extra=$extra, max_error=$max_error")
                    end
                end
            end
        end
    end
end

@testitem "QR transpose and multiply (vmap)" begin
    using BatchedKernels
    using LinearAlgebra
    using CUDA

    function qr_mul(A, B, dummy)
        res = qr(A)
        return res.Q' * B
    end

    # Test parameters
    N = 2^9

    count = 0
    for D1 in 2:10
        for D2 in 2:10
            B_D1 = D1
            for B_D2 in 2:10
                for extra in 0:1
                    global count
                    count += 1
                    D = max(D1, D2, B_D1, B_D2) + extra

                    CUDA.seed!(1234)

                    As = CUDA.rand(Float32, D1, D2, N)
                    Bs = CUDA.rand(Float32, B_D1, B_D2, N)
                    dummy = CUDA.zeros(Float32, D, D, N)

                    qr_mul_vmap = BatchedKernels.vmap(qr_mul)
                    Cs = qr_mul_vmap(As, Bs, dummy)

                    # CPU comparison
                    As_cpu = Array(As)
                    Bs_cpu = Array(Bs)
                    Cs_cpu = Array(Cs)

                    Cs_real = zeros(Float32, D1, B_D2, N)

                    for i in 1:N
                        Cs_real[:, :, i] = qr_mul(As_cpu[:, :, i], Bs_cpu[:, :, i], 0)
                    end

                    max_error = maximum(abs.(Cs_real - Cs_cpu))

                    @test max_error < 1e-4 && !isnan(max_error)

                    println("[$count/1458]: D1=$D1, D2=$D2, B_D1=$B_D1, B_D2=$B_D2, extra=$extra, max_error=$max_error")
                end
            end
        end
    end
end

@testitem "QR Decomposition Qthin (vmap, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^9 + 113

    function qr_decomp(A)
        res = qr(A)
        Q = Matrix(res.Q)
        R = res.R
        return Q, R
    end

    for D1 in 2:10
        for D2 in 2:10
            for extra in 0:1
                D = max(D1, D2) + extra

                CUDA.seed!(1234)

                # Create symmetric positive definite matrices
                As = CUDA.rand(Float32, D1, D2, N)
                As_cpu = Array(As)
                dummy = CUDA.rand(Float32, D, D, N)

                qr_vmap = BatchedKernels.vmap(qr_decomp)
                Q, R = qr_vmap(As)
                Qs_result = Array(Q)
                Rs_result = Array(R)

                Qs_real = zeros(Float32, D1, min(D1, D2), N)
                Rs_real = zeros(Float32, min(D1, D2), D2, N)
                for i in 1:N
                    Q_real, R_real = qr_decomp(As_cpu[:, :, i])
                    Qs_real[:, :, i] = Q_real
                    Rs_real[:, :, i] = R_real
                end
                
                max_error_Q = maximum(abs.(Qs_real - Qs_result))
                max_error_R = maximum(abs.(Rs_real - Rs_result))
                max_error = max(max_error_Q, max_error_R)

                @test max_error < 1e-3 && !isnan(max_error)
            end
        end
    end
end

@testitem "QR Decomposition Qthin transpose (vmap, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^9

    function qr_decomp(A)
        res = qr(A')
        Q = Matrix(res.Q)
        R = res.R
        return Q, R
    end

    for D1 in 2:10
        for D2 in 2:10
            for extra in 0:1
                D = max(D1, D2) + extra

                CUDA.seed!(1234)

                # Create symmetric positive definite matrices
                As = CUDA.rand(Float32, D1, D2, N)
                As_cpu = Array(As)
                dummy = CUDA.rand(Float32, D, D, N)

                qr_vmap = BatchedKernels.vmap(qr_decomp)
                Q, R = qr_vmap(As)
                Qs_result = Array(Q)
                Rs_result = Array(R)

                Qs_real = zeros(Float32, D2, min(D1, D2), N)
                Rs_real = zeros(Float32, min(D1, D2), D1, N)
                for i in 1:N
                    Q_real, R_real = qr_decomp(As_cpu[:, :, i])
                    Qs_real[:, :, i] = Q_real
                    Rs_real[:, :, i] = R_real
                end
                
                max_error_Q = maximum(abs.(Qs_real - Qs_result))
                max_error_R = maximum(abs.(Rs_real - Rs_result))
                max_error = max(max_error_Q, max_error_R)

                @test max_error < 1e-3 && !isnan(max_error)
            end
        end
    end
end

@testitem "QR Decomposition Qfull (vmap, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^9 + 113

    function qr_decomp(A)
        res = qr(A)
        Q = res.Q * I
        R = res.R
        return Q, R
    end

    for D1 in 2:10
        for D2 in 2:10
            for extra in 0:1
                D = max(D1, D2) + extra

                CUDA.seed!(1234)

                # Create symmetric positive definite matrices
                As = CUDA.rand(Float32, D1, D2, N)
                As_cpu = Array(As)
                dummy = CUDA.rand(Float32, D, D, N)

                qr_vmap = BatchedKernels.vmap(qr_decomp)
                Q, R = qr_vmap(As)
                Qs_result = Array(Q)
                Rs_result = Array(R)

                Qs_real = zeros(Float32, D1, D1, N)
                Rs_real = zeros(Float32, min(D1, D2), D2, N)
                for i in 1:N
                    Q_real, R_real = qr_decomp(As_cpu[:, :, i])
                    Qs_real[:, :, i] = Q_real
                    Rs_real[:, :, i] = R_real
                end
                
                max_error_Q = maximum(abs.(Qs_real - Qs_result))
                max_error_R = maximum(abs.(Rs_real - Rs_result))
                max_error = max(max_error_Q, max_error_R)

                @test max_error < 1e-3 && !isnan(max_error)
            end
        end
    end
end

@testitem "QR Decomposition Qfull  transpose(vmap, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^9 + 113

    function qr_decomp(A)
        res = qr(A')
        Q = res.Q * I
        R = res.R
        return Q, R
    end

    for D1 in 2:10
        for D2 in 2:10
            for extra in 0:1
                D = max(D1, D2) + extra

                CUDA.seed!(1234)

                # Create symmetric positive definite matrices
                As = CUDA.rand(Float32, D1, D2, N)
                As_cpu = Array(As)
                dummy = CUDA.rand(Float32, D, D, N)

                qr_vmap = BatchedKernels.vmap(qr_decomp)
                Q, R = qr_vmap(As)
                Qs_result = Array(Q)
                Rs_result = Array(R)

                Qs_real = zeros(Float32, D2, D2, N)
                Rs_real = zeros(Float32, min(D1, D2), D1, N)
                for i in 1:N
                    Q_real, R_real = qr_decomp(As_cpu[:, :, i])
                    Qs_real[:, :, i] = Q_real
                    Rs_real[:, :, i] = R_real
                end
                
                max_error_Q = maximum(abs.(Qs_real - Qs_result))
                max_error_R = maximum(abs.(Rs_real - Rs_result))
                max_error = max(max_error_Q, max_error_R)

                @test max_error < 1e-3 && !isnan(max_error)
            end
        end
    end
end