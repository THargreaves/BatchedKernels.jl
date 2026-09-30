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
        N::Int32,
    ) where {D1,D2,B_D1,B_D2,D,nthreads}
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
        grid_mtrx_id =
            warp_matrix_id +
            (wid - 1i32) * n_mats_per_warp +
            (bid - 1i32) * n_mats_per_block

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_elems = warp_shmem_size * n_warps
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_vec_elems = D * n_mats_per_warp * n_warps
        shmem_tau = CuStaticSharedArray(Float32, (shmem_vec_elems,))

        # Load A
        intermediate_layout_load!(shmem_3, As, Val(D1), Val(D2), Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(
            shmem_1, shmem_3, Val(D1), Val(D2), Val(D), Val(nthreads), N
        )

        # Load B
        intermediate_layout_load!(
            shmem_3, Bs, Val(B_D1), Val(B_D2), Val(D), Val(nthreads), N
        )
        interm_to_dual_transfer!(
            shmem_2, shmem_3, Val(B_D1), Val(B_D2), Val(D), Val(nthreads), N
        )

        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
        R = A
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id)
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id)
        tau = BatchedVector(shmem_tau, Val(D), warp_matrix_id)

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            batch_op!(qr, R, tau, A, d, Val(D1), Val(D2), Val(D), warp_matrix_id)
            batch_op!(
                Val(:qr_Q_multiply),
                Val(false),
                C,
                R,
                B,
                d,
                tau,
                Val(D1),
                Val(D2),
                Val(B_D1),
                Val(B_D2),
                Val(D),
                warp_matrix_id,
            )
        end

        # Store C
        dual_to_interm_transfer!(shmem_1, C, Val(D1), Val(B_D2), Val(D), Val(nthreads), N)
        intermediate_layout_write!(
            Cs, shmem_1, Val(D1), Val(B_D2), Val(D), Val(nthreads), N
        )

        return nothing
    end

    # Test parameters
    nthreads = 2^8
    N = 2^9

    count = 0
    # (D1, D2, B_D1, B_D2, extra): square, tall and wide factors, full and thin Q
    # rows for B, and padded layouts, spanning padded sizes D from 2 to 11.
    cases = (
        (2, 2, 2, 2, 0),
        (3, 2, 3, 5, 0),
        (3, 2, 2, 5, 1),
        (2, 7, 2, 3, 0),
        (8, 8, 8, 8, 0),
        (10, 4, 4, 9, 1),
        (10, 10, 10, 2, 1),
    )
    for (D1, D2, B_D1, B_D2, extra) in cases
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
            error(
                "ERROR: [$count/$(length(cases))]: D1=$D1, D2=$D2, B_D1=$B_D1, B_D2=$B_D2, extra=$extra, max_error=$max_error",
            )
            break
        end

        @test max_error < 1e-4

        count += 1
        println(
            "[$count/$(length(cases))]: D1=$D1, D2=$D2, B_D1=$B_D1, B_D2=$B_D2, extra=$extra, max_error=$max_error",
        )
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
        N::Int32,
    ) where {D1,D2,B_D1,B_D2,D,nthreads}
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
        grid_mtrx_id =
            warp_matrix_id +
            (wid - 1i32) * n_mats_per_warp +
            (bid - 1i32) * n_mats_per_block

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_elems = warp_shmem_size * n_warps
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_vec_elems = D * n_mats_per_warp * n_warps
        shmem_tau = CuStaticSharedArray(Float32, (shmem_vec_elems,))

        # Load A
        intermediate_layout_load!(shmem_3, As, Val(D1), Val(D2), Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(
            shmem_1, shmem_3, Val(D1), Val(D2), Val(D), Val(nthreads), N
        )

        # Load B
        intermediate_layout_load!(
            shmem_3, Bs, Val(B_D1), Val(B_D2), Val(D), Val(nthreads), N
        )
        interm_to_dual_transfer!(
            shmem_2, shmem_3, Val(B_D1), Val(B_D2), Val(D), Val(nthreads), N
        )

        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
        R = A
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id)
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id)
        tau = BatchedVector(shmem_tau, Val(D), warp_matrix_id)

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            batch_op!(qr, R, tau, A, d, Val(D1), Val(D2), Val(D), warp_matrix_id)
            batch_op!(
                Val(:qr_Q_multiply),
                Val(true),
                C,
                R,
                B,
                d,
                tau,
                Val(D1),
                Val(D2),
                Val(B_D1),
                Val(B_D2),
                Val(D),
                warp_matrix_id,
            )
        end

        # Store C
        dual_to_interm_transfer!(shmem_1, C, Val(D1), Val(B_D2), Val(D), Val(nthreads), N)
        intermediate_layout_write!(
            Cs, shmem_1, Val(D1), Val(B_D2), Val(D), Val(nthreads), N
        )

        return nothing
    end

    # Test parameters
    nthreads = 2^8
    N = 2^9

    count = 0
    # (D1, D2, B_D2, extra): square, tall and wide factors and padded layouts,
    # spanning padded sizes D from 2 to 11.
    cases = (
        (2, 2, 2, 0),
        (3, 2, 5, 0),
        (2, 7, 3, 1),
        (5, 9, 10, 0),
        (8, 8, 8, 0),
        (10, 4, 9, 1),
        (10, 10, 2, 1),
    )
    for (D1, D2, B_D2, extra) in cases
        B_D1 = D1
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
            error(
                "ERROR: [$count/$(length(cases))]: D1=$D1, D2=$D2, B_D1=$B_D1, B_D2=$B_D2, extra=$extra, max_error=$max_error",
            )
            break
        end

        @test max_error < 1e-4

        count += 1
        println(
            "[$count/$(length(cases))]: D1=$D1, D2=$D2, B_D1=$B_D1, B_D2=$B_D2, extra=$extra, max_error=$max_error",
        )
    end
end
@testitem "QR decomposition 2x1 blocks (outofplace)" setup = [SubKernelShapes] begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    function qr_kernel!(Rs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32) where {D,nthreads}
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
        grid_mtrx_id =
            warp_matrix_id +
            (wid - 1i32) * n_mats_per_warp +
            (bid - 1i32) * n_mats_per_block

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_elems = warp_shmem_size * n_warps
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load A
        intermediate_layout_load!(shmem_3, As, Val(D), Val(D), Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(D), Val(D), Val(nthreads), N)

        # Load B
        intermediate_layout_load!(shmem_3, Bs, Val(D), Val(D), Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(D), Val(D), Val(nthreads), N)

        sync_warp()

        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id)
        R = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id)

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            AB = BlockMatrix_2_1(A', B', Val(D), warp_matrix_id)
            batch_op!(qr, R, AB, d, Val(D), Val(2), Val(1), warp_matrix_id)
        end

        sync_warp()

        # Store R
        dual_to_interm_transfer!(
            shmem_1, UpperTriangular(R), Val(D), Val(D), Val(D), Val(nthreads), N
        )
        intermediate_layout_write!(Rs, shmem_1, Val(D), Val(D), Val(D), Val(nthreads), N)

        return nothing
    end

    # Test parameters
    nthreads = 2^8
    N = 2^12

    for D in SubKernelShapes.square(10)
        nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

        CUDA.seed!(1234)

        As = CUDA.rand(Float32, D, D, N)
        Bs = CUDA.rand(Float32, D, D, N)
        Rs = CUDA.zeros(Float32, D, D, N)

        As_cpu = Array(As)
        Bs_cpu = Array(Bs)
        Rs_real = Array(Rs)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks qr_kernel!(
            Rs, As, Bs, Val(Int32(D)), Val(nthreads), Int32(N)
        )

        # CPU comparison
        Rs_res = Array(Rs)

        max_error = 0.0
        for i in 1:N
            M = vcat(As_cpu[:, :, i]', Bs_cpu[:, :, i]')
            R_expected = qr(M).R

            err = maximum(abs.(R_expected .- Rs_res[:, :, i]))
            max_error = max(max_error, err)
            if max_error > 1e-4
                println("error at i=$i")
                break
            end
        end

        @test max_error < 1e-4
    end
end

@testitem "QR decomposition 2x2 blocks" setup = [SubKernelShapes] begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    function qr_kernel!(
        Rs, As, Bs, Cs, ::Val{D}, ::Val{THRESH}, ::Val{nthreads}, N::Int32
    ) where {D,THRESH,nthreads}
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
        grid_mtrx_id =
            warp_matrix_id +
            (wid - 1i32) * n_mats_per_warp +
            (bid - 1i32) * n_mats_per_block

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_elems = warp_shmem_size * n_warps
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_4 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load A
        intermediate_layout_load!(shmem_4, As, Val(D), Val(D), Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_1, shmem_4, Val(D), Val(D), Val(D), Val(nthreads), N)

        # Load B
        intermediate_layout_load!(shmem_4, Bs, Val(D), Val(D), Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_2, shmem_4, Val(D), Val(D), Val(D), Val(nthreads), N)

        # Load C
        intermediate_layout_load!(shmem_4, Cs, Val(D), Val(D), Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_3, shmem_4, Val(D), Val(D), Val(D), Val(nthreads), N)

        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id)
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id)
        # R = DualAccessMatrix(shmem_4, Val(D), warp_matrix_id)
        R = C

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            ABC = BlockMatrixLowerTrig_2_2(A', B', C, Val(D), warp_matrix_id)
            batch_op!(qr, R, ABC, d, Val(D), Val(THRESH), Val(2), Val(2), warp_matrix_id)
        end

        sync_warp()

        # Store R
        dual_to_interm_transfer!(
            shmem_1, UpperTriangular(R), Val(D), Val(D), Val(D), Val(nthreads), N
        )
        intermediate_layout_write!(Rs, shmem_1, Val(D), Val(D), Val(D), Val(nthreads), N)

        return nothing
    end

    # Test parameters
    nthreads = 2^8
    N = 2^12
    THRESH = 3

    for D in SubKernelShapes.square(10)
        nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

        CUDA.seed!(1234)

        As = CUDA.rand(Float32, D, D, N)
        Bs = CUDA.rand(Float32, D, D, N)
        Cs = CUDA.rand(Float32, D, D, N)
        Rs = CUDA.zeros(Float32, D, D, N)

        As_cpu = Array(As)
        Bs_cpu = Array(Bs)
        Cs_cpu = Array(Cs)
        Rs_real = Array(Rs)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks qr_kernel!(
            Rs, As, Bs, Cs, Val(Int32(D)), Val(Int32(THRESH)), Val(nthreads), Int32(N)
        )

        # CPU comparison
        Rs_res = Array(Rs)

        max_error = 0.0
        for i in 1:N
            M = [
                As_cpu[:, :, i]' zeros(D, D)
                Bs_cpu[:, :, i]' Cs_cpu[:, :, i]
            ]
            R_full = qr(M).R
            R_expected = R_full[(D + 1):2D, (D + 1):2D]

            err = maximum(abs.(R_expected .- Rs_res[:, :, i]))
            max_error = max(max_error, err)
            if max_error > 1e-4
                println("error at i=$i for D=$D")
                display(M)
                display(R_expected)
                display(Rs_res[:, :, i])
                break
            end
        end

        @test max_error < 1e-4
    end
end
