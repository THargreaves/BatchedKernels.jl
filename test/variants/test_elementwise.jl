@testitem "Hybrid matmul and elementwise variants" tags = [:gpu] begin
    using CUDA
    using BatchedKernels
    using LinearAlgebra
    using KernelAbstractions.Extras: @unroll
    const BK = BatchedKernels
    include(joinpath(@__DIR__, "..", "memory", "resource_checks.jl"))

    @inline function make_view(
        raw, ::Val{:register}, m, n, ::Val{D}, orientation, matrix, d
    ) where {D}
        return @inbounds BK.RegisterMatrix{Float32}(
            m, n, Val(D), orientation, (matrix - Int32(1)) * Int32(D), d
        )
    end
    @inline make_view(raw, ::Val{:single}, m, n, dim, orientation, matrix, d) =
        @inbounds BK.SingleAccessMatrix(raw, m, n, dim, orientation, Int32(1), matrix)
    @inline make_view(raw, ::Val{:shared}, m, n, dim, orientation, matrix, d) =
        BK.SharedMatrix(raw, m, n)
    @inline make_view(raw, ::Val{:dual}, m, n, dim, orientation, matrix, d) =
        BK.DualAccessMatrix(raw, dim, Int32(1), matrix)

    @inline function load_owned!(
        A, input, matrix, d, ::Val{M}, ::Val{N}, ::BK.RowOriented
    ) where {M,N}
        if d <= Int32(N)
            @unroll for k in Int32(1):Int32(M)
                BK.ours_write!(A, k, d, @inbounds(input[k, d, matrix]), BK.RowAccess())
            end
        end
        return nothing
    end
    @inline function load_owned!(
        A, input, matrix, d, ::Val{M}, ::Val{N}, ::BK.ColOriented
    ) where {M,N}
        if d <= Int32(M)
            @unroll for k in Int32(1):Int32(N)
                BK.ours_write!(A, k, d, @inbounds(input[d, k, matrix]), BK.ColAccess())
            end
        end
        return nothing
    end
    @inline function store_owned!(
        output, A, matrix, d, ::Val{M}, ::Val{N}, ::BK.RowOriented
    ) where {M,N}
        if d <= Int32(N)
            @unroll for k in Int32(1):Int32(M)
                @inbounds output[k, d, matrix] = BK.ours(A, k, d, BK.RowAccess())
            end
        end
        return nothing
    end
    @inline function store_owned!(
        output, A, matrix, d, ::Val{M}, ::Val{N}, ::BK.ColOriented
    ) where {M,N}
        if d <= Int32(M)
            @unroll for k in Int32(1):Int32(N)
                @inbounds output[d, k, matrix] = BK.ours(A, k, d, BK.ColAccess())
            end
        end
        return nothing
    end
    @inline wrap_inputs(A, B, ::Val{:none}) = (A, B)
    @inline wrap_inputs(A, B, ::Val{:upper_b}) = (A, UpperTriangular(B))
    @inline wrap_inputs(A, B, ::Val{:identity_a}) =
        (BK.IAddSubGetterMatrix(A, 2.0f0, -1.0f0), B)

    function matmul_kernel!(
        output,
        left,
        right,
        ::Val{M},
        ::Val{N},
        ::Val{P},
        ::Val{D},
        mode,
        a_residence,
        b_residence,
        c_residence,
        a_orientation,
        b_orientation,
        c_orientation,
        wrapper,
    ) where {M,N,P,D}
        # Exact per-layout helpers are already tested; allocate enough for either view.
        words = max(
            BK.single_region_elems(Val(D), Val(Int32(32))),
            BK.dual_region_elems(Val(D), Val(Int32(32))),
        )
        a_raw = CuStaticSharedArray(Float32, (words,))
        b_raw = CuStaticSharedArray(Float32, (words,))
        c_raw = CuStaticSharedArray(Float32, (words,))
        lane = threadIdx().x - Int32(1)
        matrix = lane ÷ Int32(D) + Int32(1)
        d = lane % Int32(D) + Int32(1)
        if matrix <= Int32(32 ÷ D)
            A = make_view(
                a_raw, a_residence, Val(M), Val(N), Val(D), a_orientation, matrix, d
            )
            B = make_view(
                b_raw, b_residence, Val(N), Val(P), Val(D), b_orientation, matrix, d
            )
            C = make_view(
                c_raw, c_residence, Val(M), Val(P), Val(D), c_orientation, matrix, d
            )
            load_owned!(A, left, matrix, d, Val(M), Val(N), a_orientation)
            load_owned!(B, right, matrix, d, Val(N), Val(P), b_orientation)
            # Only complete groups participate; keeping construction and use in
            # one guard avoids escaping mutable views through undefined-value phis.
            mask = typemax(UInt32) >>> Int32(32 - (32 ÷ D) * D)
            sync_warp(mask)
            Aw, Bw = wrap_inputs(A, B, wrapper)
            if mode isa Val{:legacy}
                BK.batch_op!(*, C, Aw, Bw, d, Val(M), Val(N), Val(P))
            else
                BK.variant_op!(mode, C, Aw, Bw, d, Val(M), Val(N), Val(P), Val(D))
            end
            sync_warp(mask)
            store_owned!(output, C, matrix, d, Val(M), Val(P), c_orientation)
        end
        return nothing
    end

    cases = (
        (
            3,
            4,
            2,
            6,
            :matmul_row,
            :register,
            :register,
            :register,
            BK.RowOriented(),
            BK.RowOriented(),
            BK.RowOriented(),
            :none,
        ),
        (
            5,
            3,
            3,
            6,
            :matmul_row,
            :register,
            :single,
            :register,
            BK.ColOriented(),
            BK.RowOriented(),
            BK.RowOriented(),
            :upper_b,
        ),
        (
            3,
            5,
            3,
            6,
            :matmul_col,
            :single,
            :register,
            :register,
            BK.ColOriented(),
            BK.RowOriented(),
            BK.ColOriented(),
            :identity_a,
        ),
        (
            4,
            6,
            5,
            6,
            :matmul_col,
            :register,
            :dual,
            :single,
            BK.ColOriented(),
            BK.RowOriented(),
            BK.ColOriented(),
            :none,
        ),
        (
            3,
            4,
            2,
            32,
            :matmul_row,
            :shared,
            :register,
            :register,
            BK.ColOriented(),
            BK.RowOriented(),
            BK.RowOriented(),
            :none,
        ),
    )
    for (M, N, P, D, mode, ar, br, cr, ao, bo, co, wrapper) in cases
        count = 32 ÷ D
        left = reshape(sin.(Float32.(1:(M * N * count))), M, N, count)
        right = reshape(cos.(Float32.(1:(N * P * count))), N, P, count)
        expected = zeros(Float32, M, P, count)
        for matrix in 1:count
            A, B = copy(left[:, :, matrix]), copy(right[:, :, matrix])
            if wrapper == :identity_a
                A = 2.0f0 * Matrix{Float32}(I, M, N) - A
            elseif wrapper == :upper_b
                B = Matrix(UpperTriangular(B))
            end
            expected[:, :, matrix] = A * B
        end
        output, legacy = CUDA.zeros(Float32, size(expected)),
        CUDA.zeros(Float32, size(expected))
        inputs = (CuArray(left), CuArray(right))
        dims = (Val(Int32(M)), Val(Int32(N)), Val(Int32(P)), Val(Int32(D)))
        args = (
            output,
            inputs...,
            dims...,
            Val(mode),
            Val(ar),
            Val(br),
            Val(cr),
            ao,
            bo,
            co,
            Val(wrapper),
        )
        kernel = @cuda launch = false matmul_kernel!(args...)
        if !BK.DEBUG_ACCESSORS
            @test require_register_resident(kernel).local_bytes == 0
        end
        CUDA.@sync kernel(args...; threads=32)
        legacy_args = (
            legacy,
            inputs...,
            dims...,
            Val(:legacy),
            Val(:dual),
            Val(:dual),
            Val(:dual),
            BK.RowOriented(),
            BK.RowOriented(),
            BK.RowOriented(),
            Val(wrapper),
        )
        CUDA.@sync @cuda threads = 32 matmul_kernel!(legacy_args...)
        @test Array(output) ≈ expected atol = 2.0f-6 rtol = 2.0f-6
        @test Array(output) ≈ Array(legacy) atol = 2.0f-6 rtol = 2.0f-6
    end

    function elementwise_kernel!(
        added, restored, left, right, ::Val{D}, orientation, residence, add_mode, sub_mode
    ) where {D}
        words = BK.single_region_elems(Val(D), Val(Int32(32)))
        a_raw = CuStaticSharedArray(Float32, (words,))
        b_raw = CuStaticSharedArray(Float32, (words,))
        lane = threadIdx().x - Int32(1)
        matrix = lane ÷ Int32(D) + Int32(1)
        d = lane % Int32(D) + Int32(1)
        if matrix <= Int32(32 ÷ D)
            A = make_view(
                a_raw,
                residence,
                Val(Int32(3)),
                Val(Int32(4)),
                Val(D),
                orientation,
                matrix,
                d,
            )
            B = make_view(
                b_raw,
                residence,
                Val(Int32(3)),
                Val(Int32(4)),
                Val(D),
                orientation,
                matrix,
                d,
            )
            load_owned!(A, left, matrix, d, Val(Int32(3)), Val(Int32(4)), orientation)
            load_owned!(B, right, matrix, d, Val(Int32(3)), Val(Int32(4)), orientation)
            W = BK.IAddSubGetterMatrix(B, 2.0f0, -1.0f0)
            # Same-owner shared aliases need no cross-lane communication. Register
            # mutation here tests local scratch only, not tape-level alias eligibility.
            BK.variant_op!(add_mode, A, A, W, d, Val(Int32(3)), Val(Int32(4)), Val(D))
            store_owned!(added, A, matrix, d, Val(Int32(3)), Val(Int32(4)), orientation)
            BK.variant_op!(sub_mode, A, A, W, d, Val(Int32(3)), Val(Int32(4)), Val(D))
            store_owned!(restored, A, matrix, d, Val(Int32(3)), Val(Int32(4)), orientation)
        end
        return nothing
    end
    left = reshape(sin.(Float32.(1:60)), 3, 4, 5)
    right = reshape(cos.(Float32.(1:60)), 3, 4, 5)
    expected = similar(left)
    for matrix in 1:5
        expected[:, :, matrix] =
            left[:, :, matrix] + 2.0f0 * Matrix{Float32}(I, 3, 4) - right[:, :, matrix]
    end
    for (orientation, residence, add_mode, sub_mode) in (
        (BK.RowOriented(), :register, :add_row, :sub_row),
        (BK.ColOriented(), :single, :add_col, :sub_col),
    )
        added, restored = CUDA.zeros(Float32, 3, 4, 5), CUDA.zeros(Float32, 3, 4, 5)
        args = (
            added,
            restored,
            CuArray(left),
            CuArray(right),
            Val(Int32(6)),
            orientation,
            Val(residence),
            Val(add_mode),
            Val(sub_mode),
        )
        kernel = @cuda launch = false elementwise_kernel!(args...)
        if !BK.DEBUG_ACCESSORS
            @test require_register_resident(kernel).local_bytes == 0
        end
        CUDA.@sync kernel(args...; threads=32)
        @test Array(added) ≈ expected atol = 5.0f-7
        @test Array(restored) ≈ left atol = 5.0f-7
    end
end
