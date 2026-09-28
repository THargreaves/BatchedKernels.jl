@testitem "Hybrid layout metadata and address maps" tags = [:cpu] begin
    using BatchedKernels
    using StaticArrays: MVector
    using LinearAlgebra

    const BK = BatchedKernels

    # Independent specification: no production address helper in the expected map.
    function reference_address(D, warp, matrix, i, j, row)
        count = 32 ÷ D
        interval = (32 ÷ (D & -D)) * D
        words = count * D^2
        stride = words + (words - 1) ÷ interval
        r = (matrix - 1) * D^2 + (row ? (j - 1) * D + i - 1 : (i - 1) * D + j - 1)
        return (warp - 1) * stride + r + r ÷ interval + 1
    end

    # Each case targets a different failure: padding across matrices, unused lanes,
    # transposed rectangular ownership, or the full-warp boundary.
    cases = (
        (4, 4, 4, BK.RowOriented()),
        (3, 4, 6, BK.RowOriented()),
        (4, 3, 6, BK.ColOriented()),
        (32, 32, 32, BK.ColOriented()),
    )
    for (M, N, D, physical) in cases
        count = 32 ÷ D
        interval = (32 ÷ (D & -D)) * D
        words = count * D^2
        stride = words + (words - 1) ÷ interval
        @test BK._single_warp_stride(Val(D)) == stride
        backing = fill(-1.0f0, 2 * stride)
        expected = copy(backing)
        A = @inferred BK.SingleAccessMatrix(
            backing, Val(M), Val(N), Val(D), physical, Int32(1), Int32(1)
        )
        @test fieldtype(typeof(A), :shmem) === Vector{Float32}
        @test size(A) == (M, N)
        @test Base.IndexStyle(typeof(A)) == IndexCartesian()
        @test (@inferred BK.orientation(A)) === physical
        @test_throws BoundsError A[0, 1]
        @test_throws BoundsError A[M + 1, 1] = 0.0f0
        values = reshape(Float32.(1:(M * N * count * 2)), M, N, count, 2)
        for warp in 1:2, matrix in 1:count
            A = BK.SingleAccessMatrix(
                backing, Val(M), Val(N), Val(D), physical, Int32(warp), Int32(matrix)
            )
            for j in 1:N, i in 1:M
                A[Int32(i), Int32(j)] = values[i, j, matrix, warp]
                expected[reference_address(D, warp, matrix, i, j, physical isa BK.RowOriented)] = values[
                    i, j, matrix, warp
                ]
            end
        end
        # Whole-buffer comparison catches wrong maps, overlap and overwritten padding.
        @test backing == expected
        actual = similar(values)
        for warp in 1:2, matrix in 1:count
            A = BK.SingleAccessMatrix(
                expected, Val(M), Val(N), Val(D), physical, Int32(warp), Int32(matrix)
            )
            actual[:, :, matrix, warp] = Matrix(A)
        end
        @test actual == values
    end

    @test BK.orientation(BK.DualAccessMatrix) === BK.BothOriented()
    @test BK.flip(BK.BothOriented()) === BK.BothOriented()
    @test BK.flip(BK.RowAccess()) === BK.ColAccess()
    @test BK.flip(BK.ColAccess()) === BK.RowAccess()
    @test BK.flip(BK.RowOriented()) === BK.ColOriented()
    @test BK.flip(BK.ColOriented()) === BK.RowOriented()

    # Literal masks isolate the shift boundaries and the last complete non-power-of-two group.
    for (D, base, expected) in
        ((1, 31, 0x80000000), (6, 24, 0x3f000000), (8, 24, 0xff000000), (32, 0, 0xffffffff))
        @test (@inferred BK._register_group_mask(Val(D), Int32(base))) === expected
    end

    # Device code convention uses Int32-valued Val dimensions, including the full warp.
    for (M, N, D) in ((Int32(3), Int32(4), Int32(6)), (Int32(32), Int32(31), Int32(32)))
        for physical in (BK.RowOriented(), BK.ColOriented())
            storage = zeros(Float32, Int(BK._single_warp_stride(Val(D))))
            A = @inferred BK.SingleAccessMatrix(
                storage, Val(M), Val(N), Val(D), physical, Int32(1), Int32(1)
            )
            @test size(A) === (Int(M), Int(N))
            A[Int32(1), Int32(1)] = 7.0f0
            @test (@inferred getindex(A, Int32(1), Int32(1))) === 7.0f0
            width = physical isa BK.RowOriented ? Int(M) : Int(N)
            R = @inferred BK.RegisterMatrix{Float32}(
                Val(M), Val(N), Val(D), physical, Int32(0), Int32(1)
            )
            @test fieldtype(typeof(R), :mv) === MVector{width,Float32}
            @test R.mv == zeros(Float32, width)
            supplied = MVector{width,Float32}(ntuple(k -> Float32(k), width))
            S = @inferred BK.RegisterMatrix(
                supplied,
                Val(M),
                Val(N),
                Val(D),
                physical,
                Int32(0),
                BK._register_group_mask(Val(D), Int32(0)),
                Int32(1),
            )
            @test S.mv === supplied
            @test size(S) === (Int(M), Int(N))
            @test S.base == 0 &&
                S.mask == BK._register_group_mask(Val(D), Int32(0)) &&
                S.d == 1
            wrong = MVector{width + 1,Float32}(undef)
            @test_throws ArgumentError BK.RegisterMatrix(
                wrong, Val(M), Val(N), Val(D), physical, Int32(0), S.mask, Int32(1)
            )
        end
    end

    # Shared and dual-backed IAddSub wrappers remain Both; wrapper traits must stay inferred.
    @test (@inferred BK.orientation(BK.SharedMatrix{Float32,2,2,32})) === BK.BothOriented()
    @test (@inferred BK.orientation(
        BK.IAddSubGetterMatrix{Float32,2,BK.DualAccessMatrix{Float32,2}}
    )) === BK.BothOriented()
    for physical in (BK.RowOriented(), BK.ColOriented())
        storage = zeros(Float32, Int(BK._single_warp_stride(Val(4))))
        A = BK.SingleAccessMatrix(
            storage, Val(2), Val(3), Val(4), physical, Int32(1), Int32(1)
        )
        R = BK.RegisterMatrix{Float32}(Val(2), Val(3), Val(4), physical, Int32(0), Int32(1))
        for parent in (A, R), wrap in (adjoint, transpose)
            wrapped = wrap(parent)
            @test size(wrapped) == (3, 2)
            @test (@inferred BK.orientation(wrapped)) === BK.flip(physical)
            @test (@inferred BK.orientation(typeof(wrapped))) === BK.flip(physical)
        end
        square = BK.SingleAccessMatrix(
            storage, Val(2), Val(2), Val(4), physical, Int32(1), Int32(1)
        )
        for wrap in
            (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
            wrapped = wrap(square)
            @test (@inferred BK.orientation(wrapped)) === physical
            @test (@inferred BK.orientation(typeof(wrapped))) === physical
            @test (@inferred BK.orientation(adjoint(wrapped))) === BK.flip(physical)
        end
    end

    storage = zeros(Float32, 1024)
    for (M, N, D) in ((0, 2, 4), (2, 0, 4), (5, 2, 4), (2, 5, 4), (1, 1, 0), (1, 1, 33))
        @test_throws ArgumentError BK.SingleAccessMatrix(
            storage, Val(M), Val(N), Val(D), BK.RowOriented(), Int32(1), Int32(1)
        )
    end
    @test_throws ArgumentError BK.SingleAccessMatrix(
        storage, Val(2), Val(2), Val(2), BK.RowOriented(), Int32(0), Int32(1)
    )
    @test_throws ArgumentError BK.SingleAccessMatrix(
        storage, Val(2), Val(2), Val(2), BK.RowOriented(), Int32(1), Int32(17)
    )
    @test_throws BoundsError BK.SingleAccessMatrix(
        Float32[], Val(2), Val(2), Val(2), BK.RowOriented(), Int32(1), Int32(1)
    )
    for (M, N, D) in ((0, 2, 4), (2, 0, 4), (5, 2, 4), (2, 5, 4), (1, 1, 0), (1, 1, 33))
        @test_throws ArgumentError BK.RegisterMatrix{Float32}(
            Val(M), Val(N), Val(D), BK.RowOriented(), Int32(0), Int32(1)
        )
    end
    mv = MVector{2,Float32}(0, 0)
    for (base, mask, d) in (
        (Int32(-1), UInt32(3), Int32(1)),
        (Int32(1), UInt32(6), Int32(1)),
        (Int32(32), UInt32(3), Int32(1)),
        (Int32(0), UInt32(1), Int32(1)),
        (Int32(0), UInt32(3), Int32(0)),
        (Int32(0), UInt32(3), Int32(3)),
    )
        @test_throws ArgumentError BK.RegisterMatrix(
            mv, Val(2), Val(2), Val(2), BK.RowOriented(), base, mask, d
        )
    end
end

@testitem "Hybrid single layout GPU addresses and batch tails" tags = [:gpu] begin
    using CUDA
    using BatchedKernels

    const BK = BatchedKernels

    function layout_kernel!(
        result,
        raw,
        input,
        ::Val{M},
        ::Val{N},
        ::Val{D},
        orientation,
        ::Val{THREADS},
        batches,
    ) where {M,N,D,THREADS}
        stride = BK._single_warp_stride(Val(D))
        total = stride * (THREADS ÷ 32)
        shared = CuStaticSharedArray(Float32, (total,))
        tid = threadIdx().x
        warp = (tid - Int32(1)) ÷ Int32(32) + Int32(1)
        lane = (tid - Int32(1)) % Int32(32)
        matrix = lane ÷ Int32(D) + Int32(1)
        d = lane % Int32(D) + Int32(1)
        count = Int32(32 ÷ D)
        batch =
            ((blockIdx().x - Int32(1)) * Int32(THREADS ÷ 32) + warp - Int32(1)) * count +
            matrix
        for index in tid:Int32(THREADS):total
            shared[index] = -1.0f0
        end
        sync_threads()
        if matrix <= count && batch <= batches
            A = BK.SingleAccessMatrix(
                shared, Val(M), Val(N), Val(D), orientation, warp, matrix
            )
            if d <= N
                for i in Int32(1):Int32(M)
                    A[i, d] = input[i, d, batch]
                end
            end
        end
        sync_threads()
        if matrix <= count && batch <= batches
            A = BK.SingleAccessMatrix(
                shared, Val(M), Val(N), Val(D), orientation, warp, matrix
            )
            if d <= M
                for j in Int32(1):Int32(N)
                    result[d, j, batch] = A[d, j]
                end
            end
        end
        for index in tid:Int32(THREADS):total
            raw[index, blockIdx().x] = shared[index]
        end
        return nothing
    end

    # Same distinct layout risks as the CPU checks, plus device addressing and tails.
    for (M, N, D, orientation) in (
        (4, 4, 4, BK.RowOriented()),
        (3, 4, 6, BK.RowOriented()),
        (4, 3, 6, BK.ColOriented()),
        (32, 32, 32, BK.ColOriented()),
    )
        threads = 128
        count = 32 ÷ D
        batches = 4 * count + 1 # A second block with just one active matrix.
        blocks = cld(batches, 4 * count)
        interval = (32 ÷ (D & -D)) * D
        words = count * D^2
        stride = words + (words - 1) ÷ interval
        input = reshape(Float32.(1:(M * N * batches)), M, N, batches)
        device_input = CuArray(input)
        result = CUDA.zeros(Float32, M, N, batches)
        raw = CUDA.zeros(Float32, 4 * stride, blocks)
        CUDA.@sync @cuda threads = threads blocks = blocks layout_kernel!(
            result,
            raw,
            device_input,
            Val(M),
            Val(N),
            Val(D),
            orientation,
            Val(threads),
            Int32(batches),
        )
        @test Array(result) == input
        expected = fill(-1.0f0, 4 * stride, blocks)
        for batch in 1:batches, j in 1:N, i in 1:M
            block = (batch - 1) ÷ (4 * count) + 1
            within = (batch - 1) % (4 * count)
            warp = within ÷ count
            matrix = within % count
            r = matrix * D^2 + (
                if orientation isa BK.RowOriented
                    (j - 1) * D + i - 1
                else
                    (i - 1) * D + j - 1
                end
            )
            expected[warp * stride + r + r ÷ interval + 1, block] = input[i, j, batch]
        end
        @test Array(raw) == expected
    end
end
