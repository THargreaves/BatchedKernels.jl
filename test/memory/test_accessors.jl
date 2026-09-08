@testitem "Hybrid accessor logical semantics" tags = [:cpu] begin
    using BatchedKernels
    using LinearAlgebra
    const BK = BatchedKernels

    # CPU-backed single storage permits independent inference and wrapper checks.
    for (physical, convention) in
        ((BK.RowOriented(), BK.RowAccess()), (BK.ColOriented(), BK.ColAccess()))
        A = BK.SingleAccessMatrix(
            zeros(ComplexF32, Int(BK.single_region_elems(Val(4), Val(32)))),
            Val(3),
            Val(3),
            Val(4),
            physical,
            Int32(1),
            Int32(1),
        )
        expected = ComplexF32[10i + j + (i - j)im for i in 1:3, j in 1:3]
        A[:, :] = expected
        for (k, d) in ((Int32(1), Int32(3)), (Int32(3), Int32(1)), (Int32(2), Int32(2)))
            i, j = convention isa BK.RowAccess ? (k, d) : (d, k)
            @test (@inferred BK.ours(A, k, d, convention)) == expected[i, j]
            @test (@inferred BK.theirs(A, k, d)) == expected[k, d]
            @test (@inferred BK.ours(adjoint(A), k, d, BK.flip(convention))) ==
                conj(expected[i, j])
        end
        for wrap in
            (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
            W, reference = wrap(A), wrap(expected)
            for (k, d) in ((Int32(1), Int32(3)), (Int32(3), Int32(1)), (Int32(2), Int32(2)))
                i, j = convention isa BK.RowAccess ? (k, d) : (d, k)
                @test (@inferred BK.ours(W, k, d, convention)) == reference[i, j]
                @test (@inferred BK.theirs(W, k, d)) == reference[k, d]
            end
        end
        wrapped = @inferred BK.IAddSubGetterMatrix(A, ComplexF32(2), ComplexF32(-1))
        @test fieldtype(typeof(wrapped), :parent) === typeof(A)
        @test (@inferred BK.theirs(wrapped, Int32(2), Int32(2))) == 2 - expected[2, 2]
        @test (@inferred BK.ours(wrapped, Int32(1), Int32(2), convention)) ==
            -(convention isa BK.RowAccess ? expected[1, 2] : expected[2, 1])
        @test_throws MethodError BK.ours(A, Int32(1), Int32(1), BK.flip(convention))
        BK.ours_write!(
            adjoint(A), Int32(1), Int32(2), ComplexF32(7 + 3im), BK.flip(convention)
        )
        i, j = convention isa BK.RowAccess ? (1, 2) : (2, 1)
        @test A[i, j] == 7 - 3im
        setter = BK.IAddSubSetterMatrix(A, ComplexF32(2), ComplexF32(-1))
        @test fieldtype(typeof(setter), :parent) === typeof(A)
        BK.ours_write!(setter, Int32(2), Int32(2), ComplexF32(5), convention)
        @test A[2, 2] == -3
        unit = UnitUpperTriangular(A)
        BK.ours_write!(unit, Int32(2), Int32(2), ComplexF32(1), convention)
        @test A[2, 2] == -3 # Logical unit diagonal must not overwrite backing.
        @test_throws ArgumentError BK.ours_write!(
            unit, Int32(2), Int32(2), ComplexF32(5), convention
        )
        if BK.DEBUG_ACCESSORS
            @test_throws AssertionError BK.ours_write!(
                unit, Int32(4), Int32(4), ComplexF32(1), convention
            )
        end
    end
end

@testitem "Hybrid register collectives and shared conventions" tags = [:gpu] begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra
    using KernelAbstractions.Extras: @unroll
    const BK = BatchedKernels
    include("resource_checks.jl")

    function register_accessors!(output, ::Val{D}) where {D}
        lane = (threadIdx().x - Int32(1)) % Int32(32)
        matrix = lane ÷ Int32(D) + Int32(1)
        d = lane % Int32(D) + Int32(1)
        base = lane - d + Int32(1)
        if matrix <= Int32(32 ÷ D)
            # Complete-group guard proves metadata valid; checked constructors retain
            # exception stack storage even when all MVector entries scalarize.
            R = @inbounds BK.RegisterMatrix{Float32}(
                Val(Int32(3)), Val(Int32(4)), Val(D), BK.RowOriented(), base, d
            )
            C = @inbounds BK.RegisterMatrix{Float32}(
                Val(Int32(3)), Val(Int32(4)), Val(D), BK.ColOriented(), base, d
            )
            @unroll for k in Int32(1):Int32(3)
                R.mv[k] = d <= Int32(4) ? Float32(100 * matrix + 10 * k + d) : 0.0f0
            end
            @unroll for k in Int32(1):Int32(4)
                C.mv[k] = d <= Int32(3) ? Float32(100 * matrix + 10 * d + k) : 0.0f0
            end
            # Padded lanes participate, although they own no logical output line.
            @inbounds output[d, matrix, 1] = BK.theirs(R, Int32(3), Int32(4))
            @inbounds output[d, matrix, 2] = BK.theirs(C, Int32(3), Int32(4))
            @inbounds output[d, matrix, 3] = BK.theirs(adjoint(R), Int32(4), Int32(3))
            W = BK.IAddSubGetterMatrix(R, 2.0f0, -1.0f0)
            @inbounds output[d, matrix, 4] = BK.theirs(W, Int32(2), Int32(2))
            @inbounds output[d, matrix, 5] = BK.theirs(W, Int32(3), Int32(1))
            square = @inbounds BK.RegisterMatrix(
                R.mv,
                Val(Int32(3)),
                Val(Int32(3)),
                Val(D),
                BK.RowOriented(),
                base,
                R.mask,
                d,
            )
            @inbounds output[d, matrix, 6] = BK.theirs(
                UpperTriangular(square), Int32(3), Int32(1)
            )
            @inbounds output[d, matrix, 7] = BK.theirs(
                LowerTriangular(square), Int32(3), Int32(1)
            )
            @inbounds output[d, matrix, 8] = BK.theirs(
                UnitUpperTriangular(square), Int32(2), Int32(2)
            )
            @inbounds output[d, matrix, 9] = BK.theirs(
                UnitLowerTriangular(square), Int32(1), Int32(3)
            )
            if d <= Int32(4)
                @inbounds output[d, matrix, 10] = BK.ours(R, Int32(2), d, BK.RowAccess())
                @inbounds output[d, matrix, 11] = BK.ours(W, Int32(2), d, BK.RowAccess())
                BK.ours_write!(adjoint(R), Int32(2), d, Float32(d), BK.ColAccess())
                @inbounds output[d, matrix, 12] = R.mv[2]
            end
            if d <= Int32(3)
                @inbounds output[d, matrix, 13] = BK.ours(C, Int32(2), d, BK.ColAccess())
                BK.ours_write!(C, Int32(2), d, Float32(-d), BK.ColAccess())
                @inbounds output[d, matrix, 14] = C.mv[2]
            end
        end
        return nothing
    end

    function shared_accessors!(output)
        single = CuStaticSharedArray(Float32, (Int32(16),))
        dual = CuStaticSharedArray(
            Float32, (BK.dual_region_elems(Val(Int32(4)), Val(Int32(32))),)
        )
        lane = threadIdx().x
        d = (lane - Int32(1)) % Int32(4) + Int32(1)
        matrix = (lane - Int32(1)) ÷ Int32(4) + Int32(1)
        S = BK.SharedMatrix(single, Val(Int32(4)))
        A = BK.DualAccessMatrix(dual, Val(Int32(4)), Int32(1), matrix)
        if lane <= Int32(4)
            for i in Int32(1):Int32(4)
                S[i, lane] = Float32(10 * i + lane)
            end
        end
        for i in Int32(1):Int32(4)
            A[i, d] = Float32(100 * matrix + 10 * i + d)
        end
        sync_threads()
        output[lane, 1] = BK.ours(A, Int32(2), d, BK.RowAccess())
        output[lane, 2] = BK.ours(A, Int32(2), d, BK.ColAccess())
        output[lane, 3] = BK.ours(adjoint(A), Int32(2), d, BK.RowAccess())
        output[lane, 4] = BK.ours(S, Int32(2), d, BK.RowAccess())
        output[lane, 5] = BK.ours(S, Int32(2), d, BK.ColAccess())
        output[lane, 6] = BK.theirs(A, Int32(3), Int32(4))
        sync_threads()
        BK.ours_write!(adjoint(A), Int32(2), d, Float32(-d), BK.RowAccess())
        sync_threads()
        output[lane, 7] = A[d, Int32(2)]
        return nothing
    end

    for D in (Int32(6), Int32(32))
        output = CuArray(fill(-1.0f0, Int(D), 32 ÷ Int(D), 14))
        kernel = @cuda launch = false register_accessors!(output, Val(D))
        resources = if BK.DEBUG_ACCESSORS
            register_resources(kernel)
        else
            require_register_resident(kernel)
        end
        @test BK.DEBUG_ACCESSORS || resources.local_bytes == 0
        CUDA.@sync kernel(output, Val(D); threads=32)
        expected = fill(-1.0f0, size(output))
        for matrix in 1:(32 ÷ D), d in 1:D
            expected[d, matrix, 1:9] = Float32[
                100matrix + 34,
                100matrix + 34,
                100matrix + 34,
                2 - (100matrix + 22),
                -(100matrix + 31),
                0,
                100matrix + 31,
                1,
                0,
            ]
            if d <= 4
                expected[d, matrix, 10:12] = Float32[
                    100matrix + 20 + d, (d == 2 ? 2 : 0) - (100matrix + 20 + d), d
                ]
            end
            if d <= 3
                expected[d, matrix, 13:14] = Float32[100matrix + 10d + 2, -d]
            end
        end
        @test Array(output) == expected
    end
    output = CUDA.zeros(Float32, 32, 7)
    CUDA.@sync @cuda threads = 32 shared_accessors!(output)
    expected = zeros(Float32, 32, 7)
    for lane in 1:32
        d = mod1(lane, 4)
        matrix = (lane - 1) ÷ 4 + 1
        expected[lane, :] = Float32[
            100matrix + 20 + d,
            100matrix + 10d + 2,
            100matrix + 10d + 2,
            20 + d,
            10d + 2,
            100matrix + 34,
            -d,
        ]
    end
    @test Array(output) == expected
end
