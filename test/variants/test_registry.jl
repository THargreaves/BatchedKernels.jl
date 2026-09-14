@testitem "Hybrid variant contracts and emission" tags = [:cpu] begin
    using BatchedKernels
    using LinearAlgebra
    const BK = BatchedKernels
    const M34 = BK.TraceMatrix{Float32,3,4}
    const M42 = BK.TraceMatrix{Float32,4,2}
    const M33 = BK.TraceMatrix{Float32,3,3}
    const M32 = BK.TraceMatrix{Float32,3,2}

    mul = BK.orientation_variants(*, M34, M42)
    @test map(v -> v.id, mul) == (:matmul_row, :matmul_col)
    @test map(v -> (v.input_access, v.output_access), mul) ==
        (((:any, :row), :row), ((:col, :any), :col))
    @test mul[2].mirror == :adjoint_swap
    @test all(v -> isempty(v.alias_safe_args) && :register in v.output_residences, mul)
    @test all(v -> v.alias_requirement == :all_overlaps_pointwise_identical, mul)
    add = BK.orientation_variants(+, M34, M34)
    @test all(
        v -> v.alias_safe_args == (1, 2) && v.alias_residences == (:single, :dual), add
    )
    @test all(
        v -> v.alias_safe_args == (2,),
        BK.orientation_variants(+, BK.IAddSubWrapped{Float32,3}, M33),
    )
    @test only(BK.orientation_variants(cholesky, M33)).shape_rule == :square_spd
    solves = BK.orientation_variants(\, LowerTriangular{Float32,M33}, M32)
    @test map(v -> v.id, solves) == (:solve_row, :solve_col)
    @test map(v -> (v.input_access, v.output_access), solves) ==
        (((:any, :row), :row), ((:col, :col), :col))
    wrapped_solves = BK.orientation_variants(
        \, Adjoint{Float32,UnitUpperTriangular{Float32,M33}}, M32
    )
    @test map(v -> v.id, wrapped_solves) == (:solve_row, :solve_col)
    @test all(v -> isempty(v.alias_safe_args), solves)
    wide_rhs = BK.TraceMatrix{Float32,3,5}
    @test map(
        v -> v.id, BK.orientation_variants(\, LowerTriangular{Float32,M33}, wide_rhs)
    ) == (:solve_row, :solve_col)
    col_solve = BK.emit_variant(
        solves[2], :C, [:A, :B], [LowerTriangular{Float32,M33}, M32], 6
    )
    @test col_solve == :(variant_op!(
        Val(:solve_col), C, A, B, d, Val(Int32(3)), Val(Int32(2)), Val(Int32(6))
    ))
    @test all(
        v ->
            v.synchronization == :caller_entry_and_exit_shared_fences &&
                v.participation == :complete_matrix_group,
        mul,
    )

    # Check supported precision boundaries and rejected hybrid requests. Rejection
    # does not claim that the legacy implementation supports the request.
    @test isempty(BK.orientation_variants(*, M34, M33)) # contraction mismatch
    @test isempty(BK.orientation_variants(+, M34, M33))
    @test isempty(BK.orientation_variants(cholesky, M34))
    @test isempty(BK.orientation_variants(\, M33, M32)) # factor must be triangular
    @test map(
        v -> v.id,
        BK.orientation_variants(
            *, BK.TraceMatrix{Float64,3,4}, BK.TraceMatrix{Float64,4,2}
        ),
    ) == (:matmul_row, :matmul_col)
    @test isempty(BK.orientation_variants(*, BK.TraceMatrix{Float64,3,4}, M42))
    @test isempty(
        BK.orientation_variants(*, BK.TraceMatrix{Float16,3,4}, BK.TraceMatrix{Float16,4,2})
    )
    @test isempty(
        BK.orientation_variants(+, BK.TraceMatrix{Float64,3,3}, BK.TraceMatrix{Float64,3,3})
    )
    @test isempty(BK.orientation_variants(cholesky, BK.TraceMatrix{Float64,3,3}))
    @test isempty(
        BK.orientation_variants(
            \,
            LowerTriangular{Float64,BK.TraceMatrix{Float64,3,3}},
            BK.TraceMatrix{Float64,3,2},
        ),
    )
    @test isempty(BK.orientation_variants(*, BK.TraceMatrix{Float32,33,4}, M42))
    @test isempty(BK.orientation_variants(*, Symmetric{Float32,M33}, M32))
    @test isempty(BK.orientation_variants(BK.gram, M34))
    @test isempty(BK.orientation_variants(transpose, M34))
    @test isempty(BK.orientation_variants(cholesky!, M33))
    @test isempty(BK.orientation_variants(ldiv!, LowerTriangular{Float32,M33}, M33))
    @test_throws ArgumentError BK.emit_variant(mul[1], :C, [:A, :B], [M34, M42], 3)
    @test_throws ArgumentError BK.emit_variant(mul[1], :C, [:A], [M34], 6)
    @test_throws ArgumentError BK.emit_variant(mul[1], :C, [:A, :B], [M34, M33], 6)

    # Execute emitted mirror selection against independent CPU matrix arithmetic.
    function cpu_view(data, orientation)
        M, N = size(data)
        raw = zeros(Float32, BK.single_region_elems(Val(6), Val(32)))
        A = BK.SingleAccessMatrix(
            raw, Val(M), Val(N), Val(6), orientation, Int32(1), Int32(1)
        )
        A[:, :] = data
        return A
    end
    left = reshape(Float32.(1:12), 3, 4)
    right = reshape(Float32.(1:8), 4, 2)
    A = cpu_view(left, BK.ColOriented())
    B = cpu_view(right, BK.RowOriented())
    C = cpu_view(zeros(Float32, 3, 2), BK.ColOriented())
    call = BK.emit_variant(mul[2], :C, [:A, :B], [M34, M42], 6)
    emitted = Core.eval(BK, :((C, A, B, d) -> $call))
    for d in Int32(1):Int32(6)
        emitted(C, A, B, d)
    end
    @test Matrix(C) == left * right
end
