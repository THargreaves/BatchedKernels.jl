@testitem "Forced register assignments and element pressure" tags = [:cpu] begin
    using BatchedKernels
    using LinearAlgebra
    const BK = BatchedKernels

    product(A, B) = A * B
    tape = BK.trace(
        product,
        BK.InputSpec[
            BK.LeafInput(BK.TraceMatrix{Float32,3,4}, BK.BATCHED),
            BK.LeafInput(BK.TraceMatrix{Float32,4,2}, BK.BATCHED),
        ],
    )
    calls(f) = [i for (i, n) in enumerate(tape.nodes) if n isa BK.CallNode && n.fn === f]
    inputs, result = calls(BK._single_to_dual), only(calls(*))
    register_ids = [inputs; result]
    assignment = BK.Assignment(
        tape;
        residences=Dict(i => :register for i in register_ids),
        orientations=Dict(i => :col for i in register_ids),
        variants=Dict(result => :matmul_col),
        nthreads=64,
    )
    planner = BK.plan_memory(tape, assignment; D_MAX=4)
    # ColAccess stores N scalars per lane: A=4, B=2, fresh C=2. All three
    # coexist during multiplication. This is an element count, not a predicted
    # compiler register count; it excludes temporary accumulators and pointers.
    @test planner.peak_register_elements == 8
    @test planner.num_dual_slots == 0
    @test planner.num_single_slots > 0 # Global transfers still stage in shared memory.
    @test all(i -> !haskey(planner.slots, i), register_ids)
    @test all(i -> planner.owners[i] == i, register_ids)
    @test all(i -> planner.assignment.orientations[i] === :col, register_ids)

    # RowAccess has orientation-dependent widths A=3, B=4, C=3.
    row_assignment = BK.Assignment(
        tape;
        residences=Dict(i => :register for i in register_ids),
        orientations=Dict(i => :row for i in register_ids),
        variants=Dict(result => :matmul_row),
        nthreads=64,
    )
    @test BK.plan_memory(tape, row_assignment; D_MAX=4).peak_register_elements == 10
    stage = only(calls(BK._dual_to_single))
    @test_throws ArgumentError BK.plan_memory(
        tape, BK.Assignment(tape; residences=Dict(stage => :register)); D_MAX=4
    )
    @test_throws ArgumentError BK.plan_memory(
        tape,
        BK.Assignment(
            tape;
            residences=Dict(result => :register),
            orientations=Dict(result => :both),
            variants=Dict(result => :matmul_col),
        );
        D_MAX=4,
    )
    # Register inputs do not silently enable a legacy primitive body.
    @test_throws ArgumentError BK.plan_memory(
        tape, BK.Assignment(tape; residences=Dict(i => :register for i in inputs)); D_MAX=4
    )

    # Exercise the explicit tape contract: normal composite output handling stages
    # the parent first, but a logical Symmetric staging source has no register accessor.
    function symmetric_source(A)
        wrapped = BK.register_wrapped!(A.tape, Symmetric(A, :L))
        return BK.TraceMatrix{Float32,3,3}(A.tape, wrapped)
    end
    symmetric_tape = BK.trace(
        symmetric_source,
        BK.InputSpec[BK.LeafInput(BK.TraceMatrix{Float32,3,3}, BK.BATCHED)],
    )
    symmetric_input = only([
        i for (i, n) in enumerate(symmetric_tape.nodes) if
        n isa BK.CallNode && n.fn === BK._single_to_dual
    ])
    @test_throws ArgumentError BK.plan_memory(
        symmetric_tape,
        BK.Assignment(symmetric_tape; residences=Dict(symmetric_input => :register));
        D_MAX=3,
    )

    mutate(A) = cholesky!(A).U
    mutation = BK.trace(
        mutate, BK.InputSpec[BK.LeafInput(BK.TraceMatrix{Float32,3,3}, BK.BATCHED)]
    )
    placement = only([
        i for (i, n) in enumerate(mutation.nodes) if
        n isa BK.CallNode && n.fn === BK._single_to_dual
    ])
    chol = only([
        i for (i, n) in enumerate(mutation.nodes) if n isa BK.CallNode && n.fn === cholesky!
    ])
    @test_throws ArgumentError BK.plan_memory(
        mutation,
        BK.Assignment(
            mutation;
            residences=Dict(placement => :register, chol => :register),
            orientations=Dict(placement => :row, chol => :row),
        );
        D_MAX=3,
    )
end

@testitem "Forced register fused kernels" tags = [:gpu] begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra
    const BK = BatchedKernels

    function register_assignment(f, args...; orientation=:row, shared_products=false)
        tape = BK.trace(f, BK.InputSpec[BK.input_spec(x) for x in args])
        residences, orientations, variants = Dict{Int,Symbol}(),
        Dict{Int,Symbol}(),
        Dict{Int,Symbol}()
        for (i, node) in enumerate(tape.nodes)
            node isa BK.CallNode || continue
            if node.fn === BK._single_to_dual
                residences[i], orientations[i] = :register, orientation
            elseif node.fn in (*, cholesky, (\))
                residences[i] = shared_products && node.fn === (*) ? :single : :register
                orientations[i] = orientation
                variants[i] = if node.fn === (*)
                    orientation === :row ? :matmul_row : :matmul_col
                elseif node.fn === cholesky
                    :cholesky_row
                else
                    :solve_row
                end
            end
        end
        return BK.Assignment(tape; residences, orientations, variants, nthreads=64)
    end
    cpu_matrix(A) = Float32[A[i, j] for i in axes(A, 1), j in axes(A, 2)]

    rectangular(A, B) = begin
        C = A * B
        (C, adjoint(C), C)
    end
    # D=4 has eight matrices per warp. The last block is partial, while lane
    # four must supply B input even though C has only three owned column lines.
    N = 19
    left = reshape(sin.(Float32.(1:(12N))), 3, 4, N)
    right = reshape(cos.(Float32.(1:(8N))), 4, 2, N)
    A, B = BK.BatchedCuMatrix(CuArray(left)), BK.BatchedCuMatrix(CuArray(right))
    registers = register_assignment(rectangular, A, B; orientation=:col)
    mixed = register_assignment(rectangular, A, B; orientation=:col, shared_products=true)
    result = @inferred BK.fuse(rectangular, A, B; assignment=registers)
    shared_result = @inferred BK.fuse(rectangular, A, B; assignment=mixed)
    legacy = @inferred BK.fuse(rectangular, A, B; policy=:legacy, nthreads=64)
    references = [rectangular(left[:, :, batch], right[:, :, batch]) for batch in 1:N]
    CUDA.@allowscalar for got in (result, shared_result, legacy), component in 1:3
        @test all(1:N) do batch
            isapprox(
                cpu_matrix(got[batch][component]),
                Matrix(references[batch][component]);
                rtol=5.0f-5,
                atol=5.0f-6,
            )
        end
    end

    factor_solve(A, B) = begin
        U = cholesky(A).U
        U \ (adjoint(U) \ B)
    end
    # The same register factor is read through upper and adjoint-lower wrappers.
    # The narrow RHS also exercises row-solve lanes without output ownership.
    N = 23
    seed = reshape(sin.(Float32.(1:(9N))), 3, 3, N)
    spd = cat((seed[:, :, b]' * seed[:, :, b] + 3.0f0 * I for b in 1:N)...; dims=3)
    rhs = reshape(cos.(Float32.(1:(6N))), 3, 2, N)
    F, R = BK.BatchedCuMatrix(CuArray(spd)), BK.BatchedCuMatrix(CuArray(rhs))
    factor_assignment = register_assignment(factor_solve, F, R)
    solved = @inferred BK.fuse(factor_solve, F, R; assignment=factor_assignment)
    baseline = @inferred BK.fuse(factor_solve, F, R; policy=:legacy, nthreads=64)
    reference = cat((spd[:, :, b] \ rhs[:, :, b] for b in 1:N)...; dims=3)
    @test Array(solved.data) ≈ reference rtol = 5.0f-5 atol = 5.0f-6
    @test Array(baseline.data) ≈ reference rtol = 5.0f-5 atol = 5.0f-6

    entry = BK._ensure_compiled!(
        factor_solve, (F, R); assignment=factor_assignment, nthreads=64
    )
    kernel_args = (solved.data, F.data, R.data, Int32(N))
    compiled = Base.invokelatest() do
        fn = entry.fn
        @cuda launch = false fn(kernel_args...)
    end
    @test BK.DEBUG_ACCESSORS || CUDA.memory(compiled).local == 0
    @info "Small fused register solve resources" registers = CUDA.registers(compiled) memory = CUDA.memory(
        compiled
    )
end
