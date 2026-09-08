@testitem "Forced hybrid assignments and owner lifetimes" tags = [:cpu] begin
    using BatchedKernels
    using LinearAlgebra
    const BK = BatchedKernels
    const Matrix3 = BK.TraceMatrix{Float32,3,3}
    specs(n) = BK.InputSpec[BK.LeafInput(Matrix3, BK.BATCHED) for _ in 1:n]
    calls(tape, f) =
        [i for (i, node) in enumerate(tape.nodes) if node isa BK.CallNode && node.fn === f]
    plan(tape; kwargs...) = BK.plan_memory(tape, BK.Assignment(tape; kwargs...); D_MAX=3)

    # Instantaneous total capacity two does not imply one compatible region of
    # each kind can serve these complete lifetimes. Separate pools are constructive.
    intervals = [(1, :Ms, 1, 5), (2, :Md, 2, 3), (3, :Ms, 4, 7), (4, :Md, 6, 8)]
    slots, counts = BK._allocate_owner_intervals(intervals)
    @test counts[:Ms] == 2 && counts[:Md] == 1
    @test slots[1].idx != slots[3].idx
    @test slots[2].idx == slots[4].idx
    @test maximum(
        sum(start <= t <= stop for (_, _, start, stop) in intervals) for t in 1:8
    ) == 2

    chain(A, B) = begin
        X = A * B
        Y = X + B
        Z = Y * B
        (Z, Y, Y)
    end
    tape = BK.trace(chain, specs(2))
    product_ids, add_ids = calls(tape, *), calls(tape, +)
    variants = Dict(id => :matmul_row for id in product_ids)
    variants[only(add_ids)] = :add_row
    single_ids = [
        i for (i, meta) in enumerate(tape.metas) if meta.type <: BK.TraceMatrix &&
        meta.lifecycle == BK.BATCHED &&
        !(tape.nodes[i] isa BK.InputNode)
    ]
    assignment = BK.Assignment(
        tape;
        residences=Dict(i => :single for i in single_ids),
        orientations=Dict(i => :row for i in single_ids),
        variants,
        nthreads=64,
    )
    p = BK.plan_memory(tape, assignment; D_MAX=3)
    @test p.num_single_slots > 0 && p.num_dual_slots == 0
    # Repeated outputs and no-op output staging extend the same owner, not copies.
    stages = calls(tape, BK._dual_to_single)
    repeated = [id for id in stages if only(tape.nodes[id].args).id == only(add_ids)]
    @test length(repeated) == 2
    owner = p.owners[only(add_ids)]
    @test all(id -> p.owners[id] == owner, repeated)
    @test p.owner_last_use[owner] >=
        maximum(findfirst(==(id), assignment.order) for id in repeated)
    placement = first(calls(tape, BK._single_to_dual))
    load = only(tape.nodes[placement].args).id
    @test p.owners[placement] == p.owners[load]
    owner_slots = [slot for (id, slot) in p.slots if p.owners[id] == id]
    @test length(unique((slot.kind, slot.idx) for slot in owner_slots)) <
        length(owner_slots)

    @test_throws ArgumentError plan(tape; residences=Dict(first(product_ids) => :register))
    @test_throws ArgumentError plan(tape; nthreads=48)
    @test_throws ArgumentError plan(tape; order=reverse(collect(eachindex(tape.nodes))))
    @test_throws ArgumentError plan(tape; order=fill(1, length(tape.nodes)))
    @test_throws ArgumentError plan(
        tape;
        residences=Dict(first(product_ids) => :single),
        orientations=Dict(first(product_ids) => :col),
        variants=Dict(first(product_ids) => :matmul_row),
    )
    @test_throws ArgumentError plan(tape; variants=Dict(first(product_ids) => :solve_col))
    @test_throws ArgumentError plan(
        tape; residences=Dict(length(tape.nodes) + 1 => :single)
    )

    function incompatible(A)
        # The current scalar trace surface lacks +(::TraceMatrix, ::Adjoint).
        # Build that valid registry-level input directly to isolate assignment checks.
        wrapped = BK.register_wrapped!(A.tape, adjoint(A))
        ref = BK.emit_call!(A.tape, +, BK.NodeRef[A.ref, wrapped], Matrix3)
        return Matrix3(A.tape, ref)
    end
    incompatible_tape = BK.trace(incompatible, specs(1))
    input_compute = only(calls(incompatible_tape, BK._single_to_dual))
    addition = only(calls(incompatible_tape, +))
    @test_throws ArgumentError plan(
        incompatible_tape;
        residences=Dict(input_compute => :single),
        orientations=Dict(input_compute => :row),
        variants=Dict(addition => :add_row),
    )

    mutation(A, B) = begin
        U = cholesky!(A).U
        X = ldiv!(U, B)
        (X, B, U)
    end
    mutation_tape = BK.trace(mutation, specs(2))
    mutation_plan = plan(mutation_tape; nthreads=64)
    chol, solve = only(calls(mutation_tape, cholesky!)), only(calls(mutation_tape, ldiv!))
    @test mutation_plan.owners[chol] ==
        mutation_plan.owners[only(mutation_tape.nodes[chol].args).id]
    rhs = mutation_tape.nodes[solve].args[2].id
    @test mutation_plan.owners[solve] == mutation_plan.owners[rhs]
    @test mutation_plan.owner_last_use[mutation_plan.owners[rhs]] >=
        findfirst(==(solve), collect(eachindex(mutation_tape.nodes)))
    @test_throws ArgumentError plan(
        mutation_tape; residences=Dict(chol => :single), orientations=Dict(chol => :row)
    )
    # A raw-reference consumer after mutation must not be moved before it merely
    # because the SSA operand refers to the earlier input placement.
    rhs_stage = only([
        id for id in calls(mutation_tape, BK._dual_to_single) if
        only(mutation_tape.nodes[id].args).id == rhs
    ])
    bad_order = collect(eachindex(mutation_tape.nodes))
    deleteat!(bad_order, findfirst(==(rhs_stage), bad_order))
    insert!(bad_order, findfirst(==(solve), bad_order), rhs_stage)
    @test_throws ArgumentError plan(mutation_tape; order=bad_order)

    # The scheduler's occupancy model charges forced mutation aliases to their
    # canonical owners, agreeing with the actual legacy slot allocation.
    ids = BK._collect_schedulable(mutation_tape)
    dag = BK._reduced_dag(mutation_tape, ids)
    output_bits = BK._output_leaf_mask(mutation_tape, dag.bit_of)
    pools = [BK._sched_pool(mutation_tape, id) for id in ids]
    matrix_bits = foldl(
        |, (UInt64(1) << (i - 1) for i in eachindex(ids) if pools[i] === :M); init=UInt64(0)
    )
    vector_bits = foldl(
        |, (UInt64(1) << (i - 1) for i in eachindex(ids) if pools[i] === :V); init=UInt64(0)
    )
    inplace = [BK._inplace_info(mutation_tape, id, dag.bit_of) for id in ids]
    consumers, outputs, normalized_M, normalized_V, normalized_inplace, _ = BK._canonical_schedule_storage(
        mutation_tape, ids, dag, output_bits, matrix_bits, vector_bits, inplace
    )
    @test normalized_M & (UInt64(1) << (dag.bit_of[chol] - 1)) == 0
    @test normalized_M & (UInt64(1) << (dag.bit_of[solve] - 1)) == 0
    @test normalized_inplace[dag.bit_of[solve]].forced_owner ==
        BK.canonical_storage_owners(mutation_tape)[rhs]
    score = BK._evaluate_order(
        ids,
        dag.bit_of,
        consumers,
        outputs,
        pools,
        normalized_M,
        normalized_V,
        normalized_inplace,
    )
    legacy_plan = BK.plan_memory(mutation_tape)
    @test score.M == legacy_plan.num_matrix_slots

    # Symmetric metadata must survive trace-to-kernel expression reconstruction;
    # the poisoned upper half makes an accidental default :U immediately visible.
    symmetric_tape = BK.Tape()
    input = BK.push_node!(symmetric_tape, BK.InputNode(1), BK.NodeMeta(Matrix3, BK.BATCHED))
    symmetric_ref = BK.register_wrapped!(
        symmetric_tape, Symmetric(Matrix3(symmetric_tape, input), :L)
    )
    expression = BK.arg_kernel_expr(symmetric_tape, symmetric_ref, Dict(input.id => :A))
    reconstruct = Core.eval(BK, :((A) -> $expression))
    stored = Float32[1 999 888; 2 3 777; 4 5 6]
    @test Matrix(Base.invokelatest(reconstruct, stored)) == Float32[1 2 4; 2 3 5; 4 5 6]
end

@testitem "Forced hybrid fused shared kernels" tags = [:gpu] begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra
    const BK = BatchedKernels

    function assignment_for(f, args...; mixed=false)
        tape = BK.trace(f, BK.InputSpec[BK.input_spec(x) for x in args])
        residences, orientations, variants = Dict{Int,Symbol}(),
        Dict{Int,Symbol}(),
        Dict{Int,Symbol}()
        if mixed
            for (id, node) in enumerate(tape.nodes)
                if node isa BK.CallNode && node.fn in (*, +, -)
                    residences[id], orientations[id] = :single, :row
                    variants[id] = if node.fn === (*)
                        :matmul_row
                    elseif node.fn === (+)
                        :add_row
                    else
                        :sub_row
                    end
                end
            end
        end
        return BK.Assignment(tape; residences, orientations, variants, nthreads=64)
    end
    cpu_matrix(A) = Float32[A[i, j] for i in axes(A, 1), j in axes(A, 2)]

    repeated_wrappers(A, B) = begin
        X = A * B
        Y = X + B
        Z = Y * B
        (Z, UpperTriangular(Y), adjoint(Y), Y, Y)
    end
    N = 23 # D=3: two warps own twenty matrices, so final block is partial.
    left = reshape(sin.(Float32.(1:(9N))), 3, 3, N)
    right = reshape(cos.(Float32.(1:(9N))), 3, 3, N)
    A, B = BK.BatchedCuMatrix(CuArray(left)), BK.BatchedCuMatrix(CuArray(right))
    dual = assignment_for(repeated_wrappers, A, B)
    mixed = assignment_for(repeated_wrappers, A, B; mixed=true)
    legacy = repeated_wrappers.(A, B)
    hybrid_dual = @inferred BK.fuse(repeated_wrappers, A, B; assignment=dual)
    hybrid_mixed = @inferred BK.fuse(repeated_wrappers, A, B; assignment=mixed)
    CUDA.@allowscalar for batch in (1, 20, 23)
        reference = repeated_wrappers(left[:, :, batch], right[:, :, batch])
        for result in (legacy, hybrid_dual, hybrid_mixed), component in 1:5
            @test cpu_matrix(result[batch][component]) ≈ Matrix(reference[component]) rtol =
                5.0f-5 atol = 5.0f-6
        end
    end

    four_shared(A, S1, S2, S3, S4) = ((A * S1 + S2) * S3) + S4
    shared_cpu = [reshape(Float32.(1:9), 3, 3) ./ Float32(7 + i) for i in 1:4]
    shared_gpu = [BK.SharedCuMatrix(CuArray(S), N) for S in shared_cpu]
    shared_assignment = assignment_for(four_shared, A, shared_gpu...; mixed=true)
    shared_result = @inferred BK.fuse(
        four_shared, A, shared_gpu...; assignment=shared_assignment
    )
    reference = cat(
        (four_shared(left[:, :, batch], shared_cpu...) for batch in 1:N)...; dims=3
    )
    @test Array(shared_result.data) ≈ reference rtol = 5.0f-5 atol = 5.0f-6
    shared_legacy = @inferred BK.fuse(four_shared, A, shared_gpu...; nthreads=64)
    @test Array(shared_legacy.data) ≈ reference rtol = 5.0f-5 atol = 5.0f-6

    entry = BK._ensure_compiled!(
        four_shared, (A, shared_gpu...); assignment=shared_assignment, nthreads=64
    )
    kernel_args = (shared_result.data, A.data, (S.data for S in shared_gpu)..., Int32(N))
    compiled = Base.invokelatest() do
        fn = entry.fn
        @cuda launch = false fn(kernel_args...)
    end
    shared_tape = BK.hybrid_tape(
        BK.trace(four_shared, BK.InputSpec[BK.input_spec(x) for x in (A, shared_gpu...)])
    )
    shared_plan = BK.plan_memory(shared_tape, shared_assignment; D_MAX=3)
    actual_shared = CUDA.memory(compiled).shared
    @info "Forced shared kernel storage" predicted = shared_plan.shared_bytes actual =
        actual_shared
    @test actual_shared <= shared_plan.shared_bytes

    rectangular(A, B) = A * B
    rect_left = reshape(sin.(Float32.(1:(12N))), 3, 4, N)
    rect_right = reshape(cos.(Float32.(1:(8N))), 4, 2, N)
    RA, RB = BK.BatchedCuMatrix(CuArray(rect_left)), BK.BatchedCuMatrix(CuArray(rect_right))
    rect_tape = BK.trace(rectangular, BK.InputSpec[BK.input_spec(RA), BK.input_spec(RB)])
    product = only([
        id for
        (id, node) in enumerate(rect_tape.nodes) if node isa BK.CallNode && node.fn === (*)
    ])
    placement = rect_tape.nodes[product].args[1].id
    load = only(rect_tape.nodes[placement].args).id
    col_ids = (load, placement, product)
    col_assignment = BK.Assignment(
        rect_tape;
        residences=Dict(id => :single for id in col_ids),
        orientations=Dict(id => :col for id in col_ids),
        variants=Dict(product => :matmul_col),
        nthreads=64,
    )
    rect_result = @inferred BK.fuse(rectangular, RA, RB; assignment=col_assignment)
    rect_reference = cat((rect_left[:, :, b] * rect_right[:, :, b] for b in 1:N)...; dims=3)
    @test Array(rect_result.data) ≈ rect_reference rtol = 5.0f-5 atol = 5.0f-6

    matrix_vector_scalar(A, v) = (A * v, sum(abs2, v))
    vectors = reshape(cos.(Float32.(1:(3N))), 3, N)
    V = BK.BatchedCuVector(CuArray(vectors))
    vector_assignment = assignment_for(matrix_vector_scalar, A, V)
    vector_tape = BK.trace(
        matrix_vector_scalar, BK.InputSpec[BK.input_spec(A), BK.input_spec(V)]
    )
    vector_plan = BK.plan_memory(vector_tape, vector_assignment; D_MAX=3)
    @test vector_plan.num_vector_slots >= 1 && vector_plan.num_scalar_out_slots == 1
    vector_result = @inferred BK.fuse(
        matrix_vector_scalar, A, V; assignment=vector_assignment
    )
    expected_vectors = hcat((left[:, :, b] * vectors[:, b] for b in 1:N)...)
    expected_scalars = vec(sum(abs2, vectors; dims=1))
    @test Array(vector_result.components._1.data) ≈ expected_vectors rtol = 5.0f-5 atol =
        5.0f-6
    @test Array(vector_result.components._2.data) ≈ expected_scalars rtol = 5.0f-5 atol =
        5.0f-6

    mutation(A, B) = begin
        U = cholesky!(A).U
        X = ldiv!(U, B)
        (X, B, U)
    end
    spd = cat(
        (left[:, :, batch]' * left[:, :, batch] + 3.0f0 * I for batch in 1:N)...; dims=3
    )
    spd_gpu = BK.BatchedCuMatrix(CuArray(spd))

    factor_solve(A, B) = begin
        U = cholesky(A).U
        U \ (adjoint(U) \ B)
    end
    narrow_rhs = copy(right[:, 1:2, :])
    narrow_gpu = BK.BatchedCuMatrix(CuArray(narrow_rhs))
    factor_tape = BK.trace(
        factor_solve, BK.InputSpec[BK.input_spec(spd_gpu), BK.input_spec(narrow_gpu)]
    )
    factor_ids = [
        id for (id, node) in enumerate(factor_tape.nodes) if
        node isa BK.CallNode && node.fn in (cholesky, (\))
    ]
    factor_assignment = BK.Assignment(
        factor_tape;
        residences=Dict(id => :single for id in factor_ids),
        orientations=Dict(id => :row for id in factor_ids),
        variants=Dict(
            id => (factor_tape.nodes[id].fn === cholesky ? :cholesky_row : :solve_row) for
            id in factor_ids
        ),
        nthreads=64,
    )
    factored = @inferred BK.fuse(
        factor_solve, spd_gpu, narrow_gpu; assignment=factor_assignment
    )
    factor_reference = cat((spd[:, :, b] \ narrow_rhs[:, :, b] for b in 1:N)...; dims=3)
    @test Array(factored.data) ≈ factor_reference rtol = 5.0f-5 atol = 5.0f-6
    mutation_assignment = assignment_for(mutation, spd_gpu, B)
    mutated = @inferred BK.fuse(mutation, spd_gpu, B; assignment=mutation_assignment)
    CUDA.@allowscalar for batch in (1, 23)
        U = cholesky(spd[:, :, batch]).U
        solved = U \ right[:, :, batch]
        got = mutated[batch]
        @test cpu_matrix(got[1]) ≈ solved rtol = 5.0f-5 atol = 5.0f-6
        @test cpu_matrix(got[2]) ≈ solved rtol = 5.0f-5 atol = 5.0f-6
        @test cpu_matrix(got[3]) ≈ Matrix(U) rtol = 5.0f-5 atol = 5.0f-6
    end
end
