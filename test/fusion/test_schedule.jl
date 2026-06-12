@testitem "Scheduler" begin
    # CPU-only scheduler unit tests. Operate on hand-built Tapes — no GPU /
    # trace pass needed. Verify the subset-DP scheduler finds the slot-optimal
    # ordering for a handful of canonical DAG shapes (linear chain, diamond,
    # parallel chains, auto in-place, beats-natural).

    using BatchedKernels
    using LinearAlgebra
    const BK = BatchedKernels

    const Tape = BK.Tape
    const NodeRef = BK.NodeRef
    const InputNode = BK.InputNode
    const CallNode = BK.CallNode
    const NodeMeta = BK.NodeMeta
    const TraceMatrix = BK.TraceMatrix
    const BATCHED = BK.BATCHED
    const SHARED = BK.SHARED
    const schedule = BK.schedule
    const _collect_schedulable = BK._collect_schedulable
    const _reduced_dag = BK._reduced_dag
    const _output_leaf_mask = BK._output_leaf_mask
    const _evaluate_order = BK._evaluate_order
    const _sched_pool = BK._sched_pool
    const _inplace_info = BK._inplace_info
    const _compute_lb1 = BK._compute_lb1

    # Helper: append a node to the tape and return its NodeRef.
    function add!(tape::Tape, node, T::Type, lc=BATCHED)
        Base.push!(tape.nodes, node)
        Base.push!(tape.metas, NodeMeta(T, lc))
        return NodeRef(length(tape.nodes))
    end

    # Helper: assert the returned schedule is a valid permutation of 1:n.
    function check_perm(order::Vector{Int}, n::Int)
        @test length(order) == n
        @test sort(order) == collect(1:n)
    end

    # Helper: assert the schedule respects the reduced DAG (every pred precedes
    # its consumer).
    function check_topological(tape::Tape, order::Vector{Int})
        pos = Dict(id => i for (i, id) in enumerate(order))
        sched = _collect_schedulable(tape)
        dag = _reduced_dag(tape, sched)
        for (id, bit_i) in dag.bit_of
            for j in 1:length(sched)
                if (dag.preds[bit_i] & (UInt64(1) << (j - 1))) != 0
                    pred_id = sched[j]
                    @test pos[pred_id] < pos[id]
                end
            end
        end
    end

    # Helper: evaluate a given order under the scheduler's cost model.
    function eval_order(tape::Tape, order::Vector{Int})
        sched = _collect_schedulable(tape)
        dag = _reduced_dag(tape, sched)
        leaves = _output_leaf_mask(tape, dag.bit_of)
        pool_of = [_sched_pool(tape, id) for id in sched]
        pool_M = UInt64(0); pool_V = UInt64(0)
        for (i, p) in enumerate(pool_of)
            bit = UInt64(1) << (i - 1)
            p === :M ? (pool_M |= bit) : p === :V ? (pool_V |= bit) : nothing
        end
        inplace = [_inplace_info(tape, sched[i], dag.bit_of) for i in 1:length(sched)]
        sched_order = filter(id -> id in Set(sched), order)
        return _evaluate_order(
            sched_order, dag.bit_of, dag.consumers, leaves,
            pool_of, pool_M, pool_V, inplace,
        )
    end

    @testset "empty tape returns identity" begin
        tape = Tape()
        tape.output = nothing
        @test schedule(tape) == Int[]
    end

    @testset "linear chain A -> B -> C peaks at 2" begin
        # All-batched chain with input load (L), single→dual transfer (T), two
        # generic CallNodes (B, C), and a dual→single output write (Cout).
        # Natural order is optimal here.
        tape = Tape()
        MT = TraceMatrix{Float32,3,3}
        A = add!(tape, InputNode(1), MT, BATCHED)
        Base.push!(tape.inputs, A)
        L = add!(tape, CallNode(BK._load_to_single, NodeRef[A]), MT, BATCHED)
        T = add!(tape, CallNode(BK._single_to_dual, NodeRef[L]), MT, BATCHED)
        fake_fn1(args...) = nothing
        fake_fn2(args...) = nothing
        B = add!(tape, CallNode(fake_fn1, NodeRef[T]), MT, BATCHED)
        C = add!(tape, CallNode(fake_fn2, NodeRef[B]), MT, BATCHED)
        Cout = add!(tape, CallNode(BK._dual_to_single, NodeRef[C]), MT, BATCHED)
        tape.output = Cout

        @test length(_collect_schedulable(tape)) == 5   # L, T, B, C, Cout
        o = schedule(tape)
        check_perm(o, length(tape.nodes))
        check_topological(tape, o)
        @test eval_order(tape, o).M == 2
    end

    @testset "diamond DAG peaks at 3" begin
        # A (shared) -> B, A -> C, (B, C) -> D. Shared input A doesn't draw from
        # the batched pool; D's compute forces 3 slots (B, C, D simultaneously).
        tape = Tape()
        MT = TraceMatrix{Float32,3,3}
        A = add!(tape, InputNode(1), MT, SHARED); Base.push!(tape.inputs, A)
        f(args...) = nothing
        g(args...) = nothing
        h(args...) = nothing
        B = add!(tape, CallNode(f, NodeRef[A]), MT, BATCHED)
        C = add!(tape, CallNode(g, NodeRef[A]), MT, BATCHED)
        D = add!(tape, CallNode(h, NodeRef[B, C]), MT, BATCHED)
        Dout = add!(tape, CallNode(BK._dual_to_single, NodeRef[D]), MT, BATCHED)
        tape.output = Dout

        o = schedule(tape)
        check_perm(o, length(tape.nodes))
        check_topological(tape, o)
        @test eval_order(tape, o).M == 3
    end

    @testset "parallel chains: scheduler never worse than natural" begin
        # Two independent chains A→A2→B and C→C2→D joined into E. Smart schedule
        # finishes one chain before starting the other to avoid piling up live
        # intermediates.
        tape = Tape()
        MT = TraceMatrix{Float32,3,3}
        A = add!(tape, InputNode(1), MT, SHARED); Base.push!(tape.inputs, A)
        C = add!(tape, InputNode(2), MT, SHARED); Base.push!(tape.inputs, C)
        f = (args...) -> nothing; g = (args...) -> nothing; h = (args...) -> nothing
        A2 = add!(tape, CallNode(f, NodeRef[A]), MT, BATCHED)
        C2 = add!(tape, CallNode(g, NodeRef[C]), MT, BATCHED)
        B  = add!(tape, CallNode(f, NodeRef[A2]), MT, BATCHED)
        D  = add!(tape, CallNode(g, NodeRef[C2]), MT, BATCHED)
        E  = add!(tape, CallNode(h, NodeRef[B, D]), MT, BATCHED)
        Eout = add!(tape, CallNode(BK._dual_to_single, NodeRef[E]), MT, BATCHED)
        tape.output = Eout

        o = schedule(tape)
        check_perm(o, length(tape.nodes))
        check_topological(tape, o)
        p = eval_order(tape, o)
        @test p.M == 3

        p_nat = eval_order(tape, [A2.id, C2.id, B.id, D.id, E.id, Eout.id])
        @test p_nat.M >= p.M
    end

    @testset "auto in-place fires on `+`" begin
        # `+` is registered with both args in-place-safe. With both A and B dead
        # at the `+`, the result aliases onto one of them. Verify both the
        # whole-schedule peak and the per-step during-peak at `+`.
        tape = Tape()
        MT = TraceMatrix{Float32,3,3}
        Ain = add!(tape, InputNode(1), MT, BATCHED); Base.push!(tape.inputs, Ain)
        Bin = add!(tape, InputNode(2), MT, BATCHED); Base.push!(tape.inputs, Bin)
        AL = add!(tape, CallNode(BK._load_to_single, NodeRef[Ain]), MT, BATCHED)
        AT = add!(tape, CallNode(BK._single_to_dual, NodeRef[AL]), MT, BATCHED)
        BL = add!(tape, CallNode(BK._load_to_single, NodeRef[Bin]), MT, BATCHED)
        BT = add!(tape, CallNode(BK._single_to_dual, NodeRef[BL]), MT, BATCHED)
        C  = add!(tape, CallNode(+, NodeRef[AT, BT]), MT, BATCHED)
        Cout = add!(tape, CallNode(BK._dual_to_single, NodeRef[C]), MT, BATCHED)
        tape.output = Cout

        o = schedule(tape)
        check_perm(o, length(tape.nodes))
        check_topological(tape, o)
        # The BT step is forced to 3 (AT live, BL live, BT fresh — _single_to_dual
        # can't alias). C step itself takes only 2 thanks to auto in-place.
        @test eval_order(tape, o).M == 3

        # Verify the alias fires: build state S = {AL, AT, BL, BT}, ask the
        # planner for the during-peak at C — should be 2 (not 3).
        sched_ids = _collect_schedulable(tape)
        bit_idx = Dict(id => i for (i, id) in enumerate(sched_ids))
        S = UInt64(0)
        for id in (AL.id, AT.id, BL.id, BT.id)
            S |= UInt64(1) << (bit_idx[id] - 1)
        end
        dag = _reduced_dag(tape, sched_ids)
        leaves = _output_leaf_mask(tape, dag.bit_of)
        pool_of = [_sched_pool(tape, id) for id in sched_ids]
        pool_M = UInt64(0); pool_V = UInt64(0)
        for (i, pp) in enumerate(pool_of)
            bit = UInt64(1) << (i - 1)
            pp === :M ? (pool_M |= bit) : pp === :V ? (pool_V |= bit) : nothing
        end
        ip_C = _inplace_info(tape, C.id, dag.bit_of)
        d = BK._during_peak(
            bit_idx[C.id], S, dag.consumers, leaves,
            :M, pool_M, pool_V, ip_C, dag.bit_of,
        )
        @test d.M == 2
    end

    @testset "matmul does not in-place (output is +1 over inputs)" begin
        # Generic fn with no `inplace_safe_args` registration — stands in for
        # `*` whose sub-kernel reads multiple columns of A per thread and so
        # cannot safely overwrite an input.
        tape = Tape()
        MT = TraceMatrix{Float32,3,3}
        Ain = add!(tape, InputNode(1), MT, BATCHED); Base.push!(tape.inputs, Ain)
        Bin = add!(tape, InputNode(2), MT, BATCHED); Base.push!(tape.inputs, Bin)
        AL = add!(tape, CallNode(BK._load_to_single, NodeRef[Ain]), MT, BATCHED)
        AT = add!(tape, CallNode(BK._single_to_dual, NodeRef[AL]), MT, BATCHED)
        BL = add!(tape, CallNode(BK._load_to_single, NodeRef[Bin]), MT, BATCHED)
        BT = add!(tape, CallNode(BK._single_to_dual, NodeRef[BL]), MT, BATCHED)
        fake_matmul(a, b) = nothing
        C  = add!(tape, CallNode(fake_matmul, NodeRef[AT, BT]), MT, BATCHED)
        Cout = add!(tape, CallNode(BK._dual_to_single, NodeRef[C]), MT, BATCHED)
        tape.output = Cout

        o = schedule(tape)
        check_perm(o, length(tape.nodes))
        check_topological(tape, o)
        @test eval_order(tape, o).M == 3
    end

    @testset "LB1 is a sound lower bound" begin
        # Diamond again: LB1 should be 3 (D has 2 distinct preds + its own
        # slot), and the scheduler's peak must be >= LB1.
        tape = Tape()
        MT = TraceMatrix{Float32,3,3}
        A = add!(tape, InputNode(1), MT, SHARED); Base.push!(tape.inputs, A)
        f = (args...) -> nothing; g = (args...) -> nothing; h = (args...) -> nothing
        B = add!(tape, CallNode(f, NodeRef[A]), MT, BATCHED)
        C = add!(tape, CallNode(g, NodeRef[A]), MT, BATCHED)
        D = add!(tape, CallNode(h, NodeRef[B, C]), MT, BATCHED)
        Dout = add!(tape, CallNode(BK._dual_to_single, NodeRef[D]), MT, BATCHED)
        tape.output = Dout

        sched = _collect_schedulable(tape)
        dag = _reduced_dag(tape, sched)
        pool_of = [_sched_pool(tape, id) for id in sched]
        pool_M = UInt64(0); pool_V = UInt64(0)
        for (i, pp) in enumerate(pool_of)
            bit = UInt64(1) << (i - 1)
            pp === :M ? (pool_M |= bit) : pp === :V ? (pool_V |= bit) : nothing
        end
        inplace = [_inplace_info(tape, sched[i], dag.bit_of) for i in 1:length(sched)]
        lb = _compute_lb1(sched, pool_M, pool_V, pool_of, dag.preds, inplace, dag.bit_of)
        @test lb.M == 3

        p = eval_order(tape, schedule(tape))
        @test p.M >= lb.M
    end

    @testset "scheduler beats natural by deferring an input load" begin
        # Three batched inputs A, B, C feeding two chained computes:
        #   R1 = f(A, B)      # f, g not inplace-safe — no aliasing in play
        #   R2 = g(R1, C)
        # Natural order loads all three inputs up-front, so during R1's compute
        # A_dual + B_dual + C_dual + R1 are simultaneously live → peak 4. The
        # scheduler defers C's load+transfer until after R1 has landed (A and B
        # then die before C arrives) and reaches peak 3 throughout. Pure
        # reordering win — no aliasing involved.
        tape = Tape()
        MT = TraceMatrix{Float32,3,3}
        Ain = add!(tape, InputNode(1), MT, BATCHED); Base.push!(tape.inputs, Ain)
        Bin = add!(tape, InputNode(2), MT, BATCHED); Base.push!(tape.inputs, Bin)
        Cin = add!(tape, InputNode(3), MT, BATCHED); Base.push!(tape.inputs, Cin)
        AL = add!(tape, CallNode(BK._load_to_single, NodeRef[Ain]), MT, BATCHED)
        AT = add!(tape, CallNode(BK._single_to_dual, NodeRef[AL]), MT, BATCHED)
        BL = add!(tape, CallNode(BK._load_to_single, NodeRef[Bin]), MT, BATCHED)
        BT = add!(tape, CallNode(BK._single_to_dual, NodeRef[BL]), MT, BATCHED)
        CL = add!(tape, CallNode(BK._load_to_single, NodeRef[Cin]), MT, BATCHED)
        CT = add!(tape, CallNode(BK._single_to_dual, NodeRef[CL]), MT, BATCHED)
        f = (a, b) -> nothing
        g = (a, b) -> nothing
        R1 = add!(tape, CallNode(f, NodeRef[AT, BT]), MT, BATCHED)
        R2 = add!(tape, CallNode(g, NodeRef[R1, CT]), MT, BATCHED)
        R2out = add!(tape, CallNode(BK._dual_to_single, NodeRef[R2]), MT, BATCHED)
        tape.output = R2out

        o = schedule(tape)
        check_perm(o, length(tape.nodes))
        check_topological(tape, o)
        p_sched = eval_order(tape, o)

        natural = [AL.id, AT.id, BL.id, BT.id, CL.id, CT.id, R1.id, R2.id, R2out.id]
        p_nat = eval_order(tape, natural)

        @test p_nat.M == 4
        @test p_sched.M == 3
    end

    @testset "scheduler beats natural by aliasing on `+`" begin
        # Three sources feed three producers M1, M2, M3; two `+` reductions G1,
        # G2 join them. Natural order keeps all three M's live before reducing
        # (peak 3); the optimal schedule reduces eagerly to alias-merge (peak 2).
        tape = Tape()
        MT = TraceMatrix{Float32,3,3}
        A = add!(tape, InputNode(1), MT, SHARED); Base.push!(tape.inputs, A)
        B = add!(tape, InputNode(2), MT, SHARED); Base.push!(tape.inputs, B)
        C = add!(tape, InputNode(3), MT, SHARED); Base.push!(tape.inputs, C)
        h = (x,) -> nothing
        M1 = add!(tape, CallNode(h, NodeRef[A]), MT, BATCHED)
        M2 = add!(tape, CallNode(h, NodeRef[B]), MT, BATCHED)
        M3 = add!(tape, CallNode(h, NodeRef[C]), MT, BATCHED)
        G1 = add!(tape, CallNode(+, NodeRef[M1, M2]), MT, BATCHED)
        G2 = add!(tape, CallNode(+, NodeRef[G1, M3]), MT, BATCHED)
        Gout = add!(tape, CallNode(BK._dual_to_single, NodeRef[G2]), MT, BATCHED)
        tape.output = Gout

        o = schedule(tape)
        check_perm(o, length(tape.nodes))
        check_topological(tape, o)
        p_sched = eval_order(tape, o)
        p_nat = eval_order(tape, [M1.id, M2.id, M3.id, G1.id, G2.id, Gout.id])
        @test p_nat.M == 3
        @test p_sched.M == 2
    end

    @testset "all-batched Kalman cov: scheduler trims 6 → 5 matrix slots" begin
        # Feed the all-batched Kalman cov update through the tracer, and check the scheduler 
        # interleaves loads with compute to keep at most 5 slots alive instead of 6.
        function kalman_cov(P, A, Q, H, R)
            P_pred = A * P * A' + Q
            HP_pred = H * P_pred
            S = HP_pred * H' + R
            chol = cholesky(S)
            Y = LowerTriangular(chol.U') \ HP_pred
            K_T = chol.U \ Y
            KH = K_T' * H
            return (I - KH) * P_pred
        end

        D = 3; T = Float32
        specs = BK.InputSpec[
            BK.LeafInput(BK.TraceMatrix{T,D,D}, BATCHED) for _ in 1:5
        ]
        tape = BK.trace(kalman_cov, specs)

        o = schedule(tape)
        check_perm(o, length(tape.nodes))
        check_topological(tape, o)

        sched_nodes = _collect_schedulable(tape)
        dag = _reduced_dag(tape, sched_nodes)
        leaves = _output_leaf_mask(tape, dag.bit_of)
        pool_of = [_sched_pool(tape, id) for id in sched_nodes]
        pool_M = UInt64(0); pool_V = UInt64(0)
        for (i, pp) in enumerate(pool_of)
            bit = UInt64(1) << (i - 1)
            pp === :M ? (pool_M |= bit) : pp === :V ? (pool_V |= bit) : nothing
        end
        inplace = [_inplace_info(tape, sched_nodes[i], dag.bit_of) for i in 1:length(sched_nodes)]

        sched_only = filter(id -> id in Set(sched_nodes), o)
        p_sched = BK._evaluate_order(
            sched_only, dag.bit_of, dag.consumers, leaves,
            pool_of, pool_M, pool_V, inplace,
        )
        p_nat = BK._evaluate_order(
            sched_nodes, dag.bit_of, dag.consumers, leaves,
            pool_of, pool_M, pool_V, inplace,
        )
        lb = _compute_lb1(sched_nodes, pool_M, pool_V, pool_of, dag.preds, inplace, dag.bit_of)

        @test p_nat.M == 6
        @test p_sched.M == 5
        @test p_sched.M >= lb.M  # LB1 sound
    end
end
