"""
CPU-only register-liveness audit for the Kalman covariance trace.

Run from the repository root:

    julia --project=. benchmarking/kalman/pressure_audit/kalman_liveness.jl

This traces only phantom matrices; it neither allocates a CuArray nor compiles or
launches a GPU kernel.  The reported element peak is the same logical proxy used
by `plan_memory`: one D-wide line for every live register-resident matrix owner.
It is deliberately not a prediction of physical compiler registers.
"""

using BatchedKernels
using LinearAlgebra
const BK = BatchedKernels

function covariance_step(P, A, Q, H, R)
    predicted = A * P * A' + Q
    rhs = H * predicted
    U = cholesky(rhs * H' + R).U
    gain_t = U \ (U' \ rhs)
    return (I - gain_t' * H) * predicted
end

"""The assignment used by `benchmarking/kalman/hybrid_selective.jl`, without CUDA."""
function selective_assignment(tape, policy, nthreads; staging=:col, h_input=4)
    a = BK.Assignment(tape; nthreads)
    # Match forced_assignment(tape, :register) from hybrid_m6.jl.
    for (id, node) in enumerate(tape.nodes)
        node isa BK.CallNode || continue
        if BK._isplacement(node.fn)
            a.residences[id], a.orientations[id] = :register, :row
            a.orientations[only(node.args).id] = :row
        elseif node.fn in (*, +, -, cholesky, (\))
            a.residences[id], a.orientations[id] = :register, :row
            prefix = node.fn === (*) ? :matmul : node.fn === (+) ? :add : node.fn === (-) ? :sub : node.fn === cholesky ? :cholesky : :solve
            a.variants[id] = Symbol(prefix, :_row)
        end
    end
    placements, products, sums, factors = Dict{Int,Int}(), Int[], Int[], Int[]
    for (id, node) in enumerate(tape.nodes)
        node isa BK.CallNode || continue
        if BK._isplacement(node.fn)
            input = tape.nodes[only(tape.nodes[only(node.args).id].args).id]
            placements[input.index] = id
            input.index in (2, 4) && (a.orientations[id] = :col)
        elseif node.fn === (*)
            push!(products, id)
        elseif node.fn === (+)
            push!(sums, id)
        elseif node.fn === cholesky
            push!(factors, id)
        end
    end
    for id in products[end-1:end]
        a.orientations[id] = :col
        a.variants[id] = :matmul_col
    end
    if policy in (:shared_H, :shared_H_predicted)
        id = placements[h_input]
        a.residences[id] = :single
        a.orientations[only(tape.nodes[id].args).id] = :col
    end
    policy in (:shared_predicted, :shared_H_predicted) &&
        (a.residences[first(sums)] = :single)
    policy === :shared_factor && (a.residences[only(factors)] = :single)
    for (id, node) in enumerate(tape.nodes)
        node isa BK.CallNode && BK._isstage(node.fn) && (a.orientations[id] = staging)
    end
    raw = [i for (i, node) in enumerate(tape.nodes) if !(node isa BK.NewNode)]
    return BK.Assignment(tape;
        residences=Dict(i => a.residences[i] for i in raw if haskey(a.residences, i)),
        orientations=Dict(i => a.orientations[i] for i in raw if haskey(a.orientations, i)),
        variants=a.variants, nthreads)
end

function node_name(tape, id)
    node = tape.nodes[id]
    node isa BK.CallNode || return string(nameof(typeof(node)))
    fn = node.fn
    fn === (*) && return "*"
    fn === (+) && return "+"
    fn === (-) && return "-"
    fn === (\) && return "\\"
    fn === cholesky && return "cholesky"
    fn === BK._place_input && return "place"
    fn === BK._stage_output && return "stage"
    return string(fn)
end

# Exact subset DP for the register-element objective.  This differs from
# `BK.schedule`, whose objective is shared slots.  Results are fresh SSA values:
# the output is counted during its producer even if every input dies there.
function min_register_order(tape, assignment; D=32)
    probe = BK.plan_memory(tape, assignment; D_MAX=D)
    ids = BK._collect_schedulable(tape)
    length(ids) <= 63 || error("audit DP requires at most 63 schedulable nodes")
    dag = BK._reduced_dag(tape, ids)
    k = length(ids)
    # `consumers` already collapses NewNode wrappers.  All hybrid kernels here
    # use fresh outputs (assignment.jl rejects forced mutation for variants).
    weights = zeros(Int, k)
    for (bit, id) in enumerate(ids)
        owner = probe.owners[id]
        if owner == id && get(assignment.residences, id, nothing) === :register
            shape = BK._assignment_shape(tape.metas[id].type)
            weights[bit] = BK._register_line_elements(shape, assignment.orientations[id])
        end
    end
    full = (UInt64(1) << k) - UInt64(1)
    best = Dict{UInt64,Int}(UInt64(0) => 0)
    parent = Dict{UInt64,Tuple{UInt64,Int}}()
    for count in 0:k-1
        for S in [s for s in keys(best) if count_ones(s) == count]
            current = best[S]
            for i in 1:k
                bit = UInt64(1) << (i - 1)
                S & bit != 0 && continue
                dag.preds[i] & ~S != 0 && continue
                live = 0
                for j in 1:k
                    jbit = UInt64(1) << (j - 1)
                    S & jbit == 0 && continue
                    # A producer remains live if the candidate operation, or
                    # any later operation, still reads it.  Output is a staged
                    # CallNode here, so it is represented by a normal consumer.
                    dag.consumers[j] & ~S != 0 && (live += weights[j])
                end
                candidate = max(current, live + weights[i])
                next = S | bit
                old = get(best, next, typemax(Int))
                if candidate < old
                    best[next] = candidate
                    parent[next] = (S, i)
                end
            end
        end
    end
    order_bits = Int[]
    S = full
    while S != 0
        previous, bit = parent[S]
        push!(order_bits, bit)
        S = previous
    end
    reverse!(order_bits)
    sched = ids[order_bits]
    return BK._interleave_non_schedulable(tape, sched), best[full]
end

function report(policy; D=32, nthreads=128, f=covariance_step, specs=nothing, h_input=4)
    specs === nothing && (specs = BK.InputSpec[
        BK.LeafInput(BK.TraceMatrix{Float32,D,D}, BK.BATCHED) for _ in 1:5])
    tape = BK.hybrid_tape(BK.trace(f, specs))
    assignment = selective_assignment(tape, policy, nthreads; h_input)
    natural = collect(eachindex(tape.nodes))
    optimal, exact = min_register_order(tape, assignment; D)
    plans = [("trace", natural), ("register-optimal", optimal), ("shared-slot", BK.schedule(tape))]
    println("\nPOLICY=$policy D=$D threads=$nthreads")
    for (name, order) in plans
        a = BK.Assignment(assignment.residences, assignment.orientations, assignment.variants, order, assignment.nthreads)
        p = BK.plan_memory(tape, a; D_MAX=D)
        println("$name peak=$(p.peak_register_elements) ($(p.peak_register_elements ÷ D)D) single=$(p.num_single_slots) dual=$(p.num_dual_slots)")
        if name == "register-optimal"
            println("exact_DP=$(exact) ($(exact ÷ D)D)")
            pos = Dict(id => at for (at, id) in enumerate(order))
            println("full_order=$(order)")
            for at in eachindex(order)
                live = [i for i in eachindex(tape.nodes) if p.owners[i] == i && get(a.residences, i, :none) === :register && pos[i] <= at <= p.owner_last_use[i]]
                elems = sum(BK._register_line_elements(BK._assignment_shape(tape.metas[i].type), a.orientations[i]) for i in live; init=0)
                elems == p.peak_register_elements && println("  PEAK at position=$at after id=$(order[at]): owners=$live")
            end
            for (pos, id) in enumerate(order)
                node = tape.nodes[id]
                node isa BK.CallNode || continue
                residence = get(a.residences, id, :none)
                println("  $pos: id=$id $(node_name(tape,id)) args=$(getfield.(node.args,:id)) residence=$residence")
            end
        end
    end
end

report(:register)
report(:shared_H_predicted)

# GPU-facing candidate: two batched inputs first, then three block-shared inputs.
function covariance_step_PH(P, H, A, Q, R)
    return covariance_step(P, A, Q, H, R)
end
const PH_SHARED_SPECS = BK.InputSpec[
    BK.LeafInput(BK.TraceMatrix{Float32,32,32}, BK.BATCHED),
    BK.LeafInput(BK.TraceMatrix{Float32,32,32}, BK.BATCHED),
    BK.LeafInput(BK.TraceMatrix{Float32,32,32}, BK.SHARED),
    BK.LeafInput(BK.TraceMatrix{Float32,32,32}, BK.SHARED),
    BK.LeafInput(BK.TraceMatrix{Float32,32,32}, BK.SHARED),
]
report(:register; f=covariance_step_PH, specs=PH_SHARED_SPECS, h_input=2)
report(:shared_H_predicted; f=covariance_step_PH, specs=PH_SHARED_SPECS, h_input=2)
