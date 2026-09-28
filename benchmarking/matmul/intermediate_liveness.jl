# CPU-only, exact topological-order audit; no CuArray allocation or GPU launch.
# The subset DP follows the same methodology as the Kalman liveness audit.
include("intermediate_storage.jl")
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

for f in (retained_intermediates,reused_intermediates)
    tape=BK.trace(f,BK.InputSpec[BK.LeafInput(BK.TraceMatrix{Float32,32,32},BK.BATCHED) for _=1:2])
    products=[i for (i,n) in enumerate(tape.nodes) if n isa BK.CallNode && n.fn === (*)]
    a=intermediate_assignment(tape,:register,64)
    p=BK.plan_memory(tape,a;D_MAX=32)
    # At V's completion, X,Y,U,V all have future consumers in the natural order.
    @assert all(p.owner_last_use[id]>products[4] for id in products[1:4])
    order,peak=min_register_order(tape,a)
    println("$(nameof(f)): natural=$(p.peak_register_elements), minimum=$peak, order=$order")
    @assert peak==(f===retained_intermediates ? 160 : 128)
end
