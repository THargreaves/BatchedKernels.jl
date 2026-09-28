export automatic_assignment

"""
    automatic_assignment(tape; nthreads=128)

Build a deterministic register-first assignment in tape order. Operation contracts
constrain physical orientations through wrappers. A value whose consumers require
incompatible orientations is produced in dual-access shared memory. This is a
bounded greedy legality policy, not a throughput or register-pressure optimizer.
Vectors keep shared storage; scalars use registers. Unsupported hybrid primitives
retain their legacy bodies and require dual matrix operands. Legacy restrictions
still apply and unsupported graphs fail during validation/emission.
"""
function automatic_assignment(tape::Tape; nthreads::Int=128)
    defaults = Assignment(tape; nthreads)
    # Preserve explicit mutation semantics without introducing register aliases.
    any(n -> n isa CallNode && _maybe_inplace_idx(tape, n) !== nothing, tape.nodes) &&
        return defaults

    roots = Dict{Int,Int}()
    flipped = Dict{Int,Bool}()
    for (i, node) in enumerate(tape.nodes)
        haskey(defaults.residences, i) || continue
        if node isa NewNode
            parent = only(r.id for r in node_refs(node) if haskey(roots, r.id))
            roots[i] = roots[parent]
            flipped[i] = xor(
                flipped[parent], tape.metas[i].type <: Union{Adjoint,Transpose}
            )
        else
            roots[i], flipped[i] = i, false
        end
    end
    # A zero mask is unconstrained; 1/2 require row/column; 3 requires both.
    demands = Dict{Int,Int}()
    requirements = Dict(:any => 0, :row => 1, :col => 2, :both => 3)
    function demand!(target, id, access)
        root = roots[id]
        defaults.residences[root] in (:global, :shared_input) && return nothing
        req = requirements[access]
        if flipped[id] && req in (1, 2)
            req = 3 - req
        end
        return target[root] = get(target, root, 0) | req
    end
    # Unhandled wrappers must stay on the shared accessor path.
    for (i, node) in enumerate(tape.nodes)
        haskey(roots, i) || continue
        node isa NewNode &&
            _variant_shape(tape.metas[i].type) === nothing &&
            demand!(demands, i, :both)
    end
    variants = Dict{Int,Symbol}()
    for (i, node) in enumerate(tape.nodes)
        node isa CallNode || continue
        if node.fn === _load_to_single || _isplacement(node.fn) || _isstage(node.fn)
            continue
        end
        matrixargs = [r.id for r in node.args if haskey(roots, r.id)]
        candidates = orientation_variants(
            node.fn, (tape.metas[r.id].type for r in node.args)...
        )
        if isempty(candidates)
            for p in matrixargs
                demand!(demands, p, :both)
            end
            haskey(roots, i) && demand!(demands, i, :both)
        else
            best, bestdemands, bestscore = nothing, demands, typemax(Int)
            for candidate in candidates
                trial = copy(demands)
                for (p, access) in zip(matrixargs, candidate.input_access)
                    demand!(trial, p, access)
                end
                outputs = call_result_ids(tape, i)
                accesses = if candidate.output_access isa Tuple
                    candidate.output_access
                else
                    (candidate.output_access,)
                end
                for (out, access) in zip(outputs, accesses)
                    haskey(roots, out) && demand!(trial, out, access)
                end
                # Minimize the number of elements promoted to shared storage.
                # Stable registry order breaks ties; no shape threshold or timing.
                score = sum(
                    (
                        prod(_assignment_shape(tape.metas[p].type)) for
                        (p, req) in trial if req == 3
                    );
                    init=0,
                )
                if score < bestscore
                    best, bestdemands, bestscore = candidate, trial, score
                end
            end
            variants[i] = best.id
            demands = bestdemands
        end
    end
    residences, orientations = Dict{Int,Symbol}(), Dict{Int,Symbol}()
    for (i, node) in enumerate(tape.nodes)
        haskey(roots, i) || continue
        node isa Union{NewNode,InputNode} && continue
        node isa CallNode && (node.fn === _load_to_single || _isstage(node.fn)) && continue
        req = get(demands, i, 0)
        residences[i] = req == 3 ? :dual : :register
        orientations[i] = if req == 3
            :both
        elseif req == 2
            :col
        else
            :row
        end
        if node isa CallNode && _isplacement(node.fn) && req != 3
            orientations[only(node.args).id] = orientations[i]
        end
    end
    provisional = Assignment(tape; residences, orientations, variants, nthreads)
    for (i, node) in enumerate(tape.nodes)
        node isa CallNode && _isstage(node.fn) || continue
        source = only(node.args).id
        orientation = provisional.orientations[source]
        orientations[i] = orientation === :both ? :row : orientation
    end
    return Assignment(tape; residences, orientations, variants, nthreads)
end
