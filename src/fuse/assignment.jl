# Explicit host-only assignments for the hybrid storage integration.
"""
    Assignment(tape; residences=Dict(), orientations=Dict(), variants=Dict(),
               order=collect(eachindex(tape.nodes)), nthreads=256)

Explicit host metadata for hybrid storage compilation. Matrix placements
use `:single` (`:row` or `:col`), `:dual` (`:both`), or `:register` (`:row` or
`:col`); input handles and shared inputs retain their assigned residence. Compute
calls default to `:legacy`, which requires dual-access batched operands. Named
orientation variants permit single or register storage. Multi-result calls default
to their first registered variant, with independent dual outputs born together at
the producer. Wrapper metadata follows its
parent, swapping row/column for transpose. Register values are fresh SSA values:
they have no shared slot; logical wrappers retain their parent ownership. `plan_memory` validates and snapshots the
assignment before allocating independent single/dual pools.
"""
struct Assignment
    residences::Dict{Int,Symbol}
    orientations::Dict{Int,Symbol}
    variants::Dict{Int,Symbol}
    order::Vector{Int}
    nthreads::Int
end

struct HybridPlannerOutput
    slots::Dict{Int,SlotAssignment}
    shared_slots::Dict{Int,SlotAssignment}
    scalar_output_slots::Dict{Int,SlotAssignment}
    num_single_slots::Int
    num_dual_slots::Int
    num_vector_slots::Int
    num_scalar_out_slots::Int
    num_shared_matrix_slots::Int
    num_shared_vector_slots::Int
    last_use::Dict{Int,Int}
    owners::Dict{Int,Int}
    owner_last_use::Dict{Int,Int}
    assignment::Assignment
    shared_bytes::Int
    peak_register_elements::Int
    D_MAX::Int
    element_type::DataType
end

_assignment_shape(T) = _variant_shape(T)
_assignment_shape(::Type{<:Symmetric{T,P}}) where {T,P} = _assignment_shape(P)
_isplacement(fn) = fn === _single_to_dual || fn === _place_input
_isstage(fn) = fn === _dual_to_single || fn === _stage_output
function _assignment_owner(owners, id)
    while owners[id] != id
        id = owners[id]
    end
    return id
end
_flip_assignment(o) =
    if o === :row
        :col
    elseif o === :col
        :row
    else
        :both
    end

# Each participating lane owns one logical line. The physical orientation names
# follow RegisterMatrix: row-oriented storage owns an M-wide line and col-oriented
# storage owns an N-wide line. This is an element count, deliberately separate
# from the compiler's eventual register allocation.
function _register_line_elements(shape::Tuple{Integer,Integer}, orientation::Symbol)
    m, n = shape
    orientation === :row && return m
    orientation === :col && return n
    return throw(ArgumentError("Register storage needs row or column orientation"))
end

function Assignment(
    tape::Tape;
    residences=Dict(),
    orientations=Dict(),
    variants=Dict(),
    order=collect(1:length(tape.nodes)),
    nthreads=256,
)
    rs = Dict{Int,Symbol}()
    os = Dict{Int,Symbol}()
    vs = Dict{Int,Symbol}()
    for (i, node) in enumerate(tape.nodes)
        node isa CallNode || continue
        vs[i] = :legacy
        if _is_multi_call(tape, i)
            candidates = orientation_variants(
                node.fn, (tape.metas[r.id].type for r in node.args)...
            )
            isempty(candidates) && throw(
                ArgumentError(
                    "Unsupported multi-result primitive or operand shapes at %$i"
                ),
            )
            vs[i] = first(candidates).id
        end
    end
    for (i, n) in enumerate(tape.nodes)
        m = tape.metas[i]
        _assignment_shape(m.type) === nothing && continue
        if n isa NewNode
            parents = [r.id for r in node_refs(n) if haskey(rs, r.id)]
            length(parents) == 1 || throw(
                ArgumentError("Assignment: matrix wrapper %$i needs one matrix parent")
            )
            p = only(parents)
            r = rs[p]
            o = os[p]
            m.type <: Union{Adjoint,Transpose} && (o = _flip_assignment(o))
        elseif n isa InputNode
            r = m.lifecycle == SHARED ? :shared_input : :global
            o = r === :shared_input ? :both : :row
        elseif n isa CallNode && (n.fn === _load_to_single || _isstage(n.fn))
            r = :single
            o = :row
        else
            r = :dual
            o = :both
        end
        rs[i] = get(residences, i, r)
        os[i] = get(
            orientations,
            i,
            if rs[i] === :dual
                :both
            elseif rs[i] in (:single, :register) && !(r in (:single, :register))
                :row
            else
                o
            end,
        )
    end
    all(i -> haskey(rs, i), keys(residences)) ||
        throw(ArgumentError("Assignment: residence key is not a matrix node"))
    all(i -> haskey(os, i), keys(orientations)) ||
        throw(ArgumentError("Assignment: orientation key is not a matrix node"))
    all(i -> haskey(vs, i), keys(variants)) ||
        throw(ArgumentError("Assignment: variant key is not a call node"))
    merge!(vs, variants)
    return Assignment(rs, os, vs, collect(Int, order), Int(nthreads))
end

function assignment_key(a::Assignment)
    return (
        Tuple(sort!(collect(a.residences))),
        Tuple(sort!(collect(a.orientations))),
        Tuple(sort!(collect(a.variants))),
        Tuple(a.order),
        a.nthreads,
    )
end

# Closed intervals deliberately prohibit allocating a fresh result over an input
# consumed by the same operation. Only explicit canonical aliases share storage.
function _allocate_owner_intervals(intervals)
    slots = Dict{Int,SlotAssignment}()
    counts = Dict(:Ms => 0, :Md => 0, :V => 0)
    active = Dict(k => Tuple{Int,Int}[] for k in keys(counts))
    free = Dict(k => Int[] for k in keys(counts))
    for (owner, kind, start, stop) in sort!(collect(intervals); by=x -> (x[3], x[1]))
        haskey(counts, kind) || throw(ArgumentError("Unknown storage pool $kind"))
        start <= stop || throw(ArgumentError("Invalid owner interval"))
        remaining = Tuple{Int,Int}[]
        for (endpos, slot) in active[kind]
            endpos < start ? push!(free[kind], slot) : push!(remaining, (endpos, slot))
        end
        active[kind] = remaining
        idx = isempty(free[kind]) ? (counts[kind] += 1) : pop!(free[kind])
        slots[owner] = SlotAssignment(kind, idx)
        push!(active[kind], (stop, idx))
    end
    return slots, counts
end

function plan_memory(
    tape::Tape, supplied::Assignment; D_MAX, T=Float32, max_shared_bytes=typemax(Int)
)
    D = Int(D_MAX isa Val ? typeof(D_MAX).parameters[1] : D_MAX)
    N = length(tape.nodes)
    a = Assignment(
        copy(supplied.residences),
        copy(supplied.orientations),
        copy(supplied.variants),
        copy(supplied.order),
        supplied.nthreads,
    )
    sort(a.order) == collect(1:N) ||
        throw(ArgumentError("Assignment order must be a permutation of tape nodes"))
    1 <= D <= 32 || throw(ArgumentError("D_MAX must be between 1 and 32"))
    32 <= a.nthreads <= 1024 && a.nthreads % 32 == 0 ||
        throw(ArgumentError("nthreads must be a whole number of warps, at most 1024"))
    T in (Float32, Float64) || throw(
        ArgumentError(
            "Hybrid assignments support Float32 and Float64; compute support is variant-specific",
        ),
    )
    expected = Assignment(tape)
    keys(a.residences) == keys(expected.residences) ||
        throw(ArgumentError("Incomplete or unknown residence keys"))
    keys(a.orientations) == keys(expected.orientations) ||
        throw(ArgumentError("Incomplete or unknown orientation keys"))
    keys(a.variants) == keys(expected.variants) ||
        throw(ArgumentError("Incomplete or unknown variant keys"))
    pos = Dict(i => p for (p, i) in enumerate(a.order))
    owners = canonical_storage_owners(tape)
    for (i, node) in enumerate(tape.nodes)
        all(r -> haskey(pos, r.id) && pos[r.id] < pos[i], node_refs(node)) ||
            throw(ArgumentError("Assignment violates dependency order at %$i"))
        if node isa ResultNode
            _is_multi_call(tape, node.producer.id) ||
                throw(ArgumentError("Result %$i has no multi-result producer"))
            call_result_ids(tape, node.producer.id)
            (!haskey(a.residences, i) || a.residences[i] in (:single, :dual, :register)) ||
                throw(ArgumentError("Result %$i needs fresh computed storage"))
        end
        if haskey(a.residences, i)
            logicalshape = _assignment_shape(tape.metas[i].type)
            _variant_eltype(tape.metas[i].type) === T ||
                throw(ArgumentError("Matrix element type disagrees with planner at %$i"))
            all(x -> 1 <= x <= D, logicalshape) ||
                throw(ArgumentError("Matrix shape at %$i exceeds D_MAX"))
            r, o = a.residences[i], a.orientations[i]
            r in (:global, :shared_input, :single, :dual, :register) ||
                throw(ArgumentError("Unsupported residence $r at %$i"))
            o in (:row, :col, :both) || throw(ArgumentError("Invalid orientation at %$i"))
            r in (:dual, :shared_input) &&
                o !== :both &&
                throw(ArgumentError("Dual/shared input must be both-oriented"))
            r in (:single, :register) &&
                !(o in (:row, :col)) &&
                throw(
                    ArgumentError("Single/register storage needs row or column orientation")
                )
            if node isa InputNode
                r === expected.residences[i] ||
                    throw(ArgumentError("Input residence cannot be changed at %$i"))
            elseif node isa NewNode
                r === :register &&
                    _variant_shape(tape.metas[i].type) === nothing &&
                    throw(
                        ArgumentError(
                            "Register storage is unsupported for this matrix wrapper at %$i"
                        ),
                    )
                ps = [x.id for x in node_refs(node) if haskey(a.residences, x.id)]
                length(ps) == 1 || throw(ArgumentError("Unsupported matrix wrapper at %$i"))
                p = only(ps)
                want = if tape.metas[i].type <: Union{Adjoint,Transpose}
                    _flip_assignment(a.orientations[p])
                else
                    a.orientations[p]
                end
                r === a.residences[p] && o === want || throw(
                    ArgumentError(
                        "Wrapper residence/orientation disagrees with parent at %$i"
                    ),
                )
                owners[i] = owners[p]
            elseif node isa CallNode
                fn = node.fn
                if fn === _load_to_single || _isstage(fn)
                    r === :single || throw(
                        ArgumentError("Global transfer staging must use single storage")
                    )
                elseif _isplacement(fn)
                    r in (:single, :dual, :register) ||
                        throw(ArgumentError("Invalid input placement"))
                elseif r in (:global, :shared_input)
                    throw(
                        ArgumentError(
                            "Computed values need batched shared or register storage"
                        ),
                    )
                end
                if fn === _load_to_single
                    length(node.args) == 1 ||
                        throw(ArgumentError("Load requires one operand"))
                    p = only(node.args).id
                    tape.nodes[p] isa InputNode &&
                    get(a.residences, p, nothing) === :global ||
                        throw(ArgumentError("Load requires a global matrix input"))
                    _assignment_shape(tape.metas[p].type) == logicalshape ||
                        throw(ArgumentError("Load shape mismatch"))
                end
                if _isplacement(fn) || _isstage(fn)
                    length(node.args) == 1 ||
                        throw(ArgumentError("Transfer requires one operand"))
                    p = only(node.args).id
                    haskey(a.residences, p) ||
                        throw(ArgumentError("Transfer requires matrix input"))
                    _assignment_shape(tape.metas[p].type) == logicalshape ||
                        throw(ArgumentError("Transfer shape mismatch"))
                    if _isplacement(fn) && r === :single && o !== a.orientations[p]
                        throw(
                            ArgumentError(
                                "Single input placement cannot change orientation; assign the load orientation explicitly",
                            ),
                        )
                    end
                    if _isplacement(fn)
                        a.residences[p] === :single && !(tape.nodes[p] isa NewNode) ||
                            throw(
                                ArgumentError("Input placement requires raw single storage")
                            )
                    else
                        a.residences[p] in (:single, :dual, :shared_input, :register) ||
                            throw(
                                ArgumentError(
                                    "Output staging requires a logical shared or register view",
                                ),
                            )
                    end
                    # A structured or remapped view requires materialization.
                    raw = !(tape.nodes[p] isa NewNode)
                    matching = raw && r === a.residences[p] && o === a.orientations[p]
                    future_write =
                        _isstage(fn) && any(
                            j ->
                                j > i && any(
                                    x ->
                                        _assignment_owner(owners, x.id) ==
                                        _assignment_owner(owners, p),
                                    _mutation_target_refs(tape, tape.nodes[j]),
                                ),
                            1:N,
                        )
                    matching && !future_write && (owners[i] = owners[p])
                end
                target = _maybe_inplace_idx(tape, node)
                if target !== nothing
                    p = node.args[target].id
                    logicalshape == _assignment_shape(tape.metas[p].type) ||
                        throw(ArgumentError("Forced mutation cannot change logical shape"))
                    a.residences[i] !== :register && a.residences[p] !== :register ||
                        throw(ArgumentError("Forced mutation requires shared storage"))
                    r === a.residences[p] && o === a.orientations[p] ||
                        throw(ArgumentError("Forced mutation cannot change storage map"))
                    for (k, ref) in enumerate(node.args)
                        k == target && continue
                        owners[ref.id] == owners[p] && throw(
                            ArgumentError(
                                "Forced mutation has an overlapping non-target operand"
                            ),
                        )
                    end
                    owners[i] = owners[p]
                end
            end
        end
        node isa CallNode || continue
        outputs = call_result_ids(tape, i)
        variant = a.variants[i]
        _is_multi_call(tape, i) &&
            variant === :legacy &&
            throw(ArgumentError("Multi-result primitive at %$i requires a named variant"))
        staging = node.fn === _load_to_single || _isplacement(node.fn) || _isstage(node.fn)
        if staging
            variant === :legacy ||
                throw(ArgumentError("Transfer nodes do not accept compute variants"))
            continue
        end
        matrixargs = [x.id for x in node.args if haskey(a.residences, x.id)]
        if variant === :legacy
            for p in matrixargs
                a.residences[p] in (:dual, :shared_input) || throw(
                    ArgumentError(
                        "Legacy primitive at %$i requires dual-access matrix operands"
                    ),
                )
            end
            haskey(a.residences, i) &&
                a.residences[i] !== :dual &&
                throw(ArgumentError("Legacy primitive needs dual output"))
        else
            _maybe_inplace_idx(tape, node) === nothing ||
                throw(ArgumentError("Hybrid variants require fresh outputs"))
            candidates = orientation_variants(
                node.fn, (tape.metas[x.id].type for x in node.args)...
            )
            found = findfirst(v -> v.id === variant, candidates)
            found === nothing && throw(ArgumentError("Unsupported variant $variant at %$i"))
            v = candidates[found]
            inputshapes = [_assignment_shape(tape.metas[p].type) for p in matrixargs]
            outputshape = if v.shape_rule === :matmul
                (inputshapes[1][1], inputshapes[2][2])
            elseif v.shape_rule === :triangular_solve
                inputshapes[2]
            elseif v.shape_rule in (:matvec, :vector_solve)
                (inputshapes[1][1],)
            elseif v.shape_rule === :qr_stack
                inputshapes[2]
            elseif v.shape_rule === :qr_blocks
                (inputshapes[1], (inputshapes[1][1], inputshapes[3][2]), inputshapes[3])
            elseif v.shape_rule === :qr_identity
                (inputshapes[1][1], inputshapes[1][1])
            elseif v.shape_rule === :qr_residual
                n = inputshapes[1][2]
                ((n, n), (n,), ())
            elseif v.shape_rule === :scalar_logdet
                ()
            else
                inputshapes[1]
            end
            actualshape = if _is_multi_call(tape, i)
                Tuple(shape(tape.metas[o].type) for o in outputs)
            else
                shape(tape.metas[i].type)
            end
            actualshape == outputshape ||
                throw(ArgumentError("Variant output shape mismatch at %$i"))
            length(matrixargs) == length(v.input_access) ||
                throw(ArgumentError("Variant operand mismatch"))
            for (k, p) in enumerate(matrixargs)
                a.residences[p] in v.input_residences[k] ||
                    throw(ArgumentError("Variant input residence mismatch"))
                req = v.input_access[k]
                req === :any ||
                    a.orientations[p] in (req, :both) ||
                    throw(ArgumentError("Variant input orientation mismatch at %$i"))
            end
            accesses = v.output_access isa Tuple ? v.output_access : (v.output_access,)
            length(outputs) == length(accesses) ||
                throw(ArgumentError("Variant result count mismatch"))
            for (out, access) in zip(outputs, accesses)
                haskey(a.residences, out) || continue
                a.residences[out] in v.output_residences ||
                    throw(ArgumentError("Variant output residence mismatch"))
                a.orientations[out] in (access, :both) ||
                    throw(ArgumentError("Variant output orientation mismatch at %$out"))
            end
        end
    end
    for i in eachindex(tape.nodes)
        owners[i] = _assignment_owner(owners, i)
    end
    for (before, after) in mutation_dependencies(tape; owners)
        pos[before] < pos[after] ||
            throw(ArgumentError("Assignment violates mutation ordering"))
    end
    lastuse = Dict(i => pos[i] for i in 1:N)
    for (i, node) in enumerate(tape.nodes), ref in node_refs(node)
        lastuse[ref.id] = max(lastuse[ref.id], pos[i])
    end
    tape.output === nothing && throw(ArgumentError("Tape has no output"))
    lastuse[tape.output.id] = N + 1
    for i in reverse(1:N)
        tape.nodes[i] isa NewNode || continue
        for ref in node_refs(tape.nodes[i])
            lastuse[ref.id] = max(lastuse[ref.id], lastuse[i])
        end
    end
    ownerlast = Dict{Int,Int}()
    for i in 1:N
        ownerlast[owners[i]] = max(get(ownerlast, owners[i], 0), lastuse[i])
    end
    intervals = Tuple{Int,Symbol,Int,Int}[]
    # Register values have fresh SSA ownership.  Count the line elements live at
    # every schedule point; this exposes a placement-level pressure proxy without
    # pretending it equals the compiler's physical register allocation.
    register_intervals = Tuple{Int,Int,Int}[]
    for i in 1:N
        owners[i] == i || continue
        haskey(a.residences, i) && a.residences[i] === :register || continue
        width = _register_line_elements(
            _assignment_shape(tape.metas[i].type), a.orientations[i]
        )
        push!(register_intervals, (_result_birth(tape, i, pos), ownerlast[i], width))
    end
    peak_register_elements = maximum(
        (
            sum(
                (
                    width for
                    (start, stop, width) in register_intervals if start <= at <= stop
                );
                init=0,
            ) for at in 1:(N + 1)
        );
        init=0,
    )
    shared = Dict{Int,SlotAssignment}()
    sharedM = 0
    sharedV = 0
    dedicated_elems = Int[]
    for (i, node) in enumerate(tape.nodes)
        meta = tape.metas[i]
        if node isa InputNode && meta.lifecycle == SHARED
            if haskey(a.residences, i)
                sharedM += 1
                shared[i] = SlotAssignment(:M, sharedM)
                m, n = _assignment_shape(meta.type)
                pad = (32 ÷ (m & -m)) * m
                push!(dedicated_elems, m * n + (m * n - 1) ÷ pad)
            elseif meta.type <: TraceVector
                sharedV += 1
                shared[i] = SlotAssignment(:V, sharedV)
                push!(dedicated_elems, meta.type.parameters[2])
            end
        elseif meta.lifecycle == SHARED && node isa CallNode
            throw(ArgumentError("Shared derived values are unsupported"))
        end
        owners[i] == i || continue
        if haskey(a.residences, i) && a.residences[i] in (:single, :dual)
            kind = a.residences[i] === :single ? :Ms : :Md
            push!(intervals, (i, kind, _result_birth(tape, i, pos), ownerlast[i]))
        elseif meta.lifecycle == BATCHED && meta.type <: TraceVector && !(node isa NewNode)
            push!(intervals, (i, :V, _result_birth(tape, i, pos), ownerlast[i]))
        end
    end
    owner_slots, counts = _allocate_owner_intervals(intervals)
    slots = Dict(i => owner_slots[owners[i]] for i in 1:N if haskey(owner_slots, owners[i]))
    scalar = Dict{Int,SlotAssignment}()
    next = Ref(1)
    _collect_scalar_output_slots!(scalar, next, tape, tape.output)
    groups = (32 ÷ D) * (a.nthreads ÷ 32)
    # CUDA.emit_shmem (CuStaticSharedArray's allocator) requests
    # max(32, datatype_alignment(T)) bytes of alignment for EACH global, to
    # permit WMMA/vectorized accesses. Raw region footprints omit the holes
    # between these independently allocated globals. Round every allocation up
    # individually: this is an order-independent upper bound, including a
    # conservative final tail. Do not assume the assembler preserves emission
    # order. Compiled shared-memory resource checks remain the final authority.
    alignment = max(32, Base.datatype_alignment(T))
    allocation_bytes(n) = cld(sizeof(T) * n, alignment) * alignment
    bytes =
        counts[:Ms] * allocation_bytes(Int(single_region_elems(Val(D), Val(a.nthreads)))) +
        counts[:Md] * allocation_bytes(Int(dual_region_elems(Val(D), Val(a.nthreads)))) +
        counts[:V] * allocation_bytes(groups * D) +
        length(scalar) * allocation_bytes(groups) +
        sum(allocation_bytes, dedicated_elems; init=0)
    bytes <= max_shared_bytes ||
        throw(ArgumentError("Assignment exceeds shared-memory budget"))
    return HybridPlannerOutput(
        slots,
        shared,
        scalar,
        counts[:Ms],
        counts[:Md],
        counts[:V],
        length(scalar),
        sharedM,
        sharedV,
        lastuse,
        owners,
        ownerlast,
        a,
        bytes,
        peak_register_elements,
        D,
        T,
    )
end
