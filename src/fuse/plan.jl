# =============================================================================
# Memory planner
# =============================================================================
#
# Assigns shared-memory slots to batched tape values. Strategy:
#   - Compute the last use of every node (with wrapper NewNodes extending their
#     parents' lifetimes so wrappers see the wrapped value).
#   - Shared inputs get their own dedicated slot.
#   - Batched values are placed via a greedy free-list-by-last-use allocator.
#   - Explicit mutating primitives (registered in `inplace_arg`) alias the
#     result's slot to the overwritten argument's slot.
#   - Auto in-place: for non-mutating primitives, if any operand declared
#     safe by `inplace_safe_args` is dead at the current node and directly
#     slot-backed (not wrapped through a NewNode chain — wrappers like
#     Adjoint/IAddSubWrapped remap reads and aliasing through them would
#     race against the in-place write), alias the result's slot to that
#     operand's. This is a pure slot-reuse optimisation.
#
# Matrix and vector slots come from separate pools because the shared-memory
# layouts differ: matrix slots reserve a dual-access D_MAX×D_MAX region per
# batch element, vector slots reserve a single-access D_MAX region. A slot
# assignment is tagged with its kind (`:M`, `:V`, or `:Sout`) and an index
# within its pool.
#
# Scalars (`TraceScalar`) do not occupy a shared-memory slot during compute —
# they live in registers, replicated across the D lanes of a warp-matrix. The
# only shmem they need is one `n_mats_per_block`-sized staging buffer per
# distinct scalar terminal in the output tree, used to cross from leader-lane
# registers to a coalesced cross-warp write. These are the `:Sout` slots.

struct SlotAssignment
    kind::Symbol            # :M (matrix), :V (vector), or :Sout (scalar output staging)
    idx::Int                # 1-based index within the pool
end

struct PlannerOutput
    slots::Dict{Int,SlotAssignment}             # batched node -> slot
    shared_slots::Dict{Int,SlotAssignment}      # shared input -> slot
    scalar_output_slots::Dict{Int,SlotAssignment}  # scalar tape ref -> :Sout slot
    num_matrix_slots::Int
    num_vector_slots::Int
    num_scalar_out_slots::Int
    num_shared_matrix_slots::Int
    num_shared_vector_slots::Int
    last_use::Dict{Int,Int}
end

function slot_kind(node::TapeNode, meta::NodeMeta)
    node isa ConstNode && return :literal
    node isa NewNode && return :inline
    if meta.lifecycle == BATCHED
        return meta.type <: TraceScalar ? :scalar : :batched
    elseif meta.lifecycle == SHARED
        return :shared
    else
        return :literal
    end
end

# Which slot pool (:M or :V) a (batched or shared) node belongs to, based on
# its trace-time type.
function pool_kind(meta::NodeMeta)
    T = meta.type
    T <: TraceMatrix && return :M
    T <: TraceVector && return :V
    return error("pool_kind: unsupported batched/shared trace type $T")
end

function plan_memory(tape::Tape; order::AbstractVector{Int}=1:length(tape.nodes))
    _require_single_result_calls(tape)
    N = length(tape.nodes)
    length(order) == N ||
        error("plan_memory: order length $(length(order)) ≠ tape length $N")

    # Position of each node in the chosen execution order. Last-use is in
    # *position* coordinates so the free-list semantics carry over unchanged
    # even when codegen walks the tape out of natural order.
    pos = Dict{Int,Int}()
    for (p, id) in enumerate(order)
        pos[id] = p
    end

    last_use = Dict{Int,Int}()
    for (p, id) in enumerate(order)
        node = tape.nodes[id]
        for ref in node_refs(node)
            last_use[ref.id] = max(get(last_use, ref.id, 0), p)
        end
    end
    last_use[tape.output.id] = N + 1

    # Wrapper NewNodes extend the lifetime of their parents.
    changed = true
    while changed
        changed = false
        for (i, node) in enumerate(tape.nodes)
            if node isa NewNode
                wrapper_last = get(last_use, i, 0)
                for (_, parent_ref) in node.fields
                    p_last = get(last_use, parent_ref.id, 0)
                    if wrapper_last > p_last
                        last_use[parent_ref.id] = wrapper_last
                        changed = true
                    end
                end
            end
        end
    end

    canonical = canonical_storage_owners(tape)
    owner_last = Dict{Int,Int}()
    for i in 1:N
        root = canonical[i]
        owner_last[root] = max(get(owner_last, root, 0), get(last_use, i, 0))
    end
    for i in 1:N
        last_use[i] = owner_last[canonical[i]]
    end

    shared_slots = Dict{Int,SlotAssignment}()
    next_shared_M = 1
    next_shared_V = 1
    for (i, (node, meta)) in enumerate(zip(tape.nodes, tape.metas))
        if node isa InputNode && meta.lifecycle == SHARED
            k = pool_kind(meta)
            if k === :M
                shared_slots[i] = SlotAssignment(:M, next_shared_M)
                next_shared_M += 1
            else
                shared_slots[i] = SlotAssignment(:V, next_shared_V)
                next_shared_V += 1
            end
        elseif meta.lifecycle == SHARED &&
            !(node isa InputNode || node isa NewNode || node isa ConstNode)
            error(
                "plan_memory: shared derived value at node %$i; V2 only supports shared *inputs*.",
            )
        end
    end

    slots = Dict{Int,SlotAssignment}()
    free_M = Int[]
    free_V = Int[]
    next_M = 1
    next_V = 1

    for (p, i) in enumerate(order)
        node = tape.nodes[i]
        meta = tape.metas[i]
        sk = slot_kind(node, meta)
        sk in (:batched, :scalar) || continue
        # Batched matrix InputNodes are not slot-backed: their value lives in
        # global memory and reaches a shmem slot only via the LoadNode +
        # TransferNode pair pushed at trace time. The InputNode itself is just
        # a handle to the global pointer. Vectors keep their direct
        # InputNode-backed slot (no dual-layout buffer, so no Load/Transfer
        # split).
        if node isa InputNode && meta.type <: TraceMatrix
            continue
        end

        # Destination slot allocation: only matrix/vector consumers get a slot.
        # Scalars live in registers, so we skip this for sk == :scalar — but we
        # still fall through to the free-list pass so any matrix/vector inputs
        # the scalar consumed can be recycled.
        if sk == :batched
            k = pool_kind(meta)
            free_list = k === :M ? free_M : free_V

            in_place_idx = _maybe_inplace_idx(tape, node)
            auto_inplace_slot = if in_place_idx === nothing
                _maybe_auto_inplace_slot(tape, node, p, k, slots, last_use)
            else
                nothing
            end
            if in_place_idx !== nothing
                target_ref = node.args[in_place_idx]
                owner = resolve_slot_owner(tape, target_ref)
                haskey(slots, owner) || error(
                    "plan_memory: in-place op at %$i references unassigned slot (owner %$owner).",
                )
                slots[i] = slots[owner]
            elseif auto_inplace_slot !== nothing
                slots[i] = auto_inplace_slot
            elseif isempty(free_list)
                if k === :M
                    slots[i] = SlotAssignment(:M, next_M)
                    next_M += 1
                else
                    slots[i] = SlotAssignment(:V, next_V)
                    next_V += 1
                end
            else
                slots[i] = SlotAssignment(k, pop!(free_list))
            end
        end

        # Recurse through NewNode wrappers so a value consumed only through a
        # wrapper chain (`LowerTriangular(chol.U') \ x`, `(I - KH) * P_pred`,
        # `K_T' * H`, …) still has its slot freed at the wrapper's last use.
        # Without recursion the underlying value's slot stays live to the end
        # of the kernel, inflating the slot count by one per such value.
        dest_slot = sk == :batched ? slots[i] : nothing
        freed_here = Set{Tuple{Symbol,Int}}()
        for ref in node_refs(node)
            _free_dead_arg!(
                tape, slots, last_use, free_M, free_V, freed_here, ref, p, sk, dest_slot
            )
        end
    end

    scalar_output_slots = Dict{Int,SlotAssignment}()
    next_Sout = Ref(1)
    _collect_scalar_output_slots!(scalar_output_slots, next_Sout, tape, tape.output)

    return PlannerOutput(
        slots,
        shared_slots,
        scalar_output_slots,
        next_M - 1,
        next_V - 1,
        next_Sout[] - 1,
        next_shared_M - 1,
        next_shared_V - 1,
        last_use,
    )
end

# Walk the output tree; allocate one :Sout staging slot per distinct scalar
# tape ref reached as a (non-NewNode, non-ConstNode) leaf. The same scalar
# value reused in two output positions shares one staging slot — two
# `scalar_write!` calls then read from it for two independent global writes.
function _collect_scalar_output_slots!(
    scalar_out::Dict{Int,SlotAssignment}, next_idx::Ref{Int}, tape::Tape, ref::NodeRef
)
    node = tape.nodes[ref.id]
    if node isa NewNode
        for (_, child) in node.fields
            _collect_scalar_output_slots!(scalar_out, next_idx, tape, child)
        end
    elseif node isa ConstNode
        return nothing
    else
        meta = tape.metas[ref.id]
        if meta.type <: TraceScalar && !haskey(scalar_out, ref.id)
            scalar_out[ref.id] = SlotAssignment(:Sout, next_idx[])
            next_idx[] += 1
        end
    end
    return nothing
end

node_refs(::InputNode) = NodeRef[]
node_refs(::ConstNode) = NodeRef[]
node_refs(n::CallNode) = n.args
node_refs(n::ResultNode) = NodeRef[n.producer]
node_refs(n::NewNode) = NodeRef[p.second for p in n.fields]

function resolve_slot_owner(tape::Tape, ref::NodeRef)
    n = tape.nodes[ref.id]
    if n isa NewNode
        for (_, parent_ref) in n.fields
            pn = tape.nodes[parent_ref.id]
            pn isa ConstNode && continue
            return resolve_slot_owner(tape, parent_ref)
        end
        error("resolve_slot_owner: NewNode at %$(ref.id) has no non-literal field refs")
    else
        return ref.id
    end
end

# Walk an arg ref and free the slot of every dead slot-backed value reachable
# through any NewNode wrapper chain. `freed_here` dedupes when the same
# underlying value is reached through multiple paths (e.g. `a + a`, or a tuple
# with two refs to the same node) so we don't push the same slot index onto
# the free list twice.
function _free_dead_arg!(
    tape::Tape,
    slots::Dict{Int,SlotAssignment},
    last_use::Dict{Int,Int},
    free_M::Vector{Int},
    free_V::Vector{Int},
    freed_here::Set{Tuple{Symbol,Int}},
    ref::NodeRef,
    pos::Int,
    sk::Symbol,
    dest_slot::Union{SlotAssignment,Nothing},
)
    arg_node = tape.nodes[ref.id]
    if arg_node isa NewNode
        for (_, child) in arg_node.fields
            _free_dead_arg!(
                tape, slots, last_use, free_M, free_V, freed_here, child, pos, sk, dest_slot
            )
        end
        return nothing
    end
    haskey(slots, ref.id) || return nothing
    get(last_use, ref.id, 0) == pos || return nothing
    owner_slot = slots[ref.id]
    # Don't free the dest's own slot when the current op was in-placed onto
    # this arg — the result lives in it now.
    if dest_slot !== nothing &&
        owner_slot.idx == dest_slot.idx &&
        owner_slot.kind === dest_slot.kind
        return nothing
    end
    any(j -> slots[j] == owner_slot && get(last_use, j, 0) > pos, keys(slots)) &&
        return nothing
    key = (owner_slot.kind, owner_slot.idx)
    key in freed_here && return nothing
    push!(freed_here, key)
    if owner_slot.kind === :M
        push!(free_M, owner_slot.idx)
    else
        push!(free_V, owner_slot.idx)
    end
    return nothing
end

function _maybe_inplace_idx(tape::Tape, node::TapeNode)
    node isa CallNode || return nothing
    arg_types = Tuple(meta_at(tape, r).type for r in node.args)
    applicable(inplace_arg, node.fn, arg_types...) || return nothing
    return inplace_arg(node.fn, arg_types...)
end

# Auto in-place: return a SlotAssignment to alias onto, or `nothing` if no
# eligible operand. Eligibility requires:
#   - the operand is declared safe by `inplace_safe_args` for this primitive;
#   - it is a direct slot-backed CallNode/InputNode (not wrapped through a
#     NewNode chain — Adjoint/IAddSubWrapped/etc. remap reads and aliasing
#     through them would race against the in-place write);
#   - its owner is in the batched-slot table (not shared, not unallocated);
#   - its pool kind matches the destination's;
#   - its last use is the current node (so no later consumer is lost).
function _maybe_auto_inplace_slot(
    tape::Tape,
    node::TapeNode,
    i::Int,
    dest_kind::Symbol,
    slots::Dict{Int,SlotAssignment},
    last_use::Dict{Int,Int},
)
    node isa CallNode || return nothing
    arg_types = Tuple(meta_at(tape, r).type for r in node.args)
    applicable(inplace_safe_args, node.fn, arg_types...) || return nothing
    safe_positions = inplace_safe_args(node.fn, arg_types...)
    isempty(safe_positions) && return nothing

    for k in safe_positions
        1 <= k <= length(node.args) || continue
        ref = node.args[k]
        arg_node = tape.nodes[ref.id]
        # Skip wrapped / literal operands: aliasing through a NewNode wrapper
        # would either race (the wrapper remaps reads against the destination's
        # writes) or has no slot to alias to (ConstNode).
        (arg_node isa CallNode || arg_node isa InputNode) || continue
        haskey(slots, ref.id) || continue
        slot = slots[ref.id]
        slot.kind === dest_kind || continue
        get(last_use, ref.id, 0) == i || continue
        _legacy_alias_candidate_safe(tape, node, k) || continue
        any(node.args) do other
            tape.nodes[other.id] isa NewNode || return false
            root = resolve_slot_owner(tape, other)
            return haskey(slots, root) && slots[root] == slot
        end && continue
        any(j -> slots[j] == slot && get(last_use, j, 0) > i, keys(slots)) && continue
        return slot
    end
    return nothing
end

# Identity of mutable storage, distinct from SSA names and logical wrappers.
function canonical_storage_owners(tape::Tape)
    owners = Dict{Int,Int}()
    for (i, node) in enumerate(tape.nodes)
        owner = i
        if node isa NewNode
            refs = [r for r in node_refs(node) if tape.metas[r.id].lifecycle != LITERAL]
            length(refs) == 1 && (owner = get(owners, only(refs).id, only(refs).id))
        else
            target = _maybe_inplace_idx(tape, node)
            target === nothing || (owner = owners[node.args[target].id])
        end
        owners[i] = owner
    end
    return owners
end

# An overlapping remapped view cannot share the destination of a pointwise op.
function _legacy_alias_candidate_safe(tape::Tape, node::CallNode, k::Int)
    owners = canonical_storage_owners(tape)
    target = node.args[k].id
    tape.nodes[target] isa NewNode && return false
    for ref in node.args
        owners[ref.id] == owners[target] || continue
        tape.nodes[ref.id] isa NewNode && return false
    end
    return true
end
