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
#
# Matrix and vector slots come from separate pools because the shared-memory
# layouts differ: matrix slots reserve a dual-access D_MAX×D_MAX region per
# batch element, vector slots reserve a single-access D_MAX region. A slot
# assignment is tagged with its kind (`:M` or `:V`) and an index within its
# pool.

struct SlotAssignment
    kind::Symbol            # :M (matrix) or :V (vector)
    idx::Int                # 1-based index within the pool
end

struct PlannerOutput
    slots::Dict{Int,SlotAssignment}             # batched node -> slot
    shared_slots::Dict{Int,SlotAssignment}      # shared input -> slot
    num_matrix_slots::Int
    num_vector_slots::Int
    num_shared_matrix_slots::Int
    num_shared_vector_slots::Int
    last_use::Dict{Int,Int}
end

function slot_kind(node::TapeNode, meta::NodeMeta)
    node isa ConstNode && return :literal
    node isa NewNode && return :inline
    if meta.lifecycle == BATCHED
        return :batched
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

function plan_memory(tape::Tape)
    N = length(tape.nodes)

    last_use = Dict{Int,Int}()
    for (i, node) in enumerate(tape.nodes)
        for ref in node_refs(node)
            last_use[ref.id] = max(get(last_use, ref.id, 0), i)
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

    for (i, (node, meta)) in enumerate(zip(tape.nodes, tape.metas))
        slot_kind(node, meta) == :batched || continue
        k = pool_kind(meta)
        free_list = k === :M ? free_M : free_V

        in_place_idx = _maybe_inplace_idx(tape, node)
        if in_place_idx !== nothing
            target_ref = node.args[in_place_idx]
            owner = resolve_slot_owner(tape, target_ref)
            haskey(slots, owner) || error(
                "plan_memory: in-place op at %$i references unassigned slot (owner %$owner).",
            )
            slots[i] = slots[owner]
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

        for ref in node_refs(node)
            args_meta = tape.metas[ref.id]
            args_node = tape.nodes[ref.id]
            slot_kind(args_node, args_meta) == :batched || continue
            if get(last_use, ref.id, 0) == i && ref.id != i
                owner = resolve_slot_owner(tape, ref)
                if haskey(slots, owner) && slots[owner].idx != slots[i].idx
                    owner_slot = slots[owner]
                    if owner_slot.kind === :M
                        push!(free_M, owner_slot.idx)
                    else
                        push!(free_V, owner_slot.idx)
                    end
                end
            end
        end
    end

    return PlannerOutput(
        slots,
        shared_slots,
        next_M - 1,
        next_V - 1,
        next_shared_M - 1,
        next_shared_V - 1,
        last_use,
    )
end

node_refs(::InputNode) = NodeRef[]
node_refs(::ConstNode) = NodeRef[]
node_refs(n::CallNode) = n.args
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

function _maybe_inplace_idx(tape::Tape, node::TapeNode)
    node isa CallNode || return nothing
    arg_types = Tuple(meta_at(tape, r).type for r in node.args)
    applicable(inplace_arg, node.fn, arg_types...) || return nothing
    return inplace_arg(node.fn, arg_types...)
end
