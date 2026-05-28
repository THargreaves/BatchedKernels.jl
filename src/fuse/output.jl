# =============================================================================
# Output spec
# =============================================================================
#
# The user function's return value is reflected as a tree of `OutputSpec` nodes,
# parallel to the input spec / composite handling on the input side. Leaves are
# batched-matrix slots; composites rebuild structs; literals carry a scalar.

abstract type OutputSpec end

struct LeafOutput <: OutputSpec
    slot::SlotAssignment
    trace_type::Type
    node_id::Int            # tape ref the leaf was extracted from; codegen uses
                            # it to look up the scalar local / :Sout shmem.
    # For an `aI + bM` wrapper returned directly as a broadcast output:
    # `(a, b)` are the coefficient values; `slot` holds the *parent matrix's*
    # slot (not the wrapper's, which has none). Codegen materialises the
    # wrapper by passing `IAddSubGetterMatrix(parent_view, a, b)` as the source
    # of the output transfer, so no extra slot is allocated.
    iaddsub::Union{Nothing,Tuple{Any,Any}}
end
LeafOutput(slot, trace_type, node_id) = LeafOutput(slot, trace_type, node_id, nothing)

struct CompositeOutput <: OutputSpec
    T::Type
    fields::Vector{Pair{Symbol,OutputSpec}}
end

struct LiteralOutput <: OutputSpec
    val::Any
end

function extract_output_spec(tape::Tape, planner::PlannerOutput)
    return _extract_spec(tape, planner, tape.output)
end
function _extract_spec(tape::Tape, planner::PlannerOutput, ref::NodeRef)
    node = tape.nodes[ref.id]
    if node isa NewNode
        # `IAddSubWrapped` returned directly as output: substitute the wrapper
        # view (`IAddSubGetterMatrix(parent_view, a, b)`) as the source of the
        # existing dual→interm transfer, which already accepts arbitrary
        # 2D-indexable views. No extra slot, no extra kernel.
        if node.T <: IAddSubWrapped
            return _wrapped_leaf_output(tape, planner, ref)
        end
        fields = Pair{Symbol,OutputSpec}[
            name => _extract_spec(tape, planner, child) for (name, child) in node.fields
        ]
        return CompositeOutput(node.T, fields)
    elseif node isa ConstNode
        return LiteralOutput(node.val)
    else
        meta = tape.metas[ref.id]
        # Scalar leaves go through the :Sout staging-slot pool, not the
        # compute-slot pool.
        if meta.type <: TraceScalar
            haskey(planner.scalar_output_slots, ref.id) || error(
                "extract_output_spec: scalar leaf %$(ref.id) has no :Sout slot",
            )
            return LeafOutput(planner.scalar_output_slots[ref.id], meta.type, ref.id)
        end
        haskey(planner.slots, ref.id) ||
            error("extract_output_spec: leaf %$(ref.id) has no batched slot")
        return LeafOutput(planner.slots[ref.id], meta.type, ref.id)
    end
end

function _wrapped_leaf_output(tape::Tape, planner::PlannerOutput, ref::NodeRef)
    node = tape.nodes[ref.id]::NewNode
    parent_ref = node.fields[1].second
    a_ref = node.fields[2].second
    b_ref = node.fields[3].second
    # The wrapper's parent must be a node with its own batched slot — i.e. a
    # direct CallNode/InputNode result, not another wrapper. Nesting (`I - M'`
    # or `I - (I - M)`) would need wrapper composition in the output write path
    # that is not implemented.
    haskey(planner.slots, parent_ref.id) || error(
        "BatchedKernels: `aI + bM` returned as output but its parent is not a slot-backed value (parent node %$(parent_ref.id) :: $(typeof(tape.nodes[parent_ref.id]))); only direct `I ± M` over a TraceMatrix input or primitive result is supported in this position.",
    )
    parent_slot = planner.slots[parent_ref.id]
    a_node = tape.nodes[a_ref.id]
    b_node = tape.nodes[b_ref.id]
    (a_node isa ConstNode && b_node isa ConstNode) || error(
        "BatchedKernels: `aI + bM` wrapper has non-const coefficients (a is $(typeof(a_node)), b is $(typeof(b_node))).",
    )
    return LeafOutput(
        parent_slot, tape.metas[ref.id].type, ref.id, (a_node.val, b_node.val)
    )
end

function flatten_leaves(spec::OutputSpec)
    leaves = LeafOutput[]
    _collect_leaves!(leaves, spec)
    return leaves
end
_collect_leaves!(out, spec::LeafOutput) = (push!(out, spec); out)
_collect_leaves!(out, ::LiteralOutput) = out
function _collect_leaves!(out, spec::CompositeOutput)
    for (_, s) in spec.fields
        _collect_leaves!(out, s)
    end
    return out
end
