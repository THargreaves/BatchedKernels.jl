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
end

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
        fields = Pair{Symbol,OutputSpec}[
            name => _extract_spec(tape, planner, child) for (name, child) in node.fields
        ]
        return CompositeOutput(node.T, fields)
    elseif node isa ConstNode
        return LiteralOutput(node.val)
    else
        haskey(planner.slots, ref.id) ||
            error("extract_output_spec: leaf %$(ref.id) has no batched slot")
        return LeafOutput(planner.slots[ref.id], tape.metas[ref.id].type)
    end
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
