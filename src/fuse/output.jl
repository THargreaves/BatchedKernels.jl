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
end

struct CompositeOutput <: OutputSpec
    T::Type
    fields::Vector{Pair{Symbol,OutputSpec}}
end

struct LiteralOutput <: OutputSpec
    val::Any
end

function extract_output_spec(tape::Tape, planner::Union{PlannerOutput,HybridPlannerOutput})
    return _extract_spec(tape, planner, tape.output)
end
function _extract_spec(
    tape::Tape, planner::Union{PlannerOutput,HybridPlannerOutput}, ref::NodeRef
)
    node = tape.nodes[ref.id]
    if node isa NewNode
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
            haskey(planner.scalar_output_slots, ref.id) ||
                error("extract_output_spec: scalar leaf %$(ref.id) has no :Sout slot")
            return LeafOutput(planner.scalar_output_slots[ref.id], meta.type, ref.id)
        end
        haskey(planner.slots, ref.id) ||
            error("extract_output_spec: leaf %$(ref.id) has no batched slot")
        return LeafOutput(planner.slots[ref.id], meta.type, ref.id)
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
