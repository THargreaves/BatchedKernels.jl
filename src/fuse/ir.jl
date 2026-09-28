# =============================================================================
# Tape IR
# =============================================================================
#
# The tape is a straight-line, SSA-style record of the scalar function executed
# on phantom trace values. Nodes are one of:
#   - `InputNode`: a top-level batched/shared input
#   - `CallNode`: a primitive operation on tape values
#   - `NewNode`: a structural wrapper (Adjoint, Triangular, Cholesky, tuples,
#                user structs)
#   - `ConstNode`: a trace-time literal
#
# Each node carries `NodeMeta(type, lifecycle)`. Lifecycle is the max over input
# lifecycles in `LITERAL < SHARED < BATCHED` and drives where the value lives
# (kernel constant, once-loaded shared slot, or per-batch slot).

struct NodeRef
    id::Int
end

abstract type TapeNode end

struct InputNode <: TapeNode
    index::Int
end

struct CallNode <: TapeNode
    fn::Any
    args::Vector{NodeRef}
end

struct NewNode <: TapeNode
    T::Type
    fields::Vector{Pair{Symbol,NodeRef}}
end

struct ConstNode <: TapeNode
    val::Any
end

@enum Lifecycle LITERAL SHARED BATCHED

struct NodeMeta
    type::Type
    lifecycle::Lifecycle
end

mutable struct Tape
    nodes::Vector{TapeNode}
    metas::Vector{NodeMeta}
    inputs::Vector{NodeRef}
    output::Union{Nothing,NodeRef}
end
Tape() = Tape(TapeNode[], NodeMeta[], NodeRef[], nothing)

function push_node!(tape::Tape, node::TapeNode, meta::NodeMeta)
    push!(tape.nodes, node)
    push!(tape.metas, meta)
    return NodeRef(length(tape.nodes))
end
node_at(tape::Tape, r::NodeRef) = tape.nodes[r.id]
meta_at(tape::Tape, r::NodeRef) = tape.metas[r.id]

# =============================================================================
# Tape emission helpers
# =============================================================================

# Combined lifecycle: max over {LITERAL < SHARED < BATCHED}.
function combined_lifecycle(tape::Tape, refs::Vector{NodeRef})
    lc = LITERAL
    for r in refs
        m = meta_at(tape, r)
        if Int(m.lifecycle) > Int(lc)
            lc = m.lifecycle
        end
    end
    return lc
end

# Emit a CallNode that produces an output of the given trace type.
function emit_call!(tape::Tape, fn::F, arg_refs::Vector{NodeRef}, ::Type{Out}) where {F,Out}
    lc = combined_lifecycle(tape, arg_refs)
    # Shared inputs are read-only. Computed values get ordinary per-particle
    # storage even when all operands are common to the batch. Hoisting such
    # work is an optimization; it must not be required for a valid scalar graph.
    lc == SHARED && (lc = BATCHED)
    return push_node!(tape, CallNode(fn, arg_refs), NodeMeta(Out, lc))
end

# Emit a ConstNode for a literal.
function emit_const!(tape::Tape, val)
    return push_node!(tape, ConstNode(val), NodeMeta(typeof(val), LITERAL))
end

# Emit a NewNode wrapping an inner ref. `field` is the field name expected by
# codegen / the wrapper struct.
function emit_new!(
    tape::Tape, ::Type{WrapT}, field::Symbol, inner_ref::NodeRef
) where {WrapT}
    lc = meta_at(tape, inner_ref).lifecycle
    fields = Pair{Symbol,NodeRef}[field => inner_ref]
    return push_node!(tape, NewNode(WrapT, fields), NodeMeta(WrapT, lc))
end

# =============================================================================
# Tape pretty printing
# =============================================================================

function Base.show(io::IO, tape::Tape)
    println(io, "Tape with $(length(tape.nodes)) nodes:")
    for (i, (node, meta)) in enumerate(zip(tape.nodes, tape.metas))
        lc = if meta.lifecycle == BATCHED
            "B"
        elseif meta.lifecycle == SHARED
            "S"
        else
            "L"
        end
        print(io, "  [$lc] %$i :: $(meta.type) = ")
        show_node(io, tape, node)
        println(io)
    end
    if tape.output !== nothing
        println(io, "  output: %$(tape.output.id)")
    end
end
show_node(io, tape, n::InputNode) = print(io, "input(", n.index, ")")
show_node(io, tape, n::ConstNode) = print(io, "const(", n.val, ")")
function show_node(io, tape, n::CallNode)
    print(io, n.fn, "(")
    join(io, ("%" * string(r.id) for r in n.args), ", ")
    return print(io, ")")
end
function show_node(io, tape, n::NewNode)
    print(io, "new(", n.T, "; ")
    join(io, (string(p.first, "=%", p.second.id) for p in n.fields), ", ")
    return print(io, ")")
end
