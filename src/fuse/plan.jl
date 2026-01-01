using LinearAlgebra

"Computes the last node that uses each variable so memory can be freed."
function compute_last_use(prog::IRProgram)
    M = length(prog.nodes)
    last_use = Dict{ValueId,Int}()

    var_mat_inputs = 0
    shared_mat_inputs = 0
    var_vec_inputs = 0
    shared_vec_inputs = 0

    # Inputs: initialise last_use and count kinds
    for (_, vid) in prog.inputs
        K = prog.kinds[vid]

        if K <: SharedKind
            if K <: SharedMatKind
                shared_mat_inputs += 1
            elseif K <: SharedVecKind
                shared_vec_inputs += 1
            else
                error("Unknown shared input kind: $K")
            end
            last_use[vid] = M + 1  # Don't override shared kinds
        else
            if K <: MatKind
                var_mat_inputs += 1
            elseif K <: VecKind
                var_vec_inputs += 1
            else
                error("Only matrix and vector inputs permitted, got $K")
            end
            last_use[vid] = 0
        end
    end

    # Nodes: update last_use for ValueId (matrix & vector) arguments
    for (i, node) in enumerate(prog.nodes)
        for arg in node.args
            arg isa ValueId || continue
            vid = arg::ValueId
            last_use[vid] = max(get(last_use, vid, 0), i)
        end
    end

    # Outputs: never free their memory
    for vid in prog.outputs
        last_use[vid] = M + 1
    end

    return last_use, var_mat_inputs, shared_mat_inputs, var_vec_inputs, shared_vec_inputs
end

mutable struct State
    next_shared_mat_slot::Int
    next_mat_slot::Int
    next_shared_vec_slot::Int
    next_vec_slot::Int
    next_pseudo_mat_slot::Int
    free_mat_slots::Vector{String}
    free_vec_slots::Vector{String}
    slots::Dict{ValueId,String}  # ValueId -> slot
    live_count::Dict{String,Int}  # slot -> counts
    parent::Dict{String,String}  # slot -> true slot that it points to
    last_use::Dict{ValueId,Int}
    kinds::Dict{ValueId,Type{<:SymKind}}
end

function State(
    next_shared_mat_slot::Int,
    next_mat_slot::Int,
    next_shared_vec_slot::Int,
    next_vec_slot::Int,
    next_pseudo_mat_slot::Int,
    last_use::Dict{ValueId,Int},
    kinds::Dict{ValueId,Type{<:SymKind}},
)
    return State(
        next_shared_mat_slot,
        next_mat_slot,
        next_shared_vec_slot,
        next_vec_slot,
        next_pseudo_mat_slot,
        String[],
        String[],
        Dict{ValueId,String}(),
        Dict{String,Int}(),
        Dict{String,String}(),
        last_use,
        kinds,
    )
end

"Union find"
function get_true_slot!(state::State, slot::String)
    if state.parent[slot] != slot
        state.parent[slot] = get_true_slot!(state, state.parent[slot])
    end
    return state.parent[slot]
end
get_true_slot!(state::State, vid::ValueId) = get_true_slot!(state, state.slots[vid])

####################
### SLOT HELPERS ###
####################

get_mat_slot(slot_no::Int) = "M$slot_no"
get_vec_slot(slot_no::Int) = "v$slot_no"

alloc_new_mat!(state::State) = (slot = get_mat_slot(state.next_mat_slot); state.next_mat_slot += 1; slot)
alloc_new_vec!(state::State) = (slot = get_vec_slot(state.next_vec_slot); state.next_vec_slot += 1; slot)
alloc_new_shared_mat!(state::State) = (slot = get_mat_slot(state.next_shared_mat_slot); state.next_shared_mat_slot += 1; slot)
alloc_new_shared_vec!(state::State) = (slot = get_vec_slot(state.next_shared_vec_slot); state.next_shared_vec_slot += 1; slot)
alloc_new_pseudo_mat!(state::State) = (slot = get_mat_slot(state.next_pseudo_mat_slot); state.next_pseudo_mat_slot += 1; slot)

alloc_mat_outofplace!(state::State) = isempty(state.free_mat_slots) ? alloc_new_mat!(state) : pop!(state.free_mat_slots)
alloc_vec_outofplace!(state::State) = isempty(state.free_vec_slots) ? alloc_new_vec!(state) : pop!(state.free_vec_slots)

function increment_slot!(state::State, slot::String)
    true_slot = get_true_slot!(state, slot)
    state.live_count[true_slot] = get(state.live_count, true_slot, 0) + 1
    return true_slot
end
increment_slot!(state::State, vid::ValueId) = increment_slot!(state, state.slots[vid])

function free_true_slot!(state::State, true_slot::String, K::Type{<:SymKind})
    if K <: MatKind
        push!(state.free_mat_slots, true_slot)
    elseif K <: VecKind
        push!(state.free_vec_slots, true_slot)
    end
end

function alloc_input!(state::State, vid::ValueId)
    K = state.kinds[vid]

    if K <: SharedKind
        if K <: SharedMatKind
            slot = alloc_new_shared_mat!(state)
        elseif K <: SharedVecKind
            slot = alloc_new_shared_vec!(state)
        else
            error("Unknown shared input kind: $K")
        end
    else
        if K <: MatKind
            slot = alloc_new_mat!(state)
        elseif K <: VecKind
            slot = alloc_new_vec!(state)
        else
            error("Only matrix/vector inputs supported, got $K")
        end
    end

    state.slots[vid] = slot
    state.parent[slot] = slot
    state.live_count[slot] = 1
end

#######################
### IN-PLACE POLICY ###
#######################

function can_reuse_arg1(state::State, i::Int, node::IRNode)
    arg1 = node.args[1]::ValueId
    get(state.last_use, arg1, 0) == i || return false

    slot1 = get_true_slot!(state, arg1)
    count1 = state.live_count[slot1]

    # If arg2 is alias of arg1, allow count==2
    if length(node.args) == 2 && node.args[2] isa ValueId
        arg2 = node.args[2]::ValueId
        slot2 = get_true_slot!(state, arg2)
        
        slot1 == slot2 && return count1 == 2
    end

    return count1 == 1
end

function can_reuse_arg2(state::State, i::Int, node::IRNode)
    length(node.args) == 2 || return false
    node.args[2] isa ValueId || return false
    
    arg2 = node.args[2]::ValueId

    get(state.last_use, arg2, 0) == i || return false
    return state.live_count[get_true_slot!(state, arg2)] == 1
end

function try_inplace!(state::State, i::Int, node::IRNode, outputs::Vector{ValueId})
    op = node.op
    arg1 = node.args[1]

    (
        op in (:chol, :forwardsolve, :backwardsolve)
        || (op in (:add, :sub) && arg1 isa ValueId && !(state.kinds[arg1] <: TransMatKind))
        || (op == :trans && node.out in outputs)
    ) || return false

    if op in (:chol, :add, :sub, :trans) && arg1 isa ValueId && can_reuse_arg1(state, i, node)
        state.slots[node.out] = increment_slot!(state, arg1::ValueId)
        return true
    end
    
    if op in (:add, :sub, :forwardsolve, :backwardsolve) && can_reuse_arg2(state, i, node)
        arg2 = node.args[2]
        state.slots[node.out] = increment_slot!(state, arg2::ValueId)
        return true
    end

    return false
end

is_wrapper_op(op::Symbol) = op in (:trans, :lowertrig, :uppertrig, :iminus, :iplus, :sym)

function alloc_wrapper!(state::State, node::IRNode)
    vid = node.args[1]::ValueId
    slot = alloc_new_pseudo_mat!(state)
    state.slots[node.out] = slot
    state.parent[slot] = get_true_slot!(state, vid)
    increment_slot!(state, state.parent[slot])

    return true
end

function alloc_result_outofplace!(state::State, vid::ValueId)
    K = state.kinds[vid]

    if K <: MatKind
        slot = alloc_mat_outofplace!(state)
    elseif K <: VecKind
        slot = alloc_vec_outofplace!(state)
    else
        error("Unsupported result kind $K")
    end

    state.slots[vid] = slot
    state.parent[slot] = slot
    state.live_count[slot] = 1

    return true
end

function dec_maybe_free!(state::State, i::Int, vid::ValueId)
    get(state.last_use, vid, 0) == i || return
    true_slot = get_true_slot!(state, vid)
    state.live_count[true_slot] -= 1
    state.live_count[true_slot] == 0 && free_true_slot!(state, true_slot, state.kinds[vid])
end

function free_if_unused_result!(state::State, node)
    vid = node.out
    haskey(state.last_use, vid) && return
    true_slot = get_true_slot!(state, vid)
    state.live_count[true_slot] = 0
    free_true_slot!(state, true_slot, state.kinds[vid])
end

"""
Plans the memory usage of the function via a greedy memory allocation method.

Overall flow:
1. Allocates memory for all the inputs
2. Loops through all the nodes of the IR. For each node:
    a) Checks if the operation can be done in-place. An operation can be done in-place
        if the operation supports it (e.g. addition/subtraction), and if the this node
        is the last node to use the input, so the input can be overridden.
    b) Checks if the operation is a wrapper-operation, e.g. transposition, Symmetric().
        If so, make a wrapper variable without allocating new memory.
    c) If neither of the above were successful, allocate new memory. Check if any existing
        memory slots are free, if so, use them. If not, allocate a new memory slot.
    d) For all inputs of the node, check if this node is the last one to use them. If so,
        free their memory.
    e) Checks if the output of this node is used in the future. If not, free that memory.
    
Format of the memory slots:
- Matrices: M + (slot index)
- Vectors: v + (slot index)

Pseudo slots for wrapper operation results:
Variables that don't own the underlying memory (results of wrapper operations) will be
given unique 'pseudo' memory slots, which are created for each such variable and are named
similarly (M + slot index). However, they use a different index to the real memory slots.
For example, an adjoint operation would be planned as such: M10 = adjoint(M1), where
M10 points to the same underlying memory as M1.

This is required because during tracing, each operation is handled one by one. E.g.
batch_op!(..., Symmetric(M1), ...) will be broken down to temp = Symmetric(M1);
batch_op!(..., temp, ...). When tracing any operation, the future ones are unknown.
This makes temporary variables necessary. To make this compatible with later codegen stage,
a naming convention of using pseudo slot indices will allow codegen to treat all variables
similarly, both temporary and memory-owning.

Indexing rule:
- Memory slot indices for shared matirces and vectors start from 1
- Memory slot indices for matrices and vectors start from shared_mat_inputs + 1
    and shared_vec_inputs + 1 respectively to avoid overlap
- Pseudo matrix slot indices start from length(nodes) + length(inputs) to avoid overlap

Returns:
    - slots::Dict{ValueId,String}: Slots for each variable
    - mat_slots::Vector{String}: All the batched matrix slots required throughout
    - vec_slots::Vector{String}: All the batched vector slots required throughout
    - require_extra_slot::bool: True if mat_slots aren't enough to load all the inputs,
        requiring an extra slot to transfer between intermediate & dual access layout
    - mat_load_slot_id: The matrix slot index that is used to load inputs, where the
        matrices are stored in intermediate layout
"""
function plan_memory_usage(prog::IRProgram)
    last_use, var_mat_inputs, shared_mat_inputs, _, shared_vec_inputs = compute_last_use(prog)

    state = State(
        1,
        shared_mat_inputs + 1,
        1,
        shared_vec_inputs + 1,
        length(prog.nodes) + length(prog.inputs) + 2,
        last_use,
        prog.kinds,
    )

    # Allocating input memory
    for (_, vid) in prog.inputs
        alloc_input!(state, vid)
    end

    # Main loop
    for (i, node) in enumerate(prog.nodes)
        # Allocate memory, in-place, wrapper, or out-of-place
        (
            try_inplace!(state, i, node, prog.outputs)
            || (is_wrapper_op(node.op) && alloc_wrapper!(state, node))
            || alloc_result_outofplace!(state, node.out)
        )

        # Decrement args if this was their last use
        for arg in node.args
            arg isa ValueId || continue
            dec_maybe_free!(state, i, arg::ValueId)
        end

        free_if_unused_result!(state, node)
    end

    # Gather results
    mat_slots = ["M$i" for i in (shared_mat_inputs + 1):(state.next_mat_slot - 1)]
    vec_slots = ["v$i" for i in (shared_vec_inputs + 1):(state.next_vec_slot - 1)]

    require_extra_slot = var_mat_inputs == length(mat_slots)
    mat_load_slot_id = require_extra_slot ? state.next_mat_slot : state.next_mat_slot - 1

    return state.slots, mat_slots, vec_slots, require_extra_slot, mat_load_slot_id
end
