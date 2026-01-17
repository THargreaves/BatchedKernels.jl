using LinearAlgebra

"Computes the last node that uses each variable so memory can be freed."
function compute_last_use(prog::IRProgram)
    M = length(prog.nodes)
    last_use = Dict{ValueId,Int}()

    shared_mat_inputs = 0
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
            (K <: MatKind || K <: VecKind) || error("Only matrix and vector inputs permitted, got $K")
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

    return last_use, shared_mat_inputs, shared_vec_inputs
end

"""
Tracks the allocation and liveliness of each memory slot during an execution of an `IRProgram`.

- `next_shared_mat_slot::Int`: Next available slot index for shared matrices.
- `next_mat_slot::Int`: Next available slot index for batched matrices.
- `next_shared_vec_slot::Int`: Next available slot index for shared vectors.
- `next_vec_slot:::Int`: Next available slot index for batched vectors.
- `next_pseudo_mat_slot`: Next available pseudo-slot (non-memory owning warpper variables) for matrices.
- `free_mat_slots::Vector{String}`: Pool of unused matrix slots.
- `free_vec_slots::Vector{String}`: Pool of unused vector slots.
- `slots::Dict{ValueId,String}`: Mapping from `ValueId` to its assigned slot.
- `live_count::Dict{String,Int}`: Number of references pointing to each memory slot.
- `parent::Dict{String,String}`: Union find data structure mapping a memory slot to the true underlying memory slot.
- `last_use::Dict{ValueId,Int}`: Dictionary mapping each `ValueId` to the last node's index that it was used in.
    Memory will be freed after this node
- `kinds::Dict{ValueId,Type{<:SymKind}}`: Dictionary mapping each `ValueId` to its type.
- `unalloc_bat_inputs::Set{ValueId}`: Set of input `ValueId`s whose memory haven't been allocated yet.
- `input_load_schedule::Vector{Vector{Tuple{ValueId,String}}}`: Per-node schedule for input loading
    (which inputs will be loaded at the start of node i, where i indexes this vector).
"""
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
    unalloc_bat_inputs::Set{ValueId}
    input_load_schedule::Vector{Vector{Tuple{ValueId,String}}}
end

function State(
    shared_mat_inputs::Int,
    shared_vec_inputs::Int,
    prog::IRProgram,
    last_use::Dict{ValueId,Int},
)
    unalloc_bat_inputs = Set(vid for (_, vid) in prog.inputs if prog.kinds[vid] <: BatchedKind)
    input_load_schedule::Vector{Vector{Tuple{ValueId,String}}} = [Tuple{ValueId,String}[] for _ in 1:length(prog.nodes)]

    return State(
        1,
        shared_mat_inputs + 1,
        1,
        shared_vec_inputs + 1,
        length(prog.nodes) + length(prog.inputs) + 2,
        String[],
        String[],
        Dict{ValueId,String}(),
        Dict{String,Int}(),
        Dict{String,String}(),
        last_use,
        prog.kinds,
        unalloc_bat_inputs,
        input_load_schedule,
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

function alloc_shared_input!(state::State, vid::ValueId)
    K = state.kinds[vid]
    K <: SharedKind || return

    if K <: SharedMatKind
        slot = alloc_new_shared_mat!(state)
    elseif K <: SharedVecKind
        slot = alloc_new_shared_vec!(state)
    else
        error("Unknown shared input kind: $K")
    end

    state.slots[vid] = slot
    state.parent[slot] = slot
    state.live_count[slot] = 1
end

function maybe_alloc_batched_input!(state::State, node::IRNode, i::Int)
    for arg in node.args
        arg isa ValueId || continue
        vid = arg::ValueId

        vid in state.unalloc_bat_inputs || continue 

        K = state.kinds[vid]
        if K <: MatKind
            slot = alloc_mat_outofplace!(state)
            load_slot = alloc_mat_outofplace!(state)
            push!(state.input_load_schedule[i], (vid, load_slot))
            free_true_slot!(state, load_slot, K)
        elseif K <: VecKind
            slot = alloc_vec_outofplace!(state)
            push!(state.input_load_schedule[i], (vid, ""))
        else
            error("Only matrix/vector inputs supported, got $K")
        end

        state.slots[vid] = slot
        state.parent[slot] = slot
        state.live_count[slot] = 1
        delete!(state.unalloc_bat_inputs, vid)
    end
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

function dec_arg_count!(state::State, node::IRNode, i::Int)
    for arg in node.args
        arg isa ValueId || continue
        dec_maybe_free!(state, i, arg::ValueId)
    end
end

function free_if_unused_result!(state::State, node::IRNode)
    vid = node.out
    haskey(state.last_use, vid) && return
    true_slot = get_true_slot!(state, vid)
    state.live_count[true_slot] = 0
    free_true_slot!(state, true_slot, state.kinds[vid])
end

"""
Plans the memory usage of the function via a lazy input loading and greedy memory allocation.

# Algorithm
1. Initialises `State` that keeps track of the current memory allocations.
2. Allocates memory for all shared input arguments.
3. Loops through all nodes in the `IRProgram`. For each node:
    a) Load the inputs if they are used in the current node and haven't been allocated earlier.
    b) Attempt to perform operation in-place. An operation can be performed in-place if the
        operation supports it (e.g. addition/subtraction), and if the inputs to this operation
        can be overridden (i.e they aren't used later). If successful, go to step 3.
    c) Checks if the operation is a wrapper operation (e.g. transposition/Symmetric).
        If so, make a wrapper variable without allocating new memory, and skip to step 3.
    d) Allocate new memory for the result of this operation. If free slots exist in the
        pool of free memories, use them. If not, allocate new memory slot.
    e) Decrement the usage counts of the input variables to this operation. Free the memory
        if the counts reach zero.
    f) Check if the result of this operation is used. If not, free the memory.

Format of the memory slots are `String`s of format:
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
- Pseudo matrix slot indices start from length(nodes) + length(inputs) + 2 to avoid overlap

Returns:
    - `slots::Dict{ValueId,String}`: Mapping from `ValueId` to its assigned slot.
    - `mat_slots::Vector{String}`: Vector of all matrix slots required throughout.
    - `vec_slots::Vector{String}`: Vector of all matrix slots required throughout.
    - `input_load_schedule::Vector{Vector{Tuple{ValueId,String}}}`: Per-node schedule for input loading
        (which inputs will be loaded at the start of node i, where i indexes this vector).
    - `require_extra_slot::bool`: An indicator on whether an extra memory slot is required to
        store the output matrices, as storage of each matrix requires an extra slot to store
        the intermediate layout.
    - `mat_store_slot::String`: The matrix slot index that is used to store outputs, where the
        matrices are stored in intermediate layout
"""
function plan_memory_usage(prog::IRProgram)
    last_use, shared_mat_inputs, shared_vec_inputs = compute_last_use(prog)

    state = State(
        shared_mat_inputs,
        shared_vec_inputs,
        prog,
        last_use,
    )

    # Allocating shared input memory
    for (_, vid) in prog.inputs
        alloc_shared_input!(state, vid)
    end

    # Main loop
    for (i, node) in enumerate(prog.nodes)
        # Check if arguments of the operation are loaded, if not, load them
        maybe_alloc_batched_input!(state, node, i)

        # Allocate memory, in-place, wrapper, or out-of-place
        (
            try_inplace!(state, i, node, prog.outputs)
            || (is_wrapper_op(node.op) && alloc_wrapper!(state, node))
            || alloc_result_outofplace!(state, node.out)
        )

        # Decrement args if this was their last use
        dec_arg_count!(state, node, i)  # Decrement count of the arguments, if those counts reach zero, free the memory
        free_if_unused_result!(state, node)  # Free the memory of the results if its unused
    end

    # Gather results
    mat_slots = ["M$i" for i in (shared_mat_inputs + 1):(state.next_mat_slot - 1)]
    vec_slots = ["v$i" for i in (shared_vec_inputs + 1):(state.next_vec_slot - 1)]

    require_extra_slot = length(prog.outputs) == length(mat_slots)
    mat_store_slot = get_mat_slot(require_extra_slot ? state.next_mat_slot : state.next_mat_slot - 1)

    return state.slots, mat_slots, vec_slots, state.input_load_schedule, require_extra_slot, mat_store_slot
end
