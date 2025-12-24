using LinearAlgebra

get_mat_slot(slot_no::Int) = "M$slot_no"
get_vec_slot(slot_no::Int) = "v$slot_no"

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


function plan_memory_usage(prog::IRProgram)
    last_use, var_mat_inputs, shared_mat_inputs, _, shared_vec_inputs = compute_last_use(prog)

    # States
    slots = Dict{ValueId,String}()  # ValueId -> slot
    live_count = Dict{String,Int}()  # slot -> counts
    parent = Dict{String,String}()  # slot -> true slot that it points to

    free_mat_slots = String[]
    free_vec_slots = String[]

    next_shared_mat_slot = 1
    next_mat_slot = shared_mat_inputs + 1

    next_shared_vec_slot = 1
    next_vec_slot = shared_vec_inputs + 1
    
    next_pseudo_mat_slot = length(prog.nodes) + length(prog.inputs) + 2

    # Union find
    function get_true_slot(slot::String)
        if parent[slot] != slot
            parent[slot] = get_true_slot(parent[slot])
        end
        return parent[slot]
    end
    get_true_slot(vid::ValueId) = get_true_slot(slots[vid])

    ####################
    ### SLOT HELPERS ###
    ####################

    alloc_new_mat!() = (slot = get_mat_slot(next_mat_slot); next_mat_slot += 1; slot)
    alloc_new_vec!() = (slot = get_vec_slot(next_vec_slot); next_vec_slot += 1; slot)
    alloc_new_shared_mat!() = (slot = get_mat_slot(next_shared_mat_slot); next_shared_mat_slot += 1; slot)
    alloc_new_shared_vec!() = (slot = get_vec_slot(next_shared_vec_slot); next_shared_vec_slot += 1; slot)
    alloc_new_pseudo_mat!() = (slot = get_mat_slot(next_pseudo_mat_slot); next_pseudo_mat_slot += 1; slot)

    alloc_mat_outofplace!() = isempty(free_mat_slots) ? alloc_new_mat!() : pop!(free_mat_slots)
    alloc_vec_outofplace!() = isempty(free_vec_slots) ? alloc_new_vec!() : pop!(free_vec_slots)

    function increment_slot!(slot::String)
        true_slot = get_true_slot(slot)
        live_count[true_slot] = get(live_count, true_slot, 0) + 1
        return true_slot
    end
    increment_slot!(vid::ValueId) = increment_slot!(slots[vid])

    function free_true_slot!(true_slot::String, K::Type{<:SymKind})
        if K <: MatKind
            push!(free_mat_slots, true_slot)
        elseif K <: VecKind
            push!(free_vec_slots, true_slot)
        end
    end

    function alloc_input!(vid::ValueId)
        K = prog.kinds[vid]

        if K <: SharedKind
            if K <: SharedMatKind
                slot = alloc_new_shared_mat!()
            elseif K <: SharedVecKind
                slot = alloc_new_shared_vec!()
            else
                error("Unknown shared input kind: $K")
            end
        else
            if K <: MatKind
                slot = alloc_new_mat!()
            elseif K <: VecKind
                slot = alloc_new_vec!()
            else
                error("Only matrix/vector inputs supported, got $K")
            end
        end

        slots[vid] = slot
        parent[slot] = slot
        live_count[slot] = 1
    end

    ###########################
    ### INPUT MEMPORY ALLOC ###
    ###########################

    for (_, vid) in prog.inputs
        alloc_input!(vid)
    end

    #######################
    ### IN-PLACE POLICY ###
    #######################

    function can_reuse_arg1(i::Int, node::IRNode)
        arg1 = node.args[1]::ValueId
        get(last_use, arg1, 0) == i || return false

        slot1 = get_true_slot(arg1)
        count1 = live_count[slot1]

        # If arg2 is alias of arg1, allow count==2
        if length(node.args) == 2 && node.args[2] isa ValueId
            arg2 = node.args[2]::ValueId
            slot2 = get_true_slot(arg2)
            
            slot1 == slot2 && return count1 == 2
        end

        return count1 == 1
    end

    function can_reuse_arg2(i::Int, node::IRNode)
        length(node.args) == 2 || return false
        node.args[2] isa ValueId || return false
        
        arg2 = node.args[2]::ValueId

        get(last_use, arg2, 0) == i || return false
        return live_count[get_true_slot(arg2)] == 1
    end

    function try_inplace!(i::Int, node::IRNode)
        op = node.op
        arg1 = node.args[1]

        (
            op in (:chol, :forwardsolve, :backwardsolve)
            || (op in (:add, :sub) && arg1 isa ValueId && !(prog.kinds[arg1] <: TransMatKind))
            || (op == :trans && node.out in prog.outputs)
        ) || return false

        if op in (:chol, :add, :sub, :trans) && arg1 isa ValueId && can_reuse_arg1(i, node)
            # slots[node.out] = true_slot(arg1::ValueId)
            # increment_slot(slots[node.out])
            slots[node.out] = increment_slot!(arg1::ValueId)
            return true
        end
        
        if op in (:add, :sub, :forwardsolve, :backwardsolve) && can_reuse_arg2(i, node)
            arg2 = node.args[2]
            # slots[node.out] = true_slot(arg2::ValueId)
            # increment_slot!(slots[node.out])
            slots[node.out] = increment_slot!(arg2::ValueId)
            return true
        end

        return false
    end

    is_wrapper_op(op::Symbol) = op in (:trans, :lowertrig, :uppertrig, :iminus, :iplus, :sym)

    function alloc_wrapper!(node::IRNode)
        vid = node.args[1]::ValueId
        slot = alloc_new_pseudo_mat!()
        slots[node.out] = slot
        parent[slot] = get_true_slot(vid)
        increment_slot!(parent[slot])

        return true
    end

    function alloc_result_outofplace!(vid::ValueId)
        K = prog.kinds[vid]

        if K <: MatKind
            slot = alloc_mat_outofplace!()
        elseif K <: VecKind
            slot = alloc_vec_outofplace!()
        else
            error("Unsupported result kind $K")
        end

        slots[vid] = slot
        parent[slot] = slot
        live_count[slot] = 1

        return true
    end

    function dec_maybe_free!(i::Int, vid::ValueId)
        get(last_use, vid, 0) == i || return
        true_slot = get_true_slot(vid)
        live_count[true_slot] -= 1
        live_count[true_slot] == 0 && free_true_slot!(true_slot, prog.kinds[vid])
    end

    function free_if_unused_result!(node)
        vid = node.out
        haskey(last_use, vid) && return
        true_slot = get_true_slot(vid)
        live_count[true_slot] = 0
        free_true_slot!(true_slot, prog.kinds[vid])
    end

    #################
    ### MAIN LOOP ###
    #################

    for (i, node) in enumerate(prog.nodes)
        # Allocate memory, in-place, wrapper, or out-of-place
        (
            try_inplace!(i, node)
            || (is_wrapper_op(node.op) && alloc_wrapper!(node))
            || alloc_result_outofplace!(node.out)
        )

        # Decrement args if this was their last use
        for arg in node.args
            arg isa ValueId || continue
            dec_maybe_free!(i, arg::ValueId)
        end

        free_if_unused_result!(node)
    end

    ######################
    ### GATHER RESULTS ###
    ######################

    mat_slots = ["M$i" for i in (shared_mat_inputs + 1):(next_mat_slot - 1)]
    vec_slots = ["v$i" for i in (shared_vec_inputs + 1):(next_vec_slot - 1)]

    require_extra_slot = var_mat_inputs == length(mat_slots)
    mat_load_slot_id = require_extra_slot ? next_mat_slot : next_mat_slot - 1

    return slots, shared_mat_inputs, mat_slots, shared_vec_inputs, vec_slots, require_extra_slot, mat_load_slot_id
end


# function func(A, v)
#     return Symmetric(A) * v
# end

# prog = trace(func, (Mat(:A), SharedVec(:b)); D=2, nthreads=4)

# println(prog)
# slots, shared_mat_inputs, mat_slots, shared_vec_inputs, vec_slots, require_extra_slot, mat_load_slot_id = plan_memory_usage(prog)
# println("\nPRINTOUTS:")
# println(slots)
# println("shared_vec_inputs=", shared_vec_inputs)
# println("peak=", length(mat_slots))
