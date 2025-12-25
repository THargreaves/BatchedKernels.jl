##########################
### KIND CHECK HELPERS ###
##########################

is_mat(prog::IRProgram, v::ValueId) = prog.kinds[v] <: MatLike
is_shared(prog::IRProgram, v::ValueId) = prog.kinds[v] <: SharedKind
is_shared_mat(prog::IRProgram, v::ValueId) = prog.kinds[v] <: SharedMatKind
is_shared_vec(prog::IRProgram, v::ValueId) = prog.kinds[v] <: SharedVecKind
is_vec(prog::IRProgram, v::ValueId) = prog.kinds[v] <: VecLike
is_trans(prog::IRProgram, v::ValueId) = prog.kinds[v] <: TransMatKind
is_lowertrig(prog::IRProgram, v::ValueId) = prog.kinds[v] <: LowerTrigMatKind
is_uppertrig(prog::IRProgram, v::ValueId) = prog.kinds[v] <: UpperTrigMatKind
is_scalar(prog::IRProgram, v::ValueId) = prog.kinds[v] <: ScalarKind

##################
### STEP TYPES ###
##################

abstract type KernelStep end

# Loads
struct LoadSharedMatStep <: KernelStep
    slot::String
    vid::ValueId
    wid_eq::Int
end
struct LoadSharedVecStep <: KernelStep
    slot::String
    vid::ValueId
    wid_eq::Int
end
struct LoadBatMatStep <: KernelStep
    slot::String
    vid::ValueId
end
struct LoadBatVecStep <: KernelStep
    slot::String
    vid::ValueId
end

# Active threads wrappers
struct InitMatWrapperStep <: KernelStep
    slot::String
end
struct InitVecWrapperStep <: KernelStep
    slot::String
end

# Computations
struct BatchBinaryStep{F} <: KernelStep
    f::F  # *, +, -, \
    dest::String
    a::String
    b::String
end
struct BatchUnaryStep{F} <: KernelStep
    f::F  # cholesky, transpose
    dest::String
    a::String
end
struct WrapperStep <: KernelStep
    wrapper::Symbol  # :adjoint, :LowerTriangular, :UpperTriangular, :Symmetric
    dest::String
    a::String
end
struct IAddSubStep{T} <: KernelStep
    dest::String
    a::String
    λ::T
    sign::T
end

# Stores
struct StoreMatStep <: KernelStep
    slot::String
    store_slot::String
    vid::ValueId
end
struct StoreVecStep <: KernelStep
    slot::String
    vid::ValueId
end

########################
### STEP PLANNERS ###
########################

function get_load_steps(prog::IRProgram, slots::Dict{ValueId,String})
    load_steps = KernelStep[]
    shared_input_count = 1

    for (_, vid) in prog.inputs
        if is_shared(prog, vid)
            if is_shared_mat(prog, vid)
                push!(load_steps, LoadSharedMatStep(slots[vid], vid, shared_input_count))
            elseif is_shared_vec(prog, vid)
                push!(load_steps, LoadSharedVecStep(slots[vid], vid, shared_input_count))
            else
                error("Unknown shared kind for input $vid: $(prog.kinds[vid])")
            end
            shared_input_count += 1
        else  # Batched type
            if is_mat(prog, vid)
                push!(load_steps, LoadBatMatStep(slots[vid], vid))
            elseif is_vec(prog, vid)
                push!(load_steps, LoadBatVecStep(slots[vid], vid))
            else
                error("Unsupported input kind for $vid: $(prog.kinds[vid])")
            end
        end
    end

    return load_steps
end

function get_active_init_steps(mat_slots::Vector{String}, vec_slots::Vector{String})
    active_init_steps = KernelStep[]
    for slot in mat_slots
        push!(active_init_steps, InitMatWrapperStep(slot))
    end
    for slot in vec_slots
        push!(active_init_steps, InitVecWrapperStep(slot))
    end
    return active_init_steps
end

function get_compute_steps(prog::IRProgram{T}, slots::Dict{ValueId,String}) where {T}
    compute_steps = KernelStep[]    

    for node in prog.nodes
        op = node.op

        if op == :mul
            a = node.args[1]::ValueId
            b = node.args[2]::ValueId
            (
                is_mat(prog, a) && (is_mat(prog, b) || is_vec(prog, b))
            ) || error("Multipication only supports mat*mat or mat*vec")
            push!(compute_steps, BatchBinaryStep(*, slots[node.out], slots[a], slots[b]))
        
        elseif op == :add
            a = node.args[1]::ValueId
            b = node.args[2]::ValueId
            (
                (is_mat(prog, a) && is_mat(prog, b)) || (is_vec(prog, a) && is_vec(prog, b))
            ) || error("Addition only supports mat+mat or vec+vec")
            push!(compute_steps, BatchBinaryStep(+, slots[node.out], slots[a], slots[b]))

        elseif op == :sub
            a = node.args[1]::ValueId
            b = node.args[2]::ValueId
            (
                (is_mat(prog, a) && is_mat(prog, b)) || (is_vec(prog, a) && is_vec(prog, b))
            ) || error("Subtractino only supports mat-mat or vec-vec")
            push!(compute_steps, BatchBinaryStep(-, slots[node.out], slots[a], slots[b]))
        
        elseif op == :chol
            a = node.args[1]::ValueId
            is_mat(prog, a) || error("Cholesky only defined for matrices")
            push!(compute_steps, BatchUnaryStep(cholesky, slots[node.out], slots[a]))
        
        elseif op == :trans
            a = node.args[1]::ValueId
            is_mat(prog, a) || error("Transposition only defined for matrices")
            if node.out in prog.outputs  # In-place for outputs
                push!(compute_steps, BatchUnaryStep(transpose, slots[node.out], slots[a]))
            else  # Wrapper for non-outputs
                push!(compute_steps, WrapperStep(:adjoint, slots[node.out], slots[a]))
            end

        elseif op == :lowertrig
            a = node.args[1]::ValueId
            is_mat(prog, a) || error("LowerTriangular only defined for matrices")
            push!(compute_steps, WrapperStep(:LowerTriangular, slots[node.out], slots[a]))
        
        elseif op == :uppertrig
            a = node.args[1]::ValueId
            is_mat(prog, a) || error("UpperTriangular only defined for matrices")
            push!(compute_steps, WrapperStep(:UpperTriangular, slots[node.out], slots[a]))

        elseif op == :sym
            a = node.args[1]::ValueId
            is_mat(prog, a) || error("Symmetric only defined for matrices")
            push!(compute_steps, WrapperStep(:Symmetric, slots[node.out], slots[a]))

        elseif op == :forwardsolve || op == :backwardsolve
            a = node.args[1]::ValueId
            b = node.args[2]::ValueId
            (
                is_lowertrig(prog, a) || is_uppertrig(prog, a)
            ) || error("Triangular solve requires triangular L/U")
            push!(compute_steps, BatchBinaryStep(\, slots[node.out], slots[a], slots[b]))            
        
        elseif op == :iplus || op == :iminus
            a = node.args[1]::ValueId
            λ = node.args[2]::T
            is_mat(prog, a) || error("I +- var only defined for matrices")
            sign = op == :iplus ? one(T) : -one(T)
            push!(compute_steps, IAddSubStep{T}(slots[node.out], slots[a], λ, sign))

        else
            error("Unsupported operation $op")
        end
    end

    return compute_steps
end

function get_store_steps(prog::IRProgram, slots::Dict{ValueId,String}, mat_load_slot::String)
    store_steps = KernelStep[]

    for vid in prog.outputs
        !is_shared(prog, vid) || error("Cannot return shared variables")

        if is_mat(prog, vid)
            slot = slots[vid]
            store_slot = mat_load_slot != slot ? mat_load_slot : (slot != "M1" ? "M1" : "M2")
            push!(store_steps, StoreMatStep(slots[vid], store_slot, vid))
        elseif is_vec(prog, vid)
            push!(store_steps, StoreVecStep(slots[vid], vid))
        else
            error("Unsupported output kind: $(prog.kinds[vid])")
        end
    end

    return store_steps
end

function emit_kernel_expr(
    prog::IRProgram{T};
    output_names::Vector{Symbol},
    input_names::Vector{Symbol},
) where {T}
    ############################
    ### INPUT & OUTPUT NAMES ###
    ############################

    length(input_names) == length(prog.inputs) ||
        error("Input_names length $(length(input_names)) != number of inputs $(length(prog.inputs))")
    length(output_names) == length(prog.outputs) ||
        error("Output_names length $(length(output_names)) != number of outputs $(length(prog.outputs))")

    input_index = Dict{ValueId,Int}()
    for (i, (_, vid)) in enumerate(prog.inputs)
        input_index[vid] = i
    end
    output_index = Dict{ValueId,Int}()
    for (i, vid) in enumerate(prog.outputs)
        output_index[vid] = i
    end

    input_ref(vid) = input_names[input_index[vid]]  # e.g. :_in1
    output_ref(vid) = output_names[output_index[vid]]  # e.g. :_out1

    ###################
    ### MEMORY PLAN ###
    ###################
    (
        slots,
        shared_mat_inputs,
        mat_slots,
        shared_vec_inputs,
        vec_slots,
        require_extra_slot,
        mat_load_slot_id
    ) = plan_memory_usage(prog)
    mat_load_slot = "M$mat_load_slot_id"

    load_steps = get_load_steps(prog, slots)
    active_init_steps = get_active_init_steps(mat_slots, vec_slots)
    compute_steps = get_compute_steps(prog, slots)
    store_steps = get_store_steps(prog, slots, mat_load_slot)

    #########################
    ### INITIAL VARIABLES ###
    #########################

    stmts = Expr[]

    # Thread dependent variables
    push!(stmts, :(tid = threadIdx().x))
    push!(stmts, :(bid = blockIdx().x))
    push!(stmts, :(wid = div(tid - 1i32, 32i32) + 1i32))
    push!(stmts, :(lid = mod1(tid, 32i32)))

    # Compile-time constants
    push!(stmts, :(D_i32 = Int32($(prog.D))))
    push!(stmts, :(nthreads_i32 = Int32($(prog.nthreads))))
    push!(stmts, :(n_mats_per_warp = 32i32 ÷ D_i32))
    push!(stmts, :(n_warps = nthreads_i32 ÷ 32i32))
    push!(stmts, :(n_mats_per_block = n_warps * n_mats_per_warp))
    push!(stmts, :(dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D_i32, 32i32), 32i32)))

    # Calculating matrix shared memory size
    push!(stmts, :(warp_matrix_id = div(lid - 1i32, D_i32) + 1i32))
    push!(stmts, :(d = mod1(lid, D_i32)))
    push!(stmts, :(grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block))
    push!(stmts, :(warp_shmem_size = n_mats_per_warp * D_i32 * D_i32 + dual_padding * (D_i32 - 1i32)))
    push!(stmts, :(mat_shmem_elems = warp_shmem_size * n_warps))

    # Calculating shared matrix memory size
    push!(stmts, :(pad_interval = div(32i32, D_i32 & -D_i32) * D_i32))
    push!(stmts, :(shared_mat_shmem_elems = D_i32 * D_i32 + (D_i32 * D_i32 - 1i32) ÷ pad_interval))

    # Calculating vector shared memory size
    push!(stmts, :(vec_shmem_elems = D_i32 * n_warps * n_mats_per_warp))

    # Calculating shared vector memory size
    push!(stmts, :(shared_vec_shmem_elems = D_i32))

    ####################################
    ### SHARED MEMORY INITIALISATION ###
    ####################################

    for slot_id in 1:shared_mat_inputs
        push!(stmts, Expr(
            :(=),
            Symbol("shmem_M$(slot_id)"),
            :(CuStaticSharedArray($T, (shared_mat_shmem_elems,))),
        ))
    end
    for slot_id in 1:shared_vec_inputs
        push!(stmts, Expr(
            :(=),
            Symbol("shmem_v$(slot_id)"),
            :(CuStaticSharedArray($T, (shared_vec_shmem_elems,))),
        ))
    end
    for slot in mat_slots
        push!(stmts, Expr(
            :(=),
            Symbol("shmem_$slot"),
            :(CuStaticSharedArray($T, (mat_shmem_elems,))),
        ))
    end
    for slot in vec_slots
        push!(stmts, Expr(
            :(=),
            Symbol("shmem_$slot"),
            :(CuStaticSharedArray($T, (vec_shmem_elems,))),
        ))
    end
    # Possibly extra one for intermediate layout transfer
    if require_extra_slot
        push!(stmts, Expr(
            :(=),
            Symbol("shmem_$(mat_load_slot)"),
            :(CuStaticSharedArray($T, (mat_shmem_elems,))),
        ))
    end

    ##################################
    ### EMIT INPUT LOAD STATEMENTS ###
    ##################################

    for curr_step in load_steps
        if curr_step isa LoadSharedMatStep
            step = curr_step::LoadSharedMatStep
            slot = step.slot
            # wid_eq = :(Int32($(step.wid_eq)))
            wid_eq_expr = Expr(:call, :Int32, step.wid_eq)
            input_expr = input_ref(step.vid)

            push!(stmts, Expr(
                :if,
                # :(wid == $wid_eq),
                :(wid == $wid_eq_expr),
                :(shared_matrix_load!($(Symbol("shmem_$slot")), $input_expr, Val(D_i32))),
            ))
            push!(stmts, Expr(
                :(=),
                Symbol(slot),
                :(SharedMatrix($(Symbol("shmem_$slot")), Val(D_i32))),
            ))
        
        elseif curr_step isa LoadSharedVecStep
            step = curr_step::LoadSharedVecStep
            slot = step.slot
            # wid_eq = :(Int32($(step.wid_eq)))
            wid_eq_expr = Expr(:call, :Int32, step.wid_eq)
            input_expr = input_ref(step.vid)

            push!(stmts, Expr(
                :if,
                # :(wid == $wid_eq),
                :(wid == $wid_eq_expr),
                :(shared_vector_load!($(Symbol("shmem_$slot")), $input_expr, Val(D_i32))),
            ))
            push!(stmts, Expr(
                :(=),
                Symbol(slot),
                :(SharedVector($(Symbol("shmem_$slot")), Val(D_i32))),
            ))
        
        elseif curr_step isa LoadBatMatStep
            step = curr_step::LoadBatMatStep
            slot = step.slot
            input_expr = input_ref(step.vid)

            push!(
                stmts,
                :(intermediate_layout_load!($(Symbol("shmem_$mat_load_slot")), $input_expr, Val(D_i32), Val(nthreads_i32), N, Val(:small))),
            )
            push!(
                stmts,
                :(interm_to_dual_transfer!($(Symbol("shmem_$slot")), $(Symbol("shmem_$mat_load_slot")), Val(D_i32), Val(nthreads_i32), N, Val(:small))),
            )
        
        elseif curr_step isa LoadBatVecStep
            step = curr_step::LoadBatVecStep
            slot = step.slot
            input_expr = input_ref(step.vid)
            push!(
                stmts,
                :(vector_load!($(Symbol("shmem_$slot")), $input_expr, Val(D_i32), Val(nthreads_i32), N)),
            )
        end
    end

    # Add sync_threads() call for shared matrices & vectors
    push!(stmts, Expr(:call, :sync_threads))

    ############################################
    ### ACTIVE THREADS STATEMENTS (if-block) ###
    ############################################

    active = Expr[]

    # Initialise matrix & vector wrappers
    for curr_step in active_init_steps
        if curr_step isa InitMatWrapperStep
            step = curr_step::InitMatWrapperStep
            slot = step.slot
            push!(active, Expr(
                :(=),
                Symbol(slot),
                :(DualAccessMatrix($(Symbol("shmem_$slot")), Val(D_i32), warp_matrix_id, Val(:small))),
            ))
        
        elseif curr_step isa InitVecWrapperStep
            step = curr_step::InitVecWrapperStep
            slot = step.slot
            push!(active, Expr(
                :(=),
                Symbol(slot),
                :(BatchedVector($(Symbol("shmem_$slot")), Val(D_i32), warp_matrix_id)),
            ))
        
        else
            error("Unknown init step type $(typeof(curr_step))")
        end
    end

    # Render compute steps
    for curr_step in compute_steps
        if curr_step isa BatchBinaryStep
            step = curr_step::BatchBinaryStep
            f = step.f
            dest = Symbol(step.dest)
            a = Symbol(step.a)
            b = Symbol(step.b)
            push!(
                active,
                :(batch_op!($f, $dest, $a, $b, d, Val(D_i32), Val(:small))),
            )
                    
        elseif curr_step isa BatchUnaryStep
            step = curr_step::BatchUnaryStep
            f = step.f
            dest = Symbol(step.dest)
            a = Symbol(step.a)

            if f === cholesky
                push!(
                    active,
                    :(batch_op!(cholesky, $dest, $a, d, Val(D_i32), n_mats_per_warp, warp_matrix_id, Val(:small))),
                )
            elseif f === transpose
                push!(
                    active,
                    :(batch_op!(transpose, $dest, $a, d, Val(D_i32), Val(:small))),
                )
            else
                error("Unknown unary op $f")
            end

        elseif curr_step isa WrapperStep
            step = curr_step::WrapperStep
            dest = Symbol(step.dest)
            a = Symbol(step.a)
            wrapper = step.wrapper
            push!(active, Expr(
                :(=),
                dest,
                Expr(:call, wrapper, a),
            ))
        
        elseif curr_step isa IAddSubStep{T}
            step = curr_step::IAddSubStep{T}
            dest = Symbol(step.dest)
            a = Symbol(step.a)
            λ = step.λ
            sign = step.sign
            push!(active, Expr(
                :(=),
                dest,
                Expr(:call, :IAddSubGetterMatrix, a, λ, sign),
            ))
        
        else
            error("Unknown compute step type $(typeof(curr_step))")
        end
    end

    # Masking inactive threads via if-statement
    push!(stmts, Expr(
        :if,
        :(warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N),
        Expr(:block, active...),
    ))

    ##########################
    ### EMIT OUTPUT STORES ###
    ##########################

    for curr_step in store_steps
        if curr_step isa StoreMatStep
            step = curr_step::StoreMatStep
            slot = Symbol("shmem_$(step.slot)")
            store_slot = Symbol("shmem_$(step.store_slot)")
            output_expr = output_ref(step.vid)
            push!(
                stmts,
                :(dual_to_interm_transfer!($store_slot, $slot, Val(D_i32), Val(nthreads_i32), N, Val(:small))),
            )
            push!(
                stmts,
                :(intermediate_layout_write!($output_expr, $store_slot, Val(D_i32), Val(nthreads_i32), N, Val(:small), Val(:indep))),
            )
        
        elseif curr_step isa StoreVecStep
            step = curr_step::StoreVecStep
            slot = Symbol("shmem_$(step.slot)")
            output_expr = output_ref(step.vid)
            push!(
                stmts,
                :(vector_write!($output_expr, $slot, Val(D_i32), Val(nthreads_i32), N))
            )
        else
            error("Unknown store step type $(typeof(curr_step))")
        end
    end

    push!(stmts, :(return nothing))

    return Expr(:block, stmts...)
end

"Pretty-print generated Expr for debugging."
function debug_print_expr(expr::Expr)
    println(sprint(show, expr))
end
