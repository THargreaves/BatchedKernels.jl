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
    shape::MatShape
end
struct LoadSharedVecStep <: KernelStep
    slot::String
    vid::ValueId
    wid_eq::Int
    shape::VecShape
end
struct LoadBatMatStep <: KernelStep
    slot::String
    load_slot::String
    vid::ValueId
    shape::MatShape
end
struct LoadBatVecStep <: KernelStep
    slot::String
    vid::ValueId
    shape::VecShape
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
    shape::Shape
    shape2::Union{Nothing,Shape}
    adj::Union{Nothing,Bool}
end
function BatchBinaryStep(
    f::F,
    dest::String,
    a::String,
    b::String,
    shape::Shape,
) where {F}
    return BatchBinaryStep{F}(f, dest, a, b, shape, nothing, nothing)
end
function BatchBinaryStep(
    f::F,
    dest::String,
    a::String,
    b::String,
    shape::Shape,
    shape2::Shape,
    adj::Bool
) where {F}
    return BatchBinaryStep{F}(f, dest, a, b, shape, shape2, adj)
end
struct BatchUnaryStep{F} <: KernelStep
    f::F  # cholesky, transpose
    dest::String
    a::String
    shape::Shape
end
struct WrapperStep <: KernelStep
    wrapper::Symbol  # :adjoint, :LowerTriangular, :UpperTriangular, :Symmetric
    dest::String
    a::String
    shape::MatShape
end
struct IAddSubStep{T} <: KernelStep
    dest::String
    a::String
    λ::T
    sign::T
    shape::MatShape
end

# Stores
struct StoreMatStep <: KernelStep
    slot::String
    store_slot::String
    vid::ValueId
    shape::MatShape
end
struct StoreVecStep <: KernelStep
    slot::String
    vid::ValueId
    shape::VecShape
end

########################
### STEP PLANNERS ###
########################

matshape(prog::IRProgram, vid::ValueId) = prog.shapes[vid]::MatShape
vecshape(prog::IRProgram, vid::ValueId) = prog.shapes[vid]::VecShape

function get_shared_load_steps(prog::IRProgram, slots::Dict{ValueId,String})
    shared_load_steps = KernelStep[]
    shared_input_count = 1

    for (_, vid) in prog.inputs
        if is_shared(prog, vid)
            if is_shared_mat(prog, vid)
                push!(shared_load_steps, LoadSharedMatStep(slots[vid], vid, shared_input_count, matshape(prog, vid)))
            elseif is_shared_vec(prog, vid)
                push!(shared_load_steps, LoadSharedVecStep(slots[vid], vid, shared_input_count, vecshape(prog, vid)))
            else
                error("Unknown shared kind for input $vid: $(prog.kinds[vid])")
            end
            shared_input_count += 1
        end
    end

    return shared_load_steps
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

function get_compute_steps(
    prog::IRProgram{T},
    slots::Dict{ValueId,String},
    input_load_schedule::Vector{Vector{Tuple{ValueId,String}}},
    kinds::Dict{ValueId,Type{<:SymKind}},
) where {T}
    compute_steps = KernelStep[]

    for (i, node) in enumerate(prog.nodes)
        op = node.op

        # Check if this step involves an input load
        for (vid, load_slot) in input_load_schedule[i]
            K = kinds[vid]
            if K <: MatKind
                push!(compute_steps, LoadBatMatStep(slots[vid], load_slot, vid, matshape(prog, vid)))
            elseif K <: VecKind
                push!(compute_steps, LoadBatVecStep(slots[vid], vid, vecshape(prog, vid)))
            else
                error("Invalid type for loading inputs, supported MatKind and VecKind, got $K")
            end
        end

        if op == :mul
            a = node.args[1]::ValueId
            b = node.args[2]::ValueId
            shape = prog.shapes[a]
            (
                is_mat(prog, a) && (is_mat(prog, b) || is_vec(prog, b))
            ) || error("Multipication only supports mat*mat or mat*vec")
            push!(compute_steps, BatchBinaryStep(*, slots[node.out], slots[a], slots[b], shape))
        
        elseif op == :add
            a = node.args[1]::ValueId
            b = node.args[2]::ValueId
            shape = prog.shapes[a]
            (
                (is_mat(prog, a) && is_mat(prog, b)) || (is_vec(prog, a) && is_vec(prog, b))
            ) || error("Addition only supports mat+mat or vec+vec")
            push!(compute_steps, BatchBinaryStep(+, slots[node.out], slots[a], slots[b], shape))

        elseif op == :sub
            a = node.args[1]::ValueId
            b = node.args[2]::ValueId
            shape = prog.shapes[a]
            (
                (is_mat(prog, a) && is_mat(prog, b)) || (is_vec(prog, a) && is_vec(prog, b))
            ) || error("Subtractino only supports mat-mat or vec-vec")
            push!(compute_steps, BatchBinaryStep(-, slots[node.out], slots[a], slots[b], shape))
        
        elseif op == :chol
            a = node.args[1]::ValueId
            shape = prog.shapes[a]
            is_mat(prog, a) || error("Cholesky only defined for matrices")
            push!(compute_steps, BatchUnaryStep(cholesky, slots[node.out], slots[a], shape))
        
        elseif op == :qr
            a = node.args[1]::ValueId
            shape = prog.shapes[a]
            is_mat(prog, a) || error("QR decomposition only defined for matrices")
            push!(compute_steps, BatchUnaryStep(qr, slots[node.out], slots[a], shape))
        
        elseif op == :qr_Q_thin
            a = node.args[1]::ValueId
            shape = node.args[2]::MatShape
            # shape = prog.shapes[a]
            is_mat(prog, a) || error("Invalid type to materialise")
            push!(compute_steps, BatchUnaryStep(:qr_Q_thin, slots[node.out], slots[a], shape))
        
        elseif op == :qr_Q_full
            a = node.args[1]::ValueId
            shape = node.args[2]::MatShape
            # shape = prog.shapes[a]
            is_mat(prog, a) || error("Invalid type to materialise")
            push!(compute_steps, BatchUnaryStep(:qr_Q_full, slots[node.out], slots[a], shape))

        elseif op == :qr_Q_multiply
            a = node.args[1]::ValueId
            b = node.args[2]::ValueId
            adj = node.args[3]::Bool
            shape1 = node.args[4]::MatShape
            shape2 = prog.shapes[b]
            (
                is_mat(prog, a) && is_mat(prog, b)
            ) || error("QR multipication only supports mat*mat")
            push!(compute_steps, BatchBinaryStep(:qr_Q_multiply, slots[node.out], slots[a], slots[b], shape1, shape2, adj))

        elseif op == :trans
            a = node.args[1]::ValueId
            shape = prog.shapes[a]
            is_mat(prog, a) || error("Transposition only defined for matrices")
            if node.out in prog.outputs  # In-place for outputs
                push!(compute_steps, BatchUnaryStep(transpose, slots[node.out], slots[a], shape))
            else  # Wrapper for non-outputs
                push!(compute_steps, WrapperStep(:adjoint, slots[node.out], slots[a], shape))
            end

        elseif op == :lowertrig
            a = node.args[1]::ValueId
            shape = prog.shapes[a]
            is_mat(prog, a) || error("LowerTriangular only defined for matrices")
            push!(compute_steps, WrapperStep(:LowerTriangular, slots[node.out], slots[a], shape))
        
        elseif op == :uppertrig
            a = node.args[1]::ValueId
            shape = prog.shapes[a]
            is_mat(prog, a) || error("UpperTriangular only defined for matrices")
            push!(compute_steps, WrapperStep(:UpperTriangular, slots[node.out], slots[a], shape))

        elseif op == :sym
            a = node.args[1]::ValueId
            shape = prog.shapes[a]
            is_mat(prog, a) || error("Symmetric only defined for matrices")
            push!(compute_steps, WrapperStep(:Symmetric, slots[node.out], slots[a], shape))

        elseif op == :forwardsolve || op == :backwardsolve
            a = node.args[1]::ValueId
            b = node.args[2]::ValueId
            shape = prog.shapes[b]  # Take shape of the second input
            (
                is_lowertrig(prog, a) || is_uppertrig(prog, a)
            ) || error("Triangular solve requires triangular L/U")
            push!(compute_steps, BatchBinaryStep(\, slots[node.out], slots[a], slots[b], shape))            
        
        elseif op == :iplus || op == :iminus
            a = node.args[1]::ValueId
            λ = node.args[2]::T
            shape = prog.shapes[a]
            is_mat(prog, a) || error("I +- var only defined for matrices")
            sign = op == :iplus ? one(T) : -one(T)
            push!(compute_steps, IAddSubStep{T}(slots[node.out], slots[a], λ, sign, shape))

        else
            error("Unsupported operation $op")
        end
    end

    return compute_steps
end

function get_store_steps(prog::IRProgram, slots::Dict{ValueId,String}, mat_store_slot::String, mat_slots::Vector{String})
    store_steps = KernelStep[]

    for vid in prog.outputs
        !is_shared(prog, vid) || error("Cannot return shared variables")

        if is_mat(prog, vid)
            slot = slots[vid]
            shape = matshape(prog, vid)
            store_slot = mat_store_slot != slot ? mat_store_slot : (slot != mat_slots[1] ? mat_slots[1] : mat_slots[2])
            push!(store_steps, StoreMatStep(slots[vid], store_slot, vid, shape))
        elseif is_vec(prog, vid)
            shape = vecshape(prog, vid)
            push!(store_steps, StoreVecStep(slots[vid], vid, shape))
        else
            error("Unsupported output kind: $(prog.kinds[vid])")
        end
    end

    return store_steps
end

"Generates and emits the Julia AST for the generated kernel from the IR."
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
        mat_slots,
        vec_slots,
        input_load_schedule,
        require_extra_slot,
        mat_store_slot,
    ) = plan_memory_usage(prog)

    shared_load_steps = get_shared_load_steps(prog, slots)
    active_init_steps = get_active_init_steps(mat_slots, vec_slots)
    compute_steps = get_compute_steps(prog, slots, input_load_schedule, prog.kinds)
    store_steps = get_store_steps(prog, slots, mat_store_slot, mat_slots)

    #########################
    ### INITIAL VARIABLES ###
    #########################

    stmts = Expr[]

    # Compile-time constants
    D = Int32(prog.D)
    nthreads = Int32(prog.nthreads)
    n_mats_per_warp = Int32(32) ÷ D
    n_warps = nthreads ÷ Int32(32)
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, Int32(32)), Int32(32))

    # Calculating matrix shared memory size
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - Int32(1))
    mat_shmem_elems = warp_shmem_size * n_warps

    # Calculating vector shared memory size
    vec_shmem_elems = D * n_warps * n_mats_per_warp

    # Thread dependent variables
    push!(stmts, :(tid = threadIdx().x))
    push!(stmts, :(bid = blockIdx().x))
    push!(stmts, :(lid = mod1(tid, 32i32)))
    push!(stmts, :(warp_matrix_id = div(lid - 1i32, $D) + 1i32))
    push!(stmts, :(d = mod1(lid, $D)))
    push!(stmts, :(grid_mtrx_id = warp_matrix_id + (bid - 1i32) * $n_mats_per_block))
    push!(stmts, :(active = warp_matrix_id <= $n_mats_per_warp && grid_mtrx_id <= N))
    if !isempty(shared_load_steps)
        push!(stmts, :(wid = div(tid - 1i32, 32i32) + 1i32))
    end

    ####################################
    ### SHARED MEMORY INITIALISATION ###
    ####################################

    for slot in mat_slots
        push!(stmts, Expr(
            :(=),
            Symbol("shmem_$slot"),
            :(CuStaticSharedArray($T, ($mat_shmem_elems,))),
        ))
    end
    for slot in vec_slots
        push!(stmts, Expr(
            :(=),
            Symbol("shmem_$slot"),
            :(CuStaticSharedArray($T, ($vec_shmem_elems,))),
        ))
    end
    # Possibly extra one for intermediate layout transfer during storing
    if require_extra_slot
        push!(stmts, Expr(
            :(=),
            Symbol("shmem_$(mat_store_slot)"),
            :(CuStaticSharedArray($T, ($mat_shmem_elems,))),
        ))
    end

    #########################################
    ### EMIT SHARED INPUT LOAD STATEMENTS ###
    #########################################

    for curr_step in shared_load_steps
        if curr_step isa LoadSharedMatStep
            step = curr_step::LoadSharedMatStep
            slot = step.slot
            wid_eq_expr = Expr(:call, :Int32, step.wid_eq)
            input_expr = input_ref(step.vid)
            shape = step.shape::MatShape
            D1 = Int32(shape.D1)
            D2 = Int32(shape.D2)

            pad_interval = div(32i32, D1 & -D1) * D1
            shmem_size = D1 * D2 + (D1 * D2 - 1i32) ÷ pad_interval
            push!(stmts, Expr(
                :(=),
                Symbol("shmem_$slot"),
                :(CuStaticSharedArray($T, ($shmem_size,))),
            ))

            push!(stmts, Expr(
                :if,
                :(wid == $wid_eq_expr),
                :(shared_matrix_load!($(Symbol("shmem_$slot")), $input_expr, Val($D1), Val($D2))),
            ))
            push!(stmts, Expr(
                :(=),
                Symbol(slot),
                :(SharedMatrix($(Symbol("shmem_$slot")), Val($D1), Val($D2))),
            ))
        
        elseif curr_step isa LoadSharedVecStep
            step = curr_step::LoadSharedVecStep
            slot = step.slot
            wid_eq_expr = Expr(:call, :Int32, step.wid_eq)
            input_expr = input_ref(step.vid)
            shape = step.shape::VecShape
            D1 = Int32(shape.D1)

            push!(stmts, Expr(
                :(=),
                Symbol("shmem_$slot"),
                :(CuStaticSharedArray($T, ($D1,))),
            ))

            push!(stmts, Expr(
                :if,
                :(wid == $wid_eq_expr),
                :(shared_vector_load!($(Symbol("shmem_$slot")), $input_expr, Val($D1))),
            ))
            push!(stmts, Expr(
                :(=),
                Symbol(slot),
                :(SharedVector($(Symbol("shmem_$slot")), Val($D1))),
            ))

        else
            error("Invalid shared load type")
        end
    end

    # Add sync_threads() call for shared matrices & vectors
    if !isempty(shared_load_steps)
        push!(stmts, Expr(:call, :sync_threads))
    end

    ############################################
    ### ACTIVE THREADS STATEMENTS (if-block) ###
    ############################################

    # Initialise matrix & vector wrappers
    for curr_step in active_init_steps
        if curr_step isa InitMatWrapperStep
            step = curr_step::InitMatWrapperStep
            slot = step.slot
            push!(stmts, Expr(
                :(=),
                Symbol(slot),
                :(DualAccessMatrix($(Symbol("shmem_$slot")), Val($D), warp_matrix_id, Val(:small))),
            ))
        
        elseif curr_step isa InitVecWrapperStep
            step = curr_step::InitVecWrapperStep
            slot = step.slot
            push!(stmts, Expr(
                :(=),
                Symbol(slot),
                :(BatchedVector($(Symbol("shmem_$slot")), Val($D), warp_matrix_id)),
            ))
        
        else
            error("Unknown init step type $(typeof(curr_step))")
        end
    end

    # Render compute steps
    for curr_step in compute_steps
        if curr_step isa LoadBatMatStep
            step = curr_step::LoadBatMatStep
            slot = step.slot
            mat_load_slot = step.load_slot
            input_expr = input_ref(step.vid)
            shape = step.shape::MatShape
            D1 = Int32(shape.D1)
            D2 = Int32(shape.D2)

            push!(
                stmts,
                :(intermediate_layout_load!($(Symbol("shmem_$mat_load_slot")), $input_expr, Val($D1), Val($D2), Val($D), Val($nthreads), N, Val(:small))),
            )
            push!(
                stmts,
                :(interm_to_dual_transfer!($(Symbol("shmem_$slot")), $(Symbol("shmem_$mat_load_slot")), Val($D1), Val($D2), Val($D), Val($nthreads), N, Val(:small))),
            )

        elseif curr_step isa LoadBatVecStep
            step = curr_step::LoadBatVecStep
            slot = step.slot
            input_expr = input_ref(step.vid)
            shape = step.shape::VecShape
            D1 = Int32(shape.D1)

            push!(
                stmts,
                :(vector_load!($(Symbol("shmem_$slot")), $input_expr, Val($D1), Val($D), Val($nthreads), N)),
            )

        elseif curr_step isa BatchBinaryStep
            step = curr_step::BatchBinaryStep
            f = step.f
            dest = Symbol(step.dest)
            a = Symbol(step.a)
            b = Symbol(step.b)
            shape = step.shape
            D1 = Int32(shape.D1)
            D2 = hasproperty(shape, :D2) ? Int32(shape.D2) : D1

            if f == :qr_Q_multiply
                shape2 = step.shape2::MatShape
                B_D1 = shape2.D1
                B_D2 = shape2.D2
                adj = step.adj

                push!(
                    stmts,
                    Expr(
                        :if,
                        :active,
                        Expr(
                            :block,
                            :(batch_op!(Val($(QuoteNode(f))), Val($adj), $dest, $a, $b, d, $(Symbol("tau_$a")), Val($D1), Val($D2), Val($B_D1), Val($B_D2), Val($D), warp_matrix_id, Val(:small))),
                        ),
                    ),
                )
            else
                push!(
                    stmts,
                    Expr(
                        :if,
                        :active,
                        Expr(
                            :block,
                            :(batch_op!($f, $dest, $a, $b, d, Val($D1), Val($D2), Val($D), Val(:small))),
                        ),
                    ),
                )
            end
                    
        elseif curr_step isa BatchUnaryStep
            step = curr_step::BatchUnaryStep
            f = step.f
            dest = Symbol(step.dest)
            a = Symbol(step.a)
            shape = step.shape::MatShape
            D1 = Int32(shape.D1)
            D2 = Int32(shape.D2)

            if f === cholesky
                push!(
                    stmts,
                    Expr(
                        :if,
                        :active,
                        Expr(
                            :block,
                            :(batch_op!(cholesky, $dest, $a, d, Val($D1), Val($D), warp_matrix_id, Val(:small))),
                        ),
                    ),
                )

            elseif f === transpose
                push!(
                    stmts,
                    Expr(
                        :if,
                        :active,
                        Expr(
                            :block,
                            :(batch_op!(transpose, $dest, $a, d, Val($D1), Val($D2), Val($D), Val(:small))),
                        ),
                    ),
                )

            elseif f === qr
                tau_sym = Symbol("tau_$dest")
                push!(stmts, :($(tau_sym) = 0.0f0))
                push!(
                    stmts,
                    Expr(
                        :if,
                        :active,
                        Expr(
                            :block,
                            :(
                                $(tau_sym) = batch_op!(qr, $dest, $a, d, Val($D1), Val($D2), Val($D), warp_matrix_id, Val(:small))
                            ),
                        ),
                    ),
                )
            
            elseif f === :qr_Q_thin || f === :qr_Q_full
                push!(
                    stmts,
                    Expr(
                        :if,
                        :active,
                        Expr(
                            :block,
                            :(batch_op!(Val($(QuoteNode(f))), $dest, $a, d, $(Symbol("tau_$a")), Val($D1), Val($D2), Val($D), warp_matrix_id, Val(:small))),
                        ),
                    ),
                )

            else
                error("Unknown unary op $f")
            end

        elseif curr_step isa WrapperStep
            step = curr_step::WrapperStep
            dest = Symbol(step.dest)
            a = Symbol(step.a)
            wrapper = step.wrapper
            push!(stmts, Expr(
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
            push!(stmts, Expr(
                :(=),
                dest,
                Expr(:call, :IAddSubGetterMatrix, a, λ, sign),
            ))
        
        else
            error("Unknown compute step type $(typeof(curr_step))")
        end
    end

    ##########################
    ### EMIT OUTPUT STORES ###
    ##########################

    for curr_step in store_steps
        if curr_step isa StoreMatStep
            step = curr_step::StoreMatStep
            slot = Symbol(step.slot)
            store_slot = Symbol("shmem_$(step.store_slot)")
            output_expr = output_ref(step.vid)
            shape = step.shape::MatShape
            D1 = Int32(shape.D1)
            D2 = Int32(shape.D2)

            push!(
                stmts,
                :(dual_to_interm_transfer!($store_slot, $slot, Val($D1), Val($D2), Val($D), Val($nthreads), N, Val(:small))),
            )
            push!(
                stmts,
                :(intermediate_layout_write!($output_expr, $store_slot, Val($D1), Val($D2), Val($D), Val($nthreads), N, Val(:small))),
            )
        
        elseif curr_step isa StoreVecStep
            step = curr_step::StoreVecStep
            slot = Symbol("shmem_$(step.slot)")
            output_expr = output_ref(step.vid)
            shape = step.shape::VecShape
            D1 = Int32(shape.D1)
            push!(
                stmts,
                :(vector_write!($output_expr, $slot, Val($D1), Val($D), Val($nthreads), N))
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
