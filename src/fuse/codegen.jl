# =============================================================================
# Codegen
# =============================================================================
#
# `codegen` walks the tape + planner and emits a single CUDA kernel `Expr`:
#   - allocates shared-memory slots (one per batched matrix slot, one per
#     batched vector slot, one per shared input, plus a dedicated single-access
#     load buffer — A6 in the merge plan will fold the load buffer into the
#     regular free-list);
#   - loads each batched input the first time it's needed: matrices go via
#     global→single→dual transfer; vectors go directly into single-access
#     layout, skipping the dual buffer;
#   - emits the per-primitive operation for each CallNode (under `if active`);
#   - writes each output leaf back: matrices via dual→single→global, vectors
#     directly to global.

struct KernelSignature
    fn_name::Symbol
    n_outputs::Int
    n_batched_inputs::Int
    n_shared_inputs::Int
end

function codegen(
    tape::Tape,
    planner::PlannerOutput,
    leaves::Vector{LeafOutput};
    D_MAX::Int,
    nthreads::Int,
    T::Type,
    fn_name::Symbol=:_fused_kernel,
)
    D32 = Int32(D_MAX)
    nthreads32 = Int32(nthreads)
    n_mats_per_warp = Int32(32) ÷ D32
    n_warps = nthreads32 ÷ Int32(32)
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D32, Int32(32)), Int32(32))
    mat_shmem_elems =
        (n_mats_per_warp * D32 * D32 + dual_padding * (D32 - Int32(1))) * n_warps
    # Vector slots use the single-access layout: n_vecs_per_warp == n_mats_per_warp
    # vectors of length D_MAX per warp, no padding.
    vec_shmem_elems = n_mats_per_warp * D32 * n_warps

    input_ids = [r.id for r in tape.inputs]
    batched_input_ids = filter(
        id -> tape.metas[id].lifecycle == BATCHED && tape.nodes[id] isa InputNode,
        input_ids,
    )
    shared_input_ids = filter(
        id -> tape.metas[id].lifecycle == SHARED && tape.nodes[id] isa InputNode,
        input_ids,
    )

    out_syms = [Symbol(:_out, i) for i in 1:length(leaves)]
    b_in_syms = [Symbol(:_bin, i) for i in 1:length(batched_input_ids)]
    s_in_syms = [Symbol(:_sin, i) for i in 1:length(shared_input_ids)]

    input_sym = Dict{Int,Symbol}()
    for (k, id) in enumerate(batched_input_ids)
        input_sym[id] = b_in_syms[k]
    end
    for (k, id) in enumerate(shared_input_ids)
        input_sym[id] = s_in_syms[k]
    end

    stmts = Expr[]
    push!(stmts, :(tid = threadIdx().x))
    push!(stmts, :(bid = blockIdx().x))
    push!(stmts, :(lid = mod1(tid, 32i32)))
    push!(stmts, :(wid = div(tid - 1i32, 32i32) + 1i32))
    push!(stmts, :(warp_matrix_id = div(lid - 1i32, $D32) + 1i32))
    push!(stmts, :(d = mod1(lid, $D32)))
    push!(stmts, :(block_matrix_id = warp_matrix_id + (wid - 1i32) * $n_mats_per_warp))
    push!(stmts, :(grid_mtrx_id = block_matrix_id + (bid - 1i32) * $n_mats_per_block))
    push!(stmts, :(active = warp_matrix_id <= $n_mats_per_warp && grid_mtrx_id <= N))

    # Matrix slot shmem allocations.
    matrix_slot_syms = Symbol[]
    for s in 1:(planner.num_matrix_slots)
        sym = Symbol("shmem_M", s)
        push!(matrix_slot_syms, sym)
        push!(stmts, :($sym = CuStaticSharedArray($T, ($mat_shmem_elems,))))
    end
    push!(stmts, :(shmem_load = CuStaticSharedArray($T, ($mat_shmem_elems,))))

    # Vector slot shmem allocations.
    vector_slot_syms = Symbol[]
    for s in 1:(planner.num_vector_slots)
        sym = Symbol("shmem_V", s)
        push!(vector_slot_syms, sym)
        push!(stmts, :($sym = CuStaticSharedArray($T, ($vec_shmem_elems,))))
    end

    # Scalar output staging slots: one `n_mats_per_block`-sized shmem buffer per
    # distinct scalar terminal in the output tree. Used in the output epilogue
    # to cross from leader-lane registers to a coalesced cross-warp write.
    sout_shmem_syms = Dict{Int,Symbol}()  # scalar node id -> shmem symbol
    for (node_id, slot) in planner.scalar_output_slots
        shmem_sym = Symbol("shmem_Sout", slot.idx)
        sout_shmem_syms[node_id] = shmem_sym
        push!(stmts, :($shmem_sym = CuStaticSharedArray($T, ($n_mats_per_block,))))
    end

    # Scalar tape locals: each BATCHED `TraceScalar` node lives in a Julia local
    # `s{id}`, replicated across the D lanes of a warp-matrix after the
    # producing reduction's shfl-broadcast. Pre-initialised here so the
    # variables exist outside the per-primitive `if active` blocks (a value
    # computed only when `active` would otherwise be undefined for inactive
    # lanes).
    scalar_node_sym = Dict{Int,Symbol}()
    for (i, (node, meta)) in enumerate(zip(tape.nodes, tape.metas))
        if slot_kind(node, meta) == :scalar
            sym = Symbol("s", i)
            scalar_node_sym[i] = sym
            push!(stmts, :($sym = zero($T)))
        end
    end

    # Shared-matrix and shared-vector input slot allocations + loads.
    shared_input_view_syms = Dict{Int,Symbol}()
    shared_load_order = Int[]  # parallel array; one shared input per warp
    for (k, id) in enumerate(shared_input_ids)
        push!(shared_load_order, id)
        meta = tape.metas[id]
        slot = planner.shared_slots[id]
        if slot.kind === :M
            shmem_sym = Symbol("shmem_SM", slot.idx)
            view_sym = Symbol("SM", slot.idx)
            D_M, D_N = shape(meta.type)
            D_M32 = Int32(D_M)
            pad_interval = div(Int32(32), D_M32 & -D_M32) * D_M32
            shmem_size_fixed = D_M * D_N + (D_M * D_N - 1) ÷ pad_interval
            push!(stmts, :($shmem_sym = CuStaticSharedArray($T, ($shmem_size_fixed,))))
            push!(
                stmts,
                Expr(
                    :if,
                    :(wid == $(Int32(k))),
                    :(shared_matrix_load!(
                        $shmem_sym, $(s_in_syms[k]), Val(Int32($D_M)), Val(Int32($D_N))
                    )),
                ),
            )
            shared_input_view_syms[id] = view_sym
        else
            shmem_sym = Symbol("shmem_SV", slot.idx)
            view_sym = Symbol("SV", slot.idx)
            D_M, = shape(meta.type)
            push!(stmts, :($shmem_sym = CuStaticSharedArray($T, ($D_M,))))
            push!(
                stmts,
                Expr(
                    :if,
                    :(wid == $(Int32(k))),
                    :(shared_vector_load!($shmem_sym, $(s_in_syms[k]), Val(Int32($D_M)))),
                ),
            )
            shared_input_view_syms[id] = view_sym
        end
    end

    if !isempty(shared_input_ids)
        push!(stmts, :(sync_threads()))
    end

    # Shared-input view constructors (after sync).
    for id in shared_load_order
        slot = planner.shared_slots[id]
        meta = tape.metas[id]
        if slot.kind === :M
            shmem_sym = Symbol("shmem_SM", slot.idx)
            view_sym = Symbol("SM", slot.idx)
            D_M, D_N = shape(meta.type)
            push!(
                stmts,
                :(
                    $view_sym = SharedMatrix(
                        $shmem_sym, Val(Int32($D_M)), Val(Int32($D_N))
                    )
                ),
            )
        else
            shmem_sym = Symbol("shmem_SV", slot.idx)
            view_sym = Symbol("SV", slot.idx)
            D_M, = shape(meta.type)
            push!(stmts, :($view_sym = SharedVector($shmem_sym, Val(Int32($D_M)))))
        end
    end

    # Batched matrix slot view constructors.
    for s in 1:(planner.num_matrix_slots)
        view_sym = Symbol("M", s)
        push!(
            stmts,
            :(
                $view_sym = DualAccessMatrix(
                    $(matrix_slot_syms[s]), Val($D32), warp_matrix_id, Val(:small)
                )
            ),
        )
    end

    # Batched vector slot view constructors.
    for s in 1:(planner.num_vector_slots)
        view_sym = Symbol("V", s)
        push!(
            stmts,
            :($view_sym = BatchedVector($(vector_slot_syms[s]), Val($D32), warp_matrix_id)),
        )
    end

    node_view_sym = Dict{Int,Symbol}()
    for id in shared_input_ids
        node_view_sym[id] = shared_input_view_syms[id]
    end

    function slot_view_sym(slot::SlotAssignment)
        return Symbol(slot.kind === :M ? "M" : "V", slot.idx)
    end
    function slot_shmem_sym(slot::SlotAssignment)
        return slot.kind === :M ? matrix_slot_syms[slot.idx] : vector_slot_syms[slot.idx]
    end

    loaded = Set{Int}()
    function maybe_load_batched_input!(node_id::Int)
        node_id in loaded && return nothing
        node = tape.nodes[node_id]
        meta = tape.metas[node_id]
        (node isa InputNode && meta.lifecycle == BATCHED) || return nothing
        slot = planner.slots[node_id]
        global_in = input_sym[node_id]
        if slot.kind === :M
            D_M, D_N = shape(meta.type)
            push!(
                stmts,
                :(intermediate_layout_load!(
                    shmem_load,
                    $global_in,
                    Val(Int32($D_M)),
                    Val(Int32($D_N)),
                    Val($D32),
                    Val($nthreads32),
                    N,
                    Val(:small),
                )),
            )
            push!(
                stmts,
                :(interm_to_dual_transfer!(
                    $(slot_shmem_sym(slot)),
                    shmem_load,
                    Val(Int32($D_M)),
                    Val(Int32($D_N)),
                    Val($D32),
                    Val($nthreads32),
                    N,
                    Val(:small),
                )),
            )
        else
            D_M, = shape(meta.type)
            D_M == D_MAX || error(
                "codegen: batched vector with D_M=$D_M ≠ D_MAX=$D_MAX not yet supported (masked vector view pending).",
            )
            push!(
                stmts,
                :(vector_load!(
                    $(slot_shmem_sym(slot)),
                    $global_in,
                    Val(Int32($D_M)),
                    Val($D32),
                    Val($nthreads32),
                    N,
                )),
            )
        end
        node_view_sym[node_id] = slot_view_sym(slot)
        return push!(loaded, node_id)
    end

    function maybe_load_batched_inputs_for_ref!(ref::NodeRef)
        node = tape.nodes[ref.id]
        if node isa NewNode
            for (_, child) in node.fields
                maybe_load_batched_inputs_for_ref!(child)
            end
        else
            maybe_load_batched_input!(ref.id)
        end
    end

    for (i, (node, meta)) in enumerate(zip(tape.nodes, tape.metas))
        if node isa InputNode || node isa ConstNode
            continue
        end
        if node isa NewNode
            continue
        end
        for ref in node.args
            maybe_load_batched_inputs_for_ref!(ref)
        end

        arg_exprs = Any[arg_kernel_expr(tape, ref, node_view_sym) for ref in node.args]
        arg_types = Any[tape.metas[ref.id].type for ref in node.args]

        if slot_kind(node, meta) == :scalar
            # Scalar destinations write into the Julia local — the corresponding
            # emit_primitive method returns an assignment (`s_i = …`).
            dest = scalar_node_sym[i]
        else
            dest_slot = planner.slots[i]
            dest = slot_view_sym(dest_slot)
        end
        node_view_sym[i] = dest

        emit_expr = emit_primitive(node.fn, dest, arg_exprs, arg_types, D_MAX)
        push!(stmts, Expr(:if, :active, Expr(:block, emit_expr)))
    end

    # Matrix and vector output leaves; scalar leaves are handled below.
    for (k, leaf) in enumerate(leaves)
        leaf.trace_type <: TraceScalar && continue
        out_sym = out_syms[k]
        leaf_slot = leaf.slot
        if leaf_slot.kind === :M
            leaf_view = slot_view_sym(leaf_slot)
            D_M, D_N = shape(leaf.trace_type)
            push!(
                stmts,
                :(dual_to_interm_transfer!(
                    shmem_load,
                    $leaf_view,
                    Val(Int32($D_M)),
                    Val(Int32($D_N)),
                    Val($D32),
                    Val($nthreads32),
                    N,
                    Val(:small),
                )),
            )
            push!(
                stmts,
                :(intermediate_layout_write!(
                    $out_sym,
                    shmem_load,
                    Val(Int32($D_M)),
                    Val(Int32($D_N)),
                    Val($D32),
                    Val($nthreads32),
                    N,
                    Val(:small),
                )),
            )
        else
            D_M, = shape(leaf.trace_type)
            D_M == D_MAX || error(
                "codegen: batched vector output with D_M=$D_M ≠ D_MAX=$D_MAX not yet supported.",
            )
            push!(
                stmts,
                :(vector_write!(
                    $out_sym,
                    $(slot_shmem_sym(leaf_slot)),
                    Val($D32),
                    Val($nthreads32),
                    N,
                )),
            )
        end
    end

    # Scalar output leaves. Pattern: stage each leader's register value into its
    # :Sout slot, sync_threads once, then cooperatively write each slot to its
    # global out buffer. `scalar_stage!` is leader-and-active-gated internally,
    # so we call it unconditionally outside the `if active` block.
    scalar_leaf_indices = Int[k for (k, leaf) in enumerate(leaves) if leaf.trace_type <: TraceScalar]
    for k in scalar_leaf_indices
        leaf = leaves[k]
        s_local = scalar_node_sym[leaf.node_id]
        shmem_sym = sout_shmem_syms[leaf.node_id]
        push!(
            stmts,
            :(scalar_stage!(
                $shmem_sym,
                $s_local,
                lid,
                warp_matrix_id,
                block_matrix_id,
                active,
                Val($D32),
            )),
        )
    end
    if !isempty(scalar_leaf_indices)
        push!(stmts, :(sync_threads()))
    end
    for k in scalar_leaf_indices
        leaf = leaves[k]
        out_sym = out_syms[k]
        shmem_sym = sout_shmem_syms[leaf.node_id]
        push!(stmts, :(scalar_write!($out_sym, $shmem_sym, $n_mats_per_block)))
    end

    push!(stmts, :(return nothing))

    args = [out_syms..., b_in_syms..., s_in_syms..., :(N::Int32)]
    fn_expr = Expr(:function, Expr(:call, fn_name, args...), Expr(:block, stmts...))

    return fn_expr,
    KernelSignature(
        fn_name, length(leaves), length(batched_input_ids), length(shared_input_ids)
    )
end

function arg_kernel_expr(tape::Tape, ref::NodeRef, node_view_sym::Dict{Int,Symbol})
    n = tape.nodes[ref.id]
    if n isa NewNode
        T = n.T
        if T <: Adjoint
            inner = arg_kernel_expr(tape, n.fields[1].second, node_view_sym)
            return :(adjoint($inner))
        elseif T <: LowerTriangular
            inner = arg_kernel_expr(tape, n.fields[1].second, node_view_sym)
            return :(LowerTriangular($inner))
        elseif T <: UpperTriangular
            inner = arg_kernel_expr(tape, n.fields[1].second, node_view_sym)
            return :(UpperTriangular($inner))
        elseif T <: Symmetric
            inner = arg_kernel_expr(tape, n.fields[1].second, node_view_sym)
            return :(Symmetric($inner))
        else
            error("arg_kernel_expr: unsupported wrapper type $T")
        end
    elseif n isa ConstNode
        return n.val
    elseif haskey(node_view_sym, ref.id)
        return node_view_sym[ref.id]
    else
        error("arg_kernel_expr: no slot view for node %$(ref.id) :: $(typeof(n))")
    end
end
