# =============================================================================
# Codegen
# =============================================================================
#
# `codegen` walks the tape + planner and emits a single CUDA kernel `Expr`:
#   - allocates shared-memory slots (one per batched slot, one per shared
#     input, plus a dedicated single-access load buffer — A6 in the merge plan
#     will fold the load buffer into the regular free-list);
#   - loads each batched input via global→single→dual transfer the first time
#     it's needed;
#   - emits the per-primitive operation for each CallNode (under `if active`);
#   - writes each output leaf back via dual→single→global transfer.

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
    D::Int,
    nthreads::Int,
    T::Type,
    fn_name::Symbol=:_fused_kernel,
)
    D32 = Int32(D)
    nthreads32 = Int32(nthreads)
    n_mats_per_warp = Int32(32) ÷ D32
    n_warps = nthreads32 ÷ Int32(32)
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D32, Int32(32)), Int32(32))
    mat_shmem_elems =
        (n_mats_per_warp * D32 * D32 + dual_padding * (D32 - Int32(1))) * n_warps

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
    push!(
        stmts,
        :(
            grid_mtrx_id =
                warp_matrix_id +
                (wid - 1i32) * $n_mats_per_warp +
                (bid - 1i32) * $n_mats_per_block
        ),
    )
    push!(stmts, :(active = warp_matrix_id <= $n_mats_per_warp && grid_mtrx_id <= N))

    batched_slot_syms = Symbol[]
    for s in 1:(planner.num_batched_slots)
        sym = Symbol("shmem_M", s)
        push!(batched_slot_syms, sym)
        push!(stmts, :($sym = CuStaticSharedArray($T, ($mat_shmem_elems,))))
    end
    push!(stmts, :(shmem_load = CuStaticSharedArray($T, ($mat_shmem_elems,))))

    shared_slot_syms = Symbol[]
    for (k, id) in enumerate(shared_input_ids)
        slot_idx = planner.shared_slots[id]
        sym = Symbol("shmem_S", slot_idx)
        push!(shared_slot_syms, sym)
        pad_interval = div(Int32(32), D32 & -D32) * D32
        shmem_size_fixed = D32 * D32 + (D32 * D32 - Int32(1)) ÷ pad_interval
        push!(stmts, :($sym = CuStaticSharedArray($T, ($shmem_size_fixed,))))
        push!(
            stmts,
            Expr(
                :if,
                :(wid == $(Int32(k))),
                :(shared_matrix_load!($sym, $(s_in_syms[k]), Val($D32))),
            ),
        )
    end

    if !isempty(shared_input_ids)
        push!(stmts, :(sync_threads()))
    end

    shared_view_syms = Dict{Int,Symbol}()
    for (k, id) in enumerate(shared_input_ids)
        sym = Symbol("S", planner.shared_slots[id])
        shared_view_syms[id] = sym
        push!(stmts, :($sym = SharedMatrix($(shared_slot_syms[k]), Val($D32))))
    end

    for s in 1:(planner.num_batched_slots)
        view_sym = Symbol("M", s)
        push!(
            stmts,
            :(
                $view_sym = DualAccessMatrix(
                    $(batched_slot_syms[s]), Val($D32), warp_matrix_id, Val(:small)
                )
            ),
        )
    end

    node_view_sym = Dict{Int,Symbol}()
    for id in shared_input_ids
        node_view_sym[id] = shared_view_syms[id]
    end
    slot_view_sym_by_slot(s) = Symbol("M", s)

    loaded = Set{Int}()
    function maybe_load_batched_input!(node_id::Int)
        node_id in loaded && return nothing
        node = tape.nodes[node_id]
        meta = tape.metas[node_id]
        (node isa InputNode && meta.lifecycle == BATCHED) || return nothing
        slot = planner.slots[node_id]
        view = slot_view_sym_by_slot(slot)
        global_in = input_sym[node_id]
        push!(
            stmts,
            :(intermediate_layout_load!(
                shmem_load, $global_in, Val($D32), Val($nthreads32), N, Val(:small)
            )),
        )
        push!(
            stmts,
            :(interm_to_dual_transfer!(
                $(batched_slot_syms[slot]),
                shmem_load,
                Val($D32),
                Val($nthreads32),
                N,
                Val(:small),
            )),
        )
        node_view_sym[node_id] = view
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

        dest_slot = planner.slots[i]
        dest_view = slot_view_sym_by_slot(dest_slot)
        node_view_sym[i] = dest_view

        emit_expr = emit_primitive(node.fn, dest_view, arg_exprs, arg_types)
        push!(stmts, Expr(:if, :active, Expr(:block, emit_expr)))
    end

    for (k, leaf) in enumerate(leaves)
        out_sym = out_syms[k]
        leaf_shmem = batched_slot_syms[leaf.slot]
        push!(
            stmts,
            :(dual_to_interm_transfer!(
                shmem_load, $leaf_shmem, Val($D32), Val($nthreads32), N, Val(:small)
            )),
        )
        push!(
            stmts,
            :(intermediate_layout_write!(
                $out_sym,
                shmem_load,
                Val($D32),
                Val($nthreads32),
                N,
                Val(:small),
                Val(:indep),
            )),
        )
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
