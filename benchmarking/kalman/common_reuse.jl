# Benchmark-only transformation: reuse block-common input buffers when their
# actual generated consumer intervals are disjoint. Production codegen is unchanged.
function common_reuse_entry(tape, planner, baseline)
    leaves = BK.flatten_leaves(baseline.output_spec)
    expr, sig = BK.codegen(tape, planner, leaves;
        D_MAX=baseline.D_MAX, nthreads=baseline.nthreads, T=baseline.T,
        fn_name=gensym(:common_reuse), shared_memory=:dynamic)
    body = expr.args[2].args
    contains(x, sym) = x == sym || (x isa Expr && any(a -> contains(a, sym), x.args))
    function dynamic_call(x)
        x isa Expr || return nothing
        x.head == :call && x.args[1] == :CuDynamicSharedArray && return x
        for a in x.args
            result = dynamic_call(a)
            result === nothing || return result
        end
        return nothing
    end
    regions = []
    for (index, stmt) in enumerate(body)
        stmt isa Expr && stmt.head == :(=) || continue
        name = stmt.args[1]
        name isa Symbol && startswith(string(name), "shmem_SM") || continue
        call = dynamic_call(stmt.args[2])
        call === nothing && error("Common reuse requires dynamic shared allocations")
        slot = parse(Int, replace(string(name), "shmem_SM" => ""))
        view = Symbol("SM",slot)
        load = findall(i -> body[i] isa Expr && body[i].head == :if &&
            contains(body[i], :shared_matrix_load!) && contains(body[i], name), eachindex(body))
        length(load) == 1 || error("Expected one common-input load")
        uses = findall(i -> body[i] isa Expr && body[i].head == :if &&
            body[i].args[1] == :active && contains(body[i], view), eachindex(body))
        isempty(uses) && error("Common input must have active-guarded compute consumers")
        # Reject other consumers, including direct common outputs, rather than
        # silently missing their lifetime in this deliberately narrow experiment.
        for i in eachindex(body)
            contains(body[i],view) || continue
            i in uses && continue
            body[i] isa Expr && body[i].head == :(=) && body[i].args[1] == view && continue
            error("Unsupported common-input consumer")
        end
        elems = call.args[3].args[1]
        alignment = max(32,Base.datatype_alignment(baseline.T))
        bytes = cld(sizeof(baseline.T)*elems,alignment)*alignment
        push!(regions, (; index, load=only(load), call, first=first(uses), last=last(uses), bytes))
    end
    length(regions) == 3 || error("Experiment expects three block-common matrices")
    sort!(regions; by=r->r.first)
    all(regions[i].last < regions[i+1].first for i in 1:length(regions)-1) ||
        error("Common input lifetimes overlap")
    offsets = sort([r.call.args[4] for r in regions])
    start = first(offsets)
    # Common allocations must form the final contiguous region of the arena.
    total = sum(r.bytes for r in regions)
    start + total == sig.dynamic_shared_bytes || error("Common buffers are not the arena tail")
    by_offset = sort(regions; by=r->r.call.args[4])
    all(by_offset[i].call.args[4]+by_offset[i].bytes == by_offset[i+1].call.args[4]
        for i in 1:length(by_offset)-1) || error("Common arena is not contiguous")
    for r in regions
        r.call.args[4] = start
    end
    rewritten = Any[]
    removed = Set(r.load for r in regions)
    for (i,stmt) in enumerate(body)
        i in removed && continue
        for r in regions
            i == r.first || continue
            # Every lane participates, including inactive tail matrices. The
            # first fence protects readers in other warps before buffer overwrite.
            push!(rewritten, :(sync_threads()),body[r.load],:(sync_threads()))
        end
        push!(rewritten,stmt)
    end
    expr.args[2] = Expr(:block,rewritten...)
    bytes = start + maximum(r.bytes for r in regions)
    newsig = BK.KernelSignature(sig.fn_name,sig.n_outputs,sig.n_batched_inputs,
        sig.n_shared_inputs,bytes)
    println("COMMON_REUSE,threads=$(baseline.nthreads),before=$(sig.dynamic_shared_bytes),after=$bytes")
    fn = Core.eval(BK, expr)
    return BK.CompiledKernel(fn,newsig,baseline.D_MAX,baseline.nthreads,baseline.T,
        baseline.output_spec,Base.get_world_counter())
end
