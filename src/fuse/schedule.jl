# =============================================================================
# Slot-minimising schedule search
# =============================================================================
#
# Given a tape, choose a topological ordering of its schedulable nodes that
# minimises the peak number of simultaneously-live shared-memory slots, with
# auto in-place reuse folded into the cost (per A10/B1 of the merge plan).
#
# Objective: lex (peak_M, peak_V) — matrix slots dominate shmem since each is
# `D_MAX^2 * n_mats_per_block` bytes vs `D_MAX * n_mats_per_block` for vectors.
#
# Schedulable nodes:
#   - every CallNode (compute primitives, `_load_to_single` / `_single_to_dual`
#     / `_dual_to_single` layout transitions, `_alloc_vec`)
#   - every batched-vector InputNode (its scheduled position is the lazy
#     vector_load!)
#
# Non-schedulable nodes (NewNode, ConstNode, batched matrix InputNodes, shared
# InputNodes) generate no code at their tape position and so don't participate
# in the search. NewNode wrappers are collapsed when building the reduced DAG —
# a CallNode that consumes a wrapped value depends on the wrapped value's
# underlying slot-backed producer.
#
# Algorithm:
#   1. Identify schedulable nodes, index them 1..k.
#   2. Build the reduced DAG: edges u → v iff u is reachable from v.args
#      through NewNode chains, and u is itself schedulable.
#   3. Mark output leaves (schedulable nodes reached as terminals of
#      tape.output through NewNode chains) — these stay live until kernel end.
#   4. Compute LB1: per-CallNode local bottleneck, separately for :M and :V.
#   5. Compute UB by evaluating the natural tape order.
#   6. Subset DP over UInt64 bitsets of scheduled nodes. State value is the
#      lex-min (peak_M, peak_V) achievable to reach that state; transition
#      to S∪{v} for each ready v (preds(v) ⊆ S) with cost = (peak during
#      v's compute, accounting for alive carries and auto/forced in-place).
#      Prune any state whose value ≥ current UB. Stop expanding a state
#      that has reached the LB (further moves can only tie or worsen).
#   7. Reconstruct the schedule from the argmin terminal.
#
# Returned: a Vector{Int} permutation of `1:length(tape.nodes)`. Non-
# schedulable nodes keep their natural relative position interleaved between
# schedulable nodes — they emit nothing in plan/codegen so their position is
# cosmetic.

const _PoolPeak = NamedTuple{(:M, :V),Tuple{Int,Int}}

_lex_lt(a::_PoolPeak, b::_PoolPeak) = a.M < b.M || (a.M == b.M && a.V < b.V)
_lex_max(a::_PoolPeak, b::_PoolPeak) = (M = max(a.M, b.M), V = max(a.V, b.V))

# -----------------------------------------------------------------------------
# Schedulable-node identification and pool assignment
# -----------------------------------------------------------------------------

# Pool that a node's slot draws from, or :none if it isn't slot-backed.
# Mirrors the rules in plan.jl: BATCHED matrix CallNodes / TransferNodes /
# LoadNodes get :M; BATCHED vector CallNodes and InputNodes get :V; BATCHED
# scalar CallNodes get :none (register-resident, :Sout staging is post-hoc).
function _sched_pool(tape::Tape, id::Int)
    node = tape.nodes[id]
    meta = tape.metas[id]
    (node isa CallNode || node isa InputNode) || return :none
    meta.lifecycle == BATCHED || return :none
    meta.type <: TraceScalar && return :none
    if node isa InputNode && meta.type <: TraceMatrix
        return :none  # loaded via _load_to_single CallNode
    end
    return meta.type <: TraceMatrix ? :M : :V
end

function _is_schedulable(tape::Tape, id::Int)
    node = tape.nodes[id]
    if node isa CallNode
        return true
    elseif node isa InputNode
        meta = tape.metas[id]
        return meta.lifecycle == BATCHED && meta.type <: TraceVector
    end
    return false
end

function _collect_schedulable(tape::Tape)
    ids = Int[]
    for i in eachindex(tape.nodes)
        _is_schedulable(tape, i) && push!(ids, i)
    end
    return ids
end

# -----------------------------------------------------------------------------
# Reduced DAG: edges among schedulable nodes only, NewNodes collapsed.
# -----------------------------------------------------------------------------

# Walk `ref` collecting schedulable producer tape ids. NewNodes are
# transparent (recurse into fields); ConstNodes / non-schedulable InputNodes
# are skipped.
function _collect_underlying!(out::Set{Int}, tape::Tape, ref::NodeRef)
    n = tape.nodes[ref.id]
    if n isa NewNode
        for (_, child) in n.fields
            _collect_underlying!(out, tape, child)
        end
    elseif n isa ConstNode
        return out
    else
        if _is_schedulable(tape, ref.id)
            push!(out, ref.id)
        end
    end
    return out
end

# For each schedulable node, the set of schedulable nodes that consume it
# (transitively through NewNodes).
function _reduced_dag(tape::Tape, schedulable::Vector{Int})
    n = length(schedulable)
    bit_of = Dict{Int,Int}()
    for (i, id) in enumerate(schedulable)
        bit_of[id] = i
    end

    preds = [UInt64(0) for _ in 1:n]      # preds[i] = bitmask of i's deps
    consumers = [UInt64(0) for _ in 1:n]  # consumers[i] = bitmask of users

    # Direct args (only relevant for CallNodes — vector InputNodes have none).
    work = Set{Int}()
    for (i, id) in enumerate(schedulable)
        node = tape.nodes[id]
        node isa CallNode || continue
        empty!(work)
        for ref in node.args
            _collect_underlying!(work, tape, ref)
        end
        for u in work
            u == id && continue  # defensive
            haskey(bit_of, u) || continue
            j = bit_of[u]
            preds[i] |= UInt64(1) << (j - 1)
            consumers[j] |= UInt64(1) << (i - 1)
        end
    end

    return (; bit_of, preds, consumers)
end

# Schedulable nodes that are reached as leaves of `tape.output` through
# NewNode chains. These stay live until kernel end (the output write reads
# from their slot after all compute is done).
function _output_leaf_mask(tape::Tape, bit_of::Dict{Int,Int})
    mask = UInt64(0)
    leaves = Set{Int}()
    if tape.output !== nothing
        _collect_underlying!(leaves, tape, tape.output)
    end
    for id in leaves
        haskey(bit_of, id) || continue
        mask |= UInt64(1) << (bit_of[id] - 1)
    end
    return mask
end

# -----------------------------------------------------------------------------
# In-place / aliasing analysis (per node, schedule-independent)
# -----------------------------------------------------------------------------

# For a schedulable CallNode i, identify whether it can alias its destination
# slot onto one of its direct args (in the reduced DAG), and which pool the
# alias is in. Two flavours:
#   - `forced` (inplace_arg): mandatory alias to a specific arg's underlying
#     slot. Currently `cholesky!`, `ldiv!`. Returns the underlying schedulable
#     id (NewNode-walked) or `nothing` if not registered.
#   - `auto` (inplace_safe_args): a tuple of arg positions where the
#     sub-kernel is safe to write-over the input *if* that input is dead at
#     this op. Mirrors `_maybe_auto_inplace_slot` in plan.jl: the direct
#     `node.args[k]` ref must be a CallNode or InputNode (not a NewNode
#     wrapper), and must be slot-backed in the same pool as the destination.
#
# We precompute the *candidate* alias underlying ids for each node; the DP
# checks at evaluation time whether the candidate is "dead at i" given the
# current state, and whether the pool matches.
struct _InplaceInfo
    forced_owner::Union{Nothing,Int}      # underlying schedulable id, or nothing
    auto_candidates::Vector{Int}          # schedulable ids of direct, slot-backed safe args
end

function _inplace_info(tape::Tape, i::Int, bit_of::Dict{Int,Int})
    node = tape.nodes[i]
    node isa CallNode || return _InplaceInfo(nothing, Int[])
    arg_types = Tuple(meta_at(tape, r).type for r in node.args)

    forced_owner = nothing
    if applicable(inplace_arg, node.fn, arg_types...)
        k = inplace_arg(node.fn, arg_types...)
        target_ref = node.args[k]
        owner = resolve_slot_owner(tape, target_ref)
        haskey(bit_of, owner) && (forced_owner = owner)
    end

    auto_candidates = Int[]
    if applicable(inplace_safe_args, node.fn, arg_types...)
        safe_positions = inplace_safe_args(node.fn, arg_types...)
        for k in safe_positions
            1 <= k <= length(node.args) || continue
            ref = node.args[k]
            arg_node = tape.nodes[ref.id]
            # Only direct, slot-backed CallNode/InputNode args qualify —
            # NewNode-wrapped args are rejected because the wrapper remaps
            # reads and aliasing through it would race with the in-place
            # write. (Same rule as plan.jl::_maybe_auto_inplace_slot.)
            (arg_node isa CallNode || arg_node isa InputNode) || continue
            haskey(bit_of, ref.id) || continue
            push!(auto_candidates, ref.id)
        end
    end
    return _InplaceInfo(forced_owner, auto_candidates)
end

# -----------------------------------------------------------------------------
# Live-count helpers (used by both LB1 evaluator and the DP cost function)
# -----------------------------------------------------------------------------

# Bitcount per pool: bit i set in `mask` contributes 1 to pool[pools[i]].
@inline function _popcount_by_pool(mask::UInt64, pool_M::UInt64, pool_V::UInt64)
    return (M = count_ones(mask & pool_M), V = count_ones(mask & pool_V))
end

# -----------------------------------------------------------------------------
# Per-candidate cost evaluation: peak during transition S → S∪{v}
# -----------------------------------------------------------------------------

# Returns:
#   (during_peak_M, during_peak_V): slot occupancy by pool during v's compute.
# `i` is v's bit index (1-based). `S` is the bitset of scheduled-before-v.
#
# Rules (matching plan.jl, with NewNode-wrapped args excluded from auto
# in-place per the comment in `_maybe_auto_inplace_slot`):
#   - alive_before = { u ∈ S : (consumers[u] & ~S) != 0 OR u ∈ output_leaves }
#       Note v ∉ S, so v itself is a consumer of u for u ∈ preds[v] — that
#       keeps preds(v) ⊆ alive_before by construction.
#   - v's destination slot:
#       * forced inplace_arg → alias to forced_owner's slot (same pool).
#       * else auto in-place: among auto_candidates, find one in S, same
#         pool as v, that has no consumer outside S∪{v} (i.e., would die
#         at v) AND is not an output_leaf.
#       * else fresh slot in v's pool.
#   - during_peak[p] = |alive_before ∩ pool p| + (pool(v)=p AND v fresh ? 1 : 0)
function _during_peak(
    i::Int,
    S::UInt64,
    consumers::Vector{UInt64},
    output_leaves::UInt64,
    pool_v::Symbol,
    pool_M::UInt64,
    pool_V::UInt64,
    inplace::_InplaceInfo,
    bit_of::Dict{Int,Int},
)
    v_bit = UInt64(1) << (i - 1)
    S_plus_v = S | v_bit
    not_S = ~S

    # alive_before: bits in S whose consumers reach outside S, plus output leaves.
    alive_before = UInt64(0)
    # Iterate set bits of S
    s = S
    while s != 0
        u_bit = s & (-s)
        s ⊻= u_bit
        u_idx = trailing_zeros(u_bit) + 1
        if (consumers[u_idx] & not_S) != 0 || (u_bit & output_leaves) != 0
            alive_before |= u_bit
        end
    end
    counts_before = _popcount_by_pool(alive_before, pool_M, pool_V)

    # Determine whether v allocates a fresh slot in its pool.
    fresh = true
    if pool_v !== :none
        if inplace.forced_owner !== nothing
            fresh = false
        else
            for cand_id in inplace.auto_candidates
                haskey(bit_of, cand_id) || continue
                j = bit_of[cand_id]
                cand_bit = UInt64(1) << (j - 1)
                (cand_bit & S) != 0 || continue
                # Same pool as v?
                if pool_v === :M
                    (cand_bit & pool_M) != 0 || continue
                else
                    (cand_bit & pool_V) != 0 || continue
                end
                # Output leaves can't be overwritten.
                (cand_bit & output_leaves) != 0 && continue
                # Dies at v iff no consumer outside S∪{v}.
                if (consumers[j] & ~S_plus_v) == 0
                    fresh = false
                    break
                end
            end
        end
    end

    add_M = (pool_v === :M && fresh) ? 1 : 0
    add_V = (pool_v === :V && fresh) ? 1 : 0
    return (M = counts_before.M + add_M, V = counts_before.V + add_V)
end

# -----------------------------------------------------------------------------
# LB1: per-node local bottleneck (cheap, sound)
# -----------------------------------------------------------------------------

# For each schedulable node v, peak ≥ |distinct pool-p inputs of v| +
# (v in pool p AND v can't alias onto any pool-p arg ? 1 : 0).
# The +1 for v itself reflects that v's slot must exist during its compute;
# alias eligibility here is "any safe arg in pool p" (optimistic — we ignore
# the death-at-v requirement, since LB just needs to be sound).
function _compute_lb1(
    schedulable::Vector{Int},
    pool_M::UInt64,
    pool_V::UInt64,
    pool_of::Vector{Symbol},
    preds::Vector{UInt64},
    inplace::Vector{_InplaceInfo},
    bit_of::Dict{Int,Int},
)
    lb_M = 0
    lb_V = 0
    for i in eachindex(schedulable)
        # Pool-restricted pred counts.
        pred_mask = preds[i]
        pM = count_ones(pred_mask & pool_M)
        pV = count_ones(pred_mask & pool_V)

        # v adds 1 to its own pool unless it can alias.
        addM = 0
        addV = 0
        if pool_of[i] !== :none
            can_alias = false
            if inplace[i].forced_owner !== nothing
                can_alias = true
            else
                # Any safe-arg candidate in the same pool as v?
                for cand_id in inplace[i].auto_candidates
                    haskey(bit_of, cand_id) || continue
                    j = bit_of[cand_id]
                    cand_bit = UInt64(1) << (j - 1)
                    if pool_of[i] === :M && (cand_bit & pool_M) != 0
                        can_alias = true
                        break
                    elseif pool_of[i] === :V && (cand_bit & pool_V) != 0
                        can_alias = true
                        break
                    end
                end
            end
            if !can_alias
                pool_of[i] === :M ? (addM = 1) : (addV = 1)
            end
        end

        lb_M = max(lb_M, pM + addM)
        lb_V = max(lb_V, pV + addV)
    end
    return (M = lb_M, V = lb_V)
end

# -----------------------------------------------------------------------------
# Greedy schedule (the trace's natural order) to get an initial UB.
# -----------------------------------------------------------------------------

function _evaluate_order(
    order::Vector{Int},
    bit_of::Dict{Int,Int},
    consumers::Vector{UInt64},
    output_leaves::UInt64,
    pool_of::Vector{Symbol},
    pool_M::UInt64,
    pool_V::UInt64,
    inplace::Vector{_InplaceInfo},
)
    S = UInt64(0)
    peak = (M = 0, V = 0)
    for tape_id in order
        haskey(bit_of, tape_id) || continue
        i = bit_of[tape_id]
        d = _during_peak(
            i, S, consumers, output_leaves,
            pool_of[i], pool_M, pool_V, inplace[i], bit_of,
        )
        peak = _lex_max(peak, d)
        S |= UInt64(1) << (i - 1)
    end
    return peak
end

# -----------------------------------------------------------------------------
# Subset DP
# -----------------------------------------------------------------------------

# Hard cap on schedulable-node count so we fail loudly rather than hang.
# k = 60 leaves 4 spare bits in UInt64 and gives 2^60 absolute upper bound on
# state space; in practice the reachable-set count is much smaller, but a
# real DAG with 60 schedulable nodes would also blow up the reachable count
# beyond what's tractable. Raise this and switch to BitSet-keyed states if
# this is ever insufficient.
const _SCHEDULE_MAX_NODES = 60

function schedule(tape::Tape)::Vector{Int}
    schedulable = _collect_schedulable(tape)
    k = length(schedulable)
    if k == 0
        return collect(1:length(tape.nodes))
    end
    k <= _SCHEDULE_MAX_NODES || error(
        "schedule: $k schedulable nodes exceeds the bitset cap of $_SCHEDULE_MAX_NODES; raise the cap or use a fallback heuristic.",
    )

    dag = _reduced_dag(tape, schedulable)
    bit_of = dag.bit_of
    preds = dag.preds
    consumers = dag.consumers
    output_leaves = _output_leaf_mask(tape, bit_of)

    pool_of = [_sched_pool(tape, id) for id in schedulable]
    pool_M = UInt64(0)
    pool_V = UInt64(0)
    for (i, p) in enumerate(pool_of)
        bit = UInt64(1) << (i - 1)
        if p === :M
            pool_M |= bit
        elseif p === :V
            pool_V |= bit
        end
    end

    inplace = [_inplace_info(tape, schedulable[i], bit_of) for i in 1:k]

    lb = _compute_lb1(schedulable, pool_M, pool_V, pool_of, preds, inplace, bit_of)

    # Initial UB from the natural order; that's a valid schedule by construction.
    ub = _evaluate_order(
        schedulable, bit_of, consumers, output_leaves,
        pool_of, pool_M, pool_V, inplace,
    )

    full = k == 64 ? typemax(UInt64) : (UInt64(1) << k) - UInt64(1)

    # f[S] = best (peak_M, peak_V) lex to reach S; parent[S] = (prev_S, v_bit_idx)
    f = Dict{UInt64,_PoolPeak}()
    parent = Dict{UInt64,Tuple{UInt64,Int}}()
    f[UInt64(0)] = (M = 0, V = 0)

    # BFS by |S|: enumerate states in increasing popcount order so each state's
    # value is finalised before we transition out of it.
    levels = [Vector{UInt64}() for _ in 0:k]
    push!(levels[1], UInt64(0))

    best_terminal::_PoolPeak = ub

    # If the initial UB already matches LB, the natural order is provably
    # optimal — skip the DP and return it directly.
    if !_lex_lt(lb, best_terminal)
        return _interleave_non_schedulable(tape, schedulable)
    end

    for sz in 0:(k - 1)
        states = levels[sz + 1]
        for S in states
            f_S = get(f, S, nothing)
            f_S === nothing && continue
            # Prune: skip only when this state's peak strictly exceeds the
            # best known terminal — ties must propagate, otherwise we discard
            # the very path the UB came from.
            _lex_lt(best_terminal, f_S) && continue

            # Enumerate ready v: preds(v) ⊆ S, v ∉ S.
            not_S = ~S
            ready_mask = UInt64(0)
            remaining = not_S & full
            r = remaining
            while r != 0
                v_bit = r & (-r)
                r ⊻= v_bit
                i = trailing_zeros(v_bit) + 1
                if (preds[i] & not_S) == 0
                    ready_mask |= v_bit
                end
            end

            rm = ready_mask
            while rm != 0
                v_bit = rm & (-rm)
                rm ⊻= v_bit
                i = trailing_zeros(v_bit) + 1
                d = _during_peak(
                    i, S, consumers, output_leaves,
                    pool_of[i], pool_M, pool_V, inplace[i], bit_of,
                )
                cand = _lex_max(f_S, d)
                # Branch-and-bound: skip only if strictly worse than UB.
                # Ties must be admitted so the path attaining the UB survives.
                _lex_lt(best_terminal, cand) && continue
                S_new = S | v_bit
                prev = get(f, S_new, nothing)
                if prev === nothing || _lex_lt(cand, prev)
                    f[S_new] = cand
                    parent[S_new] = (S, i)
                    if S_new == full
                        best_terminal = cand
                        # LB termination: any terminal that hits LB is
                        # provably optimal — no further exploration needed.
                        if !_lex_lt(lb, best_terminal)
                            @goto done
                        end
                    end
                    push!(levels[sz + 2], S_new)
                end
            end
        end
    end
    @label done

    haskey(f, full) || error("schedule: DP failed to reach the full state — DAG cycle?")

    # Reconstruct: walk parents from `full` back to 0.
    order_bits = Int[]  # bit indices in execution order (will reverse)
    cur = full
    while cur != UInt64(0)
        (prev, i) = parent[cur]
        push!(order_bits, i)
        cur = prev
    end
    reverse!(order_bits)
    order = [schedulable[i] for i in order_bits]

    return _interleave_non_schedulable(tape, order)
end

# Splice non-schedulable nodes into the reordered schedulable sequence.
# Strategy: walk tape.nodes in natural order, accumulating non-schedulable
# nodes; whenever we hit a schedulable node, emit *all* pending non-
# schedulable nodes followed by that schedulable node — *but* indexed by the
# new schedule, not the natural one. Concretely: produce a permutation that
# places non-schedulable nodes immediately *before* the schedulable node they
# originally preceded in natural order. (Their position is cosmetic — plan
# and codegen skip them in their main per-node loop.)
function _interleave_non_schedulable(tape::Tape, sched_order::Vector{Int})
    n = length(tape.nodes)
    sched_set = Set(sched_order)
    # Map each schedulable node to its index in sched_order.
    sched_pos = Dict{Int,Int}()
    for (i, id) in enumerate(sched_order)
        sched_pos[id] = i
    end

    # For each non-schedulable node, find the first schedulable node that
    # follows it in natural order. We then attach the non-schedulable node
    # to that schedulable node's new position.
    attached = [Int[] for _ in 1:length(sched_order)]
    trailing = Int[]  # non-schedulable nodes after the last natural schedulable
    pending = Int[]
    # Walk natural order, accumulating non-schedulable nodes; on hitting a
    # schedulable, flush the buffer to that schedulable's new position.
    for id in 1:n
        if id in sched_set
            target = sched_pos[id]
            append!(attached[target], pending)
            empty!(pending)
        else
            push!(pending, id)
        end
    end
    # Any leftover non-schedulable nodes after the last natural schedulable
    # go to the end.
    append!(trailing, pending)

    out = Int[]
    sizehint!(out, n)
    for (i, sched_id) in enumerate(sched_order)
        append!(out, attached[i])
        push!(out, sched_id)
    end
    append!(out, trailing)
    @assert length(out) == n
    return out
end
