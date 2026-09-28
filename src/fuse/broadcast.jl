# =============================================================================
# Broadcast surface
# =============================================================================
#
# The user-facing entry point is `f.(args...)` where `args` are batched
# containers. `Base.copy(::Broadcasted{BatchedStyle})` traces, plans, compiles,
# caches, and launches; the cache key includes the function, input specialization,
# storage policy, launch geometry, shared allocation mode and CUDA device.
#
# Type stability is obtained via `Core.Compiler.return_type` on the scalar
# function with trace-time argument types, mapped to the runtime output type
# via `batchify_type` and asserted at the call site (`result::BR`). This
# constant-folds when `F` is a singleton function and inputs are concrete; the
# surface explicitly rejects closures and callable structs.

struct BatchedStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:BatchedCuMatrix}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:BatchedCuVector}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:BatchedCuScalar}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:SharedCuMatrix}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:SharedCuVector}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:SharedValue}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:BatchedStruct}) = BatchedStyle()
Base.BroadcastStyle(::BatchedStyle, ::Broadcast.DefaultArrayStyle{0}) = BatchedStyle()
Base.BroadcastStyle(::BatchedStyle, ::BatchedStyle) = BatchedStyle()

struct CompiledKernel
    fn::Any
    sig::KernelSignature
    D_MAX::Int
    nthreads::Int
    T::Type
    output_spec::OutputSpec
    world::UInt
end

const KERNEL_CACHE = Dict{Any,CompiledKernel}()

# `D_MAX` is the max over every per-input row/col extent across all leaf
# trace-matrix inputs. It drives slot allocation and the warp→matrix mapping.
# Per-operand dims live in each TraceMatrix's type parameters and are extracted
# at emit_primitive time via `shape`.
function _infer_D_MAX_T_from_specs!(D_ref::Ref, T_ref::Ref, specs)
    for spec in specs
        _infer_D_MAX_T_from_spec!(D_ref, T_ref, spec)
    end
    return D_ref[], T_ref[]
end

function _infer_D_MAX_T_from_spec!(D_ref::Ref, T_ref::Ref, spec::LeafInput)
    TT = spec.trace_type
    (TT <: Union{TraceMatrix,TraceVector,TraceScalar}) || return nothing
    leaf_max = TT <: TraceScalar ? 1 : maximum(shape(TT))
    t_param = eltype(TT)
    if D_ref[] === nothing
        D_ref[] = leaf_max
        T_ref[] = t_param
    else
        D_ref[] = max(D_ref[], leaf_max)
        T_ref[] == t_param || error("Inconsistent eltype across inputs")
    end
    return nothing
end
_infer_D_MAX_T_from_spec!(::Ref, ::Ref, ::LiteralInput) = nothing
function _infer_D_MAX_T_from_spec!(D_ref::Ref, T_ref::Ref, spec::CompositeInput)
    _infer_D_MAX_T_from_specs!(D_ref, T_ref, (child for (_, child) in spec.fields))
    return nothing
end

function _set_or_check_batch_n!(N_ref::Ref, n::Int)
    if N_ref[] === nothing
        N_ref[] = n
    else
        N_ref[] == n || error("Inconsistent batch size")
    end
    return nothing
end

function _collect_runtime_inputs!(
    batched_args::Vector, shared_args::Vector, N_ref::Ref, x::BatchedCuMatrix
)
    _set_or_check_batch_n!(N_ref, batch_size(x))
    push!(batched_args, x.data)
    return nothing
end
function _collect_runtime_inputs!(
    batched_args::Vector,
    shared_args::Vector,
    N_ref::Ref,
    x::Union{BatchedCuVector,BatchedCuScalar},
)
    _set_or_check_batch_n!(N_ref, batch_size(x))
    push!(batched_args, x.data)
    return nothing
end
function _collect_runtime_inputs!(
    batched_args::Vector, shared_args::Vector, N_ref::Ref, x::SharedCuMatrix
)
    _set_or_check_batch_n!(N_ref, batch_size(x))
    push!(shared_args, x.data)
    return nothing
end
function _collect_runtime_inputs!(
    batched_args::Vector, shared_args::Vector, N_ref::Ref, x::SharedCuVector
)
    _set_or_check_batch_n!(N_ref, batch_size(x))
    push!(shared_args, x.data)
    return nothing
end
function _collect_runtime_inputs!(::Vector, ::Vector, N_ref::Ref, x::SharedValue)
    _set_or_check_batch_n!(N_ref, batch_size(x))
    return nothing
end
function _collect_runtime_inputs!(
    batched_args::Vector, shared_args::Vector, N_ref::Ref, x::BatchedStruct
)
    for component in values(getfield(x, :components))
        _collect_runtime_inputs!(batched_args, shared_args, N_ref, component)
    end
    return nothing
end
function _collect_runtime_inputs!(::Vector, ::Vector, ::Ref, x)
    return error("Unsupported batched broadcast input component of type $(typeof(x))")
end

function _ensure_compiled!(
    f,
    args::Tuple;
    assignment=nothing,
    policy::Symbol=:auto,
    nthreads::Int=128,
    shared_memory::Symbol=:static,
)
    policy in (:auto, :legacy) || throw(ArgumentError("policy must be :auto or :legacy"))
    assignment !== nothing &&
        policy === :legacy &&
        throw(ArgumentError("explicit assignment cannot be combined with policy=:legacy"))
    shared_memory in (:static, :dynamic) ||
        throw(ArgumentError("shared_memory must be :static or :dynamic"))
    shared_memory === :dynamic &&
        assignment === nothing &&
        policy === :legacy &&
        throw(
            ArgumentError(
                "dynamic shared memory requires an automatic or explicit hybrid assignment"
            ),
        )
    32 <= nthreads <= 1024 && nthreads % 32 == 0 ||
        throw(ArgumentError("nthreads must be a multiple of 32 in 32:1024"))
    assignment === nothing ||
        nthreads == assignment.nthreads ||
        throw(ArgumentError("nthreads must match the forced assignment"))
    input_specs = InputSpec[input_spec(arg) for arg in args]
    input_types = Type[input_trace_type(spec) for spec in input_specs]
    cache_policy = if assignment === nothing
        (policy, nthreads, shared_memory, CUDA.device())
    else
        (:forced_hybrid, assignment_key(assignment), shared_memory, CUDA.device())
    end
    key = (f, Tuple(input_cache_key(spec) for spec in input_specs), cache_policy)

    if haskey(KERNEL_CACHE, key)
        cached = KERNEL_CACHE[key]
        ms = Base.methods(f, Tuple(input_types))
        if length(ms) == 1 && only(ms).primary_world <= cached.world
            return cached
        end
    end

    D_ref = Ref{Any}(nothing)
    T_ref = Ref{Any}(nothing)
    _infer_D_MAX_T_from_specs!(D_ref, T_ref, input_specs)
    D_MAX = D_ref[]
    T = T_ref[]
    D_MAX === nothing && error("Could not infer matrix dimension from inputs")
    T === nothing && error("Could not infer element type from inputs")
    1 <= D_MAX <= 32 || throw(ArgumentError("matrix group dimension must be in 1:32"))

    tape = trace(f, input_specs)
    # Geometry must cover intermediate extents as well as input extents.
    for meta in tape.metas
        if meta.type <: Union{TraceMatrix,TraceVector}
            D_MAX = max(D_MAX, maximum(shape(meta.type)))
        end
    end
    1 <= D_MAX <= 32 ||
        throw(ArgumentError("intermediate matrix group dimension must be in 1:32"))
    if assignment === nothing && policy === :auto
        assignment = automatic_assignment(tape; nthreads)
    end
    if assignment === nothing
        order = schedule(tape)
        planner = plan_memory(tape; order=order)
    else
        tape = hybrid_tape(tape)
        order = assignment.order
        planner = plan_memory(tape, assignment; D_MAX, T)
        limit = CUDA.attribute(
            CUDA.device(),
            if shared_memory === :dynamic
                CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
            else
                CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
            end,
        )
        planner.shared_bytes <= limit || throw(
            ArgumentError(
                "forced assignment needs $(planner.shared_bytes) shared bytes per block; device $shared_memory limit is $limit",
            ),
        )
    end
    output_spec = extract_output_spec(tape, planner)
    leaves = flatten_leaves(output_spec)
    fn_expr, sig = codegen(
        tape,
        planner,
        leaves;
        D_MAX=D_MAX,
        nthreads=nthreads,
        T=T,
        fn_name=gensym(:fused_kernel),
        order=order,
        shared_memory,
    )
    compiled_fn = Core.eval(@__MODULE__, fn_expr)
    entry = CompiledKernel(
        compiled_fn, sig, D_MAX, nthreads, T, output_spec, Base.get_world_counter()
    )
    KERNEL_CACHE[key] = entry
    return entry
end

# The function attribute is context-local and must be set before occupancy queries
# or launches above the default shared limit. Include any compiler static storage.
function _configure_dynamic_shared!(kernel, entry::CompiledKernel)
    bytes = entry.sig.dynamic_shared_bytes
    if bytes > 0
        limit = CUDA.attribute(
            CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        )
        total = bytes + Int(CUDA.memory(kernel).shared)
        total <= limit || throw(
            ArgumentError(
                "compiled kernel needs $total shared bytes; device opt-in limit is $limit",
            ),
        )
        attrs = CUDA.attributes(kernel.fun)
        if attrs[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] < bytes
            attrs[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = bytes
        end
    end
    return bytes
end

# `_elem_types(Args)` — compute the tuple-of-trace-element-types as a Type
# literal at compile time. Used by the broadcast surface's
# `Core.Compiler.return_type` call.
@generated function _elem_types(::Type{Args}) where {Args<:Tuple}
    types = Type[trace_element_type(t) for t in Args.parameters]
    Tup = Tuple{types...}
    return :($Tup)
end

# Type-stable entry point. `Core.Compiler.return_type` constant-folds at
# inference time when its inputs are concrete (which they are here — `F` is a
# singleton function type, and `_elem_types(Args)` is a constant Type). The
# `result::BR` annotation then fixes the inferred return type at the call site.
function Base.copy(bc::Broadcasted{BatchedStyle,A,F,Args}) where {A,F<:Function,Args<:Tuple}
    isdefined(F, :instance) || error(
        "Batched broadcast only supports singleton function objects; closures and callable structs are not supported in this prototype.",
    )
    elem_types = _elem_types(Args)
    R = Core.Compiler.return_type(F.instance, elem_types)
    BR = batchify_type(R)
    result = _broadcast_impl(bc)
    return result::BR
end

function Base.copy(::Broadcasted{BatchedStyle,A,F,Args}) where {A,F,Args<:Tuple}
    return error(
        "Batched broadcast only supports singleton function objects; closures and callable structs are not supported in this prototype.",
    )
end

export fuse, Assignment

"""
    fuse(f, args...; policy=:auto, assignment=nothing,
         nthreads=assignment === nothing ? 128 : assignment.nthreads, shared_memory=:static)

Execute the same scalar function and return the same inferred batched output type
as `f.(args...)`. The default `policy=:auto` uses deterministic register-first
planning with dual shared storage for conflicting orientations. `policy=:legacy`
selects the original scheduler/planner. A host `Assignment` overrides automatic
selection with validated storage and variant choices. Build assignments
against `trace(f, InputSpec[input_spec(x) for x in args])`; staging normalization
preserves node IDs. Dictionary/layout choices never enter the device kernel.

With automatic or explicit hybrid planning, `shared_memory=:dynamic` uses an aligned dynamic
shared arena and opts into the device per-block capacity. The default is `:static`.
Allocation mode is part of the compilation cache key.

Single, dual and register matrix storage are supported by the audited variants.
Forced mutation uses shared storage. A forced choice is rejected if unsupported,
without silently selecting another variant. Register placement does not guarantee
that the device compiler avoids spills; inspect compiled resources before tuning.
"""
function fuse(
    f::F,
    args::Vararg{Any,N};
    assignment::Union{Nothing,Assignment}=nothing,
    nthreads::Int=assignment === nothing ? 128 : assignment.nthreads,
    shared_memory::Symbol=:static,
    policy::Symbol=:auto,
) where {F<:Function,N}
    isdefined(F, :instance) || error("fuse only supports singleton function objects")
    elem_types = _elem_types(typeof(args))
    R = Core.Compiler.return_type(F.instance, elem_types)
    BR = batchify_type(R)
    bc = Broadcasted{BatchedStyle}(f, args)
    result = _broadcast_impl(bc; assignment, nthreads, shared_memory, policy)
    return result::BR
end

function _broadcast_impl(
    bc::Broadcasted{BatchedStyle};
    assignment=nothing,
    nthreads::Int=128,
    shared_memory::Symbol=:static,
    policy::Symbol=:auto,
)
    f = bc.f
    args = bc.args

    entry = _ensure_compiled!(f, args; assignment, nthreads, shared_memory, policy)
    compiled_fn = entry.fn
    D_MAX, nthreads, T = entry.D_MAX, entry.nthreads, entry.T

    N_ref = Ref{Union{Nothing,Int}}(nothing)
    batched_args = Any[]
    shared_args = Any[]
    for a in args
        _collect_runtime_inputs!(batched_args, shared_args, N_ref, a)
    end
    N = N_ref[]
    N === nothing && error("At least one batched matrix input required")

    leaves = flatten_leaves(entry.output_spec)
    # Outputs are fully written by the kernel (every in-bounds element gets a
    # store; the per-write `grid_mtrx_write <= N` / `grid_vec_store <= N`
    # guards only handle out-of-bounds tail lanes), so we skip the memset and
    # allocate uninitialised.
    leaf_arrays = [
        let s = shape(leaf.trace_type)
            if length(s) == 2
                CuArray{T}(undef, s[1], s[2], N)
            elseif length(s) == 1
                CuArray{T}(undef, s[1], N)
            else
                CuArray{T}(undef, N)  # scalar
            end
        end for leaf in leaves
    ]

    nblocks = cld(N, (nthreads ÷ 32) * (32 ÷ D_MAX))
    if N > 0
        Base.invokelatest() do
            launch_args = (leaf_arrays..., batched_args..., shared_args..., Int32(N))
            kernel = @cuda launch = false compiled_fn(launch_args...)
            shmem = _configure_dynamic_shared!(kernel, entry)
            kernel(launch_args...; threads=nthreads, blocks=nblocks, shmem)
        end
    end

    leaf_iter = Ref(0)
    return _assemble_output(entry.output_spec, leaf_arrays, leaf_iter, N)
end

# -----------------------------------------------------------------------------
# Output assembly: leaf_arrays + spec → BatchedCuMatrix / BatchedStruct tree
# -----------------------------------------------------------------------------

function _assemble_output(spec::LeafOutput, leaf_arrays::Vector, idx::Ref{Int}, N::Int)
    idx[] += 1
    arr = leaf_arrays[idx[]]
    if ndims(arr) == 3
        return BatchedCuMatrix(arr)
    elseif ndims(arr) == 2
        return BatchedCuVector(arr)
    else
        return BatchedCuScalar(arr)
    end
end
_assemble_output(spec::LiteralOutput, _leaf_arrays, _idx, N::Int) = SharedValue(spec.val, N)
function _assemble_output(spec::CompositeOutput, leaf_arrays::Vector, idx::Ref{Int}, N::Int)
    field_names = Symbol[p.first for p in spec.fields]
    field_vals = Any[_assemble_output(p.second, leaf_arrays, idx, N) for p in spec.fields]
    components = NamedTuple{Tuple(field_names)}(Tuple(field_vals))
    runtime_T = _runtime_composite_type(spec.T, components)
    C = typeof(components)
    return BatchedStruct{runtime_T,C}(components, N)
end

function _runtime_composite_type(::Type{T}, components::NamedTuple) where {T}
    if T <: Tuple
        elts = ntuple(i -> _component_element_type(components[i]), length(components))
        return Tuple{elts...}
    end
    replacements = Dict(
        name => _component_element_type(value) for (name, value) in pairs(components)
    )
    return _replace_composite_field_types(T, replacements)
end

_component_element_type(c::BatchedCuMatrix{T,D1,D2}) where {T,D1,D2} = AbstractMatrix{T}
_component_element_type(c::BatchedCuVector{T,D}) where {T,D} = AbstractVector{T}
_component_element_type(c::BatchedCuScalar{T}) where {T} = T
_component_element_type(c::BatchedStruct{T}) where {T} = T
_component_element_type(c::SharedValue{T}) where {T} = T
_component_element_type(c) = typeof(c)

# =============================================================================
# scalar_form and batchify_type — scalar→batched type maps
# =============================================================================
#
# `scalar_form(T)` — the per-batch-element scalar type. Mirrors the runtime
# `_component_element_type` so that BatchedStruct's `T` parameter computed at
# trace time matches the type the runtime actually assembles.
#
# `batchify_type(T)` — the user-visible Julia type of the batched output.
# Both are `@generated` for the composite cases so the resulting names/types
# fold at inference time (otherwise the NamedTuple names become free type
# variables and the `result::BR` assertion in Base.copy fires).

# Leaf scalar_form rules
scalar_form(::Type{T}) where {T<:Union{Number,AbstractChar,Bool,Nothing}} = T
scalar_form(::Type{TraceMatrix{T,D_M,D_N}}) where {T,D_M,D_N} = AbstractMatrix{T}
scalar_form(::Type{TraceVector{T,D_M}}) where {T,D_M} = AbstractVector{T}
scalar_form(::Type{TraceScalar{T}}) where {T} = T

@generated function scalar_form(::Type{TT}) where {TT<:Tuple}
    sfs = Type[scalar_form(p) for p in TT.parameters]
    result = Tuple{sfs...}
    return :($result)
end

# Generic composite: rebuild with each field replaced by its scalar_form.
@generated function scalar_form(::Type{TT}) where {TT}
    fnames = fieldnames(TT)
    isempty(fnames) && return :($TT)
    sfs = Type[scalar_form(fieldtype(TT, f)) for f in fnames]
    result = _replace_composite_field_types(TT, Dict(zip(fnames, sfs)))
    return :($result)
end

# Leaf batchify_type rules
# A failed scalar trace inference should report an unsupported graph rather
# than the ambiguous bottom-type dispatch among the output mapping methods.
function batchify_type(::Type{Union{}})
    throw(
        ArgumentError(
            "Scalar function has no supported return type for these traced inputs"
        ),
    )
end
batchify_type(::Type{T}) where {T<:Union{Number,AbstractChar,Bool,Nothing}} = SharedValue{T}
function batchify_type(::Type{TraceMatrix{T,D_M,D_N}}) where {T,D_M,D_N}
    return BatchedCuMatrix{
        T,D_M,D_N,CuArray{T,3,CUDA.DeviceMemory},CuArray{T,2,CUDA.DeviceMemory}
    }
end
function batchify_type(::Type{TraceVector{T,D_M}}) where {T,D_M}
    return BatchedCuVector{
        T,D_M,CuArray{T,2,CUDA.DeviceMemory},CuArray{T,1,CUDA.DeviceMemory}
    }
end
function batchify_type(::Type{TraceScalar{T}}) where {T}
    return BatchedCuScalar{T,CuArray{T,1,CUDA.DeviceMemory}}
end

@generated function batchify_type(::Type{TT}) where {TT<:Tuple}
    bts = Type[batchify_type(p) for p in TT.parameters]
    scalar_T = scalar_form(TT)
    names = Tuple(Symbol("_", i) for i in 1:length(TT.parameters))
    result = BatchedStruct{scalar_T,NamedTuple{names,Tuple{bts...}}}
    return :($result)
end

@generated function batchify_type(::Type{TT}) where {TT}
    fnames = fieldnames(TT)
    isempty(fnames) && return :($TT)
    bts = Type[batchify_type(fieldtype(TT, f)) for f in fnames]
    scalar_T = scalar_form(TT)
    nt_names = Tuple(fnames)
    result = BatchedStruct{scalar_T,NamedTuple{nt_names,Tuple{bts...}}}
    return :($result)
end
