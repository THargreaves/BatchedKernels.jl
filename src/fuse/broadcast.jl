# =============================================================================
# Broadcast surface
# =============================================================================
#
# The user-facing entry point is `f.(args...)` where `args` are batched
# containers. `Base.copy(::Broadcasted{BatchedStyle})` traces, plans, compiles,
# caches, and launches; the kernel cache key is `(f, input_cache_key(specs))`.
#
# Type stability is obtained via `Core.Compiler.return_type` on the scalar
# function with trace-time argument types, mapped to the runtime output type
# via `batchify_type` and asserted at the call site (`result::BR`). This
# constant-folds when `F` is a singleton function and inputs are concrete; the
# surface explicitly rejects closures and callable structs.

struct BatchedStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:BatchedCuMatrix}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:BatchedCuVector}) = BatchedStyle()
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
    (TT <: TraceMatrix || TT <: TraceVector) || return nothing
    leaf_max = maximum(shape(TT))
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
    batched_args::Vector, shared_args::Vector, N_ref::Ref, x::BatchedCuVector
)
    _set_or_check_batch_n!(N_ref, batch_size(x))
    push!(batched_args, x.data)
    return nothing
end
function _collect_runtime_inputs!(
    batched_args::Vector, shared_args::Vector, N_ref::Ref, x::SharedCuMatrix
)
    N_ref[] !== nothing && _set_or_check_batch_n!(N_ref, batch_size(x))
    push!(shared_args, x.data)
    return nothing
end
function _collect_runtime_inputs!(
    batched_args::Vector, shared_args::Vector, N_ref::Ref, x::SharedCuVector
)
    N_ref[] !== nothing && _set_or_check_batch_n!(N_ref, batch_size(x))
    push!(shared_args, x.data)
    return nothing
end
function _collect_runtime_inputs!(::Vector, ::Vector, N_ref::Ref, x::SharedValue)
    N_ref[] !== nothing && _set_or_check_batch_n!(N_ref, batch_size(x))
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

function _ensure_compiled!(f, args::Tuple)
    input_specs = InputSpec[input_spec(arg) for arg in args]
    input_types = Type[input_trace_type(spec) for spec in input_specs]
    key = (f, Tuple(input_cache_key(spec) for spec in input_specs))

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
    nthreads = 256

    tape = trace(f, input_specs)
    planner = plan_memory(tape)
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
    )
    compiled_fn = Core.eval(@__MODULE__, fn_expr)
    entry = CompiledKernel(
        compiled_fn, sig, D_MAX, nthreads, T, output_spec, Base.get_world_counter()
    )
    KERNEL_CACHE[key] = entry
    return entry
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

function _broadcast_impl(bc::Broadcasted{BatchedStyle})
    f = bc.f
    args = bc.args

    entry = _ensure_compiled!(f, args)
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
            length(s) == 2 ? CuArray{T}(undef, s[1], s[2], N) : CuArray{T}(undef, s[1], N)
        end for leaf in leaves
    ]

    nblocks = cld(N, (nthreads ÷ 32) * (32 ÷ D_MAX))
    Base.invokelatest() do
        @cuda threads = nthreads blocks = nblocks compiled_fn(
            leaf_arrays..., batched_args..., shared_args..., Int32(N)
        )
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
    return ndims(arr) == 3 ? BatchedCuMatrix(arr) : BatchedCuVector(arr)
end
_assemble_output(spec::LiteralOutput, _leaf_arrays, _idx, N::Int) =
    SharedValue(spec.val, N)
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
    base = Base.typename(T).wrapper
    new_params = Any[]
    for p in T.parameters
        replaced = false
        for f in fieldnames(T)
            if hasfield(T, f)
                if fieldtype(T, f) === p
                    push!(new_params, _component_element_type(components[f]))
                    replaced = true
                    break
                end
            end
        end
        replaced || push!(new_params, p)
    end
    return base{new_params...}
end

_component_element_type(c::BatchedCuMatrix{T,D1,D2}) where {T,D1,D2} = AbstractMatrix{T}
_component_element_type(c::BatchedCuVector{T,D}) where {T,D} = AbstractVector{T}
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
    base = Base.typename(TT).wrapper
    new_params = Any[]
    for p in TT.parameters
        replaced = false
        for (i, f) in enumerate(fnames)
            if fieldtype(TT, f) === p
                push!(new_params, sfs[i])
                replaced = true
                break
            end
        end
        replaced || push!(new_params, p)
    end
    result = base{new_params...}
    return :($result)
end

# Leaf batchify_type rules
batchify_type(::Type{T}) where {T<:Union{Number,AbstractChar,Bool,Nothing}} =
    SharedValue{T}
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
