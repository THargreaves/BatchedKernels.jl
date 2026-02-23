export vmap

const InType = Union{Val{:batched}, Val{:shared}}

@inline function infer_from_input(input, tag::Union{Nothing,InType})
    dims = ndims(input)
    T = input isa Number ? typeof(input) : Base.eltype(input)
    tag = tag === nothing ? Val(:batched) : tag

    if tag === Val(:batched)
        if dims == 3
            D1, D2, N = size(input)
            return max(D1, D2), N, T
        elseif dims == 2
            D1, N = size(input)
            return D1, N, T
        elseif dims == 1
            N = length(input)
            return nothing, N, T
        else
            return nothing, nothing, T
        end
    else
        if dims == 2
            D1, D2 = size(input)
            return max(D1, D2), nothing, T
        elseif dims == 1
            D1 = length(input)
            return D1, nothing, T
        else
            return nothing, nothing, T
        end
    end
end

"Helper function to infer D and N from the inputs."
function infer_D_N(inputs...; in_type::Union{Nothing,Tuple{Vararg{InType}}})
    D_inferred = nothing
    N_inferred = nothing
    T_inferred = nothing

    for (i, x) in enumerate(inputs)
        tag = in_type !== nothing ? in_type[i] : nothing
        D, N, T = infer_from_input(x, tag)

        if D !== nothing
            if D_inferred === nothing
                D_inferred = D
            else
                D_inferred = max(D_inferred, D)
            end
        end
        if N !== nothing
            if N_inferred === nothing
                N_inferred = N
            elseif N != N_inferred
                error("Inconsistent batch sizes across inputs")
            end
        end
        if T_inferred === nothing
            T_inferred = T
        elseif T != T_inferred
            error("Inconsistent input types")
        end
    end

    D_inferred !== nothing || error("Could not infer D from inputs")
    N_inferred !== nothing || error("Could not infer N from inputs")

    return D_inferred::Int, N_inferred::Int, T_inferred::DataType
end

function get_spec(input, i::Int, in_type::Union{Nothing,Tuple{Vararg{InType}}})
    dims = ndims(input)
    sym = Symbol(:_param, i)
    tag = in_type === nothing ? Val(:batched) : in_type[i]

    if tag === Val(:batched)
        if dims == 3
            D1, D2, _ = size(input)
            return Mat(sym, D1, D2)
        elseif dims == 2
            D1, _ = size(input)
            return Vec(sym, D1)
        elseif dims == 1
            return Scal(sym)
        else
            error("Each input must have 1 <= ndims <= 3")
        end
    else
        if dims == 2
            D1, D2 = size(input)
            return SharedMat(sym, D1, D2)
        elseif dims == 1
            D1 = length(input)
            return SharedVec(sym, D1)
        elseif dims == 0
            return SharedScal(sym)
        else
            error("Non-batched shared input must have ndims <= 2")
        end
    end
end

get_output(::Type{<:MatKind}, shape::MatShape; T::Type, N::Int) = CUDA.zeros(T, shape.D1, shape.D2, N)
get_output(::Type{<:VecKind}, shape::VecShape; T::Type, N::Int) = CUDA.zeros(T, shape.D1, N)
get_output(::Type{<:SymKind}, ::ScalShape; T::Type, N::Int) = error("Unsupported output kind")

"Cache key for kernels."
struct KernelKey
    fid::UInt
    inTs::DataType
    D::Int
    threads::Int
end
@inline function Base.:(==)(a::KernelKey, b::KernelKey)
    a.fid == b.fid && a.inTs === b.inTs && a.D == b.D && a.threads == b.threads
end
@inline function Base.hash(k::KernelKey, h::UInt)
    hash(k.threads, hash(k.D, hash(k.inTs, hash(k.fid, h))))
end

"JIT wrapper object (callable)."
mutable struct VMap{F}
    f::F
    fid::UInt
    in_type::Union{Nothing,Tuple{Vararg{InType}}}
    debug::Bool
    cache::Dict{KernelKey,Tuple{UInt64,Tuple,Tuple}}  # Values are (pid, out_kinds, out_shapes)
end

"Returns a callable object that launches a fused batched kernel for f."
function vmap(
    f;
    in_type::Union{Nothing,Tuple{Vararg{Symbol}}} = nothing,
    debug::Bool = false,
)
    in_type_tags = in_type === nothing ? nothing :
        Tuple(map(in_type) do in_axis::Symbol
            in_axis === :batched && return Val(:batched)
            in_axis === :shared && return Val(:shared)
            error("in_type must either be nothing or entries must be :batched or :shared")
        end)
    return VMap(f, UInt(objectid(f)), in_type_tags, debug, Dict{KernelKey,Tuple{UInt64,Tuple,Tuple}}())
end

"Entry point of the vmapped function."
function (g::VMap)(inputs...; threads::Int = 256)
    inTs = typeof(inputs)
    D, N, T = infer_D_N(inputs...; in_type = g.in_type)
    D <= 32 || error("D=$D unsupported, must be <= 32")
    threads % 32 == 0 || error("Number of threads must be a multiple of 32")
    key = KernelKey(g.fid, inTs, D, threads)
    nblocks = cld(N, (threads ÷ 32) * (32 ÷ D))

    (pid, out_kinds, out_shapes) = get!(g.cache, key) do 
        specs = ntuple(i -> get_spec(getfield(inputs, i), i, g.in_type), length(inputs))
        prog = trace(g.f, specs; T=T, D=D, nthreads=threads)

        out_kinds = Tuple(prog.kinds[vid] for vid in prog.outputs)
        out_shapes = Tuple(prog.shapes[vid] for vid in prog.outputs)
        pid = register_program!(prog, g.debug)

        return pid, out_kinds, out_shapes
    end

    outputs = Tuple(get_output(kind, shape; T=T, N=N) for (kind, shape) in zip(out_kinds, out_shapes))

    Base.invokelatest() do
        @cuda threads=threads blocks=nblocks BatchedKernels._fused_kernel(Val(pid), outputs..., inputs..., Int32(N))
    end
    
    return length(outputs) == 1 ? outputs[1] : outputs
end