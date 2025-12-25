export vmap

const InType = Union{Val{:batched}, Val{:shared}}

@inline function infer_from_input(x)
    nd = ndims(x)
    T = x isa Number ? typeof(x) : Base.eltype(x)
    if nd == 3
        D1, D2, N = size(x)
        D1 == D2 || error("Only square matrices supported")
        return D1, N, T
    elseif nd == 2
        D, N = size(x)
        if D == N  # Assume shared matrix
            return D, nothing, T
        else
            return D, N, T
        end
    elseif nd == 1
        D = length(x)
        return D, nothing, T
    else
        return nothing, nothing, T
    end
end

"Helper function to infer D and N from the inputs."
function infer_D_N(inputs...)
    D_inferred = nothing
    N_inferred = nothing
    T_inferred = nothing

    for x in inputs
        D, N, T = infer_from_input(x)

        if D !== nothing
            if D_inferred === nothing
                D_inferred = D
            elseif D != D_inferred
                error("Inconsistent shapes across inputs")
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
        dims == 3 && return Mat(sym)
        dims == 2 && return Vec(sym)
        dims == 1 && return Scal(sym)
        error("Each input must have 1 <= ndims <= 3")
    else
        dims <= 2 || error("Non-batched shared input must have ndims <= 2")
        dims == 2 && return SharedMat(sym)
        dims == 1 && return SharedVec(sym)
        return SharedScal(sym)
    end
end

get_output(::Type{<:MatKind}; T::Type, D::Int, N::Int) = CUDA.zeros(T, D, D, N)
get_output(::Type{<:VecKind}; T::Type, D::Int, N::Int) = CUDA.zeros(T, D, N)
get_output(::Type{<:SymKind}; T::Type, D::Int, N::Int) = error("Unsupported output kind")

"Cache key for kernels"
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

"JIT wrapper object (callable)"
mutable struct VMap{F}
    f::F
    fid::UInt
    in_type::Union{Nothing,Tuple{Vararg{InType}}}
    debug::Bool
    cache::Dict{KernelKey, Tuple{UInt64, Tuple}}  # Values are (pid, out_kinds)
end

"Returns a callable object that launches a fused batched kernel for `f`."
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
    return VMap(f, UInt(objectid(f)), in_type_tags, debug, Dict{KernelKey,Tuple{UInt64, Tuple}}())
end

"Entry point of the vmapped function."
function (g::VMap)(inputs...; threads::Int = 256)
    inTs = typeof(inputs)
    D, N, T = infer_D_N(inputs...)
    D <= 32 || error("D=$D unsupported, must be <= 32")
    threads % 32 == 0 || error("Number of threads must be a multiple of 32")
    key = KernelKey(g.fid, inTs, D, threads)
    nblocks = cld(N, (threads ÷ 32) * (32 ÷ D))

    (pid, out_kinds) = get!(g.cache, key) do 
        specs = ntuple(i -> get_spec(getfield(inputs, i), i, g.in_type), length(inputs))
        prog = trace(g.f, specs; T=T, D=D, nthreads=threads)

        out_kinds = Tuple(prog.kinds[vid] for vid in prog.outputs)
        pid = register_program!(prog, g.debug)

        return pid, out_kinds
    end

    outputs = Tuple(get_output(kind; T=T, D=D, N=N) for kind in out_kinds)

    Base.invokelatest() do
        @cuda threads=threads blocks=nblocks BatchedKernels._fused_kernel(Val(pid), outputs..., inputs..., Int32(N))
    end
    
    return length(outputs) == 1 ? outputs[1] : outputs
end