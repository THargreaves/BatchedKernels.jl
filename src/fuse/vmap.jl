export vmap

function infer_from_input(x::BatchedCuMatrix{T,M}) where {T,M}
    D1 = size(x.data, 1)
    D2 = size(x.data, 2)
    D = max(D1, D2)
    N = size(x.data, 3)
    return D, N, T
end

function infer_from_input(x::BatchedCuVector{T,M}) where {T,M}
    D1 = size(x.data, 1)
    N = size(x.data, 2)
    return D1, N, T
end

function infer_from_input(x::SharedCuMatrix{T,M}) where {T,M}
    D1 = size(x.data, 1)
    D2 = size(x.data, 2)
    D = max(D1, D2)
    return D, nothing, T
end

function infer_from_input(x::SharedCuVector{T,M}) where {T,M}
    D1 = length(x.data)
    return D1, nothing, T
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

get_sym(i::Int) = Symbol(:_param, i)

function get_spec(x::BatchedCuMatrix{T,M}, i::Int) where {T,M}
    sym = get_sym(i)
    D1 = size(x.data, 1)
    D2 = size(x.data, 2)
    return Mat(sym, D1, D2)
end

function get_spec(x::BatchedCuVector{T,M}, i::Int) where {T,M}
    sym = get_sym(i)
    D1 = size(x.data, 1)
    return Vec(sym, D1)
end

function get_spec(x::SharedCuMatrix{T,M}, i::Int) where {T,M}
    sym = get_sym(i)
    D1 = size(x.data, 1)
    D2 = size(x.data, 2)
    return SharedMat(sym, D1, D2)
end

function get_spec(x::SharedCuVector{T,M}, i) where {T,M}
    sym = get_sym(i)
    D1 = length(x.data)
    return SharedVec(sym, D1)
end

get_output(::Type{<:MatKind}, shape::MatShape; T::Type, N::Int) = BatchedCuMatrix(CUDA.zeros(T, shape.D1, shape.D2, N))
get_output(::Type{<:VecKind}, shape::VecShape; T::Type, N::Int) = BatchedCuVector(CUDA.zeros(T, shape.D1, N))
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
    debug::Bool
    cache::Dict{KernelKey,Tuple{UInt64,Tuple,Tuple}}  # Values are (pid, out_kinds, out_shapes)
end

"Returns a callable object that launches a fused batched kernel for f."
function vmap(
    f;
    debug::Bool = false,
)
    return VMap(f, UInt(objectid(f)), debug, Dict{KernelKey,Tuple{UInt64,Tuple,Tuple}}())
end

"Entry point of the vmapped function."
function (g::VMap)(inputs...; threads::Int = 256)
    inTs = typeof(inputs)
    D, N, T = infer_D_N(inputs...)
    D <= 32 || error("D=$D unsupported, must be <= 32")
    threads % 32 == 0 || error("Number of threads must be a multiple of 32")
    key = KernelKey(g.fid, inTs, D, threads)
    nblocks = cld(N, (threads ÷ 32) * (32 ÷ D))

    (pid, out_kinds, out_shapes) = get!(g.cache, key) do 
        specs = ntuple(i -> get_spec(getfield(inputs, i), i), length(inputs))
        prog = trace(g.f, specs; T=T, D=D, nthreads=threads)

        out_kinds = Tuple(prog.kinds[vid] for vid in prog.outputs)
        out_shapes = Tuple(prog.shapes[vid] for vid in prog.outputs)
        pid = register_program!(prog, g.debug)

        return pid, out_kinds, out_shapes
    end

    outputs = Tuple(get_output(kind, shape; T=T, N=N) for (kind, shape) in zip(out_kinds, out_shapes))

    Base.invokelatest() do
        @cuda threads=threads blocks=nblocks BatchedKernels._fused_kernel(Val(pid), getproperty.(outputs, :data)..., getproperty.(inputs, :data)..., Int32(N))
    end
    
    return length(outputs) == 1 ? outputs[1] : outputs
end