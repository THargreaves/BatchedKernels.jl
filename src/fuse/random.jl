# RNG inputs have runtime kernel arguments, but no numerical storage slot. Their
# LITERAL lifecycle means storage-free here; sampling results explicitly become
# BATCHED even though they have no batched numerical operands.
struct RNGInput <: InputSpec end
struct TraceRNG <: Random.AbstractRNG
    tape::Tape
    ref::NodeRef
end
input_spec(::BatchedRNG) = RNGInput()
input_trace_type(::RNGInput) = TraceRNG
input_cache_key(::RNGInput) = (:rng,)
trace_element_type(::Type{BatchedRNG}) = TraceRNG
function _reconstruct_trace_arg!(tape::Tape, ::RNGInput)
    ref = push_node!(tape, InputNode(length(tape.inputs) + 1), NodeMeta(TraceRNG, LITERAL))
    push!(tape.inputs, ref)
    return TraceRNG(tape, ref)
end
function result_to_ref!(::Tape, ::TraceRNG)
    return throw(ArgumentError("An RNG cannot be returned from a fused function"))
end

struct RandomSpec{T,Shape,Normal}
    site::UInt32
end
_random_shape(::Type{RandomSpec{T,S,N}}) where {T,S,N} = S
function _sample_random end

Base.@constprop :aggressive function _trace_random(
    rng::TraceRNG, ::Type{T}, dims::Tuple, ::Val{Normal}
) where {T,Normal}
    T in (Float32, Float64) ||
        throw(ArgumentError("fused sampling supports Float32 and Float64"))
    length(dims) <= 2 ||
        throw(ArgumentError("fused sampling supports scalars, vectors and matrices"))
    all(d -> 1 <= d <= 32, dims) ||
        throw(ArgumentError("random array dimensions must be in 1:32"))
    shape = map(Int, dims)
    Out = if isempty(shape)
        TraceScalar{T}
    elseif length(shape) == 1
        TraceVector{T,shape[1]}
    else
        TraceMatrix{T,shape[1],shape[2]}
    end
    tape = rng.tape
    site = count(n -> n isa CallNode && n.fn === _sample_random, tape.nodes)
    site < 1 << 22 || throw(ArgumentError("too many sampling calls in one trace"))
    spec = emit_const!(tape, RandomSpec{T,shape,Normal}(UInt32(site)))
    ref = push_node!(
        tape, CallNode(_sample_random, NodeRef[rng.ref, spec]), NodeMeta(Out, BATCHED)
    )
    return Out(tape, ref)
end

# Match Random's concrete scalar float methods and restrict array sampling to
# real built-in floats. Float16 is intercepted to give an explicit rejection.
for (fn, normal) in ((:rand, false), (:randn, true))
    @eval begin
        Base.@constprop :aggressive Random.$fn(
            rng::TraceRNG, T::Union{Type{Float16},Type{Float32},Type{Float64}}
        ) = _trace_random(rng, T, (), Val($normal))
        Base.@constprop :aggressive Random.$fn(
            rng::TraceRNG, ::Type{T}, d::Integer, dims::Integer...
        ) where {T<:Union{Float16,Float32,Float64}} =
            _trace_random(rng, T, (d, dims...), Val($normal))
        Base.@constprop :aggressive function Random.$fn(
            rng::TraceRNG, ::Type{T}, dims::Dims
        ) where {T<:Union{Float16,Float32,Float64}}
            isempty(dims) && throw(
                ArgumentError(
                    "zero-dimensional random arrays are unsupported; omit dimensions for a scalar",
                ),
            )
            return _trace_random(rng, T, dims, Val($normal))
        end
        Base.@constprop :aggressive Random.$fn(rng::TraceRNG) =
            _trace_random(rng, Float64, (), Val($normal))
        Base.@constprop :aggressive Random.$fn(rng::TraceRNG, dims::Integer...) =
            _trace_random(rng, Float64, dims, Val($normal))
        Random.$fn(::TraceRNG, ::Type{T}) where {T<:Integer} =
            throw(ArgumentError("fused sampling supports Float32 and Float64"))
    end
end

# Unlike randn, rand(rng, tuple) samples from a collection, so only the
# explicitly typed uniform overload above interprets a tuple as dimensions.
Base.@constprop :aggressive Random.randn(rng::TraceRNG, dims::Dims) =
    Random.randn(rng, Float64, dims)
function Random.randn(::TraceRNG, ::Type{Complex{T}}) where {T<:AbstractFloat}
    return throw(ArgumentError("fused sampling supports real Float32 and Float64"))
end

# Output orientation is a physical choice; the random address always uses the
# logical column-major element index, including in the column-owned variant.
function orientation_variants(
    ::typeof(_sample_random), ::Type{TraceRNG}, ::Type{RandomSpec{T,S,Normal}}
) where {T,S,Normal}
    length(S) == 2 || return ()
    return (
        _orientation_variant(:random_row, (), :row, :random),
        _orientation_variant(:random_col, (), :col, :random),
    )
end

function _emit_random(
    dest, args, ::Type{RandomSpec{T,S,Normal}}, D_MAX; orientation=:row
) where {T,S,Normal}
    rng, spec = args
    site = spec.site
    draw(element) = :(_random_sample(
        $T, Val($Normal), $rng, UInt32(grid_mtrx_id - 1i32), $site, UInt32($element)
    ))
    if isempty(S)
        # All lanes use the same address, preserving TraceScalar replication.
        return :($dest = $(draw(0)))
    elseif length(S) == 1
        return :(
            if d <= $(Int32(S[1]))
                @inbounds $dest[d] = $(draw(:(d - 1i32)))
            end
        )
    end
    m, n = S
    row = orientation === :row
    owned, extent = row ? (n, m) : (m, n)
    access = row ? :(RowAccess()) : :(ColAccess())
    element = if row
        :(k - 1i32 + (d - 1i32) * $(Int32(m)))
    else
        :(d - 1i32 + (k - 1i32) * $(Int32(m)))
    end
    return :(
        if d <= $(Int32(owned))
            @unroll for k in 1i32:($(Int32(extent)))
                ours_write!($dest, k, d, $(draw(element)), $access)
            end
        end
    )
end
function emit_primitive(
    ::typeof(_sample_random), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    return _emit_random(dest, args, types[2], D_MAX)
end
