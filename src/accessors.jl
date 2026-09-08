using Preferences: @load_preference

export ours, ours_write!, theirs

const DEBUG_ACCESSORS = @load_preference("debug_accessors", false)

# CPU layout checks need no warp diagnostics. The device versions are deliberately
# separate so enabling diagnostics does not introduce device intrinsics on the CPU.
@inline _debug_uniform(mask::UInt32, base::Int32, x::Int32) = nothing
CUDA.@device_override @inline function _debug_uniform(mask::UInt32, base::Int32, x::Int32)
    if DEBUG_ACCESSORS
        reference = shfl_sync(mask, x, base + 1i32)
        @assert vote_all_sync(mask, x == reference) "accessor index must be group-uniform"
    end
    return nothing
end

@inline _debug_own_uniform(::Val{D}, k::Int32) where {D} = nothing
CUDA.@device_override @inline function _debug_own_uniform(::Val{D}, k::Int32) where {D}
    if DEBUG_ACCESSORS
        lid = mod1(threadIdx().x, 32i32)
        base = ((lid - 1i32) ÷ Int32(D)) * Int32(D)
        # Own-line calls may be predicated for rectangular lanes without a line.
        # This checks currently executing lanes, not convergence of absent lanes.
        mask = active_mask() & _register_group_mask(Val(D), base)
        source = Int32(trailing_zeros(mask)) + 1i32
        reference = shfl_sync(mask, k, source)
        @assert vote_all_sync(mask, k == reference) "owned-line index must be uniform"
    end
    return nothing
end

@inline _debug_lane(A::RegisterMatrix) = nothing
CUDA.@device_override @inline function _debug_lane(A::RegisterMatrix)
    if DEBUG_ACCESSORS
        @assert mod1(threadIdx().x, 32i32) == A.base + A.d "register lane metadata mismatch"
    end
    return nothing
end

@inline _debug_owned(A, k::Int32, d::Int32, c) = nothing
@inline function _debug_owned(
    A::Union{SingleAccessMatrix{T,M,N,D},RegisterMatrix{T,M,N,D}}, k::Int32, d::Int32, c
) where {T,M,N,D}
    if DEBUG_ACCESSORS
        i, j = _access_coords(k, d, c)
        @assert 1i32 <= i <= Int32(M) && 1i32 <= j <= Int32(N) "owned access outside logical matrix"
        _debug_own_uniform(Val(D), k)
    end
    return nothing
end
@inline function _debug_owned(A::DualAccessMatrix{T,D}, k::Int32, d::Int32, c) where {T,D}
    if DEBUG_ACCESSORS
        @assert 1i32 <= k <= Int32(D) && 1i32 <= d <= Int32(D) "owned access outside dual matrix"
        _debug_own_uniform(Val(D), k)
    end
    return nothing
end

@inline _access_coords(k::Int32, d::Int32, ::RowAccess) = (k, d)
@inline _access_coords(k::Int32, d::Int32, ::ColAccess) = (d, k)

"""
    ours(A, k, d, convention)

Read logical A[k,d] under RowAccess or A[d,k] under ColAccess. Single/register
layouts only implement their matching convention; dual and batch-shared matrices
implement both. k must be uniform among participating lanes. Callers predicate
owned accesses in lanes without a logical line, separately from broadcasts.
"""
@inline function ours(
    A::Union{DualAccessMatrix,SharedMatrix},
    k::Int32,
    d::Int32,
    c::Union{RowAccess,ColAccess},
)
    _debug_owned(A, k, d, c)
    i, j = _access_coords(k, d, c)
    return @inbounds A[i, j]
end
@inline function ours(
    A::SingleAccessMatrix{T,M,N,D,RowOriented}, k::Int32, d::Int32, c::RowAccess
) where {T,M,N,D}
    _debug_owned(A, k, d, c)
    return @inbounds A[k, d]
end
@inline function ours(
    A::SingleAccessMatrix{T,M,N,D,ColOriented}, k::Int32, d::Int32, c::ColAccess
) where {T,M,N,D}
    _debug_owned(A, k, d, c)
    return @inbounds A[d, k]
end
@inline function _register_ours(A::RegisterMatrix, k::Int32, d::Int32, c)
    _debug_owned(A, k, d, c)
    if DEBUG_ACCESSORS
        @assert d == A.d "owned register access must use the owning lane"
        _debug_lane(A)
    end
    return @inbounds A.mv[k]
end
@inline ours(
    A::RegisterMatrix{T,M,N,D,RowOriented}, k::Int32, d::Int32, c::RowAccess
) where {T,M,N,D} = _register_ours(A, k, d, c)
@inline ours(
    A::RegisterMatrix{T,M,N,D,ColOriented}, k::Int32, d::Int32, c::ColAccess
) where {T,M,N,D} = _register_ours(A, k, d, c)

"""Write an owned logical element using the same convention contract as `ours`. Returns A."""
@inline function ours_write!(
    A::Union{DualAccessMatrix,SharedMatrix},
    k::Int32,
    d::Int32,
    v,
    c::Union{RowAccess,ColAccess},
)
    _debug_owned(A, k, d, c)
    i, j = _access_coords(k, d, c)
    @inbounds A[i, j] = v
    return A
end
@inline function ours_write!(
    A::SingleAccessMatrix{T,M,N,D,RowOriented}, k::Int32, d::Int32, v, c::RowAccess
) where {T,M,N,D}
    _debug_owned(A, k, d, c)
    @inbounds A[k, d] = v
    return A
end
@inline function ours_write!(
    A::SingleAccessMatrix{T,M,N,D,ColOriented}, k::Int32, d::Int32, v, c::ColAccess
) where {T,M,N,D}
    _debug_owned(A, k, d, c)
    @inbounds A[d, k] = v
    return A
end
@inline function _register_write!(A::RegisterMatrix, k::Int32, d::Int32, v, c)
    _debug_owned(A, k, d, c)
    if DEBUG_ACCESSORS
        @assert d == A.d "owned register write must use the owning lane"
        _debug_lane(A)
    end
    @inbounds A.mv[k] = v
    return A
end
@inline ours_write!(
    A::RegisterMatrix{T,M,N,D,RowOriented}, k::Int32, d::Int32, v, c::RowAccess
) where {T,M,N,D} = _register_write!(A, k, d, v, c)
@inline ours_write!(
    A::RegisterMatrix{T,M,N,D,ColOriented}, k::Int32, d::Int32, v, c::ColAccess
) where {T,M,N,D} = _register_write!(A, k, d, v, c)

@inline _debug_broadcast(A, i::Int32, j::Int32) = nothing
@inline function _debug_broadcast(
    A::SingleAccessMatrix{T,M,N,D}, i::Int32, j::Int32
) where {T,M,N,D}
    if DEBUG_ACCESSORS
        @assert 1i32 <= i <= Int32(M) && 1i32 <= j <= Int32(N) "broadcast outside logical matrix"
        _debug_own_uniform(Val(D), i)
        _debug_own_uniform(Val(D), j)
    end
    return nothing
end
@inline function _debug_broadcast(A::DualAccessMatrix{T,D}, i::Int32, j::Int32) where {T,D}
    if DEBUG_ACCESSORS
        @assert 1i32 <= i <= Int32(D) && 1i32 <= j <= Int32(D) "broadcast outside dual matrix"
        _debug_own_uniform(Val(D), i)
        _debug_own_uniform(Val(D), j)
    end
    return nothing
end

"""
    theirs(A, i, j)

Read group-uniform logical A[i,j]. Register broadcasts require every lane named in
A.mask to execute the same collective and the source lane to offer an initialized
value. CUDA.jl shuffle sources are one-based. These helpers provide no shared-memory
fence; callers synchronize shared producers and consumers explicitly.

Debug collectives diagnose index uniformity and mask membership but cannot prove
convergence. SharedMatrix has no lane-group geometry, so its index-uniformity contract
is caller-validated. Resource and throughput measurements must disable diagnostics.
"""
@inline function theirs(
    A::Union{SingleAccessMatrix,DualAccessMatrix,SharedMatrix}, i::Int32, j::Int32
)
    _debug_broadcast(A, i, j)
    return @inbounds A[i, j]
end
@inline function _debug_register_broadcast(
    A::RegisterMatrix{T,M,N}, i::Int32, j::Int32, source::Int32
) where {T,M,N}
    if DEBUG_ACCESSORS
        @assert 1i32 <= i <= Int32(M) && 1i32 <= j <= Int32(N) "broadcast outside logical matrix"
        @assert 1i32 <= source <= 32i32 &&
            (A.mask & (UInt32(1) << (source - 1i32))) != UInt32(0) "broadcast source absent from mask"
        _debug_lane(A)
        _debug_uniform(A.mask, A.base, i)
        _debug_uniform(A.mask, A.base, j)
    end
    return nothing
end
# CUDA.jl subtracts an Int literal before converting the one-based source to
# UInt32. Express the validated source range explicitly so that conversion cannot
# retain an InexactError call frame in otherwise register-resident kernels.
@inline _shuffle_source(source::Int32) =
    (((source - 1i32) % UInt32) & UInt32(31)) + UInt32(1)

@inline function theirs(
    A::RegisterMatrix{T,M,N,D,RowOriented}, i::Int32, j::Int32
) where {T,M,N,D}
    source = A.base + j
    _debug_register_broadcast(A, i, j, source)
    return shfl_sync(A.mask, @inbounds(A.mv[i]), _shuffle_source(source))
end
@inline function theirs(
    A::RegisterMatrix{T,M,N,D,ColOriented}, i::Int32, j::Int32
) where {T,M,N,D}
    source = A.base + i
    _debug_register_broadcast(A, i, j, source)
    return shfl_sync(A.mask, @inbounds(A.mv[j]), _shuffle_source(source))
end

@inline ours(A::Adjoint, k::Int32, d::Int32, c::AccessConvention) =
    conj(ours(parent(A), k, d, flip(c)))
@inline ours(A::Transpose, k::Int32, d::Int32, c::AccessConvention) =
    ours(parent(A), k, d, flip(c))
@inline theirs(A::Adjoint, i::Int32, j::Int32) = conj(theirs(parent(A), j, i))
@inline theirs(A::Transpose, i::Int32, j::Int32) = theirs(parent(A), j, i)
@inline function ours_write!(A::Adjoint, k::Int32, d::Int32, v, c::AccessConvention)
    ours_write!(parent(A), k, d, conj(v), flip(c))
    return A
end
@inline function ours_write!(A::Transpose, k::Int32, d::Int32, v, c::AccessConvention)
    ours_write!(parent(A), k, d, v, flip(c))
    return A
end

const AccessorTriangular = Union{
    LowerTriangular,UpperTriangular,UnitLowerTriangular,UnitUpperTriangular
}
@inline _tri_stored(::Union{LowerTriangular,UnitLowerTriangular}, i::Int32, j::Int32) =
    i >= j
@inline _tri_stored(::Union{UpperTriangular,UnitUpperTriangular}, i::Int32, j::Int32) =
    i <= j
@inline _tri_unit(::Union{LowerTriangular,UpperTriangular}) = false
@inline _tri_unit(::Union{UnitLowerTriangular,UnitUpperTriangular}) = true
@inline function _tri_read(A, value, i::Int32, j::Int32)
    return ifelse(
        _tri_unit(A) && i == j, one(value), ifelse(_tri_stored(A, i, j), value, zero(value))
    )
end
@inline function ours(A::AccessorTriangular, k::Int32, d::Int32, c::AccessConvention)
    value = ours(parent(A), k, d, c)
    i, j = _access_coords(k, d, c)
    return _tri_read(A, value, i, j)
end
@inline function theirs(A::AccessorTriangular, i::Int32, j::Int32)
    value = theirs(parent(A), i, j) # Execute any register collective before masking.
    return _tri_read(A, value, i, j)
end
@inline _check_access_convention(::RowOriented, ::RowAccess) = nothing
@inline _check_access_convention(::ColOriented, ::ColAccess) = nothing
@inline _check_access_convention(::BothOriented, ::Union{RowAccess,ColAccess}) = nothing

# Structured no-op writes must still validate the logical access and lane owner.
@inline function _debug_write_owned(A, k::Int32, d::Int32, c)
    if DEBUG_ACCESSORS
        i, j = _access_coords(k, d, c)
        @assert 1i32 <= i <= size(A, 1) && 1i32 <= j <= size(A, 2) "structured write outside logical matrix"
        _debug_owned(A, k, d, c)
    end
    return nothing
end
@inline function _debug_write_owned(A::RegisterMatrix, k::Int32, d::Int32, c)
    if DEBUG_ACCESSORS
        _debug_owned(A, k, d, c)
        @assert d == A.d "structured register write must use the owning lane"
        _debug_lane(A)
    end
    return nothing
end
@inline _debug_write_owned(A::Union{Adjoint,Transpose}, k::Int32, d::Int32, c) =
    _debug_write_owned(parent(A), k, d, flip(c))
@inline _debug_write_owned(
    A::Union{AccessorTriangular,IAddSubGetterMatrix,IAddSubSetterMatrix},
    k::Int32,
    d::Int32,
    c,
) = _debug_write_owned(parent(A), k, d, c)

@inline function ours_write!(
    A::AccessorTriangular, k::Int32, d::Int32, v, c::AccessConvention
)
    _check_access_convention(orientation(A), c)
    _debug_write_owned(parent(A), k, d, c)
    i, j = _access_coords(k, d, c)
    if _tri_unit(A) && i == j
        isone(v) || throw(ArgumentError("unit triangular diagonal must remain one"))
    elseif _tri_stored(A, i, j)
        ours_write!(parent(A), k, d, v, c)
    else
        iszero(v) ||
            throw(ArgumentError("cannot write a nonzero outside the triangular structure"))
    end
    return A
end

@inline function ours(A::IAddSubGetterMatrix, k::Int32, d::Int32, c::AccessConvention)
    value = ours(parent(A), k, d, c)
    i, j = _access_coords(k, d, c)
    return wrapper_get(A, value, i, j)
end
@inline function theirs(A::IAddSubGetterMatrix, i::Int32, j::Int32)
    return wrapper_get(A, theirs(parent(A), i, j), i, j)
end
@inline ours(A::IAddSubSetterMatrix, k::Int32, d::Int32, c::AccessConvention) =
    ours(parent(A), k, d, c)
@inline theirs(A::IAddSubSetterMatrix, i::Int32, j::Int32) = theirs(parent(A), i, j)
@inline function ours_write!(
    A::IAddSubSetterMatrix, k::Int32, d::Int32, v, c::AccessConvention
)
    i, j = _access_coords(k, d, c)
    ours_write!(parent(A), k, d, wrapper_set(A, v, i, j), c)
    return A
end
