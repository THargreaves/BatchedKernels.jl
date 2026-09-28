@inline function variant_op!(
    ::Val{:qr_identity_col}, U, C, d::Int32, ::Val{M}, ::Val{N}, ::Val{D}
) where {M,N,D}
    T = eltype(C)
    col = MVector{Int(M + N),T}(undef)
    dummy = MVector{Int(M + N),T}(undef)
    @inbounds @unroll for i in (1i32):Int32(M)
        col[i] = i == d ? one(T) : zero(T)
    end
    @inbounds @unroll for i in (1i32):Int32(N)
        col[Int32(M) + i] = d <= Int32(M) ? ours(C, i, d, ColAccess()) : zero(T)
    end
    _qr_columns!(col, dummy, d, Val(M + N), Val(M), Val(0), Val(D))
    if d <= Int32(M)
        @inbounds @unroll for i in (1i32):Int32(M)
            ours_write!(U, i, d, col[i], RowAccess())
        end
    end
    return nothing
end

@inline function variant_compress_residual!(
    U, s, B, r, d::Int32, m::Val, n::Val, p::Val, group::Val
)
    return variant_compress_residual!(U, s, B, r, nothing, nothing, d, m, n, p, group)
end
@inline function variant_compress_residual!(
    U, s, B, r, C, q, d::Int32, ::Val{M}, ::Val{N}, ::Val{P}, ::Val{D}
) where {M,N,P,D}
    T = eltype(B)
    R = max(M + P, N)
    col = MVector{Int(R),T}(undef)
    rhs = MVector{Int(R),T}(undef)
    @inbounds @unroll for i in (1i32):Int32(R)
        col[i] = zero(T)
        rhs[i] = zero(T)
        if i <= Int32(M)
            d <= Int32(N) && (col[i] = ours(B, i, d, RowAccess()))
            d == 1i32 && (rhs[i] = r[i])
        elseif P > 0 && i <= Int32(M + P)
            d <= Int32(N) && (col[i] = ours(C, i - Int32(M), d, RowAccess()))
            d == 1i32 && (rhs[i] = q[i - Int32(M)])
        end
    end
    # Only matrix columns need reflectors. The residual column is transformed
    # alongside them; its tail supplies the discarded energy without cancellation.
    _qr_columns!(col, rhs, d, Val(R), Val(N), Val(1), Val(D), Val(N))
    if U !== nothing && d <= Int32(N)
        @inbounds @unroll for i in (1i32):Int32(N)
            ours_write!(U, i, d, col[i], RowAccess())
        end
    end
    base = mod1(threadIdx().x, 32i32) - d
    mask = _matrix_group_mask(Val(D), base)
    if s !== nothing
        @inbounds @unroll for i in (1i32):Int32(N)
            value = shfl_sync(mask, rhs[i], _shuffle_source(base + 1i32))
            d == i && (s[i] = value)
        end
    end
    energy = zero(T)
    if d == 1i32
        @inbounds @unroll for i in (1i32):Int32(R)
            i > Int32(N) && (energy = muladd(rhs[i], rhs[i], energy))
        end
    end
    return shfl_sync(mask, energy, _shuffle_source(base + 1i32))
end
