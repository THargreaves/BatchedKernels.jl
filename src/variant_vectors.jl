# Matrix consumers used by complete Gaussian updates. Vectors retain the existing
# shared layout; all D lanes participate in collectives, including padded lanes.
@inline function variant_op!(
    ::Val{:matvec_col}, y, A, x, d::Int32, ::Val{M}, ::Val{N}, ::Val{D}
) where {M,N,D}
    value = zero(eltype(y))
    @unroll for k in (1i32):Int32(N)
        if d <= Int32(M)
            value = muladd(ours(A, k, d, ColAccess()), @inbounds(x[k]), value)
        end
    end
    if d <= Int32(M)
        @inbounds y[d] = value
    end
    return y
end

@inline function variant_op!(
    ::Val{:solve_vector_col}, y, A, x, d::Int32, ::Val{N}, ::Val{D}
) where {N,D}
    T = eltype(y)
    base = mod1(threadIdx().x, 32i32) - d
    mask = _matrix_group_mask(Val(D), base)
    value = d <= Int32(N) ? @inbounds(x[d]) : zero(T)
    @unroll for step in (1i32):Int32(N)
        k = _solve_forward(A) ? step : Int32(N) - step + 1i32
        if !_solve_unit(A)
            diagonal = theirs(A, k, k)
            if d == k
                value /= diagonal
            end
        end
        pivot = shfl_sync(mask, value, _shuffle_source(base + k))
        if d <= Int32(N) && (_solve_forward(A) ? d > k : d < k)
            value -= ours(A, k, d, ColAccess()) * pivot
        end
    end
    if d <= Int32(N)
        @inbounds y[d] = value
    end
    return y
end

# A full group mask is needed even when the logical vector is shorter than D.
@inline _matrix_group_mask(::Val{D}, base::Int32) where {D} =
    (typemax(UInt32) >> (32 - Int(D))) << (base % UInt32)

@inline function _group_sum(value, d::Int32, ::Val{D}) where {D}
    base = mod1(threadIdx().x, 32i32) - d
    mask = _matrix_group_mask(Val(D), base)
    total = warp_reduce_sum(mask, value, d, Val(Int32(D)))
    return shfl_sync(mask, total, _shuffle_source(base + 1i32))
end

@inline function variant_logdet(A, d::Int32, ::Val{N}, ::Val{D}) where {N,D}
    # Each diagonal broadcast has uniform indices, including for register inputs.
    value = zero(eltype(A))
    @unroll for k in (1i32):Int32(N)
        diagonal = theirs(A, k, k)
        if d == k
            value = eltype(A)(2) * log(diagonal)
        end
    end
    return _group_sum(value, d, Val(D))
end

@inline function variant_norm_sq(x, d::Int32, ::Val{N}, ::Val{D}) where {N,D}
    value = d <= Int32(N) ? abs2(@inbounds(x[d])) : zero(eltype(x))
    return _group_sum(value, d, Val(D))
end

@inline function variant_op!(
    ::Val{:symmetric_row}, C, A, d::Int32, ::Val{N}, ::Val{D}
) where {N,D}
    if d <= Int32(N)
        @unroll for k in (1i32):Int32(N)
            value =
                (ours(A, k, d, RowAccess()) + ours(A, k, d, ColAccess())) * eltype(A)(0.5)
            ours_write!(C, k, d, value, RowAccess())
        end
    end
    return C
end

@inline function variant_dot(x, y, d::Int32, ::Val{N}, ::Val{D}) where {N,D}
    value = d <= Int32(N) ? conj(@inbounds(x[d])) * @inbounds(y[d]) : zero(eltype(x))
    return _group_sum(value, d, Val(D))
end
