# R-only Householder QR over two row blocks. A lane owns one row in each block;
# logical row/column counts can exceed D. Working fragments are private. Complete
# groups participate in every shuffle, including lanes outside either block.
@inline function _qr_group_max(x, d::Int32, ::Val{D}) where {D}
    base = mod1(threadIdx().x, 32i32) - d
    mask = @inbounds _register_group_mask(Val(D), base)
    @unroll for s in 0:4
        offset = Int32(1 << s)
        other = shfl_sync(mask, x, _shuffle_source(base + min(d + offset, Int32(D))))
        x = max(x, other)
    end
    return shfl_sync(mask, x, _shuffle_source(base + 1i32))
end

@inline function _qr_two_rows!(
    top, bot, d::Int32, ::Val{M}, ::Val{N}, ::Val{K}, ::Val{D}
) where {M,N,K,D}
    T = eltype(top)
    base = mod1(threadIdx().x, 32i32) - d
    mask = @inbounds _register_group_mask(Val(D), base)
    @inbounds @unroll for j in (1i32):Int32(K)
        xt = j <= d <= Int32(M) ? top[j] : zero(T)
        xb = j <= d + Int32(M) && d <= Int32(N) ? bot[j] : zero(T)
        scale = _qr_group_max(max(abs(xt), abs(xb)), d, Val(D))
        # Scale before squaring; all-zero columns have the identity reflector.
        divisor = scale == zero(T) ? one(T) : scale
        xt /= divisor
        xb /= divisor
        magnitude = sqrt(_group_sum(xt * xt + xb * xb, d, Val(D)))
        alpha = if j <= Int32(M)
            shfl_sync(mask, xt, _shuffle_source(base + j))
        else
            shfl_sync(mask, xb, _shuffle_source(base + j - Int32(M)))
        end
        beta = -copysign(magnitude, alpha)
        v1 = alpha - beta
        tau = magnitude == zero(T) ? zero(T) : (beta - alpha) / beta
        vd = magnitude == zero(T) ? one(T) : v1
        vt = d == j && j <= Int32(M) ? one(T) : xt / vd
        vb = d + Int32(M) == j ? one(T) : xb / vd
        @unroll for k in (1i32):Int32(K)
            if k > j
                projection = _group_sum(vt * top[k] + vb * bot[k], d, Val(D))
                update = tau * projection
                top[k] -= vt * update
                bot[k] -= vb * update
            end
        end
        diagonal = beta * scale
        if d >= j
            top[j] = d == j ? diagonal : zero(T)
        end
        if d + Int32(M) >= j
            bot[j] = d + Int32(M) == j ? diagonal : zero(T)
        end
        # Normalize the entire completed row, including the cross block. Zero
        # diagonals deliberately keep +1 so a nonzero trailing row is preserved.
        sign = diagonal < zero(T) ? -one(T) : one(T)
        @unroll for k in (1i32):Int32(K)
            if k >= j
                d == j && (top[k] *= sign)
                d + Int32(M) == j && (bot[k] *= sign)
            end
        end
    end
    return nothing
end

@inline function variant_op!(
    ::Val{:qr_stack_col}, R, A, B, d::Int32, ::Val{M}, ::Val{N}, ::Val{D}
) where {M,N,D}
    T = eltype(B)
    top = MVector{Int(N),T}(undef)
    bot = MVector{Int(N),T}(undef)
    @inbounds @unroll for k in (1i32):Int32(N)
        top[k] = d <= Int32(N) ? ours(B, k, d, ColAccess()) : zero(T)
        bot[k] = d <= Int32(M) ? ours(A, k, d, ColAccess()) : zero(T)
    end
    _qr_two_rows!(top, bot, d, Val(N), Val(M), Val(N), Val(D))
    if d <= Int32(N)
        @inbounds @unroll for k in (1i32):Int32(N)
            ours_write!(R, k, d, top[k], ColAccess())
        end
    end
    return nothing
end

@inline function variant_op!(
    ::Val{:qr_blocks_col}, outputs, A, B, C, d::Int32, ::Val{M}, ::Val{N}, ::Val{D}
) where {M,N,D}
    T = eltype(A)
    top = MVector{Int(M + N),T}(undef)
    bot = MVector{Int(M + N),T}(undef)
    @inbounds @unroll for k in (1i32):Int32(M)
        top[k] = d <= Int32(M) ? ours(A, k, d, ColAccess()) : zero(T)
        bot[k] = d <= Int32(N) ? ours(B, k, d, ColAccess()) : zero(T)
    end
    @inbounds @unroll for k in (1i32):Int32(N)
        top[Int32(M) + k] = zero(T)
        bot[Int32(M) + k] = d <= Int32(N) ? ours(C, k, d, ColAccess()) : zero(T)
    end
    _qr_two_rows!(top, bot, d, Val(M), Val(N), Val(M + N), Val(D))
    R11, R12, R22 = outputs
    if d <= Int32(M)
        if R11 !== nothing
            @inbounds @unroll for k in (1i32):Int32(M)
                ours_write!(R11, k, d, top[k], ColAccess())
            end
        end
        if R12 !== nothing
            @inbounds @unroll for k in (1i32):Int32(N)
                ours_write!(R12, k, d, top[Int32(M) + k], ColAccess())
            end
        end
    end
    if d <= Int32(N) && R22 !== nothing
        @inbounds @unroll for k in (1i32):Int32(N)
            ours_write!(R22, k, d, bot[Int32(M) + k], ColAccess())
        end
    end
    return nothing
end
