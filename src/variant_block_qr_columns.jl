# Positive IEEE encodings have the same order as their unsigned integers.
@inline _qr_absmax(a::Float32, b::Float32) = max(a, abs(b))
@inline _qr_absmax(a::Float64, b::Float64) =
    reinterpret(Float64, max(reinterpret(UInt64, a), reinterpret(UInt64, abs(b))))
# Column-owned Householder QR (RowAccess). Each lane owns a column in each
# panel. Dot products are local FMAs; only the pivot reflector is broadcast.
@inline function _qr_columns!(
    left, right, d::Int32, ::Val{R}, ::Val{P}, ::Val{Q}, ::Val{D}
) where {R,P,Q,D}
    return _qr_columns!(left, right, d, Val(R), Val(P), Val(Q), Val(D), Val(P + Q))
end

@inline function _qr_columns!(
    left, right, d::Int32, ::Val{R}, ::Val{P}, ::Val{Q}, ::Val{D}, ::Val{K}
) where {R,P,Q,D,K}
    T = eltype(left)
    base = mod1(threadIdx().x, 32i32) - d
    mask = @inbounds _register_group_mask(Val(D), base)
    # Broadcast once, then reuse for both the projection and rank-one update.
    # Keeping this explicit avoids a second shuffle of every reflector element.
    reflector = MVector{Int(R),T}(undef)
    @inbounds @unroll for j in (1i32):Int32(K)
        # Select individual elements, not an alias of the left/right MVector:
        # a conditional mutable-array alias prevents GPU scalar replacement.
        lane = j <= Int32(P) ? j : j - Int32(P)
        diagonal = zero(T)
        weight = zero(T)
        if d == lane
            scale = zero(T)
            @unroll for i in (1i32):Int32(R)
                if i >= j
                    scale = _qr_absmax(scale, (j <= Int32(P) ? left[i] : right[i]))
                end
            end
            # Normalizing a subnormal scale via inv(scale) can overflow.
            # An exact power-of-two adjustment makes its reciprocal finite;
            # preserve multiplication order so tiny inputs are scaled first.
            adjust = scale < floatmin(T) ? inv(floatmin(T)) : one(T)
            divisor = scale == zero(T) ? one(T) : scale * adjust
            rscale = inv(divisor)
            norm2 = zero(T)
            @unroll for i in (1i32):Int32(R)
                if i >= j
                    x = ((j <= Int32(P) ? left[i] : right[i]) * adjust) * rscale
                    if j <= Int32(P)
                        left[i] = x
                    else
                        right[i] = x
                    end
                    norm2 = muladd(x, x, norm2)
                end
            end
            magnitude = sqrt(norm2)
            beta = -copysign(magnitude, (j <= Int32(P) ? left[j] : right[j]))
            v1 = (j <= Int32(P) ? left[j] : right[j]) - beta
            # v = x/scale - beta*e_j; H = I + v*v'/(beta*v1).
            # For a nonzero column, |beta| and |v1| are >= 1 (up to rounding),
            # so the coefficient needs no small-denominator repair. A zero
            # column uses the identity reflector and keeps its trailing row.
            weight = magnitude == zero(T) ? zero(T) : inv(beta * v1)
            if j <= Int32(P)
                left[j] = v1
            else
                right[j] = v1
            end
            diagonal = beta * scale
        end
        weight = shfl_sync(mask, weight, _shuffle_source(base + lane))
        diagonal = shfl_sync(mask, diagonal, _shuffle_source(base + lane))
        pl = zero(T)
        pr = zero(T)
        @unroll for i in (1i32):Int32(R)
            if i >= j
                v = shfl_sync(
                    mask, (j <= Int32(P) ? left[i] : right[i]), _shuffle_source(base + lane)
                )
                reflector[i] = v
                if j < d <= Int32(P)
                    pl = muladd(v, left[i], pl)
                end
                if j < d + Int32(P) && d <= Int32(Q)
                    pr = muladd(v, right[i], pr)
                end
            end
        end
        pl *= weight
        pr *= weight
        @unroll for i in (1i32):Int32(R)
            if i >= j
                v = reflector[i]
                if j < d <= Int32(P)
                    left[i] = muladd(v, pl, left[i])
                end
                if j < d + Int32(P) && d <= Int32(Q)
                    right[i] = muladd(v, pr, right[i])
                end
            end
        end
        if d == lane
            @unroll for i in (1i32):Int32(R)
                if i >= j
                    if j <= Int32(P)
                        left[i] = i == j ? diagonal : zero(T)
                    else
                        right[i] = i == j ? diagonal : zero(T)
                    end
                end
            end
        end
        # Normalize the complete R row, including the cross block. Zero
        # diagonals retain +1 even if the row has nonzero trailing elements.
        sign = diagonal < zero(T) ? -one(T) : one(T)
        left[j] *= sign
        Q > 0 && (right[j] *= sign)
    end
    return nothing
end

@inline function variant_op!(
    ::Val{:qr_stack_row}, R, A, B, d::Int32, ::Val{M}, ::Val{N}, ::Val{D}
) where {M,N,D}
    T = eltype(B)
    col = MVector{Int(M + N),T}(undef)
    dummy = MVector{Int(M + N),T}(undef)
    @inbounds @unroll for i in (1i32):Int32(N)
        col[i] = d <= Int32(N) ? ours(B, i, d, RowAccess()) : zero(T)
    end
    @inbounds @unroll for i in (1i32):Int32(M)
        col[Int32(N) + i] = d <= Int32(N) ? ours(A, i, d, RowAccess()) : zero(T)
    end
    _qr_columns!(col, dummy, d, Val(M + N), Val(N), Val(0), Val(D))
    if d <= Int32(N)
        @inbounds @unroll for i in (1i32):Int32(N)
            ours_write!(R, i, d, col[i], RowAccess())
        end
    end
    return nothing
end

@inline function variant_op!(
    ::Val{:qr_blocks_row}, outputs, A, B, C, d::Int32, ::Val{M}, ::Val{N}, ::Val{D}
) where {M,N,D}
    T = eltype(A)
    left = MVector{Int(M + N),T}(undef)
    right = MVector{Int(M + N),T}(undef)
    @inbounds @unroll for i in (1i32):Int32(M)
        left[i] = d <= Int32(M) ? ours(A, i, d, RowAccess()) : zero(T)
        right[i] = zero(T)
    end
    @inbounds @unroll for i in (1i32):Int32(N)
        left[Int32(M) + i] = d <= Int32(M) ? ours(B, i, d, RowAccess()) : zero(T)
        right[Int32(M) + i] = d <= Int32(N) ? ours(C, i, d, RowAccess()) : zero(T)
    end
    _qr_columns!(left, right, d, Val(M + N), Val(M), Val(N), Val(D))
    R11, R12, R22 = outputs
    if R11 !== nothing && d <= Int32(M)
        @inbounds @unroll for i in (1i32):Int32(M)
            ours_write!(R11, i, d, left[i], RowAccess())
        end
    end
    if d <= Int32(N)
        if R12 !== nothing
            @inbounds @unroll for i in (1i32):Int32(M)
                ours_write!(R12, i, d, right[i], RowAccess())
            end
        end
        if R22 !== nothing
            @inbounds @unroll for i in (1i32):Int32(N)
                ours_write!(R22, i, d, right[Int32(M) + i], RowAccess())
            end
        end
    end
    return nothing
end
