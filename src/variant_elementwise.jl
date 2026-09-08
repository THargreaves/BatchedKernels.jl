# Compute-only variants. Callers synchronize shared producers before entry and
# shared consumers/reuse after exit, and invoke every lane of each active group.
@inline function variant_op!(
    ::Val{:matmul_row}, C, A, B, d::Int32, ::Val{M}, ::Val{N}, ::Val{P}, ::Val{D}
) where {M,N,P,D}
    # Snapshot each owned B line once. This bounds shared input traffic independently
    # of alias analysis and exposes constant register indices after unrolling.
    b_line = MVector{Int(N),eltype(B)}(undef)
    @unroll for k in (1i32):Int32(N)
        @inbounds b_line[k] = d <= Int32(P) ? ours(B, k, d, RowAccess()) : zero(eltype(B))
    end
    @unroll for i in (1i32):Int32(M)
        accumulator = zero(eltype(C))
        @unroll for k in (1i32):Int32(N)
            # Source lanes can lie beyond the output's P columns. Every lane must
            # offer A, even when it owns no B or C column.
            a = theirs(A, i, k)
            b = @inbounds b_line[k]
            accumulator = muladd(a, b, accumulator)
        end
        if d <= Int32(P)
            ours_write!(C, i, d, accumulator, RowAccess())
        end
    end
    return C
end

@inline function variant_op!(
    ::Val{:matmul_col}, C, A, B, d::Int32, ::Val{M}, ::Val{N}, ::Val{P}, ::Val{D}
) where {M,N,P,D}
    variant_op!(
        Val(:matmul_row),
        adjoint(C),
        adjoint(B),
        adjoint(A),
        d,
        Val(P),
        Val(N),
        Val(M),
        Val(D),
    )
    return C
end

@inline function variant_op!(
    ::Val{:add_row}, C, A, B, d::Int32, ::Val{M}, ::Val{N}, ::Val{D}
) where {M,N,D}
    if d <= Int32(N)
        @unroll for k in (1i32):Int32(M)
            value = ours(A, k, d, RowAccess()) + ours(B, k, d, RowAccess())
            ours_write!(C, k, d, value, RowAccess())
        end
    end
    return C
end

@inline function variant_op!(
    ::Val{:sub_row}, C, A, B, d::Int32, ::Val{M}, ::Val{N}, ::Val{D}
) where {M,N,D}
    if d <= Int32(N)
        @unroll for k in (1i32):Int32(M)
            value = ours(A, k, d, RowAccess()) - ours(B, k, d, RowAccess())
            ours_write!(C, k, d, value, RowAccess())
        end
    end
    return C
end

@inline function variant_op!(
    ::Val{:add_col}, C, A, B, d::Int32, ::Val{M}, ::Val{N}, ::Val{D}
) where {M,N,D}
    variant_op!(
        Val(:add_row), adjoint(C), adjoint(A), adjoint(B), d, Val(N), Val(M), Val(D)
    )
    return C
end

@inline function variant_op!(
    ::Val{:sub_col}, C, A, B, d::Int32, ::Val{M}, ::Val{N}, ::Val{D}
) where {M,N,D}
    variant_op!(
        Val(:sub_row), adjoint(C), adjoint(A), adjoint(B), d, Val(N), Val(M), Val(D)
    )
    return C
end
