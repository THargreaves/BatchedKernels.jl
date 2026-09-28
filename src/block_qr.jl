using StaticArrays: SOneTo, SUnitRange

export qr_upper_stack, qr_upper_blocks, covariance_root_logdet

# Keep triangular masking while preserving StaticArrays dispatch and mixed scalar
# types. In particular, parent(U) may contain nonzero values below its diagonal.
_qr_dense(A) = A
function _qr_dense(
    A::Union{UpperTriangular{T,<:StaticMatrix{M,N}},LowerTriangular{T,<:StaticMatrix{M,N}}}
) where {T,M,N}
    return SMatrix{M,N,T}(A)
end
_positive_qr_upper(A) = begin
    R = qr(A).R
    signs = map(x -> x < zero(x) ? -one(x) : one(x), diag(R))
    Diagonal(signs) * R
end

"""
    qr_upper_stack(A, B)

Return a dense upper root `R` with `R'R == A'A + B'B` (up to rounding) and
nonnegative diagonal. `B` is square; `A` may be rectangular. The CPU uses QR of
`[B; A]` and preserves static arrays. The fused implementation keeps both row
blocks implicit and supports real Float32/Float64 blocks with extents in 1:32.
Zero columns and rank-deficient roots are supported; no pivoting or jitter is used.
"""
function qr_upper_stack(A::AbstractMatrix, B::AbstractMatrix)
    size(B, 1) == size(B, 2) == size(A, 2) ||
        throw(DimensionMismatch("qr_upper_stack needs A with n columns and square B"))
    return _positive_qr_upper(vcat(_qr_dense(B), _qr_dense(A)))
end

function _qr_update_matrix(A, B, C)
    T = promote_type(eltype(A), eltype(B), eltype(C))
    return vcat(hcat(A, zeros(T, size(A, 1), size(C, 2))), hcat(B, C))
end
function _qr_update_matrix(
    A::StaticMatrix{M,M}, B::StaticMatrix{N,M}, C::StaticMatrix{N,N}
) where {M,N}
    T = promote_type(eltype(A), eltype(B), eltype(C))
    return vcat(hcat(A, zero(SMatrix{M,N,T})), hcat(B, C))
end
function _qr_split(R, ::Val{M}, ::Val{N}) where {M,N}
    return (R[1:M, 1:M], R[1:M, (M + 1):(M + N)], R[(M + 1):(M + N), (M + 1):(M + N)])
end
function _qr_split(R::StaticMatrix, ::Val{M}, ::Val{N}) where {M,N}
    return (
        R[SOneTo(M), SOneTo(M)],
        R[SOneTo(M), SUnitRange(M + 1, M + N)],
        R[SUnitRange(M + 1, M + N), SUnitRange(M + 1, M + N)],
    )
end

"""
    qr_upper_blocks(A, B, C) -> (R11, R12, R22)

R-only QR of `[A 0; B C]`, where A is m×m, B is n×m and C is n×n.
Returns the three dense upper-block factors, with nonnegative diagonals in R11
and R22. Row sign correction applies jointly to R11 and R12. Static CPU inputs
produce static outputs. Fused QR supports Float32/Float64, m,n in 1:32, including
m+n > 32, without constructing the assembled matrix or Q. Inputs are unmodified.
"""
function qr_upper_blocks(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    m, n = size(A, 1), size(C, 1)
    size(A) == (m, m) && size(B) == (n, m) && size(C) == (n, n) ||
        throw(DimensionMismatch("qr_upper_blocks needs m×m, n×m, n×n blocks"))
    R = _positive_qr_upper(_qr_update_matrix(_qr_dense(A), _qr_dense(B), _qr_dense(C)))
    return _qr_split(R, Val(m), Val(n))
end

"""Log determinant of `U'U`, for a square upper covariance root with nonnegative diagonal."""
function covariance_root_logdet(U::AbstractMatrix)
    size(U, 1) == size(U, 2) || throw(DimensionMismatch("Root must be square"))
    return 2sum(log, diag(U))
end
