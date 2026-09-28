export qr_identity_plus, qr_compress_residual

_qr_identity(C::AbstractMatrix{T}) where {T} = Matrix{T}(I, size(C, 1), size(C, 1))
_qr_identity(C::StaticMatrix{M,N,T}) where {M,N,T} = one(SMatrix{M,M,T})

"""
    qr_identity_plus(C)

Return an upper root of `I + C*C'` using QR of the implicit stack `[I; C']`.
Preserves StaticArrays on CPU. The fused implementation does not form the Gram
matrix and supports rectangular real Float32/Float64 inputs with extents in 1:32.
"""
function qr_identity_plus(C::AbstractMatrix)
    return qr_upper_stack(_qr_dense(C)', _qr_identity(_qr_dense(C)))
end

function _residual_pad(B::AbstractMatrix, r::AbstractVector)
    T = promote_type(eltype(B), eltype(r))
    n = size(B, 2)
    return vcat(hcat(B, r), zeros(T, n + 1, n + 1))
end
function _residual_pad(B::StaticMatrix{M,N}, r::StaticVector{M}) where {M,N}
    T = promote_type(eltype(B), eltype(r))
    return vcat(hcat(B, r), zero(SMatrix{N + 1,N + 1,T}))
end
function _residual_split(R, ::Val{N}) where {N}
    return R[1:N, 1:N], R[1:N, N + 1], abs2(R[N + 1, N + 1])
end
function _residual_split(R::StaticMatrix, ::Val{N}) where {N}
    return R[SOneTo(N), SOneTo(N)], R[SOneTo(N), N + 1], abs2(R[N + 1, N + 1])
end

"""
    qr_compress_residual(B, r) -> (R, s, energy)
    qr_compress_residual(B, r, C, q) -> (R, s, energy)

Compress a residual, or two vertically stacked residuals, so that
`norm(B*z-r)^2 [+ norm(C*z-q)^2] == norm(R*z-s)^2 + energy` up to rounding.
R is n×n upper triangular, s has length n, and energy is nonnegative. Supports
underdetermined and rank-deficient matrices without jitter. CPU static inputs
produce static outputs. GPU blocks each have extents in 1:32; their stack may
exceed the lane-group width. Inputs are unmodified.
"""
function qr_compress_residual(B::AbstractMatrix, r::AbstractVector)
    size(B, 1) == length(r) || throw(DimensionMismatch("Residual row count mismatch"))
    return _residual_split(
        _positive_qr_upper(_residual_pad(_qr_dense(B), r)), Val(size(B, 2))
    )
end
function qr_compress_residual(
    B::AbstractMatrix, r::AbstractVector, C::AbstractMatrix, q::AbstractVector
)
    size(B, 1) == length(r) && size(C, 1) == length(q) && size(B, 2) == size(C, 2) ||
        throw(DimensionMismatch("Stacked residual dimensions mismatch"))
    return qr_compress_residual(vcat(_qr_dense(B), _qr_dense(C)), vcat(r, q))
end
