export symmetric_part, covariance_pushforward

"""
    symmetric_part(A)

Return `(A + A') / 2` for a real square matrix. This semantic operation also
has a fused implementation; it does not repair positive definiteness.
"""
symmetric_part(A::AbstractMatrix) = (A + A') / 2

"""
    covariance_pushforward(X, A)

Compute `X * A * X'` from a covariance matrix, without introducing a
factorization. The scalar composition is traced into ordinary matrix products.
"""
covariance_pushforward(X::AbstractMatrix, A::AbstractMatrix) = (X * A) * X'
