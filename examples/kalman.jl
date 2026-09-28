using BatchedKernels
using LinearAlgebra

"""
    joseph_kalman_step(μ, P, A, b, Q, H, c, R, y)

One predict/update step, returning `(mean, covariance, loglikelihood_increment)`.
All inputs have the same real floating-point element type. State and observation
dimensions may differ. The innovation covariance must be positive definite.
This is ordinary scalar Julia code, also usable with `fuse` and batched inputs.
"""
function joseph_kalman_step(μ, P, A, b, Q, H, c, R, y)
    μp = A * μ + b
    Pp = symmetric_part(covariance_pushforward(A, P) + Q)
    e = y - H * μp - c
    HP = H * Pp
    S = symmetric_part(HP * H' + R)
    C = cholesky(Symmetric(S))
    # Solve S * K' = H * Pp; reuse C for whitening and log determinant.
    Kt = C \ HP
    K = Kt'
    J = I - K * H
    Pf = symmetric_part(covariance_pushforward(J, Pp) + covariance_pushforward(K, R))
    μf = μp + K * e
    z = C.L \ e
    T = eltype(μ)
    ll = -T(0.5) * (T(length(y)) * log(T(2) * T(π)) + logdet(C) + sum(abs2, z))
    return μf, Pf, ll
end

"""
    srkf_step(μ, U, A, b, UQ, H, c, UR, y)

Square-root predict/update, returning `(mean, upper_root, loglikelihood_increment)`.
Covariances are `U'U`, `UQ'UQ`, and `UR'UR`. UQ may have fewer rows than state
columns; UR must be a nonsingular square observation root. U and UR are dense
upper roots (the returned root is dense with a zero lower triangle). The scalar
implementation preserves StaticArrays; fuse batches the same code automatically.
"""
function srkf_step(μ, U, A, b, UQ, H, c, UR, y)
    μp = A * μ + b
    Up = qr_upper_stack(UQ, U * A')
    US, C, Uf = qr_upper_blocks(UR, Up * H', Up)
    e = y - H * μp - c
    w = LowerTriangular(US') \ e
    μf = μp + C' * w
    T = eltype(w)
    ll =
        -T(0.5) *
        (sum(abs2, w) + covariance_root_logdet(US) + T(length(y)) * log(T(2) * T(π)))
    return μf, Uf, ll
end
