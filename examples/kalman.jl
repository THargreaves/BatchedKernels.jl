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

"""
    sqrt_backward_initialise(H, c, UR, y) -> (B, r, logscale)

Normalized observation likelihood in residual form
`exp(logscale - sum(abs2, B*z-r)/2)`. UR is the upper observation covariance
root. B is square with state dimension, including for underobserved states.
"""
function sqrt_backward_initialise(H, c, UR, y)
    L = LowerTriangular(UR')
    B, r, energy = qr_compress_residual(L \ H, L \ (y - c))
    T = eltype(y)
    logscale =
        -T(0.5) * (T(length(y)) * log(T(2) * T(π)) + covariance_root_logdet(UR) + energy)
    return B, r, logscale
end

"""
    sqrt_backward_predict(B, r, logscale, A, b, UQ)

Integrate the next-state likelihood through `z_next = A*z + b + noise`, with
process covariance UQ'UQ. Zero and rectangular rank-deficient roots are valid;
no inverse of the process covariance or backward information matrix is needed.
"""
function sqrt_backward_predict(B, r, logscale, A, b, UQ)
    Bt, rt, ct = _sqrt_backward_transform(B, r, logscale, A, b, UQ)
    Bp, rp, energy = qr_compress_residual(Bt, rt)
    return Bp, rp, ct - eltype(r)(0.5) * energy
end

# An uncompressed residual is already a valid likelihood representation. The
# combined step can feed it straight into the observation compression.
function _sqrt_backward_transform(B, r, logscale, A, b, UQ)
    U = qr_identity_plus(B * UQ')
    L = LowerTriangular(U')
    return L \ (B * A),
    L \ (r - B * b),
    logscale - eltype(r)(0.5) * covariance_root_logdet(U)
end

"""Multiply a backward message by the current observation likelihood."""
function sqrt_backward_update(B, r, logscale, H, c, UR, y)
    L = LowerTriangular(UR')
    Bt, rt, energy = qr_compress_residual(B, r, L \ H, L \ (y - c))
    T = eltype(y)
    return Bt,
    rt,
    logscale -
    T(0.5) * (T(length(y)) * log(T(2) * T(π)) + covariance_root_logdet(UR) + energy)
end

"""One complete backward transition and observation update, suitable for fusion."""
function sqrt_backward_step(B, r, logscale, A, b, UQ, H, c, UR, y)
    Bp, rp, cp = _sqrt_backward_transform(B, r, logscale, A, b, UQ)
    return sqrt_backward_update(Bp, rp, cp, H, c, UR, y)
end

"""
    sqrt_backward_overlap(μ, U, B, r[, logscale])

Log integral of a Gaussian N(μ,U'U) against the residual message. The four-arg
form omits the suffix-common logscale for relative particle weights; the five-arg
form returns the normalized value. Singular forward roots are supported.
"""
function sqrt_backward_overlap(μ, U, B, r)
    V = qr_identity_plus(B * U')
    v = LowerTriangular(V') \ (r - B * μ)
    return -eltype(μ)(0.5) * (covariance_root_logdet(V) + sum(abs2, v))
end
sqrt_backward_overlap(μ, U, B, r, logscale) = logscale + sqrt_backward_overlap(μ, U, B, r)

"""
    sqrt_backward_weight(μ, U, A, b, UQ, B, r, logweight, logtransition)

Ancestor-sampling/backward-simulation candidate log weight. B,r describe the
fixed suffix at t+1; μ,U describe the candidate filter at t. Adds the supplied
particle log weight and outer-state transition log density to the Gaussian
contribution. The suffix-common logscale is omitted. Selection and suffix
orchestration are the caller's responsibility in either algorithm.
"""
function sqrt_backward_weight(μ, U, A, b, UQ, B, r, logweight, logtransition)
    μp = A * μ + b
    Up = qr_upper_stack(UQ, U * A')
    return logweight + logtransition + sqrt_backward_overlap(μp, Up, B, r)
end

"""Covariance-form candidate weight; P and Q may be positive semidefinite."""
function kalman_backward_weight(μ, P, A, b, Q, B, r, logweight, logtransition)
    μp = A * μ + b
    Pp = symmetric_part(covariance_pushforward(A, P) + Q)
    V = symmetric_part(I + covariance_pushforward(B, Pp))
    C = cholesky(Symmetric(V))
    v = C.L \ (r - B * μp)
    return logweight + logtransition - eltype(μ)(0.5) * (logdet(C) + sum(abs2, v))
end
