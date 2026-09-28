"""
A reference integration boundary using real GeneralisedFilters state types.

The numerical definitions are shared with the scalar examples. These singleton
functions can run on CPU StaticArrays or be passed to `fuse`. No GeneralisedFilters
methods are replaced. Resolve models and prepare noise roots before this boundary.
"""
module GeneralisedFiltersKernels
using BatchedKernels
using LinearAlgebra
import GeneralisedFilters as GF
include(joinpath(@__DIR__, "../../examples/kalman.jl"))

function covariance_step(state::GF.GaussianState, d, o, y)
    μ, P, ll = joseph_kalman_step(state.μ, state.Σ, d.A, d.b, d.Q, o.H, o.c, o.R, y)
    return GF.GaussianState(μ, P), ll
end

# UQ'UQ and UR'UR are the process and observation covariances. Supplying these
# explicitly avoids refactorizing a common noise covariance for every particle.
function root_step(state::GF.SqrtGaussianState, A, b, UQ, H, c, UR, y)
    μ, U, ll = srkf_step(state.μ, state.U, A, b, UQ, H, c, UR, y)
    return GF.SqrtGaussianState(μ, UpperTriangular(U)), ll
end

function backward_start(H, c, UR, y)
    return GF.SqrtInformationLikelihood(sqrt_backward_initialise(H, c, UR, y)...)
end
function backward_step(message::GF.SqrtInformationLikelihood, A, b, UQ, H, c, UR, y)
    return GF.SqrtInformationLikelihood(
        sqrt_backward_step(message.B, message.r, message.logscale, A, b, UQ, H, c, UR, y)...
    )
end
function root_weight(
    state::GF.SqrtGaussianState, A, b, UQ, message, logweight, logtransition
)
    return sqrt_backward_weight(
        state.μ, state.U, A, b, UQ, message.B, message.r, logweight, logtransition
    )
end
function covariance_weight(state::GF.GaussianState, d, message, logweight, logtransition)
    return kalman_backward_weight(
        state.μ, state.Σ, d.A, d.b, d.Q, message.B, message.r, logweight, logtransition
    )
end
function normalized_overlap(state::GF.SqrtGaussianState, message)
    return sqrt_backward_overlap(state.μ, state.U, message.B, message.r, message.logscale)
end
end
