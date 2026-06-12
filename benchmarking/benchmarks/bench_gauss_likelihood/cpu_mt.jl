using StaticArrays
using BenchmarkTools
using LinearAlgebra

# Per-batch Gaussian log-likelihood, threaded with SMatrix/SVector dispatch.
# Mirrors the CPU reference:
#   δ = x − μ
#   C = cholesky(Σ)
#   log|Σ| = 2 · Σ log diag(C.U)
#   mahal = |C.L \ δ|²
#   p = -0.5·(D·log(2π) + log|Σ| + mahal)

function gauss_likelihood_cpu_mt!(
    p_out::Array{T,1}, x_in::Array{T,2}, μ_in::Array{T,2}, Σ_in::Array{T,3},
) where {T}
    D, N = size(x_in)

    # Reinterpret as vectors of SVector / SMatrix for static dispatch.
    x_reinterp = reinterpret(reshape, SVector{D, T}, x_in)
    μ_reinterp = reinterpret(reshape, SVector{D, T}, μ_in)

    Σ_squashed = reshape(Σ_in, D^2, N)
    Σ_reinterp = reinterpret(reshape, SMatrix{D, D, T, D * D}, Σ_squashed)

    log_2pi_D = T(D) * log(T(2π))

    Threads.@threads for i in 1:N
        @inbounds begin
            xi = x_reinterp[i]
            μi = μ_reinterp[i]
            Σi = Σ_reinterp[i]

            δ = xi - μi
            C = cholesky(Σi)
            log_det = 2 * sum(log, diag(C.U))
            y = C.L \ δ
            mahal = sum(abs2, y)

            p_out[i] = -T(0.5) * (log_2pi_D + log_det + mahal)
        end
    end
end

function gauss_likelihood_timing(p_out, x, μ, Σ, _, _, ::Val{:cpu_mt})
    N = size(x, 2)

    bench_results = @benchmark begin
        gauss_likelihood_cpu_mt!($p_out, $x, $μ, $Σ)
    end

    return median(bench_results.times) / 1e9 / N
end
