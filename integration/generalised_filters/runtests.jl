using Test, Random, LinearAlgebra, StaticArrays, CUDA, BatchedKernels
import GeneralisedFilters as GF
include("adapter.jl")
const GK = GeneralisedFiltersKernels

# Test-only host/device packing. Production callers already own their arrays;
# wrap those arrays once and retain the structure-of-arrays representation.
function pack(xs::AbstractVector{<:AbstractMatrix})
    return BatchedCuMatrix(CuArray(cat(xs...; dims=3)))
end
pack(xs::AbstractVector{<:AbstractVector}) = BatchedCuVector(CuArray(hcat(xs...)))
pack(xs::AbstractVector{<:Number}) = BatchedCuScalar(CuArray(xs))
function pack(xs::AbstractVector{<:UpperTriangular})
    c = (data=pack(map(Matrix, xs)),)
    T = UpperTriangular{eltype(first(xs)),eltype(c.data)}
    return BatchedStruct{T,typeof(c)}(c, length(xs))
end
const Composite = Union{
    GF.GaussianState,
    GF.SqrtGaussianState,
    GF.SqrtInformationLikelihood,
    GF.LinearGaussianDynamics,
    GF.LinearGaussianObservation,
}
function pack(xs::AbstractVector{<:Composite})
    names = fieldnames(eltype(xs))
    c = NamedTuple{names}(map(n -> pack(map(x -> getfield(x, n), xs)), names))
    wrapper = Base.typename(eltype(xs)).wrapper
    T = wrapper{map(eltype, values(c))...}
    return BatchedStruct{T,typeof(c)}(c, length(xs))
end

# Explicit copies permit inspection without enabling device scalar indexing.
host(x::BatchedCuMatrix, i) = Array(x.data)[:, :, i]
host(x::BatchedCuVector, i) = Array(x.data)[:, i]
host(x::BatchedCuScalar, i) = Array(x.data)[i]
host(x::SharedValue, i) = x.value
function host(x::BatchedStruct{T}, i) where {T}
    xs = map(c -> host(c, i), values(x.components))
    return T <: Tuple ? xs : Base.typename(T).wrapper(xs...)
end

# A QR message has a sign freedom. Compare its complete log quadratic, including
# the normalizer, rather than asserting one choice of factors.
function check_message(a, b; rtol, atol)
    @test isapprox(a.B'a.B, b.B'b.B; rtol, atol)
    @test isapprox(a.B'a.r, b.B'b.r; rtol, atol)
    @test isapprox(
        a.logscale - sum(abs2, a.r) / 2, b.logscale - sum(abs2, b.r) / 2; rtol, atol
    )
end
function check_state(a, b; rtol, atol)
    @test isapprox(a.μ, b.μ; rtol, atol)
    Pa = a isa GF.SqrtGaussianState ? a.U'a.U : a.Σ
    Pb = b isa GF.SqrtGaussianState ? b.U'b.U : b.Σ
    @test isapprox(Pa, Pb; rtol, atol)
end

const cpu_only = get(ENV, "BATCHEDKERNELS_TEST_CPU_ONLY", "false") == "true"
const gpu = !cpu_only && CUDA.functional()
gpu && CUDA.allowscalar(false)
println("Julia ", VERSION, "; CUDA.jl ", pkgversion(CUDA), "; GPU checks: ", gpu)
@testset "GeneralisedFilters integration" begin
    for T in (Float32, Float64), noise in (:full, :rank_one, :zero)
        @testset "$T / $noise" begin
            rng = MersenneTwister(791)
            n, m, N = 3, 2, 5
            smat(a) = SMatrix{size(a, 1),size(a, 2),T}(a)
            svec(a) = SVector{length(a),T}(a)
            μs = [svec(randn(rng, T, n)) for _ in 1:N]
            Us = [
                smat(cholesky(Symmetric((X -> X'X + I)(randn(rng, T, n, n)))).U) for
                _ in 1:N
            ]
            states = [GF.GaussianState(μ, U'U) for (μ, U) in zip(μs, Us)]
            roots = [GF.SqrtGaussianState(μ, UpperTriangular(U)) for (μ, U) in zip(μs, Us)]
            A = smat(randn(rng, T, n, n) / T(3))
            b = svec(randn(rng, T, n))
            H = smat(randn(rng, T, m, n))
            c = svec(randn(rng, T, m))
            UR = smat(T(0.6) * Matrix{T}(I, m, m))
            # GeneralisedFilters CovarianceFactor stores F*F'; kernels take UQ=F'.
            F = if noise === :rank_one
                smat(randn(rng, T, n, 1))
            elseif noise === :zero
                zero(SMatrix{3,3,T})
            else
                smat(T(0.3) * Matrix{T}(I, n, n))
            end
            UQ = smat(F')
            droot = GF.LinearGaussianDynamics(A, b, GF.CovarianceFactor(F))
            d = GF.LinearGaussianDynamics(A, b, F * F')
            o = GF.LinearGaussianObservation(H, c, UR'UR)
            ys = [svec(randn(rng, T, m)) for _ in 1:N]
            bp = GF.SqrtBackwardInformationPredictor()
            tol = T === Float32 ? 3e-4 : 2e-11
            kw = (; rtol=tol, atol=tol)
            rootrefs = [
                GF.srkf_update(GF.srkf_predict(s, droot), o, y) for (s, y) in zip(roots, ys)
            ]
            covrefs = [GF.kalman_step(s, d, o, y) for (s, y) in zip(states, ys)]
            messages = [GF.backward_initialise(bp, o, y) for y in ys]
            nextmsgs = [
                GF.backward_update(bp, GF.backward_predict(bp, l, droot), o, y) for
                (l, y) in zip(messages, ys)
            ]
            for i in 1:N
                sr, ll = GK.root_step(roots[i], A, b, UQ, H, c, UR, ys[i])
                check_state(sr, rootrefs[i][1]; kw...)
                @test isapprox(ll, rootrefs[i][2]; rtol=tol, atol=tol)
                @test sr.μ isa SVector
                @test parent(sr.U) isa SMatrix
                sc, lc = GK.covariance_step(states[i], d, o, ys[i])
                check_state(sc, covrefs[i][1]; kw...)
                @test isapprox(lc, covrefs[i][2]; rtol=tol, atol=tol)
                @test sc.Σ isa SMatrix
                l = GK.backward_start(H, c, UR, ys[i])
                @test l.B isa SMatrix
                @test l.r isa SVector
                check_message(l, messages[i]; kw...)
                check_message(
                    GK.backward_step(l, A, b, UQ, H, c, UR, ys[i]), nextmsgs[i]; kw...
                )
            end
            if gpu
                sh(x::AbstractMatrix) = SharedCuMatrix(CuArray(Matrix(x)), N)
                sh(x::AbstractVector) = SharedCuVector(CuArray(Vector(x)), N)
                gd, go, gy = pack(fill(d, N)), pack(fill(o, N)), pack(ys)
                rs = fuse(
                    GK.root_step,
                    pack(roots),
                    sh(A),
                    sh(b),
                    sh(UQ),
                    sh(H),
                    sh(c),
                    sh(UR),
                    gy,
                )
                cs = fuse(GK.covariance_step, pack(states), gd, go, gy)
                # Feed a generated wrapped state directly back into the fuser.
                rs2 = fuse(
                    GK.root_step,
                    rs.components[1],
                    sh(A),
                    sh(b),
                    sh(UQ),
                    sh(H),
                    sh(c),
                    sh(UR),
                    gy,
                )
                cs2 = fuse(
                    GK.covariance_step,
                    cs.components[1],
                    gd,
                    go,
                    gy;
                    policy=(T === Float32 ? :legacy : :auto),
                )
                msg = fuse(GK.backward_start, sh(H), sh(c), sh(UR), gy)
                msg2 = fuse(
                    GK.backward_step,
                    msg,
                    sh(A),
                    sh(b),
                    sh(UQ),
                    sh(H),
                    sh(c),
                    sh(UR),
                    gy;
                    shared_memory=:dynamic,
                )
                # AS/BS candidates share one fixed future suffix. Keep its
                # matrices common while candidate states and weights are batched.
                suffix_fields = (
                    B=sh(messages[1].B),
                    r=sh(messages[1].r),
                    logscale=SharedValue(T(messages[1].logscale), N),
                )
                suffix_type = GF.SqrtInformationLikelihood{
                    eltype(suffix_fields.B),eltype(suffix_fields.r),T
                }
                suffix = BatchedStruct{suffix_type,typeof(suffix_fields)}(suffix_fields, N)
                lw, lt = T.(-1:-1:(-N)), T.(-2:-1:(-(N + 1)))
                wr = fuse(
                    GK.root_weight,
                    pack(roots),
                    sh(A),
                    sh(b),
                    sh(UQ),
                    suffix,
                    pack(lw),
                    pack(lt),
                )
                wc = fuse(
                    GK.covariance_weight, pack(states), gd, suffix, pack(lw), pack(lt)
                )
                ov = fuse(GK.normalized_overlap, pack(roots), msg2)
                for i in 1:N
                    for (got, ref) in
                        ((host(rs, i), rootrefs[i]), (host(cs, i), covrefs[i]))
                        check_state(got[1], ref[1]; kw...)
                        @test isapprox(got[2], ref[2]; rtol=tol, atol=tol)
                    end
                    check_state(
                        host(rs2, i)[1],
                        GF.srkf_update(GF.srkf_predict(rootrefs[i][1], droot), o, ys[i])[1];
                        kw...,
                    )
                    check_state(
                        host(cs2, i)[1],
                        GF.kalman_step(covrefs[i][1], d, o, ys[i])[1];
                        kw...,
                    )
                    check_message(host(msg, i), messages[i]; kw...)
                    check_message(host(msg2, i), nextmsgs[i]; kw...)
                    refw =
                        lw[i] +
                        lt[i] +
                        GF.compute_marginal_predictive_likelihood(
                            GF.srkf_predict(roots[i], droot), messages[1]
                        )
                    @test isapprox(host(wr, i), refw; rtol=tol, atol=tol)
                    @test isapprox(host(wc, i), refw; rtol=tol, atol=tol)
                    refo = GF.compute_marginal_predictive_likelihood(
                        roots[i], nextmsgs[i]; include_constant=true
                    )
                    @test isapprox(host(ov, i), refo; rtol=tol, atol=tol)
                end
            end
        end
    end
end
