# A ~1 GiB logical I/O working set, following the historical Kalman benchmark.
# Run with OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/hybrid_m6_streaming.jl
include("hybrid_m6.jl")
BK.DEBUG_ACCESSORS && error("Use production preferences")
CUDA.versioninfo()
D, N = 32, 131072
rng = MersenneTwister(62027)
P = Array{Float32}(undef,D,D,N)
for n in 1:N
    X = randn(rng,Float32,D,D)/sqrt(Float32(D))
    P[:,:,n] = X*X' + I
end
A = Matrix{Float32}(I,D,D) + .02f0*randn(rng,Float32,D,D)
H = .1f0*randn(rng,Float32,D,D)
Q, R = .2f0*Matrix{Float32}(I,D,D), Matrix{Float32}(I,D,D)
reference = similar(P)
for n in 1:N
    reference[:,:,n] = covariance_reference(P[:,:,n],A,Q,H,R)
end
args = (BK.BatchedCuMatrix(CuArray(P)), (BK.SharedCuMatrix(CuArray(x),N) for x in (A,Q,H,R))...)
records = []
for (policy,threads) in ((:legacy,64),(:register,128),(:balanced_mul,64))
    r = prepare(covariance_step,args,policy,reference; nthreads=threads)
    r === nothing || push!(records,r)
end
measure!(records,rng)
all(Array(x.data) == original for (x,original) in zip(args,(P,A,Q,H,R))) || error("Input changed during timing")
