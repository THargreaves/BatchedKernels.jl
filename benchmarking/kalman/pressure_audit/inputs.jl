include(joinpath(@__DIR__,"../hybrid_selective.jl"))
BK.DEBUG_ACCESSORS && error("production only")
CUDA.versioninfo()
D,N=32,8193
rng=MersenneTwister(62028)
inputs=[Array{Float32}(undef,D,D,N) for _=1:5]
P,A,Q,H,R=inputs
reference=similar(P)
for n=1:N
    X=randn(rng,Float32,D,D)/sqrt(Float32(D))
    P[:,:,n]=X*X'+I
    A[:,:,n]=Matrix{Float32}(I,D,D)+.02f0*randn(rng,Float32,D,D)
    Q[:,:,n]=.2f0*Matrix{Float32}(I,D,D)
    H[:,:,n]=.1f0*randn(rng,Float32,D,D)
    R[:,:,n]=Matrix{Float32}(I,D,D)
    reference[:,:,n]=covariance_reference((x[:,:,n] for x in inputs)...)
end
args=Tuple(BK.BatchedCuMatrix(CuArray(x)) for x in inputs)
tape=BK.trace(covariance_step,BK.InputSpec[BK.input_spec(x) for x in args])
