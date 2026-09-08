# Example: ncu --profile-from-start off --metrics \
# sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum,dram__bytes.sum.per_second,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum \
# julia --project=. benchmarking/kalman/hybrid_m6_profile.jl
include("hybrid_m6.jl")
BK.DEBUG_ACCESSORS && error("Use production preferences")
D, N = 16, 8192
rng = MersenneTwister(62026)
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
for policy in (:legacy,:single,:register)
    r = prepare(covariance_step,args,policy,reference; nthreads=128)
    r === nothing && continue
    kernel_time(r)
    CUDA.synchronize()
    println("PROFILE,$policy"); flush(stdout)
    CUDA.@profile CUDA.@sync r.kernel(r.ka...;threads=r.nthreads,blocks=r.blocks)
end
