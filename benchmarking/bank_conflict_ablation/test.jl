using BatchedKernels
using LinearAlgebra

include("original_kalman.jl")

D = 8
T = Float32
nthreads = 2^8
nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))
n_steps = 1

N = Int(ceil(1e9 / (4 * 2 * D^2)))

P_out_cpu = zeros(T, D, D, N)
P_in_cpu = zeros(T, D, D, N)
for i in 1:N
    P_i = rand(T, D, D) / T(D)
    P_i = P_i * P_i' + 0.1f0 * I
    P_in_cpu[:, :, i] = P_i
end

# A = rand(T, D, D) / Float32(D)
A_cpu = T.(Matrix(qr(randn(T, D, D)).Q))

Q_elem = rand(Float32, D, D) / Float32(D)^2
Q_cpu = Q_elem * Q_elem' + 0.01f0 * I

H_cpu = rand(Float32, D, D) / Float32(D)
R_elem = rand(Float32, D, D) / Float32(D)^2
R_cpu = R_elem * R_elem' + 0.01f0 * I

P_out = cu(P_out_cpu)
P_in = cu(P_in_cpu)
A = cu(A_cpu)
Q = cu(Q_cpu)
H = cu(H_cpu)
R = cu(R_cpu)

n_mats_per_warp = 32 ÷ D
n_warps = nthreads ÷ 32
padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
pad_interval = div(32, D & -D) * D

shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
shmem_size_fixed = D * D + (D * D - 1) ÷ pad_interval

shmem_bytes = sizeof(Float32) * (
    3 * shmem_elems + 4 * shmem_size_fixed
)

kernel = @cuda launch=false kernel_kalman_orig!(
    P_out, P_in, A, Q, H, R,
    Val(Int32(D)), Val(Int32(nthreads)), Int32(n_steps), Int32(N), Val(:small),
)
CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)


CUDA.@sync kernel(
    P_out, P_in, A, Q, H, R,
    Val(Int32(D)), Val(Int32(nthreads)), Int32(n_steps), Int32(N),
    Val(:small), Val(:indep);
    threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
)