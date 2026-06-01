using BatchedKernels
using LinearAlgebra

include("original_kalman.jl")
include("conflcit_kalman.jl")
include("../config/Schedule.jl")

function main(D::Int, n_warmups::Int, nthreads::Int, which::String, n_steps::Int)
    N = Int(ceil(1e9 / (4 * 2 * D^2)))
    T = Float32

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    P_out_cpu = zeros(T, D, D, N)
    P_in_cpu = zeros(T, D, D, N)
    for i in 1:N
        P_i = rand(T, D, D) / T(D)
        P_i = P_i * P_i' + 0.1f0 * I
        P_in_cpu[:, :, i] = P_i
    end

    A_cpu = rand(T, D, D, N) / T(D)
    Q_cpu = zeros(T, D, D, N)
    for i in 1:N
        Q_elem = rand(T, D, D) / T(D)^2
        Q_cpu[:, :, i] = Q_elem * Q_elem' + 0.01f0 * I
    end

    H_cpu = rand(T, D, D, N) / T(D)
    R_cpu = zeros(T, D, D, N)
    for i in 1:N
        R_elem = rand(T, D, D) / T(D)^2
        R_cpu[:, :, i] = R_elem * R_elem' + 0.01f0 * I
    end

    P_out = cu(P_out_cpu)
    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    shmem_elems = let
        n_mats_per_warp = 32 ÷ D
        n_warps = nthreads ÷ 32
        dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        warp_shmem_size * n_warps
    end
    shmem_bytes = 5 * shmem_elems * sizeof(T)

    if which == "orig"
        kernel = @cuda launch=false kernel_kalman_orig!(
            P_out, P_in, A, Q, H, R,
            Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
        )
    elseif which == "conflict"
        kernel = @cuda launch=false kernel_kalman_conflict!(
            P_out, P_in, A, Q, H, R,
            Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
        )
    end
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    # Warm-up
    for _ in 1:n_warmups
        CUDA.@sync kernel(
            P_out, P_in, A, Q, H, R,
            Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),;
            threads = nthreads, blocks = nblocks, shmem = shmem_bytes,
        )
    end
    CUDA.synchronize()

    # Profile
    CUDA.@sync kernel(
        P_out, P_in, A, Q, H, R,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
        threads = nthreads, blocks = nblocks, shmem = shmem_bytes,
    )
    CUDA.synchronize()
end


# D = 23
# n_warmups = 2
# nthreads = Schedule.best_nthreads("kalman", D)
# which = "orig"
# n_steps = 1
which = ARGS[1]
D = parse(Int, ARGS[2])
n_warmups = 2
# nthreads = Schedule.best_nthreads("kalman", D)
nthreads = 128
n_steps = 1
main(D, n_warmups, nthreads, which, n_steps)

# ncu --set full --target-processes all --import-source yes \
#     --kernel-name-base demangled \
#     --launch-skip 2 --launch-count 1 \
#     -o profiles/ours_23 -f \
#     julia --project=../../../. profile.jl

# ncu --import profiles/ours_23.ncu-rep --page source --print-source sass --csv > profiles/ours_23_sass.csv

# ncu --import profiles/ours_23.ncu-rep --csv --page raw > profiles/ours_23.csv
