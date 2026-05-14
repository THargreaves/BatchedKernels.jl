using BatchedKernels
using CUDA
using CUDA: i32
using Random
using LinearAlgebra

include("original_kalman.jl")
include("conflict_kalman.jl")

# === Configure ===
const D = 15
const N = Int(ceil(1e9 / (4 * 2 * D^2)))
const NTHREADS = 256
n_steps = 1
# =================

function main()
    Random.seed!(1234)
    T = Float32

    P_in_cpu = zeros(T, D, D, N)
    for k in 1:N
        X = randn(T, D, D) / T(D)
        P_in_cpu[:, :, k] = X * X' + T(0.1) * I
    end

    A_cpu = randn(T, D, D) / T(D)
    X = randn(T, D, D) / T(D); Q_cpu = X * X' + T(0.01) * I
    H_cpu = randn(T, D, D) / T(D)
    X = randn(T, D, D) / T(D); R_cpu = X * X' + T(0.01) * I

    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    nblocks = cld(N, NTHREADS ÷ 32 * (32 ÷ D))
    n_mats_per_warp = 32 ÷ D
    n_warps = NTHREADS ÷ 32

    # --- No-conflict setup ---
    P_out_nc = CUDA.zeros(T, D, D, N)
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    pad_interval = div(32, D & -D) * D
    shmem_elems_nc = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_fixed_nc = D * D + (D * D - 1) ÷ pad_interval
    shmem_bytes_nc = sizeof(T) * (3 * shmem_elems_nc + 4 * shmem_fixed_nc)

    # k_nc = @cuda launch=false kernel_kalman_orig!(
    #     P_out_nc, P_in, A, Q, H, R,
    #     Val(Int32(D)), Val(Int32(NTHREADS)), Int32(n_steps), Int32(N), Val(:small),
    # )
    # CUDA.cuFuncSetAttribute(k_nc.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes_nc)

    k_nc = @cuda launch=false kernel_kalman_orig!(
        P_out_nc, P_in, A, Q, H, R,
        Val(Int32(D)), Val(Int32(NTHREADS)), Int32(n_steps), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(k_nc.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes_nc)

    println("Registers: ", CUDA.registers(k_nc))
    println("Local memory: ", CUDA.memory(k_nc).local)

    ###
    # println("Registers: ", CUDA.registers(k_nc))
    # println("Local memory: ", CUDA.memory(k_nc).local)  # bytes per thread
    # Full PTX dump — search for ".local" allocations
    # CUDA.code_ptx(
    #     kernel_kalman_orig!,
    #     Tuple{
    #         typeof(P_out_nc), typeof(P_in), typeof(A), typeof(Q), typeof(H), typeof(R),
    #         Val{Int32(D)}, Val{Int32(NTHREADS)}, Int32, Int32, Val{:small},
    #     };
    #     dump_module=true
    # ) |> println
    ###


    # --- Conflict setup ---
    P_out_c = CUDA.zeros(T, D, D, N)
    n_mats_per_block = n_warps * n_mats_per_warp
    shmem_elems_c = n_mats_per_block * D * D
    shmem_fixed_c = D * D
    shmem_bytes_c = sizeof(T) * (3 * shmem_elems_c + 4 * shmem_fixed_c)

    k_c = @cuda launch=false kernel_kalman_conflict!(
        P_out_c, P_in, A, Q, H, R,
        Val(Int32(D)), Val(Int32(NTHREADS)), Int32(n_steps), Int32(N), Val(:small),
    )
    # TODO: Changed shemm amount
    CUDA.cuFuncSetAttribute(k_c.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes_nc)

    # --- Warmup both ---
    CUDA.@sync k_nc(
        P_out_nc, P_in, A, Q, H, R,
        Val(Int32(D)), Val(Int32(NTHREADS)), Int32(n_steps), Int32(N), Val(:small), Val(:indep);
        threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes_nc,
    )
    # TODO: changed shmem amount
    CUDA.@sync k_c(
        P_out_c, P_in, A, Q, H, R,
        Val(Int32(D)), Val(Int32(NTHREADS)), Int32(n_steps),Int32(N), Val(:small), Val(:indep);
        threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes_nc,
    )

    # === PROFILE: comment/uncomment one ===

    # --- No-conflict ---
    CUDA.@profile begin
        CUDA.@sync k_nc(
            P_out_nc, P_in, A, Q, H, R,
            Val(Int32(D)), Val(Int32(NTHREADS)), Int32(n_steps), Int32(N), Val(:small), Val(:indep);
            threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes_nc,
        )
    end

    # --- Conflict ---
    # TODO: changed shmem amount
    # CUDA.@profile begin
    #     CUDA.@sync k_c(
    #         P_out_c, P_in, A, Q, H, R,
    #         Val(Int32(D)), Val(Int32(NTHREADS)), Int32(n_steps), Int32(N), Val(:small), Val(:indep);
    #         threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes_nc,
    #     )
    # end

    println("Done profiling D=$D, N=$N")
end

main()

# ncu \
#   --set full \
#   --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st,l1tex__data_bank_conflicts_pipe_lsu_mem_shared \
#   --profile-from-start off \
#   --target-processes all \
#   --import-source yes \
#   --kernel-name-base demangled \
#   -o kalman_noconflict -f \
#   julia --project=. bank_conflict_ablation/profile.jl

# ncu \
#   --set full \
#   --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st,l1tex__data_bank_conflicts_pipe_lsu_mem_shared \
#   --profile-from-start off \
#   --target-processes all \
#   --import-source yes \
#   --kernel-name-base demangled \
#   -o kalman_conflict -f \
#   julia --project=. bank_conflict_ablation/profile.jl