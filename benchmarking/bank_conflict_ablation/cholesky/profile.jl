using BatchedKernels
using CUDA
using CUDA: i32
using Random
using LinearAlgebra

include("original_matmul.jl")
include("conflict_matmul.jl")

# === Configure ===
const D = 16
const N = Int(ceil(1e9 / (4 * 2 * D^2)))
const NTHREADS = 256
# =================

function main()
    Random.seed!(1234)
    T = Float32

    As_cpu = rand(T, D, D, N)
    Bs_cpu = rand(T, D, D, N)

    As = cu(As_cpu)
    Bs = cu(Bs_cpu)

    nblocks = cld(N, NTHREADS ÷ 32 * (32 ÷ D))
    n_mats_per_warp = 32 ÷ D
    n_warps = NTHREADS ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_bytes = sizeof(T) * 3 * shmem_elems

    # --- No-conflict (dual-access) ---
    Cs_nc = CUDA.zeros(T, D, D, N)

    k_nc = @cuda launch=false kernel_matmul_orig!(
        Cs_nc, As, Bs,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(k_nc.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    println("No-conflict registers: ", CUDA.registers(k_nc))
    println("No-conflict local memory: ", CUDA.memory(k_nc).local)

    # --- Conflict (naive) ---
    Cs_c = CUDA.zeros(T, D, D, N)

    k_c = @cuda launch=false kernel_matmul_conflict!(
        Cs_c, As, Bs,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(k_c.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    println("Conflict registers: ", CUDA.registers(k_c))
    println("Conflict local memory: ", CUDA.memory(k_c).local)

    # --- Warmup both ---
    CUDA.@sync k_nc(
        Cs_nc, As, Bs,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small);
        threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes,
    )
    CUDA.@sync k_c(
        Cs_c, As, Bs,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small);
        threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes,
    )

    # === PROFILE: comment/uncomment one ===

    # --- No-conflict (dual-access) ---
    # CUDA.@profile begin
    #     CUDA.@sync k_nc(
    #         Cs_nc, As, Bs,
    #         Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small);
    #         threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes,
    #     )
    # end

    # --- Conflict (naive) ---
    CUDA.@profile begin
        CUDA.@sync k_c(
            Cs_c, As, Bs,
            Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small);
            threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes,
        )
    end

    println("Done profiling matmul D=$D, N=$N")
end

main()


# ncu \
#   --set full \
#   --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st,l1tex__data_bank_conflicts_pipe_lsu_mem_shared \
#   --profile-from-start off \
#   --target-processes all \
#   --import-source yes \
#   --kernel-name-base demangled \
#   -o matmul_noconflict -f \
#   julia --project=. bank_conflict_ablation/matmul/profile.jl

# ncu \
#   --set full \
#   --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st,l1tex__data_bank_conflicts_pipe_lsu_mem_shared \
#   --profile-from-start off \
#   --target-processes all \
#   --import-source yes \
#   --kernel-name-base demangled \
#   -o matmul_conflict -f \
#   julia --project=. bank_conflict_ablation/matmul/profile.jl