using BatchedKernels
using CUDA
using CUDA: i32
using Random
using LinearAlgebra

include("backsolve_conflict.jl")
include("backsolve_orig.jl")

# === Configure ===
const D = 15
const N = Int(ceil(1e9 / (4 * 2 * D^2)))
const NTHREADS = 256
# =================

function main()
    Random.seed!(1234)
    T = Float32

    Us_cpu = zeros(T, D, D, N)
    for i in 1:N
        U_temp = randn(T, D, D)
        Us_cpu[:, :, i] = UpperTriangular(U_temp) + T(0.5) * I
    end
    Bs_cpu = rand(T, D, D, N)

    Us = cu(Us_cpu)
    Bs = cu(Bs_cpu)

    nblocks = cld(N, NTHREADS ÷ 32 * (32 ÷ D))
    n_mats_per_warp = 32 ÷ D
    n_warps = NTHREADS ÷ 32
    n_mats_per_block = n_warps * n_mats_per_warp

    # Dual-access shmem
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    shmem_elems_nc = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_bytes_nc = sizeof(T) * 3 * shmem_elems_nc

    # Naive shmem
    shmem_elems_c = n_mats_per_block * D * D
    shmem_bytes_c = sizeof(T) * 3 * shmem_elems_c

    # --- No-conflict (dual-access) ---
    Cs_nc = CUDA.zeros(T, D, D, N)

    k_nc = @cuda launch=false kernel_backward_solve_orig!(
        Cs_nc, Us, Bs, Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(k_nc.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes_nc)

    println("No-conflict registers: ", CUDA.registers(k_nc))
    println("No-conflict local memory: ", CUDA.memory(k_nc).local)

    # --- Conflict (naive) ---
    Cs_c = CUDA.zeros(T, D, D, N)

    k_c = @cuda launch=false kernel_backward_solve_conflict!(
        Cs_c, Us, Bs, Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(k_c.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes_c)

    println("Conflict registers: ", CUDA.registers(k_c))
    println("Conflict local memory: ", CUDA.memory(k_c).local)

    # --- Warmup both ---
    CUDA.@sync k_nc(
        Cs_nc, Us, Bs, Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small);
        threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes_nc,
    )
    CUDA.@sync k_c(
        Cs_c, Us, Bs, Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small);
        threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes_c,
    )

    # === PROFILE: comment/uncomment one ===

    # --- No-conflict (dual-access) ---
    # CUDA.@profile begin
    #     CUDA.@sync k_nc(
    #         Cs_nc, Us, Bs, Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small);
    #         threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes_nc,
    #     )
    # end

    # --- Conflict (naive) ---
    CUDA.@profile begin
        CUDA.@sync k_c(
            Cs_c, Us, Bs, Val(Int32(D)), Val(Int32(D)), Val(Int32(NTHREADS)), Int32(N), Val(:small);
            threads=NTHREADS, blocks=nblocks, shmem=shmem_bytes_c,
        )
    end

    println("Done profiling backward solve D=$D, N=$N")
end

main()

# ncu \
#   --set full \
#   --profile-from-start off \
#   --target-processes all \
#   --import-source yes \
#   --kernel-name-base demangled \
#   -o backsolve_noconflict -f \
#   julia --project=. bank_conflict_ablation/trig_solve/profile.jl

# ncu \
#   --set full \
#   --profile-from-start off \
#   --target-processes all \
#   --import-source yes \
#   --kernel-name-base demangled \
#   -o backsolve_conflict -f \
#   julia --project=. bank_conflict_ablation/trig_solve/profile.jl