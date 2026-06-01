using CUDA
using CUDA: i32
using LinearAlgebra
using Random

# Include both implementations
include("../cublas.jl")  # has backsolve_cublas!, _batch_ptrs
include("../ours.jl")    # has kernel_backsolve!, BatchedKernels, etc.

function test_backsolve_correctness(; D=4, N=128, T=Float32, atol=1e-3, rtol=1e-3)
    # ── Generate inputs (same construction as benchmark caller) ──
    Random.seed!(1234)
    U_in_cpu = zeros(T, D, D, N)
    for i in 1:N
        U_temp = rand(T, D, D)
        U_in_cpu[:, :, i] = UpperTriangular(U_temp) + 0.5f0 * I
    end
    B_in_cpu = rand(T, D, D, N)

    # ── Run "ours" ──
    C_out_ours = CUDA.zeros(T, D, D, N)
    U_in_d = cu(U_in_cpu)
    B_in_d = cu(B_in_cpu)

    nthreads = 256
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1)
    shmem_elems = warp_shmem_size * n_warps
    shmem_bytes = sizeof(T) * 3 * shmem_elems

    kernel = @cuda launch=false kernel_backsolve!(
        C_out_ours, U_in_d, B_in_d,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    CUDA.@sync kernel(
        C_out_ours, U_in_d, B_in_d,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small);
        threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
    )
    result_ours = Array(C_out_ours)

    # ── Run cuBLAS ──
    # trsm is in-place on B, so seed C with B
    U_in_cublas = cu(U_in_cpu)
    C_out_cublas = cu(B_in_cpu)
    DD = D * D

    dU = _batch_ptrs(U_in_cublas, N, DD)
    dC = _batch_ptrs(C_out_cublas, N, DD)

    backsolve_cublas!(
        dU, dC, D, N,
    )
    CUDA.synchronize()
    result_cublas = Array(C_out_cublas)

    # ── Compare full matrices (both implementations output full C) ──
    max_err = 0.0f0
    n_bad = 0
    for i in 1:N
        err = maximum(abs.(result_ours[:, :, i] .- result_cublas[:, :, i]))
        max_err = max(max_err, err)
        if !isapprox(result_ours[:, :, i], result_cublas[:, :, i]; atol, rtol)
            n_bad += 1
            if n_bad <= 3  # print first few mismatches
                println("Mismatch at batch $i:")
                println("  ours:   ", result_ours[:, :, i])
                println("  cublas: ", result_cublas[:, :, i])
                println("  max elem diff: $err")
            end
        end
    end

    println("\nD=$D, N=$N")
    println("Max element-wise error: $max_err")
    println("Mismatched batches: $n_bad / $N (atol=$atol, rtol=$rtol)")
    if n_bad == 0
        println("✓ PASS")
    else
        println("✗ FAIL")
    end

    return n_bad == 0
end

# Run for a few sizes
for D in 2:16
    test_backsolve_correctness(D=D, N=256)
    println()
end
