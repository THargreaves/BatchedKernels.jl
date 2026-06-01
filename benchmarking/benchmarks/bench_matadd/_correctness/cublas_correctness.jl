using CUDA
using CUDA: i32
using LinearAlgebra

# Include both implementations
include("../cublas.jl")  # has matadd_cublas!
include("../ours.jl")    # has kernel_matadd!, BatchedKernels, etc.

function test_matadd_correctness(; D=4, N=128, T=Float32, atol=1e-3, rtol=1e-3)
    # ── Generate inputs (same as benchmark caller) ──
    A_in_cpu = rand(T, D, D, N)
    B_in_cpu = rand(T, D, D, N)

    # ── Run "ours" ──
    C_out_ours = CUDA.zeros(T, D, D, N)
    A_in_d = cu(A_in_cpu)
    B_in_d = cu(B_in_cpu)

    nthreads = 256
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_bytes = sizeof(T) * 3 * shmem_elems

    kernel = @cuda launch=false kernel_matadd!(
        C_out_ours, A_in_d, B_in_d,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    CUDA.@sync kernel(
        C_out_ours, A_in_d, B_in_d,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small);
        threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
    )
    result_ours = Array(C_out_ours)

    # ── Run cuBLAS ──
    C_out_cublas = CUDA.zeros(T, D, D, N)
    A_in_cublas = cu(A_in_cpu)
    B_in_cublas = cu(B_in_cpu)
    M = D * D * N

    matadd_cublas!(
        reshape(C_out_cublas, :),
        reshape(A_in_cublas, :),
        reshape(B_in_cublas, :),
        M,
    )
    CUDA.synchronize()
    result_cublas = Array(C_out_cublas)

    # ── Compare ──
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
    test_matadd_correctness(D=D, N=256)
    println()
end
