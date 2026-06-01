using CUDA
using CUDA: i32
using LinearAlgebra
using Random

# Include both implementations
include("../cusolver.jl")  # has cholesky_cusolver!, _batch_ptrs
include("../ours.jl")    # has kernel_cholesky!, BatchedKernels, etc.

function test_cholesky_correctness(; D=4, N=128, T=Float32, atol=1e-3, rtol=1e-3)
    # ── Generate inputs (same SPD construction as benchmark caller) ──
    Random.seed!(1234)
    A_in_cpu = zeros(T, D, D, N)
    for i in 1:N
        A_temp = rand(T, D, D)
        A_in_cpu[:, :, i] = A_temp * A_temp' + 0.1f0 * I
    end

    # ── Run "ours" ──
    U_out_ours = CUDA.zeros(T, D, D, N)
    A_in_d = cu(A_in_cpu)

    nthreads = 256
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_bytes = sizeof(T) * 2 * shmem_elems

    kernel = @cuda launch=false kernel_cholesky!(
        U_out_ours, A_in_d,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    CUDA.@sync kernel(
        U_out_ours, A_in_d,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small);
        threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
    )
    result_ours = Array(U_out_ours)

    # ── Run cuSOLVER ──
    # spotrf is in-place, so seed U with A
    U_out_cusolver = cu(A_in_cpu)
    DD = D * D

    dU = _batch_ptrs(U_out_cusolver, N, DD)
    info_d = CUDA.zeros(Cint, N)

    cholesky_cusolver!(
        dU, info_d, D, N,
    )
    CUDA.synchronize()
    result_cusolver = Array(U_out_cusolver)

    # ── Compare upper triangles only (lower-triangle data is unused) ──
    max_err = 0.0f0
    n_bad = 0
    for i in 1:N
        U_ours_i = UpperTriangular(result_ours[:, :, i])
        U_cusolver_i = UpperTriangular(result_cusolver[:, :, i])
        err = maximum(abs.(U_ours_i .- U_cusolver_i))
        max_err = max(max_err, err)
        if !isapprox(U_ours_i, U_cusolver_i; atol, rtol)
            n_bad += 1
            if n_bad <= 3  # print first few mismatches
                println("Mismatch at batch $i:")
                println("  ours:   ", U_ours_i)
                println("  cusolver: ", U_cusolver_i)
                println("  max elem diff: $err")
            end
        end
    end

    println("\nD=$D, N=$N")
    println("Max element-wise error (upper triangle): $max_err")
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
    test_cholesky_correctness(D=D, N=256)
    println()
end
