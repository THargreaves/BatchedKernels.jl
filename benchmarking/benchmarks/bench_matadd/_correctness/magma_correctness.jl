using CUDA
using CUDA: i32
using LinearAlgebra
using Magma

# Include both implementations
include("../magma.jl")   # has matadd_magma!, magmablas_sgeadd_batched!
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

    # ── Run MAGMA ──
    # matadd_magma! computes B := A + B in-place (one batched sgeadd), so
    # we read the result back from B_in_magma after the call rather than
    # from a separate C buffer.
    A_in_magma = cu(A_in_cpu)
    B_in_magma = cu(B_in_cpu)

    dA = CUDA.CUBLAS.unsafe_strided_batch(A_in_magma)
    dB = CUDA.CUBLAS.unsafe_strided_batch(B_in_magma)

    Magma.LibMagma.magma_init()
    queue_ptr = Ref{Magma.LibMagma.magma_queue_t}()
    device = 0
    Magma.LibMagma.magma_queue_create_internal(
        device,
        queue_ptr,
        C_NULL,  # func
        C_NULL,  # file
        0,       # line
    )

    matadd_magma!(
        dA, dB,
        D, N, queue_ptr,
    )
    CUDA.synchronize()
    result_magma = Array(B_in_magma)

    # ── Compare ──
    max_err = 0.0f0
    n_bad = 0
    for i in 1:N
        err = maximum(abs.(result_ours[:, :, i] .- result_magma[:, :, i]))
        max_err = max(max_err, err)
        if !isapprox(result_ours[:, :, i], result_magma[:, :, i]; atol, rtol)
            n_bad += 1
            if n_bad <= 3  # print first few mismatches
                println("Mismatch at batch $i:")
                println("  ours:  ", result_ours[:, :, i])
                println("  magma: ", result_magma[:, :, i])
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
