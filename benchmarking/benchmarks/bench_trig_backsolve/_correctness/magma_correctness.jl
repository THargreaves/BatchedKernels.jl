using CUDA
using CUDA: i32
using LinearAlgebra
using Magma
using Random

# Include both implementations
include("../magma.jl")   # has backsolve_magma!, magmablas_strsm_batched!
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

    # ── Run MAGMA ──
    # trsm is in-place on B, so seed C with B. Pad with SLACK extra batches
    # to absorb MAGMA's last-batch overflow (see magma.jl for details).
    SLACK = 32

    U_in_magma = CUDA.zeros(T, D, D, N + SLACK)
    copyto!(view(U_in_magma, :, :, 1:N), cu(U_in_cpu))

    C_out_magma = CUDA.zeros(T, D, D, N + SLACK)
    copyto!(view(C_out_magma, :, :, 1:N), cu(B_in_cpu))

    dU = CUDA.CUBLAS.unsafe_strided_batch(U_in_magma)
    dC = CUDA.CUBLAS.unsafe_strided_batch(C_out_magma)

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

    backsolve_magma!(
        dU, dC, U_in_magma, C_out_magma, D, N, queue_ptr,
    )
    CUDA.synchronize()

    # Read back only the first N (real) batch slots — the padding contains
    # write garbage from MAGMA's overflow and should be ignored.
    result_magma = Array(view(C_out_magma, :, :, 1:N))

    # ── Compare full matrices (both implementations output full C) ──
    # Note: MAGMA's intra-batch overflow ALSO corrupts adjacent batches'
    # data within the first N slots (not just the trailing padding). So for
    # mid-batch indices the comparison can spuriously fail. We expect only
    # the LAST batch (index N) to be definitively correct here — all earlier
    # batches may have been clobbered by their successor's overflow. So the
    # comparison below is more of a "no-crash" check than a strict equality.
    # If you want a clean comparison, set N small and re-pad more
    # generously, or run with `batchCount=1` to defeat cross-batch overflow.
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
    test_backsolve_correctness(D=D, N=256)
    println()
end
