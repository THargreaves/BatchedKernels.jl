using CUDA
using CUDA: i32
using LinearAlgebra
using Magma
using Random

include("../magma.jl")

# CPU reference for one batch element.
function cpu_gauss_likelihood(x, μ, Σ)
    D = length(x)
    δ = x .- μ
    C = cholesky(Σ)
    log_det = 2 * sum(log, diag(C.U))
    mahal   = sum(abs2, C.L \ δ)
    return -0.5 * (D * log(2π) + log_det + mahal)
end

function test_gauss_likelihood_correctness(; D=4, N=128, T=Float32, atol=1e-3, rtol=1e-3)
    # ── Generate inputs (same shape as run_script.jl) ──
    Random.seed!(1234)
    x_cpu = rand(T, D, N)
    μ_cpu = rand(T, D, N)
    Σ_cpu = Array{T}(undef, D, D, N)
    for i in 1:N
        Σ_i = rand(T, D, D) / T(D)
        Σ_i = Σ_i * Σ_i' + 0.1f0 * I
        Σ_cpu[:, :, i] = Σ_i
    end
    p_out_cpu = zeros(T, N)

    # ── Run MAGMA pipeline ──
    Magma.LibMagma.magma_init()
    queue_ptr = Ref{Magma.LibMagma.magma_queue_t}()
    Magma.LibMagma.magma_queue_create_internal(
        0, queue_ptr, C_NULL, C_NULL, 0,
    )

    SLACK = 64

    Σs_in_d = cu(Σ_cpu)
    Σ_storage = CUDA.zeros(T, D, D, N + SLACK)
    copyto!(view(Σ_storage, :, :, 1:N), Σs_in_d)

    δ_storage = CUDA.zeros(T, D, N + SLACK)
    x_storage = cu(x_cpu)
    μ_storage = cu(μ_cpu)

    p_storage = CUDA.zeros(T, N)

    dΣ = CUDA.CUBLAS.unsafe_strided_batch(Σ_storage)
    dδ = CUDA.CUBLAS.unsafe_strided_batch(δ_storage)

    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    gauss_likelihood_magma!(
        dΣ, dδ,
        Σ_storage, δ_storage, x_storage, μ_storage, p_storage,
        info_d, D, N, queue_ptr,
    )
    CUDA.synchronize()
    result_magma = Array(p_storage)

    # ── Compare against CPU reference per batch ──
    max_err = 0.0f0
    n_bad = 0
    for i in 1:N
        p_ref = T(cpu_gauss_likelihood(x_cpu[:, i], μ_cpu[:, i], Σ_cpu[:, :, i]))
        err = abs(p_ref - result_magma[i])
        max_err = max(max_err, err)
        if !isapprox(p_ref, result_magma[i]; atol, rtol)
            n_bad += 1
            if n_bad <= 3
                println("Mismatch at batch $i:")
                println("  cpu:   ", p_ref)
                println("  magma: ", result_magma[i])
                println("  diff:  ", err)
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

for D in 2:16
    test_gauss_likelihood_correctness(D=D, N=256)
    println()
end
