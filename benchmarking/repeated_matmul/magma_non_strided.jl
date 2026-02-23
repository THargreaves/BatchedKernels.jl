using CUDA
using Magma
using BenchmarkTools

include("../matmul/magma_non_strided.jl")


function magma_repeated_matmul!(
    dC, dA, dB, D, N, queue_ptr,
)
    NT = Magma.LibMagma.MagmaNoTrans

    magmablas_sgemm_batched!(
        NT, NT,
        D, D, D,
        1.0f0, dA, D,
        dB, D,
        0.0f0, dC, D,
        N,
        queue_ptr[],
    )

    magmablas_sgemm_batched!(
        NT, NT,
        D, D, D,
        1.0f0, dC, D,
        dB, D,
        0.0f0, dA, D,
        N,
        queue_ptr[],
    )

    magmablas_sgemm_batched!(
        NT, NT,
        D, D, D,
        1.0f0, dA, D,
        dB, D,
        0.0f0, dC, D,
        N,
        queue_ptr[],
    )

    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
end

function repeated_matmul_timing(C_cpu, A_cpu, B_cpu, queue_ptr, ::Val{:magma_non_strided})
    D, _, N = size(C_cpu)

    A = cu(A_cpu)
    B = cu(B_cpu)
    C = cu(C_cpu)

    dA = CUDA.CUBLAS.unsafe_strided_batch(A)
    dB = CUDA.CUBLAS.unsafe_strided_batch(B)
    dC = CUDA.CUBLAS.unsafe_strided_batch(C)

    bench_results = @benchmark begin
        magma_repeated_matmul!($dC, $dA, $dB, $D, $N, $queue_ptr)
    end

    return median(bench_results.times) / 1e9 / N
end

