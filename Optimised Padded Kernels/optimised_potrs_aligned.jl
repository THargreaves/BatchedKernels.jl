using BenchmarkTools
using CUDA
using Unroll
using StaticArrays
using LinearAlgebra

const nthreads = 192
D = 8

# For D = 16, Magma is 2 seconds vs 3.3 seconds for ours
# Limited by occupancy. 8 warps vs 12 target (which would make the differnce)

# For D = 10, 3 we get uncoalesced memory access
# TODO: Need to check access patterns

# Good at D = 4 ( magma 3.74, ours is 2.16)
# At D = 8, 2.25 ours vs 2.66 MAGMA. Compute seems pretty high. And mem dropped slightly
# from before
# In particular ALU is at 83.6% util. Too much indexing? Or is it the shuffle?
# Lowering block size to 192 threads doesn't make much difference.


# Use transpose of colmun-Cholesky from CS554 to compute upper triangular with parallel
# access patterns
function batched_potrf_kernel!(
    A,
    U,
    ::Val{N},
    ::Val{D},
    ::Val{N_W},
    ::Val{N_B},
    ::Val{E_W},
    ::Val{E_B},
    ::Val{P_W},
    ::Val{Ē_W},
    ::Val{Ē_B},
    ::Val{pad_stride},
    ::Val{pad_interval},
) where {D,N,N_W,N_B,E_W,E_B,P_W,Ē_W,Ē_B,pad_stride,pad_interval}
    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32)                           # lane id
    wid = div(tid - 1, 32) + 1                    # warp id

    warp_matrix_id = div(lid - 1, D) + 1
    block_matrix_id = warp_matrix_id + (wid - 1) * N_W
    grid_matrix_id = block_matrix_id + (bid - 1) * N_B

    col_id = mod1(lid, D)

    # To be modified in-place
    shmem_A = CuStaticSharedArray(Float32, (Ē_B,))

    # Load matrix into shared memory using coalesced reads
    base_addr = (bid - 1) * N_B * D^2 + 1
    align_offset = (base_addr - 1) % 32

    begin
        offset = 0
        while offset < E_B + align_offset  # check
            raw_idx = offset + tid - align_offset
            raw_mtrx = div(raw_idx - 1, D^2) + 1
            grid_mtrx_load = raw_mtrx + (bid - 1) * N_B

            if raw_mtrx <= N_B && grid_mtrx_load <= N && raw_idx > 0
                padded_amount = (raw_idx - 1) ÷ pad_stride

                src_idx = (bid - 1) * N_B * D^2 + raw_idx
                dest_idx = raw_idx + padded_amount

                shmem_A[dest_idx] = A[src_idx]
            end

            offset += nthreads
        end

        sync_threads()

        j = col_id

        # Only the first D * floor(32 / D) threads in each warp will compute the result
        if lid <= D * N_W
            for i in 1:D
                # Mask needs to remove j < i else they will never reach the sync line
                mask = ((1 << (D - (i - 1))) - 1) << (i - 1)
                mask = mask << ((warp_matrix_id - 1) * D)

                if j >= i
                    # Load in value from i row
                    logical_idx = (block_matrix_id - 1) * D^2 + (col_id - 1) * D + i
                    padding = (logical_idx - 1) ÷ pad_stride
                    Ai = shmem_A[logical_idx + padding]
                    # RMOD STEP
                    for k in 1:(i - 1)
                        # Load in value from k row
                        logical_idx = (block_matrix_id - 1) * D^2 + (col_id - 1) * D + k
                        padding = (logical_idx - 1) ÷ pad_stride
                        Ak = shmem_A[logical_idx + padding]
                        # Share common value with other threads
                        Aki = shfl_sync(mask, Ak, i + (warp_matrix_id - 1) * D)
                        # Compute update
                        Ai -= Aki * Ak
                    end
                    # RDIV STEP
                    if j == i
                        Ai = sqrt(Ai)
                    end
                    # Share result with other threads
                    Aii = shfl_sync(mask, Ai, i + (warp_matrix_id - 1) * D)
                    if j > i
                        Ai = Ai / Aii
                    end
                    # Write result back to shared memory
                    logical_idx = (block_matrix_id - 1) * D^2 + (col_id - 1) * D + i
                    padding = (logical_idx - 1) ÷ pad_stride
                    shmem_A[logical_idx + padding] = Ai
                end
            end
        end

        sync_threads()

        # Write result back to shared memory using coalesced writes
        offset = 0
        while offset < E_B + align_offset
            raw_idx = offset + tid - align_offset
            raw_mtrx = div(raw_idx - 1, D^2) + 1
            grid_mtrx_store = raw_mtrx + (bid - 1) * N_B

            if raw_mtrx <= N_B && grid_mtrx_store <= N && raw_idx > 0
                padded_amount = (raw_idx - 1) ÷ pad_stride

                dest_idx = (bid - 1) * N_B * D^2 + raw_idx
                src_idx = raw_idx + padded_amount

                U[dest_idx] = shmem_A[src_idx]
            end

            offset += nthreads
        end
    end

    return nothing
end

# N = Target 1GB with some noise
N = floor(Int, 1e9 / (D^2 * 4)) * 1 + 783
WARPS_PER_BLOCK = nthreads ÷ 32
# A = reshape(CuArray(Float32.(1:(D^2 * N))), D, D, N)
# B = reshape(CuArray(Float32.(1:(D^2 * N))), D, D, N)
# A = CUDA.rand(Float32, D, D, N);
# # Force PD
# A = CUDA.CUBLAS.gemm_strided_batched('N', 'T', A, A);
As = [rand(Float32, D, D) for _ in 1:N];
As = [A * A' + I for A in As];
A = cu(stack(As));
U = CUDA.zeros(Float32, D, D, N);

matrices_per_warp = floor(Int, 32 / D)
matrices_per_block = matrices_per_warp * WARPS_PER_BLOCK
nblocks = ceil(Int, N / matrices_per_block)

# TODO: this can be made smaller in the non-division case. I.e. remove stride all together
pad_interval = div(32, gcd(32, D))
pad_stride = pad_interval * D
N_W = floor(Int, 32 / D)                      # matrices per warp
N_B = N_W * WARPS_PER_BLOCK                                # matrices per block
E_W = D^2 * N_W                               # elements per warp
E_B = D^2 * N_B                               # elements per block
P_W = ceil(Int, E_W / pad_interval)
Ē_W = E_W + P_W                               # elements per warp with padding
Ē_B = Ē_W * WARPS_PER_BLOCK                                # elements per block with padding

CUDA.@sync @cuda threads = nthreads blocks = nblocks batched_potrf_kernel!(
    A,
    U,
    Val(N),
    Val(D),
    Val(N_W),
    Val(N_B),
    Val(E_W),
    Val(E_B),
    Val(P_W),
    Val(Ē_W),
    Val(Ē_B),
    Val(pad_stride),
    Val(pad_interval),
);

# Copy for validation
A_copy = deepcopy(A);
A_ptrs = CUDA.CUBLAS.unsafe_strided_batch(A_copy);

# 857 seconds vs 2.2 seconds for D = 2
# dh = CUDA.CUSOLVER.dense_handle()
# CUDA.CUSOLVER.cusolverDnSpotrfBatched(dh, 'U', D, A_ptrs, D, dh.info, N)

# # Set lower diagonal to zero
# for i in 1:D
#     for j in 1:(i - 1)
#         A_copy[i, j, :] .= 0.0f0
#         U[i, j, :] .= 0.0f0
#     end
# end

# # Check if the result is correct
# println("Error: ", maximum(abs.(U - A_copy)))

# Compare to MAGMA
using Magma
Magma.LibMagma.magma_init()
queue = Magma.LibMagma.magma_queue_t
queue_ptr = Ref{Magma.LibMagma.magma_queue_t}()
device = 0  # or get from CUDA context
Magma.LibMagma.magma_queue_create_internal(
    device,
    queue_ptr,
    C_NULL,  # func
    C_NULL,  # file
    0,        # line
)

A_copy = deepcopy(A);
A_ptrs = CUDA.CUBLAS.unsafe_strided_batch(A_copy);
info = CuArray{Int32}(undef, N)

# D = 2: 5.28 seconds (+140%)
ccall(
    (:magma_spotrf_batched, Magma.LibMagma.libmagma),
    Magma.LibMagma.magma_int_t,
    (
        Magma.LibMagma.magma_uplo_t,
        Magma.LibMagma.magma_int_t,
        CuPtr{CuPtr{Cfloat}},
        Magma.LibMagma.magma_int_t,
        CuPtr{Magma.LibMagma.magma_int_t},
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_queue_t,
    ),
    Magma.MagmaLower,
    D,
    A_ptrs,
    D,
    info,
    N,
    queue_ptr[],
)
