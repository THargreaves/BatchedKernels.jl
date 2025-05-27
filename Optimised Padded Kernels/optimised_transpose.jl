using BenchmarkTools
using CUDA
using Unroll
using StaticArrays
using LinearAlgebra

const nthreads = 192
D = 4

# Use transpose of colmun-Cholesky from CS554 to compute upper triangular with parallel
# access patterns
function batched_transpose_kernel!(
    B,
    A,
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

    shmem_A = CuStaticSharedArray(Float32, (Ē_B,))
    shmem_B = CuStaticSharedArray(Float32, (Ē_B,))

    # Load matrix into shared memory using coalesced reads
    base_addr = (bid - 1) * N_B * D^2 + 1
    align_offset = (base_addr - 1) % 32

    # TODO: might not even need shared memory in one direction
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

        # TODO: needs to be made much cleaner
        offset = 0
        while offset < E_W  # check
            raw_idx = offset + lid
            if raw_idx <= E_W  # might be accessing unused shmem but that's fine
                warp_matrix_id = div(raw_idx - 1, D^2) + 1
                block_matrix_id = warp_matrix_id + (wid - 1) * N_W

                # raw_idx already includes warp_matrix_id
                logical_idx = (wid - 1) * N_W * D^2 + raw_idx
                padded_amount = (logical_idx - 1) ÷ pad_stride
                # if threadIdx().x == 1
                #     CUDA.@cuprintln(
                #         "logical_idx: $logical_idx, padded_amount: $padded_amount, index: $(padded_amount + logical_idx), raw_idx: $raw_idx, Ē_B: $Ē_B, E_B: $E_B"
                #     )
                # end
                Aij = shmem_A[logical_idx + padded_amount]
                # if threadIdx().x == 1 && bid == 1
                #     CUDA.@cuprintln(Aij)
                # end

                # Compute the transposed index
                within_matrix_idx = mod1(raw_idx, D^2)
                within_warp_matrix_idx = div(raw_idx - 1, D^2) + 1
                raw_row = mod1(within_matrix_idx, D)
                raw_col = div(within_matrix_idx - 1, D) + 1
                transposed_idx = (raw_row - 1) * D + raw_col
                logical_idx = (
                    (wid - 1) * N_W * D^2 +
                    (within_warp_matrix_idx - 1) * D^2 +
                    transposed_idx
                )
                padding = (logical_idx - 1) ÷ pad_stride

                # if threadIdx().x == 1 && bid == 1
                #     CUDA.@cuprintln("raw_row: $raw_row, raw_col: $raw_col, transposed_idx: $transposed_idx, logical_idx: $logical_idx, padding: $padding")
                # end

                shmem_B[logical_idx + padding] = Aij
            end

            offset += 32
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

                B[dest_idx] = shmem_B[src_idx]
            end

            offset += nthreads
        end
    end

    return nothing
end

# N = Target 1GB with some noise
N = floor(Int, 1e8 / (D^2 * 4)) * 1 + 783
WARPS_PER_BLOCK = nthreads ÷ 32
A = CUDA.rand(Float32, D, D, N);
# B = CuArray{Float32}(undef, D, D, N);
B = CUDA.rand(Float32, D, D, N);

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

CUDA.@sync @cuda threads = nthreads blocks = nblocks batched_transpose_kernel!(
    B,
    A,
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

# Validate
println("Error: ", maximum(abs.(B - permutedims(A, (2, 1, 3)))))
