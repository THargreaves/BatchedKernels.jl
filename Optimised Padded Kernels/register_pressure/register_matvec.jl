using BenchmarkTools
using CUDA
using Unroll
using StaticArrays

using CUDA: i32

CUDA.cache_config!(CUDA.FUNC_CACHE_PREFER_SHARED)

const nthreads = 256
const nthreads_i32 = Int32(nthreads)
D = 4

# Bottlenecked by registers
# Should be an easy fix

function batched_matvec_kernel!(
    y,
    A,
    x,
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
    lid = mod1(tid, 32i32)                           # lane id
    wid = div(tid - 1i32, 32i32) + 1i32                    # warp id

    warp_matrix_id = div(lid - 1i32, D) + 1i32
    block_matrix_id = warp_matrix_id + (wid - 1i32) * N_W

    col_id = mod1(lid, D)

    shmem_A = CuStaticSharedArray(Float32, (Ē_B,))
    shmem_AT = CuStaticSharedArray(Float32, (Ē_B,))
    # No padding for vectors
    shmem_x = CuStaticSharedArray(Float32, (D * N_B,))
    shmem_y = CuStaticSharedArray(Float32, (D * N_B,))

    # Load matrices into shared memory using coalesced reads
    base_addr = (bid - 1i32) * N_B * D^2 + 1i32
    align_offset = (base_addr - 1i32) % 32i32

    offset = 0i32
    while offset < E_B + align_offset  # check
        raw_idx = offset + tid - align_offset
        raw_mtrx = div(raw_idx - 1i32, D^2) + 1i32
        grid_mtrx_load = raw_mtrx + (bid - 1i32) * N_B

        if raw_mtrx <= N_B && grid_mtrx_load <= N && raw_idx > 0
            padded_amount = (raw_idx - 1i32) ÷ pad_stride

            src_idx = (bid - 1i32) * N_B * D^2 + raw_idx
            dest_idx = raw_idx + padded_amount

            # if bid == 1
            #     @cuprintln(mod1(src_idx, 32), " ", mod1(tid, 32))
            # end

            shmem_A[dest_idx] = A[src_idx]
        end

        offset += nthreads_i32
    end

    offset = 0i32
    while offset < E_B + align_offset  # check
        raw_idx = offset + tid - align_offset
        raw_vec = div(raw_idx - 1i32, D) + 1i32
        grid_vec_load = raw_vec + (bid - 1i32) * N_B

        if raw_vec <= N_B && grid_vec_load <= N && raw_idx > 0i32
            src_idx = (bid - 1i32) * N_B * D + raw_idx
            dest_idx = raw_idx
            shmem_x[dest_idx] = x[src_idx]
        end

        offset += nthreads_i32
    end

    sync_threads()

    # if threadIdx().x == 1 && bid == 1
    #     CUDA.@cuprintln("$(shmem_A[2]), $(shmem_x[6])")
    # end

    # First transpose to make memory access more efficient
    # TODO: this might be more elegant if we use row-major
    offset = 0i32
    while offset < E_W  # check
        raw_idx = offset + lid
        if raw_idx <= E_W  # might be accessing unused shmem but that's fine
            warp_matrix_id = div(raw_idx - 1i32, D^2) + 1i32

            # raw_idx already includes warp_matrix_id
            logical_idx = (wid - 1i32) * N_W * D^2 + raw_idx
            padded_amount = (logical_idx - 1i32) ÷ pad_stride
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
            within_warp_matrix_idx = div(raw_idx - 1i32, D^2) + 1i32
            raw_row = mod1(within_matrix_idx, D)
            raw_col = div(within_matrix_idx - 1i32, D) + 1i32
            transposed_idx = (raw_row - 1i32) * D + raw_col
            logical_idx = (
                (wid - 1i32) * N_W * D^2 + (within_warp_matrix_idx - 1i32) * D^2 + transposed_idx
            )
            padding = (logical_idx - 1i32) ÷ pad_stride

            # if threadIdx().x == 1 && bid == 1
            #     CUDA.@cuprintln("raw_row: $raw_row, raw_col: $raw_col, transposed_idx: $transposed_idx, logical_idx: $logical_idx, padding: $padding")
            # end

            shmem_AT[logical_idx + padding] = Aij
        end

        offset += 32i32
    end

    # Only the first D * floor(32 / D) threads in each warp will compute the result
    if lid <= D * N_W
        v = 0.0f0
        for k in 1:D
            # Load vector element
            xk = (block_matrix_id - 1i32) * D + k

            logical_idx = (block_matrix_id - 1i32) * D^2 + (col_id - 1i32) * D + k
            padding = (logical_idx - 1i32) ÷ pad_stride

            v += shmem_AT[logical_idx + padding] * shmem_x[xk]
        end

        # Write result back to shared memory using coalesced writes
        shmem_y[(block_matrix_id - 1i32) * D + col_id] = v
    end

    # if threadIdx().x == 1 && bid == 1
    #     CUDA.@cuprintln("shmem_y: ", shmem_y[1], " block_matrix_id: $block_matrix_id, col_id: $col_id")
    # end

    sync_threads()

    offset = 0i32
    while offset < E_B + align_offset  # check
        raw_idx = offset + tid - align_offset
        raw_vec = div(raw_idx - 1i32, D) + 1i32
        grid_vec_load = raw_vec + (bid - 1i32) * N_B

        if raw_vec <= N_B && grid_vec_load <= N && raw_idx > 0i32
            src_idx = (bid - 1i32) * N_B * D + raw_idx
            dest_idx = raw_idx
            y[src_idx] = shmem_y[dest_idx]
        end

        offset += nthreads
    end

    return nothing
end

# N = Target 1GB with some noise
N = floor(Int, 1e9 / (D^2 * 4)) * 1 + 783
WARPS_PER_BLOCK = nthreads ÷ 32
# A = reshape(CuArray(Float32.(1:(D^2 * N))), D, D, N)
# B = reshape(CuArray(Float32.(1:(D^2 * N))), D, D, N)
A = CUDA.rand(Float32, D, D, N);
x = CUDA.rand(Float32, D, N);
y = CUDA.zeros(Float32, D, N);

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
Ē_B = Ē_W * WARPS_PER_BLOCK                   # elements per block with padding

CUDA.@sync @cuda threads = nthreads blocks = nblocks batched_matvec_kernel!(
    y,
    A,
    x,
    Val(Int32(N)),
    Val(Int32(D)),
    Val(Int32(N_W)),
    Val(Int32(N_B)),
    Val(Int32(E_W)),
    Val(Int32(E_B)),
    Val(Int32(P_W)),
    Val(Int32(Ē_W)),
    Val(Int32(Ē_B)),
    Val(Int32(pad_stride)),
    Val(Int32(pad_interval)),
);

y_truth = CUDA.zeros(Float32, D, N);
CUDA.CUBLAS.gemv_strided_batched!('N', 1.0f0, A, x, 0.0f0, y_truth)

println("Error: ", maximum(abs.(y - y_truth)))

registers = CUDA.registers(@cuda threads = nthreads blocks = nblocks batched_matvec_kernel!(
    y,
    A,
    x,
    Val(Int32(N)),
    Val(Int32(D)),
    Val(Int32(N_W)),
    Val(Int32(N_B)),
    Val(Int32(E_W)),
    Val(Int32(E_B)),
    Val(Int32(P_W)),
    Val(Int32(Ē_W)),
    Val(Int32(Ē_B)),
    Val(Int32(pad_stride)),
    Val(Int32(pad_interval)),
);)

println("Registers used: ", registers)

