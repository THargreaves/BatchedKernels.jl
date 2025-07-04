export batched_matmul!

function batched_matmul_kernel!(
    C, A, B, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    # Computed derived constants
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    n_elements_per_warp = n_mats_per_warp * D^2
    n_elements_per_block = n_mats_per_block * D^2

    # Access thread indices
    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)                # lane id
    wid = div(tid - 1i32, 32i32) + 1i32   # warp id

    # Calculate responsibility for this thread
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    block_matrix_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp
    matrix_thread = mod1(lid, D)  # thread within each matrix operation

    # Define padding and stride
    # Equivalent to div(32, gcd(32, D)) (for D < 64) but avoids using gcd which prevents constant propagation
    # TODO: Would generated functions be a cleaner approach?
    pad_interval = div(32i32, D & -D)
    pad_stride = pad_interval * D
    padding_per_warp = cld(n_elements_per_warp, pad_interval)

    # Define shared memory 
    shmem_elems = n_elements_per_block + padding_per_warp * n_warps
    shmem_A = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_B = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_C = CuStaticSharedArray(Float32, (shmem_elems,))

    begin
        # Load matrices into shared memory using coalesced reads
        # Warps cooperate and cross matrix boundaries to mask latency

        # Offset global memory addresses so threads are naturally aligned
        # TODO: the additional indexing logic might not be worth the small savings in L1 cache use
        base_addr = (bid - 1i32) * n_mats_per_block * D^2 + 1i32
        align_offset = (base_addr - 1i32) % 32i32

        offset = 0i32
        while offset < n_elements_per_block + align_offset
            raw_idx = offset + tid - align_offset  # force thread 1 address == 1 (mod1 32)
            raw_mtrx = div(raw_idx - 1i32, D^2) + 1i32
            grid_mtrx_load = raw_mtrx + (bid - 1i32) * n_mats_per_block

            if raw_mtrx <= n_mats_per_block && grid_mtrx_load <= N && raw_idx > 0i32
                padded_amount = (raw_idx - 1i32) ÷ pad_stride

                src_idx = (bid - 1i32) * n_mats_per_block * D^2 + raw_idx
                dest_idx = raw_idx + padded_amount

                shmem_A[dest_idx] = A[src_idx]
                shmem_B[dest_idx] = B[src_idx]
            end

            offset += nthreads
        end

        sync_threads()

        # Perform computation
        # Only the first D * floor(32 / D) threads in each warp will compute the result
        if lid <= D * n_mats_per_warp

            # Load column of B into registers
            col = matrix_thread
            B_col = @MVector zeros(Float32, Int64(D))
            for row in (1i32):D
                logical_idx = (block_matrix_id - 1i32) * D^2 + (col - 1i32) * D + row
                padding = (logical_idx - 1i32) ÷ pad_stride
                B_col[row] = shmem_B[logical_idx + padding]
            end

            # Perform matrix multiplication
            for row in (1i32):D
                C_val = 0.0f0

                # Each thread loads the value from its column
                logical_idx = (block_matrix_id - 1i32) * D^2 + (col - 1i32) * D + row
                padding = (logical_idx - 1i32) ÷ pad_stride
                local_v = shmem_A[logical_idx + padding]

                # Multiply and accumulate
                for k in (1i32):D
                    # Share value with other threads in warp if this one loaded it
                    mask = (UInt32(1) << D) - UInt32(1)
                    mask = mask << ((warp_matrix_id - 1i32) * D)
                    v = shfl_sync(mask, local_v, k + (warp_matrix_id - 1i32) * D)

                    C_val += v * B_col[k]
                end

                # Store result in shared memory
                shmem_C[logical_idx + padding] = C_val
            end
        end

        sync_threads()

        # Write result back to shared memory using coalesced writes
        # Again, warps cooperate and cross matrix boundaries to mask latency
        offset = 0i32
        while offset < n_elements_per_block + align_offset
            raw_idx = offset + tid - align_offset
            raw_mtrx = div(raw_idx - 1i32, D^2) + 1i32
            grid_mtrx_store = raw_mtrx + (bid - 1i32) * n_mats_per_block

            if raw_mtrx <= n_mats_per_block && grid_mtrx_store <= N && raw_idx > 0i32
                padded_amount = (raw_idx - 1i32) ÷ pad_stride

                dest_idx = (bid - 1i32) * n_mats_per_block * D^2 + raw_idx
                src_idx = raw_idx + padded_amount

                C[dest_idx] = shmem_C[src_idx]
            end

            offset += nthreads
        end
    end

    return nothing
end

function batched_matmul!(
    C::CuArray{Float32,3}, A::CuArray{Float32,3}, B::CuArray{Float32,3}; nthreads::Int=256
)
    # Validate dimensions
    A_m, A_n, A_b = size(A)
    B_m, B_n, B_b = size(B)
    C_m, C_n, C_b = size(C)

    # Batch sizes must match
    if A_b != B_b || A_b != C_b
        throw(ArgumentError("Batch sizes of A, B, and C must match."))
    end

    # Matrix dimensions must match
    if A_n != B_m || C_m != A_m || C_n != B_n
        throw(ArgumentError("Matrix dimensions do not match for multiplication."))
    end

    # Only support square matrices for now
    if A_m != A_n || B_m != B_n || C_m != C_n
        throw(ArgumentError("Only square matrices are currently supported."))
    end

    nthreads % 32 == 0 ||
        throw(ArgumentError("Number of threads must be a multiple of 32."))

    D = A_m
    N = A_b
    warps_per_block = nthreads ÷ 32
    matrices_per_warp = div(32, D)
    matrices_per_block = matrices_per_warp * warps_per_block
    nblocks = cld(N, matrices_per_block)
    @cuda threads = nthreads blocks = nblocks batched_matmul_kernel!(
        C, A, B, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
    )

    return C
end
