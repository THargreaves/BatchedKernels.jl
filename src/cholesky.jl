export batched_cholesky!, batch_trisolve

function batched_cholesky_kernel!(
    U, A, ::Val{D}, ::Val{nthreads}, N::Int32
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

    # Define shared memory, to be modified in-place
    shmem_elems = n_elements_per_block + padding_per_warp * n_warps
    shmem_A = CuStaticSharedArray(Float32, (shmem_elems,))

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
            end

            offset += nthreads
        end

        sync_threads()

        # Perform computation
        # Only the first D * floor(32 / D) threads in each warp will compute the result
        # Use transpose of column-Cholesky from CS554 to compute upper triangular with parallel
        # access patterns
        if lid <= D * n_mats_per_warp
            j = matrix_thread
            for i in 1:D
                # Mask needs to remove j < i else they will never reach the sync line
                mask = ((1 << (D - (i - 1))) - 1) << (i - 1)
                mask = mask << ((warp_matrix_id - 1) * D)

                if j >= i
                    # Load in value from i row
                    logical_idx = (block_matrix_id - 1) * D^2 + (j - 1) * D + i
                    padding = (logical_idx - 1) ÷ pad_stride
                    Ai = shmem_A[logical_idx + padding]
                    # RMOD STEP
                    for k in 1:(i - 1)
                        # Load in value from k row
                        logical_idx = (block_matrix_id - 1) * D^2 + (j - 1) * D + k
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
                    logical_idx = (block_matrix_id - 1) * D^2 + (j - 1) * D + i
                    padding = (logical_idx - 1) ÷ pad_stride
                    shmem_A[logical_idx + padding] = Ai
                end
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

                U[dest_idx] = shmem_A[src_idx]
            end

            offset += nthreads
        end
    end

    return nothing
end

function batched_cholesky!(U::CuArray{T,3}, A::CuArray{T,3}; nthreads::Int=256) where {T}
    # Validate dimensions
    A_m, A_n, A_b = size(A)
    U_m, U_n, U_b = size(U)

    # Batch sizes must match
    if A_b != U_b
        throw(ArgumentError("Batch sizes of A and U must match."))
    end

    # Matrix dimensions must match
    if A_m != A_n || U_m != A_m || U_n != A_n
        throw(ArgumentError("A must be square and U must match A's dimensions."))
    end

    nthreads % 32 == 0 ||
        throw(ArgumentError("Number of threads must be a multiple of 32."))

    D = A_m
    N = A_b
    warps_per_block = nthreads ÷ 32
    matrices_per_warp = div(32, D)
    matrices_per_block = warps_per_block * matrices_per_warp
    nblocks = cld(N, matrices_per_block)
    @cuda threads = nthreads blocks = nblocks batched_cholesky_kernel!(
        U, A, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
    )

    return U
end

function batch_trisolve(L::CuArray{T,3}, B::CuArray{T,3}) where {T}
    D, N = size(L, 1), size(L, 3)
    X = CuArray{T}(undef, D, D, N)
    potrs_static_shmem_element!(X, L, B)
    return X
end

function potrs_static_shmem_element!(
    X::CuArray{T,3}, L::CuArray{T,3}, B::CuArray{T,3}
) where {T}
    D, N = size(L, 1), size(L, 3)
    if D > 32
        error("Too many threads required for a $D x $D matrix")
    end
    matrices_per_block = div(1024, D^2)
    threads = matrices_per_block * D^2
    blocks = ceil(Int, N / matrices_per_block)

    @cuda blocks = blocks threads = threads potrs_kernel!(
        X, L, B, Int32(N), Val(Int32(D)), Val(Int32(matrices_per_block))
    )
end

function potrs_kernel!(
    X::CuDeviceArray{T,3},
    L::CuDeviceArray{T,3},
    B::CuDeviceArray{T,3},
    N::Int32,
    ::Val{D},
    ::Val{M},
) where {T,D,M}
    tid = threadIdx().x
    local_matrix = div(tid - 1i32, D^2) + 1i32
    matrix_thread = mod1(tid, D^2)  # For coalesced memory ops

    # These positions are used for coalesced memory operations
    row = mod1(matrix_thread, D)
    col = div(matrix_thread - 1i32, D) + 1i32
    matrix_idx = (blockIdx().x - 1i32) * M + local_matrix

    # Shared memory for the current matrices we're working with
    # shmem_l stores our Cholesky factor L
    # shmem_b initially stores B, will later store our final result
    # shmem_x stores intermediate results from forward substitution
    shmem_l = @cuStaticSharedMem(Float32, (D, D, M))
    shmem_b = @cuStaticSharedMem(Float32, (D, D, M))
    shmem_x = @cuStaticSharedMem(Float32, (D, D, M))

    # Coalesced load of input matrices L and B
    if matrix_idx <= N
        shmem_l[row, col, local_matrix] = L[row, col, matrix_idx]
        shmem_b[row, col, local_matrix] = B[row, col, matrix_idx]
        shmem_x[row, col, local_matrix] = 0.0f0
    end

    sync_threads()

    # Compute matrix and column for this thread
    compute_matrix = div(tid - 1i32, D) + 1i32
    compute_col = mod1(tid, D)

    # Only the first M*D threads do computation (one column per thread)
    if compute_matrix <= N && tid <= (M * D)
        # Step 1: Forward substitution (Ly = b)
        # Results stored in shmem_x
        for i in (1i32):D
            sum = 0.0f0
            for j in (1i32):(i - 1)
                sum +=
                    shmem_l[i, j, compute_matrix] * shmem_x[j, compute_col, compute_matrix]
            end
            shmem_x[i, compute_col, compute_matrix] = (
                1.0f0 / shmem_l[i, i, compute_matrix] *
                (shmem_b[i, compute_col, compute_matrix] - sum)
            )
        end
    end

    # We need a sync here to ensure all threads have completed 
    # forward substitution before starting backward substitution
    sync_threads()

    # Reset shmem_b to zeros
    if matrix_idx <= N
        shmem_b[row, col, local_matrix] = 0.0f0
    end
    sync_threads()

    if compute_matrix <= N && tid <= (M * D)
        # Step 2: Backward substitution (L'x = y)
        # Results stored in shmem_b (reusing space since we don't need B anymore)
        for i in D:-1:(1i32)
            sum = 0.0f0
            for j in (i + 1):D
                sum +=
                    shmem_l[j, i, compute_matrix] * shmem_b[j, compute_col, compute_matrix]
            end
            if i == D
                # First iteration - copy from shmem_x
                shmem_b[i, compute_col, compute_matrix] = (
                    1.0f0 / shmem_l[i, i, compute_matrix] *
                    shmem_x[i, compute_col, compute_matrix]
                )
            else
                shmem_b[i, compute_col, compute_matrix] = (
                    1.0f0 / shmem_l[i, i, compute_matrix] *
                    (shmem_x[i, compute_col, compute_matrix] - sum)
                )
            end
        end
    end

    sync_threads()

    # All threads participate in coalesced write-back of final result
    if matrix_idx <= N
        X[row, col, matrix_idx] = shmem_b[row, col, local_matrix]
    end

    return nothing
end

function batch_trisolve(L::CuArray{T,3}, b::CuArray{T,2}) where {T}
    D, N = size(L, 1), size(L, 3)
    X = CuArray{T}(undef, D, N)
    potrs_static_shmem_element!(X, L, b)
    return X
end

function potrs_static_shmem_element!(
    X::CuArray{T,2}, L::CuArray{T,3}, b::CuArray{T,2}
) where {T}
    D, N = size(L, 1), size(L, 3)
    if D > 32
        error("Too many threads required for a $D x $D matrix")
    end
    matrices_per_block = div(1024, D^2)
    threads = matrices_per_block * D^2
    blocks = ceil(Int, N / matrices_per_block)

    @cuda blocks = blocks threads = threads potrs_kernel!(
        X, L, b, Int32(N), Val(Int32(D)), Val(Int32(matrices_per_block))
    )
end

function potrs_kernel!(
    X::CuDeviceArray{T,2},
    L::CuDeviceArray{T,3},
    B::CuDeviceArray{T,2},
    N::Int32,
    ::Val{D},
    ::Val{M},
) where {T,D,M}
    tid = threadIdx().x
    local_matrix = div(tid - 1i32, D^2) + 1i32
    matrix_thread = mod1(tid, D^2)  # For coalesced memory ops

    # These positions are used for coalesced memory operations
    row = mod1(matrix_thread, D)
    col = div(matrix_thread - 1i32, D) + 1i32
    matrix_idx = (blockIdx().x - 1i32) * M + local_matrix

    # Shared memory for the current matrices we're working with
    # shmem_l stores our Cholesky factor L
    # shmem_b initially stores B, will later store our final result
    # shmem_x stores intermediate results from forward substitution
    shmem_l = @cuStaticSharedMem(Float32, (D, D, M))
    shmem_b = @cuStaticSharedMem(Float32, (D, M))
    shmem_x = @cuStaticSharedMem(Float32, (D, M))

    # Coalesced load of input matrices L and B
    if matrix_idx <= N
        shmem_l[row, col, local_matrix] = L[row, col, matrix_idx]
        if col == 1
            shmem_b[row, local_matrix] = B[row, matrix_idx]
            shmem_x[row, local_matrix] = 0.0f0
        end
    end

    sync_threads()

    compute_matrix = (blockIdx().x - 1i32) * M + tid

    # Only the first M*D threads do computation (one column per thread)
    if compute_matrix <= N && tid <= M
        # Step 1: Forward substitution (Ly = b)
        # Results stored in shmem_x
        for i in (1i32):D
            sum = 0.0f0
            for j in (1i32):(i - 1)
                sum += shmem_l[i, j, tid] * shmem_x[j, tid]
            end
            shmem_x[i, tid] = (1.0f0 / shmem_l[i, i, tid] * (shmem_b[i, tid] - sum))
        end
    end

    # We need a sync here to ensure all threads have completed 
    # forward substitution before starting backward substitution
    sync_threads()

    # Reset shmem_b to zeros
    if matrix_idx <= N && col == 1
        shmem_b[row, local_matrix] = 0.0f0
    end
    sync_threads()

    if compute_matrix <= N && tid <= M
        # Step 2: Backward substitution (L'x = y)
        # Results stored in shmem_b (reusing space since we don't need B anymore)
        for i in D:-1:(1i32)
            sum = 0.0f0
            for j in (i + 1):D
                sum += shmem_l[j, i, tid] * shmem_b[j, tid]
            end
            if i == D
                # First iteration - copy from shmem_x
                shmem_b[i, tid] = (1.0f0 / shmem_l[i, i, tid] * shmem_x[i, tid])
            else
                shmem_b[i, tid] = (1.0f0 / shmem_l[i, i, tid] * (shmem_x[i, tid] - sum))
            end
        end
    end

    sync_threads()

    # All threads participate in coalesced write-back of final result
    if matrix_idx <= N && col == 1
        X[row, matrix_idx] = shmem_b[row, local_matrix]
    end

    return nothing
end
