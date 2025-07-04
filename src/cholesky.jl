export batch_cholesky, batch_trisolve

function batch_cholesky(A::CuArray{T,3}) where {T}
    D, N = size(A, 1), size(A, 3)
    L = CuArray{T}(undef, D, D, N)
    cholesky_static_shmem_element!(L, A)
    return L
end

function cholesky_static_shmem_element!(L::CuArray{T,3}, A::CuArray{T,3}) where {T}
    D, N = size(A, 1), size(A, 3)
    if D > 32
        error("Too many threads required for a $D x $D matrix")
    end
    matrices_per_block = div(1024, D^2)
    threads = matrices_per_block * D^2  # Keep same thread count for coalesced memory ops
    blocks = ceil(Int, N / matrices_per_block)

    @cuda blocks = blocks threads = threads cholesky_static_shmem_element_kernel!(
        L, A, Int32(N), Val(Int32(D)), Val(Int32(matrices_per_block))
    )
end

function cholesky_static_shmem_element_kernel!(
    L, A, N::Int32, ::Val{D}, ::Val{M}
) where {D,M}
    tid = threadIdx().x
    local_matrix = div(tid - 1i32, D^2) + 1i32
    matrix_thread = mod1(tid, D^2)  # Still needed for coalesced memory ops

    # These positions are used for coalesced memory operations
    row = mod1(matrix_thread, D)
    col = div(matrix_thread - 1i32, D) + 1i32

    matrix_idx = (blockIdx().x - 1i32) * M + local_matrix

    # Shared memory buffers
    shmem_a = @cuStaticSharedMem(Float32, (D, D, M))
    shmem_l = @cuStaticSharedMem(Float32, (D, D, M))

    # Coalesced load of input matrix A
    if matrix_idx <= N
        shmem_a[row, col, local_matrix] = A[row, col, matrix_idx]
        shmem_l[row, col, local_matrix] = 0.0f0
    end

    sync_threads()

    compute_matrix = (blockIdx().x - 1i32) * M + tid

    # Only the first M threads do computation (one per matrix)
    if compute_matrix <= N && tid <= M
        # Sequential Cholesky algorithm for this matrix
        for i in (1i32):D
            for j in (1i32):i
                sum = 0.0f0
                for k in (1i32):(j - 1)
                    sum += shmem_l[i, k, tid] * shmem_l[j, k, tid]
                end

                if i == j
                    shmem_l[i, j, tid] = sqrt(shmem_a[i, i, tid] - sum)
                else
                    shmem_l[i, j, tid] = (
                        1.0f0 / shmem_l[j, j, tid] * (shmem_a[i, j, tid] - sum)
                    )
                end
            end
        end
    end

    sync_threads()

    # All threads participate in coalesced write-back
    if matrix_idx <= N
        L[row, col, matrix_idx] = shmem_l[row, col, local_matrix]
    end

    return nothing
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
