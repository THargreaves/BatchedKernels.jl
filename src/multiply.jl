export batch_matvec, batch_matmul, batch_cov_update

function batch_matvec(A::CuArray{T,3}, b::CuArray{T,2}) where {T}
    D, N = size(A, 1), size(A, 3)
    c = CuArray{T}(undef, D, N)
    gemv_static_shmem_element!(c, A, b)
    return c
end

function gemv_static_shmem_element_kernel!(
    c, A, b, N::Int32, ::Val{D}, ::Val{M}
) where {D,M}
    tid = threadIdx().x
    local_matrix = div(tid - 1i32, D^2) + 1i32
    matrix_thread = mod1(tid, D^2)

    row = mod1(matrix_thread, D)
    col = div(matrix_thread - 1i32, D) + 1i32

    matrix_idx = (blockIdx().x - 1i32) * M + local_matrix

    # Shared memory for matrices and vectors in this block
    shmem_a = @cuStaticSharedMem(Float32, (D, D, M))
    shmem_b = @cuStaticSharedMem(Float32, (D, M))

    # Load matrix data
    if matrix_idx <= N
        shmem_a[row, col, local_matrix] = A[row, col, matrix_idx]
    end

    # Load vector data (only first M x D threads)
    if tid <= M * D && matrix_idx <= N
        vec_local_matrix = div(tid - 1i32, D) + 1i32
        vec_row = mod1(tid, D)
        matrix_idx_vec = (blockIdx().x - 1i32) * M + vec_local_matrix
        if matrix_idx_vec <= N
            shmem_b[vec_row, vec_local_matrix] = b[vec_row, matrix_idx_vec]
        end
    end

    sync_threads()

    # Compute (only first M x D threads)
    if tid <= M * D && matrix_idx <= N
        vec_local_matrix = div(tid - 1i32, D) + 1i32
        vec_row = mod1(tid, D)
        matrix_idx_vec = (blockIdx().x - 1i32) * M + vec_local_matrix

        if matrix_idx_vec <= N
            @inbounds begin
                # Compute dot product for this position
                result = 0.0f0
                for k in (1i32):D
                    result +=
                        shmem_a[vec_row, k, vec_local_matrix] * shmem_b[k, vec_local_matrix]
                end

                # Store result
                c[vec_row, matrix_idx_vec] = result
            end
        end
    end

    return nothing
end

function gemv_static_shmem_element!(
    c::CuArray{T,2}, A::CuArray{T,3}, b::CuArray{T,2}
) where {T}
    D, N = size(A, 1), size(A, 3)
    if D > 32
        error("Too many threads required for a $D x $D matrix")
    end
    matrices_per_block = div(1024, D^2)
    threads = matrices_per_block * D^2
    blocks = ceil(Int, N / matrices_per_block)
    shmem = (D^2 + D) * matrices_per_block * sizeof(T)

    @cuda blocks = blocks threads = threads gemv_static_shmem_element_kernel!(
        c, A, b, Int32(N), Val(Int32(D)), Val(Int32(matrices_per_block))
    )
end

function batch_matmul(
    A::CuArray{T,3}, B::CuArray{T,3}; A_trans=false, B_trans=false
) where {T}
    D, N = size(A, 1), size(A, 3)
    C = CuArray{T}(undef, D, D, N)
    gemm_trans_static_shmem_element!(C, A, B; A_trans, B_trans)
    return C
end

function gemm_trans_static_shmem_element_kernel!(
    C, A, B, N::Int32, ::Val{D}, ::Val{M}, ::Val{A_trans}, ::Val{B_trans}
) where {D,M,A_trans,B_trans}
    tid = threadIdx().x
    local_matrix = div(tid - 1i32, D^2) + 1i32
    matrix_thread = mod1(tid, D^2)

    row = mod1(matrix_thread, D)
    col = div(matrix_thread - 1i32, D) + 1i32

    matrix_idx = (blockIdx().x - 1i32) * M + local_matrix

    # Shared memory for all matrices in this block
    shmem_a = @cuStaticSharedMem(Float32, (D, D, M))
    shmem_b = @cuStaticSharedMem(Float32, (D, D, M))

    # Load data
    if matrix_idx <= N
        shmem_a[row, col, local_matrix] = A[row, col, matrix_idx]
        shmem_b[row, col, local_matrix] = B[row, col, matrix_idx]
    end

    sync_threads()

    # Compute
    if matrix_idx <= N
        @inbounds begin
            # Compute dot product for this position
            result = 0.0f0
            for k in (1i32):D
                if !A_trans & !B_trans
                    result += shmem_a[row, k, local_matrix] * shmem_b[k, col, local_matrix]
                elseif A_trans & !B_trans
                    result += shmem_a[k, row, local_matrix] * shmem_b[k, col, local_matrix]
                elseif !A_trans & B_trans
                    result += shmem_a[row, k, local_matrix] * shmem_b[col, k, local_matrix]
                elseif A_trans & B_trans
                    result += shmem_a[k, row, local_matrix] * shmem_b[col, k, local_matrix]
                end
            end

            # Store result
            C[row, col, matrix_idx] = result
        end
    end

    return nothing
end

function gemm_trans_static_shmem_element!(
    C::CuArray{T,3}, A::CuArray{T,3}, B::CuArray{T,3}; A_trans=false, B_trans=false
) where {T}
    D, N = size(A, 1), size(A, 3)
    if D > 32
        error("Too many threads required for a $D x $D matrix")
    end
    matrices_per_block = div(1024, D^2)
    threads = matrices_per_block * D^2
    blocks = ceil(Int, N / matrices_per_block)
    shmem = 2 * threads * sizeof(T)

    @cuda blocks = blocks threads = threads gemm_trans_static_shmem_element_kernel!(
        C,
        A,
        B,
        Int32(N),
        Val(Int32(D)),
        Val(Int32(matrices_per_block)),
        Val(A_trans),
        Val(B_trans),
    )
end

function batch_cov_update(A::CuArray{T,3}, P::CuArray{T,3}, Q::CuArray{T,3}) where {T}
    D, N = size(A, 1), size(A, 3)
    C = CuArray{T}(undef, D, D, N)
    cov_update_combined!(C, A, P, Q)
    return C
end

function cov_update_combined_kernel!(C, A, P, Q, N::Int32, ::Val{D}, ::Val{M}) where {D,M}
    tid = threadIdx().x
    local_matrix = div(tid - 1i32, D^2) + 1i32
    matrix_thread = mod1(tid, D^2)

    row = mod1(matrix_thread, D)
    col = div(matrix_thread - 1i32, D) + 1i32

    matrix_idx = (blockIdx().x - 1i32) * M + local_matrix

    # Shared memory for all matrices in this block
    shmem_a = @cuStaticSharedMem(Float32, (D, D, M))
    shmem_p = @cuStaticSharedMem(Float32, (D, D, M))
    shmem_tmp = @cuStaticSharedMem(Float32, (D, D, M))

    # Load data
    if matrix_idx <= N
        shmem_a[row, col, local_matrix] = A[row, col, matrix_idx]
        shmem_p[row, col, local_matrix] = P[row, col, matrix_idx]
    end

    sync_threads()

    # Compute AP into tmp
    if matrix_idx <= N
        begin
            # Compute dot product for this position
            result = 0.0f0
            for k in (1i32):D
                result += shmem_a[row, k, local_matrix] * shmem_p[k, col, local_matrix]
            end

            # Store result
            shmem_tmp[row, col, local_matrix] = result
        end
    end

    sync_threads()

    # Compute APA^T + Q into C
    if matrix_idx <= N
        begin
            # Compute dot product for this position
            result = 0.0f0
            for k in (1i32):D
                result += shmem_tmp[row, k, local_matrix] * shmem_a[col, k, local_matrix]
            end

            # Store result
            C[row, col, matrix_idx] = result + Q[row, col, matrix_idx]
        end
    end

    return nothing
end

function cov_update_combined!(
    C::CuArray{T,3}, A::CuArray{T,3}, P::CuArray{T,3}, Q::CuArray{T,3}
) where {T}
    D, N = size(A, 1), size(A, 3)
    if D > 32
        error("Too many threads required for a $D x $D matrix")
    end
    matrices_per_block = div(1024, D^2)
    threads = matrices_per_block * D^2
    blocks = ceil(Int, N / matrices_per_block)

    @cuda blocks = blocks threads = threads cov_update_combined_kernel!(
        C, A, P, Q, Int32(N), Val(Int32(D)), Val(Int32(matrices_per_block))
    )
end
