function batch_matmul(
    A::CuArray{T,3}, B::CuArray{T,3}; A_trans::Bool=false, B_trans::Bool=false
) where {T}
    C = CuArray{T}(undef, size(A, 1), size(B, 2), size(A, 3))
    return batch_matmul!(C, A, B; A_trans, B_trans)
end

function batch_matmul!(
    C::CuArray{T,3},
    A::CuArray{T,3},
    B::CuArray{T,3};
    A_trans::Bool=false,
    B_trans::Bool=false,
) where {T}
    # Validate input dimensions
    A_M, A_N, A_B = size(A)
    B_M, B_N, B_B = size(B)
    C_M, C_N, C_B = size(C)

    @assert A_N == B_M
    @assert A_M == C_M
    @assert B_N == C_N

    @assert A_B == B_B == C_B

    # Launch kernel with one thread per row, one thread-block per matrix
    @cuda threads = C_M blocks = C_B batch_matmul_kernel!(
        C, A, B, C_M, C_N, A_N, A_trans, B_trans
    )

    return nothing
end

function batch_matmul_kernel!(
    C::CuArray{T,3},
    A::CuArray{T,3},
    B::CuArray{T,3},
    C_M::Int,
    C_N::Int,
    A_N::Int,
    A_trans::Bool,
    B_trans::Bool,
) where {T}
    b = blockIdx().x
    i = threadIdx().x

    if i <= C_M
        @inbounds for j in 1:C_N
            sum = T(0.0)
            for k in 1:A_N
                A_elem = A_trans ? A[i, k, b] : A[k, i, b]
                B_elem = B_trans ? B[k, j, b] : B[j, k, b]
                sum += A_elem * B_elem
            end
            C[i, j, batch_idx] = sum
        end
    end

    return nothing
end
