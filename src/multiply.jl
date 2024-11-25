export batch_matmul, batch_matmul!

function batch_matmul(
    A::CuArray{T,3}, B::CuArray{T,3}; A_trans::Bool=false, B_trans::Bool=false
) where {T}
    C_M = A_trans ? size(A, 2) : size(A, 1)
    C_N = B_trans ? size(B, 1) : size(B, 2)
    C = CuArray{T}(undef, C_M, C_N, size(A, 3))
    batch_matmul!(C, A, B; A_trans, B_trans)
    return C
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

    @assert (A_trans ? A_M : A_N) == (B_trans ? B_N : B_M)
    @assert C_M == (A_trans ? A_N : A_M)
    @assert C_N == (B_trans ? B_M : B_N)

    @assert A_B == B_B == C_B

    # Launch kernel with one thread per row, one thread-block per matrix
    M, N, P = C_M, (A_trans ? A_M : A_N), C_N
    @cuda threads = A_M blocks = A_B batch_matmul_kernel!(
        C, A, B, M, N, P, A_trans, B_trans
    )

    return nothing
end

function batch_matmul_kernel!(
    C::CuDeviceArray{T,3},
    A::CuDeviceArray{T,3},
    B::CuDeviceArray{T,3},
    M::Int,
    N::Int,
    P::Int,
    A_trans::Bool,
    B_trans::Bool,
) where {T}
    b = blockIdx().x
    i = threadIdx().x

    if i <= M
        @inbounds for j in 1:P
            sum = T(0.0)
            for k in 1:N
                A_elem = A_trans ? A[k, i, b] : A[i, k, b]
                B_elem = B_trans ? B[j, k, b] : B[k, j, b]
                sum += A_elem * B_elem
            end
            C[i, j, b] = sum
        end
    end

    return nothing
end
