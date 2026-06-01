using BatchedKernels
using CUDA
using CUDA: i32
using LinearAlgebra

@inline function kernel_kalman!(
    Ps_out,
    Ps_in,
    A_global,
    Q_global,
    H_global,
    R_global,
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
    ::Val{:small},
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))
    shmem_3 = CuDynamicSharedArray(Float32, shmem_elems, 2 * shmem_elems * sizeof(Float32))

    pad_interval = div(32i32, D & -D) * D
    shmem_size_fixed = D * D + (D * D - 1i32) ÷ pad_interval

    shmem_A = CuDynamicSharedArray(Float32, shmem_size_fixed, 3 * shmem_elems * sizeof(Float32))
    shmem_Q = CuDynamicSharedArray(Float32, shmem_size_fixed, 3 * shmem_elems * sizeof(Float32) + shmem_size_fixed * sizeof(Float32))
    shmem_H = CuDynamicSharedArray(Float32, shmem_size_fixed, 3 * shmem_elems * sizeof(Float32) + 2 * shmem_size_fixed * sizeof(Float32))
    shmem_R = CuDynamicSharedArray(Float32, shmem_size_fixed, 3 * shmem_elems * sizeof(Float32) + 3 * shmem_size_fixed * sizeof(Float32))

    # Load fixed matrices in parallel
    if wid == 1i32
        shared_matrix_load!(shmem_A, A_global, Val(D), Val(D))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_Q, Q_global, Val(D), Val(D))
    end
    if wid == 3i32
        shared_matrix_load!(shmem_H, H_global, Val(D), Val(D))
    end
    if wid == 4i32
        shared_matrix_load!(shmem_R, R_global, Val(D), Val(D))
    end
    sync_threads()

    A = SharedMatrix(shmem_A, Val(D))
    Q = SharedMatrix(shmem_Q, Val(D))
    H = SharedMatrix(shmem_H, Val(D))
    R = SharedMatrix(shmem_R, Val(D))

    # Load P
    intermediate_layout_load!(shmem_1, Ps_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_3, shmem_1, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    B1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    B2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
    B3 = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        ######################
        #### PREDICT STEP ####
        ######################

        batch_op!(*, B2, A, B3, d, Val(D), Val(D), Val(D), Val(:small))
        batch_op!(*, B3, B2, A', d, Val(D), Val(D), Val(D), Val(:small))
        batch_op!(+, B1, B3, Q, d, Val(D), Val(D), Val(D), Val(:small))
        # B1 now contains P_pred. We keep this until the final update step.

        #####################
        #### KALMAN GAIN ####
        #####################

        # K = P_pred * H' / S where S = H * P_pred * H' + R
        # We compute K' = S^{-1} * H * P_pred, then K = K'

        # H * P_pred → B1
        batch_op!(*, B3, H, B1, d, Val(D), Val(D), Val(D), Val(:small))
        # B1 now contains H * P_pred = (P_pred * H')' since P_pred is symmetric

        # H * P_pred * H' → B2
        batch_op!(*, B2, B3, H', d, Val(D), Val(D), Val(D), Val(:small))

        # S = H*P_pred*H' + R → B2
        batch_op!(+, B2, B2, R, d, Val(D), Val(D), Val(D), Val(:small))

        # In-place Cholesky of S (B2 becomes U where S = U'*U)
        batch_op!(cholesky, B2, B2, d, Val(D), Val(D), warp_matrix_id, Val(:small))

        # In-place forward solve U' \ B1 → B1 (X = (U')^{-1} * H*P_pred)
        # We need U' for forward solve
        batch_op!(\, B3, LowerTriangular(B2'), B3, d, Val(D), Val(D), Val(D), Val(:small))
        # Backward solve U \ B1 → B1 (K' = U^{-1} * X = S^{-1} * H * P_pred)
        batch_op!(\, B3, UpperTriangular(B2), B3, d, Val(D), Val(D), Val(D), Val(:small))

        #####################
        #### UPDATE STEP ####
        #####################

        # Use P_new = (I - K*H) * P_pred form of update

        # Transpose K' in B2 to get K = P_pred * H' / S
        batch_op!(*, IAddSubSetterMatrix(B2, 1.0f0, -1.0f0), B3', H, d, Val(D), Val(D), Val(D), Val(:small))
        batch_op!(*, B3, B2, B1, d, Val(D), Val(D), Val(D), Val(:small))
        # B3 now contains P_new = (I - K*H) * P_pred
    end
    
    sync_warp()
    # Write P_new (final output) - B2/shmem_2 contains P_new
    dual_to_interm_transfer!(shmem_2, B3, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Ps_out, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end


function main(D::Int, n_warmups::Int, nthreads::Int)
    N = Int(ceil(1e9 / (4 * 2 * D^2)))
    T = Float32

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    P_out_cpu = zeros(T, D, D, N)
    P_in_cpu = zeros(T, D, D, N)
    for i in 1:N
        P_i = rand(T, D, D) / T(D)
        P_i = P_i * P_i' + 0.1f0 * I
        P_in_cpu[:, :, i] = P_i
    end

    A_cpu = rand(T, D, D) / Float32(D)

    Q_elem = rand(Float32, D, D) / Float32(D)^2
    Q_cpu = Q_elem * Q_elem' + 0.01f0 * I

    H_cpu = rand(Float32, D, D) / Float32(D)
    R_elem = rand(Float32, D, D) / Float32(D)^2
    R_cpu = R_elem * R_elem' + 0.01f0 * I

    P_out = cu(P_out_cpu)
    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    shmem_elems = let
        n_mats_per_warp = 32 ÷ D
        n_warps = nthreads ÷ 32
        dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        warp_shmem_size * n_warps
    end
    shmem_size_fixed = let
        pad_interval = div(32, D & -D) * D
        D * D + (D * D - 1) ÷ pad_interval
    end
    shmem_bytes = (3 * shmem_elems + 4 * shmem_size_fixed) * sizeof(T)

    kernel = @cuda launch=false kernel_kalman!(
        P_out, P_in, A, Q, H, R,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    # Warm-up
    for _ in 1:n_warmups
        CUDA.@sync kernel(
            P_out, P_in, A, Q, H, R,
            Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small);
            threads = nthreads, blocks = nblocks, shmem = shmem_bytes,
        )
    end
    CUDA.synchronize()

    # Profile
    CUDA.@sync kernel(
        P_out, P_in, A, Q, H, R,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small);
        threads = nthreads, blocks = nblocks, shmem = shmem_bytes,
    )
    CUDA.synchronize()
end


D = parse(Int, ARGS[1])
n_warmups = parse(Int, ARGS[2])
nthreads = parse(Int, ARGS[3])
main(D, n_warmups, nthreads)