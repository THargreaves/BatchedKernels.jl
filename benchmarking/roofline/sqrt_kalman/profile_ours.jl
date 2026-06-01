using CUDA
using CUDA: i32
using LinearAlgebra
using BatchedKernels

@inline function kernel_sqrt_kalman!(
    Ss_out,
    Ss_in,
    A_glob,
    S_Q_glob,
    H_glob,
    S_R_glob,
    ::Val{D},
    ::Val{THRESH},
    ::Val{nthreads},
    N::Int32,
    ::Val{:small},
) where {D,THRESH,nthreads}
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

    pad_interval = div(32i32, D & -D) * D
    shmem_size_fixed = D * D + (D * D - 1i32) ÷ pad_interval

    shmem_A = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32))
    shmem_S_Q = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + shmem_size_fixed * sizeof(Float32))
    shmem_H = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + 2 * shmem_size_fixed * sizeof(Float32))
    shmem_S_R = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + 3 * shmem_size_fixed * sizeof(Float32))

    # Load fixed matrices in parallel
    if wid == 1i32
        shared_matrix_load!(shmem_A, A_glob, Val(D), Val(D))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_S_Q, S_Q_glob, Val(D), Val(D))
    end
    if wid == 3i32
        shared_matrix_load!(shmem_H, H_glob, Val(D), Val(D))
    end
    if wid == 4i32
        shared_matrix_load!(shmem_S_R, S_R_glob, Val(D), Val(D))
    end
    sync_threads()

    B1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    B2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))

    A = SharedMatrix(shmem_A, Val(D), Val(D))
    S_Q = SharedMatrix(shmem_S_Q, Val(D), Val(D))
    H = SharedMatrix(shmem_H, Val(D), Val(D))
    S_R = SharedMatrix(shmem_S_R, Val(D), Val(D))

    # Load S (lower tri)
    intermediate_layout_load!(shmem_2, Ss_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        ######################
        #### PREDICT STEP ####
        ######################

        # X = A * S = A * B1
        batch_op!(*, B2, A, LowerTriangular(B1), d, Val(D), Val(D), Val(D), Val(:small))

        # Form predict pre-array:
        # M_pred = [(AS)'; S_Q'] = [B2'; S_Q'] (2D x D)
        # QR of M_pred -> R, stored in B1
        M_pred = BlockMatrix_2_1(B2', S_Q', Val(D), warp_matrix_id, Val(:small))

        batch_op!(qr, B1, M_pred, d, Val(D), Val(2), Val(1), warp_matrix_id, Val(:small))
        # B1 = R = U_pred (upper tri)

        # Y = H * S_pred = H * B1'
        batch_op!(*, B2, H, LowerTriangular(B1'), d, Val(D), Val(D), Val(D), Val(:small))

        # Form update pre-array:
        # M_upd =   [S_R'   0       ]
        #           [Y'     S_pred' ]
        # =
        #           [S_R'   - ]
        #           [Y'     B1]
        M_upd = BlockMatrixLowerTrig_2_2(S_R', B2', UpperTriangular(B1), Val(D), warp_matrix_id, Val(:small))
        
        batch_op!(qr, B1, M_upd, d, Val(D), Val(THRESH), Val(2), Val(2), warp_matrix_id, Val(:small))
        # B1' = R_22' = L_new
    end

    sync_warp()

    # Write S_out (final output)
    dual_to_interm_transfer!(shmem_2, LowerTriangular(B1'), Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Ss_out, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end


function main(D::Int, n_warmups::Int, nthreads::Int)
    N = Int(ceil(1e9 / (4 * 2 * D^2)))
    T = Float32
    THRESH = 10

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    S_in_cpu = Array{Float32}(undef, D, D, N)
    for i in 1:N
        P_i = rand(Float32, D, D) / Float32(D)
        P_i = P_i * P_i' + 0.1f0 * I
        S_in_cpu[:, :, i] = Float32.(Matrix(cholesky(P_i).L))
    end
    S_out_cpu = zeros(Float32, D, D, N)

    A_cpu = rand(T, D, D) / Float32(D)

    Q_elem = rand(Float32, D, D) / Float32(D)^2
    Q_cpu = Q_elem * Q_elem' + 0.01f0 * I

    H_cpu = rand(Float32, D, D) / Float32(D)
    R_elem = rand(Float32, D, D) / Float32(D)^2
    R_cpu = R_elem * R_elem' + 0.01f0 * I

    Ss_out = cu(S_out_cpu)
    Ss_in = cu(S_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    shmem_elems = let
        n_mats_per_warp = 32 ÷ D
        n_warps = nthreads ÷ 32
        dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1)
        warp_shmem_size * n_warps
    end
    shmem_size_fixed = let
        pad_interval = div(32, D & -D) * D
        D * D + (D * D - 1) ÷ pad_interval
    end
    shmem_bytes = (2 * shmem_elems + 4 * shmem_size_fixed) * sizeof(T)

    kernel = @cuda launch=false kernel_sqrt_kalman!(
        Ss_out, Ss_in, A, Q, H, R,
        Val(Int32(D)), Val(Int32(THRESH)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    # Warm-up
    for _ in 1:n_warmups
        CUDA.@sync kernel(
            Ss_out, Ss_in, A, Q, H, R,
            Val(Int32(D)), Val(Int32(THRESH)), Val(Int32(nthreads)), Int32(N),
            Val(:small);
            threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
        )
    end
    CUDA.synchronize()

    # Profile
    CUDA.@sync kernel(
        Ss_out, Ss_in, A, Q, H, R,
        Val(Int32(D)), Val(Int32(THRESH)), Val(Int32(nthreads)), Int32(N),
        Val(:small);
        threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
    )
    CUDA.synchronize()
end


D = parse(Int, ARGS[1])
n_warmups = parse(Int, ARGS[2])
nthreads = parse(Int, ARGS[3])
main(D, n_warmups, nthreads)