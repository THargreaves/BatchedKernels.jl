# ======================================================================
# tune_nthreads/ops/sqrt_kalman.jl
#
# Per-operation adapter for the nthreads tuning pipeline. See
# ops/gauss_likelihood.jl for the contract. Everything except
# kernel_sqrt_kalman! is exported by BatchedKernels; the kernel itself
# is pasted in below.
#
# NOTE: sqrt_kalman has an extra compile-time parameter THRESH (the
# register-vs-shared cutoff for the update QR's R_BR vector). The
# profile script fixes THRESH = 10; this adapter does the same.
# ======================================================================

using BatchedKernels
using CUDA
using CUDA: i32
using LinearAlgebra

const THRESH = 10        # register-vs-shared cutoff, matches profile script

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
        shared_matrix_load!(shmem_A, A_glob, Val(D))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_S_Q, S_Q_glob, Val(D))
    end
    if wid == 3i32
        shared_matrix_load!(shmem_H, H_glob, Val(D))
    end
    if wid == 4i32
        shared_matrix_load!(shmem_S_R, S_R_glob, Val(D))
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
    dual_to_interm_transfer!(shmem_2, B1', Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Ss_out, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

# ----------------------------------------------------------------------
# end paste region
# ----------------------------------------------------------------------

# shared-memory bytes -- sqrt_kalman shares kalman's shared layout
# (3 per-warp buffers + 4 fixed buffers). Same formula as the profile.
function _shmem_bytes(D::Int, nthreads::Int)
    n_mats_per_warp  = 32 ÷ D
    n_warps          = nthreads ÷ 32
    dual_padding     = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    warp_shmem_size  = n_mats_per_warp * D * D + dual_padding * (D - 1)
    shmem_elems      = warp_shmem_size * n_warps
    pad_interval     = div(32, D & -D) * D
    shmem_size_fixed = D * D + (D * D - 1) ÷ pad_interval
    return sizeof(Float32) * (3 * shmem_elems + 4 * shmem_size_fixed)
end

# allocate the batch ONCE per (op, D). Mirrors the profile main():
# the per-element input is the Cholesky FACTOR S = chol(P).L, not P.
function setup_inputs(D::Integer)
    N = Int(ceil(1e9 / (4 * 2 * D^2)))      # batch_size_sqrt_kalman(D)
    T = Float32

    S_in_cpu = Array{Float32}(undef, D, D, N)
    for i in 1:N
        P_i = rand(Float32, D, D) / Float32(D)
        P_i = P_i * P_i' + 0.1f0 * I                       # SPD
        S_in_cpu[:, :, i] = Float32.(Matrix(cholesky(P_i).L))
    end
    S_out_cpu = zeros(Float32, D, D, N)

    A_cpu = rand(T, D, D) / Float32(D)
    Q_elem = rand(Float32, D, D) / Float32(D)^2
    Q_cpu  = Q_elem * Q_elem' + 0.01f0 * I
    H_cpu  = rand(Float32, D, D) / Float32(D)
    R_elem = rand(Float32, D, D) / Float32(D)^2
    R_cpu  = R_elem * R_elem' + 0.01f0 * I

    return (; D = D, N = Int32(N),
              Ss_out = cu(S_out_cpu), Ss_in = cu(S_in_cpu),
              A = cu(A_cpu), Q = cu(Q_cpu), H = cu(H_cpu), R = cu(R_cpu))
end

# compile the kernel for a SPECIFIC nthreads, set the carveout.
function make_kernel(D::Integer, nthreads::Integer, st)
    N       = st.N
    nblocks = cld(Int(N), nthreads ÷ 32 * (32 ÷ D))
    shmem_b = _shmem_bytes(Int(D), Int(nthreads))

    kernel = @cuda launch=false kernel_sqrt_kalman!(
        st.Ss_out, st.Ss_in, st.A, st.Q, st.H, st.R,
        Val(Int32(D)), Val(Int32(THRESH)), Val(Int32(nthreads)), N,
        Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun,
        CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_b)

    cfg = (; threads = nthreads, blocks = nblocks, shmem = shmem_b,
             D = D, N = N)
    return kernel, cfg
end

# the timed region: launch + synchronize, nothing else.
function run_kernel(kernel, cfg, st)
    CUDA.@sync kernel(
        st.Ss_out, st.Ss_in, st.A, st.Q, st.H, st.R,
        Val(Int32(cfg.D)), Val(Int32(THRESH)),
        Val(Int32(cfg.threads)), cfg.N, Val(:small);
        threads = cfg.threads, blocks = cfg.blocks, shmem = cfg.shmem,
    )
    return nothing
end
