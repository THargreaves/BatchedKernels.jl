# ======================================================================
# tune_nthreads/ops/gauss_likelihood.jl
#
# Per-operation adapter for the nthreads tuning pipeline. Provides three
# functions that benchmark_nthreads.jl calls:
#
#   setup_inputs(D)               -> state   (device arrays, allocated once)
#   make_kernel(D, nthreads, st)  -> (kernel, cfg)   (compiled per nthreads)
#   run_kernel(kernel, cfg, st)               (the timed launch + sync)
#
# Everything except the top-level kernel_gauss_likelihood! is exported by
# BatchedKernels. The kernel itself is pasted in below (research folder --
# a local copy is fine and keeps the tuning pipeline self-contained).
# ======================================================================

using BatchedKernels
using CUDA
using CUDA: i32
using Random
using LinearAlgebra

function kernel_gauss_likelihood!(
    ps_out,
    xs_in,
    µs_in,
    Σs_in,
    ::Val{D1},
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
) where {D1,D,nthreads}
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
    block_matrix_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp
    grid_mtrx_id = block_matrix_id + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))

    shmem_vec_elems = D * n_warps * n_mats_per_warp
    shmem_vec_1 = CuDynamicSharedArray(Float32, shmem_vec_elems, 2 * shmem_elems * sizeof(Float32))
    shmem_vec_2 = CuDynamicSharedArray(Float32, shmem_vec_elems, (2 * shmem_elems + shmem_vec_elems) * sizeof(Float32))

    # Load x and µ
    vector_load!(shmem_vec_1, xs_in, Val(D1), Val(D), Val(nthreads), N)
    vector_load!(shmem_vec_2, µs_in, Val(D1), Val(D), Val(nthreads), N)

    # Load Σ
    intermediate_layout_load!(shmem_2, Σs_in, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))

    M1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    v1 = BatchedVector(shmem_vec_1, Val(D), warp_matrix_id)
    v2 = BatchedVector(shmem_vec_2, Val(D), warp_matrix_id)

    log_prob = 0.0f0
    active = warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
    if active
        # δ = x - µ -> v1
        batch_op!(-, v1, v1, v2, d, Val(D1), Val(D1), Val(D), Val(:small))

        # cholesky(Σ).U -> M1
        batch_op!(cholesky, M1, M1, d, Val(D1), Val(D), warp_matrix_id, Val(:small))

        # log det of Σ -> v2
        log_det = batch_op!(Val(:log_det), UpperTriangular(M1), d, Val(D1), Val(D), warp_matrix_id, Val(:small))
        # Leader threads will have log_det in their registers, others will have garbage

        # y = L \ δ -> v1
        batch_op!(\, v1, LowerTriangular(M1'), v1, d, Val(D1), Val(D1), Val(D), warp_matrix_id, Val(:small))

        # mahal = |y|^2
        mahal_dist = batch_op!(Val(:mahal_dist), v1, d, Val(D1), Val(D), warp_matrix_id, Val(:small))
        # Leader threads will have mahal in their registers

        log_prob = -0.5f0 * (D1 * log(Float32(2π)) + log_det + mahal_dist)        
    end    

    sync_threads()

    scalar_stage!(shmem_vec_1, log_prob, lid, warp_matrix_id, block_matrix_id, active, Val(D))
    
    sync_threads()

    # v1 -> global
    scalar_write!(ps_out, shmem_vec_1, n_mats_per_block)

    return nothing
end

# ----------------------------------------------------------------------
# end paste region
# ----------------------------------------------------------------------

# shared-memory bytes for the dynamic-shared launch attribute / shmem arg.
# Same formula the profile script uses; scales with nthreads.
function _shmem_bytes(D::Int, nthreads::Int)
    n_mats_per_warp = 32 ÷ D
    n_warps         = nthreads ÷ 32
    dual_padding    = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1)
    shmem_elems     = warp_shmem_size * n_warps
    shmem_vec_elems = D * n_warps * n_mats_per_warp
    return sizeof(Float32) * (2 * shmem_elems + 2 * shmem_vec_elems)
end

# allocate the batch ONCE per (op, D). Mirrors the profile main().
function setup_inputs(D::Integer)
    Random.seed!(1234)
    N = Int(ceil(1e9 / (4 * 1 * D^2)))      # batch_size_gauss_likelihood(D)

    xs_in_cpu  = rand(Float32, D, N)
    µs_in_cpu  = rand(Float32, D, N)
    ps_out_cpu = zeros(Float32, N)

    Σs_in_cpu = Array{Float32}(undef, D, D, N)
    for i in 1:N
        Σ_i = rand(Float32, D, D) / Float32(D)
        Σ_i = Σ_i * Σ_i' + 0.1f0 * I          # SPD
        Σs_in_cpu[:, :, i] = Σ_i
    end

    return (; D = D, N = Int32(N),
              ps_out = cu(ps_out_cpu), xs_in = cu(xs_in_cpu),
              µs_in  = cu(µs_in_cpu),  Σs_in = cu(Σs_in_cpu))
end

# compile the kernel for a SPECIFIC nthreads (Val{nthreads} is a
# compile-time type parameter, so each nthreads needs its own compile),
# set the large-shared-memory carveout, return kernel + launch config.
function make_kernel(D::Integer, nthreads::Integer, st)
    N        = st.N
    nblocks  = cld(Int(N), nthreads ÷ 32 * (32 ÷ D))
    shmem_b  = _shmem_bytes(Int(D), Int(nthreads))

    kernel = @cuda launch=false kernel_gauss_likelihood!(
        st.ps_out, st.xs_in, st.µs_in, st.Σs_in,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(nthreads)), N,
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
        st.ps_out, st.xs_in, st.µs_in, st.Σs_in,
        Val(Int32(cfg.D)), Val(Int32(cfg.D)),
        Val(Int32(cfg.threads)), cfg.N;
        threads = cfg.threads, blocks = cfg.blocks, shmem = cfg.shmem,
    )
    return nothing
end
