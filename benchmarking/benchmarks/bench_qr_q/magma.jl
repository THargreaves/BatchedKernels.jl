using CUDA
using CUDA: i32
using BenchmarkTools
using LinearAlgebra

# ─── ccall wrappers ──────────────────────────────────────────────────
#
# Three-step pipeline to materialise Q:
#   1. magma_sgeqrf_batched_smallsq   factorise A → packed (R + V), τ
#   2. magma_slarft_batched           form block-T from V, τ
#   3. magma_slarfb_gemm_batched      apply (I − V T Vᵀ) to identity → Q
#
# MAGMA 2.9 has no `*orgqr_batched`, so steps 2–3 do the assembly manually.
# Step 1b (between geqrf and slarft) builds an explicit V buffer: MAGMA's
# slarfb_gemm requires V's diagonal to be explicit 1's and upper triangle
# explicit 0's, not LAPACK's implicit-1 convention.
#
# Using the smallsq fast path for D ≤ 32 (we target D = 2..16). Note the
# signature has offset args (Ai, Aj, taui) so it can be reused as a panel
# kernel inside blocked routines; for a standalone batched QR they're all 0.

function magma_sgeqrf_batched_smallsq!(
    n::Integer,
    dA_array,
    Ai::Integer,
    Aj::Integer,
    ldda::Integer,
    dtau_array,
    taui::Integer,
    info_array,
    batchCount::Integer,
    queue::Magma.LibMagma.magma_queue_t,
)
    return ccall(
        (:magma_sgeqrf_batched_smallsq, Magma.LibMagma.libmagma),
        Magma.LibMagma.magma_int_t,
        (
            Magma.LibMagma.magma_int_t,           # n
            CuPtr{CuPtr{Cfloat}},                 # dA_array
            Magma.LibMagma.magma_int_t,           # Ai
            Magma.LibMagma.magma_int_t,           # Aj
            Magma.LibMagma.magma_int_t,           # ldda
            CuPtr{CuPtr{Cfloat}},                 # dtau_array
            Magma.LibMagma.magma_int_t,           # taui
            CuPtr{Magma.LibMagma.magma_int_t},    # info_array
            Magma.LibMagma.magma_int_t,           # batchCount
            Magma.LibMagma.magma_queue_t,
        ),
        n,
        dA_array, Ai, Aj, ldda,
        dtau_array, taui,
        info_array, batchCount, queue,
    )
end

function magma_slarft_batched!(
    n::Integer, k::Integer, stair_T::Integer,
    v_array, ldv::Integer,
    tau_array,
    T_array, ldt::Integer,
    work_array, lwork::Integer,
    batchCount::Integer,
    queue::Magma.LibMagma.magma_queue_t,
)
    return ccall(
        (:magma_slarft_batched, Magma.LibMagma.libmagma),
        Magma.LibMagma.magma_int_t,
        (
            Magma.LibMagma.magma_int_t,    # n
            Magma.LibMagma.magma_int_t,    # k
            Magma.LibMagma.magma_int_t,    # stair_T
            CuPtr{CuPtr{Cfloat}},          # v_array
            Magma.LibMagma.magma_int_t,    # ldv
            CuPtr{CuPtr{Cfloat}},          # tau_array
            CuPtr{CuPtr{Cfloat}},          # T_array
            Magma.LibMagma.magma_int_t,    # ldt
            CuPtr{CuPtr{Cfloat}},          # work_array
            Magma.LibMagma.magma_int_t,    # lwork
            Magma.LibMagma.magma_int_t,    # batchCount
            Magma.LibMagma.magma_queue_t,
        ),
        n, k, stair_T,
        v_array, ldv,
        tau_array,
        T_array, ldt,
        work_array, lwork,
        batchCount, queue,
    )
end

function magma_slarfb_gemm_batched!(
    side::Magma.LibMagma.magma_side_t,
    trans::Magma.LibMagma.magma_trans_t,
    direct::Magma.LibMagma.magma_direct_t,
    storev::Magma.LibMagma.magma_storev_t,
    m::Integer, n::Integer, k::Integer,
    dV_array, lddv::Integer,
    dT_array, lddt::Integer,
    dC_array, lddc::Integer,
    dwork_array, ldwork::Integer,
    dworkvt_array, ldworkvt::Integer,
    batchCount::Integer,
    queue::Magma.LibMagma.magma_queue_t,
)
    return ccall(
        (:magma_slarfb_gemm_batched, Magma.LibMagma.libmagma),
        Magma.LibMagma.magma_int_t,
        (
            Magma.LibMagma.magma_side_t,
            Magma.LibMagma.magma_trans_t,
            Magma.LibMagma.magma_direct_t,
            Magma.LibMagma.magma_storev_t,
            Magma.LibMagma.magma_int_t,    # m
            Magma.LibMagma.magma_int_t,    # n
            Magma.LibMagma.magma_int_t,    # k
            CuPtr{CuPtr{Cfloat}},          # dV_array
            Magma.LibMagma.magma_int_t,    # lddv
            CuPtr{CuPtr{Cfloat}},          # dT_array
            Magma.LibMagma.magma_int_t,    # lddt
            CuPtr{CuPtr{Cfloat}},          # dC_array
            Magma.LibMagma.magma_int_t,    # lddc
            CuPtr{CuPtr{Cfloat}},          # dwork_array
            Magma.LibMagma.magma_int_t,    # ldwork
            CuPtr{CuPtr{Cfloat}},          # dworkvt_array
            Magma.LibMagma.magma_int_t,    # ldworkvt
            Magma.LibMagma.magma_int_t,    # batchCount
            Magma.LibMagma.magma_queue_t,
        ),
        side, trans, direct, storev,
        m, n, k,
        dV_array, lddv,
        dT_array, lddt,
        dC_array, lddc,
        dwork_array, ldwork,
        dworkvt_array, ldworkvt,
        batchCount, queue,
    )
end

# ─── Identity-batch helper ───────────────────────────────────────────
#
# `dC_array` for step 3 must start as a batch of D×D identity matrices.
# We set diagonals via a small kernel after fill!(0).

function _set_diag_kernel!(Qs, D::Int32, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        d_zero = mod(idx - 1i32, D)
        batch_zero = div(idx - 1i32, D)
        @inbounds Qs[d_zero + 1i32, d_zero + 1i32, batch_zero + 1i32] = 1.0f0
    end
    return
end

function _init_identity_batch!(Qs::CuArray{Float32, 3}, D::Int, N::Int)
    fill!(Qs, 0f0)
    nthreads = 256
    total = D * N
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _set_diag_kernel!(Qs, Int32(D), Int32(total))
end

# ─── Explicit-V builder ──────────────────────────────────────────────
#
# After geqrf, dA holds R in the upper triangle (incl. diagonal) and v values
# in the strict lower triangle. LAPACK's slarft/slarfb routines treat V's
# diagonal as an implicit 1, but MAGMA's `slarfb_gemm_batched` documents:
# "All elements including 0's and 1's are stored, unlike LAPACK." So we must
# build a separate V where the diagonal is an explicit 1 and the upper
# triangle is explicit 0. Lower triangle is copied from dA.

function _build_v_explicit_kernel!(V, A, D::Int32, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        DD = D * D
        elem_in_batch = mod(idx - 1i32, DD)
        col_zero = div(elem_in_batch, D)
        row_zero = mod(elem_in_batch, D)
        @inbounds begin
            if row_zero == col_zero
                V[idx] = 1.0f0                   # explicit 1 on diagonal
            elseif row_zero > col_zero
                V[idx] = A[idx]                  # v values copied from A
            else
                V[idx] = 0.0f0                   # explicit 0 above diagonal
            end
        end
    end
    return
end

function _build_v_explicit!(V::CuArray{Float32, 3}, A::CuArray{Float32, 3}, D::Int, N::Int)
    nthreads = 256
    total = D * D * N
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _build_v_explicit_kernel!(V, A, Int32(D), Int32(total))
end

# ─── Main qr_q ───────────────────────────────────────────────────────

function qr_q_magma!(
    dA_array, dV_array, dtau_array, dT_array, dwslarft_array,
    dC_array, dwork_array, dworkvt_array,
    A_storage, V_storage, Qs,
    tau_storage, T_storage, wslarft_storage,
    work_storage, workvt_storage,
    info_d, D, N, queue_ptr,
)
    LEFT = Magma.LibMagma.MagmaLeft
    NT   = Magma.LibMagma.MagmaNoTrans
    FWD  = Magma.LibMagma.MagmaForward
    COLW = Magma.LibMagma.MagmaColumnwise

    # GC.@preserve keeps every pointer-array target alive across MAGMA calls.
    # Same rationale as qr_r: pointer-arrays hold raw device pointers; Julia's
    # GC doesn't follow them, and BenchmarkTools' inter-sample gcscrub will
    # free unreferenced storages otherwise.
    GC.@preserve A_storage V_storage Qs tau_storage T_storage wslarft_storage work_storage workvt_storage begin
        # Step 1: factorise. R lives in the upper triangle of dA, Householder
        # reflectors V live below the diagonal, τ scalars in dtau.
        # Ai = Aj = taui = 0: factor the whole matrix (no sub-block offset).
        magma_sgeqrf_batched_smallsq!(
            D, dA_array, 0, 0, D, dtau_array, 0, info_d, N, queue_ptr[],
        )

        # Step 1b: build the V buffer that slarfb_gemm expects. MAGMA's
        # slarfb_gemm requires explicit 1's on the diagonal and 0's above
        # (unlike LAPACK's implicit-1 convention); see the source docstring
        # for magma_slarfb_gemm_batched. The custom kernel runs on the CUDA
        # default stream, so we sync MAGMA's queue before (so geqrf's writes
        # to A are visible) and the default stream after (so V is visible to
        # the next MAGMA call).
        Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
        _build_v_explicit!(V_storage, A_storage, D, N)
        CUDA.synchronize()

        # Step 2: form the block-T matrix from V and τ.
        # lwork = D*D satisfies MAGMA's validation; work buffer itself is only
        # touched in the (k > nb) branch which never fires at D ≤ 16.
        magma_slarft_batched!(
            D, D, 0,
            dV_array, D,
            dtau_array,
            dT_array, D,
            dwslarft_array, D * D,
            N, queue_ptr[],
        )

        # Step 3: apply (I − V T Vᵀ) to the identity batch.
        # With C = I, the result is Q. Overwrites dC in-place.
        magma_slarfb_gemm_batched!(
            LEFT, NT, FWD, COLW,
            D, D, D,
            dV_array, D,
            dT_array, D,
            dC_array, D,
            dwork_array, D,
            dworkvt_array, D,
            N, queue_ptr[],
        )

        Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
        CUDA.synchronize()
    end
end

function qr_q_timing(Qs_cpu, As_cpu, queue_ptr, _, ::Val{:magma})
    D, _, N = size(Qs_cpu)

    # Reference buffers used to refresh A_storage and Qs each iteration.
    As = cu(As_cpu)
    identity_batch = CUDA.zeros(Float32, D, D, N)
    _init_identity_batch!(identity_batch, D, N)

    # Mutable buffers (overwritten by the pipeline each iter).
    A_storage = cu(As_cpu)
    Qs = cu(Qs_cpu)

    # Per-batch scratch storages.
    V_storage       = CUDA.zeros(Float32, D, D, N)      # V in explicit form (built each iter)
    tau_storage     = CUDA.zeros(Float32, D, N)         # τ scalars
    T_storage       = CUDA.zeros(Float32, D, D, N)      # block-T
    wslarft_storage = CUDA.zeros(Float32, D, D, N)      # slarft work (dead at our D)
    work_storage    = CUDA.zeros(Float32, D, D, N)      # slarfb W = VᵀC
    workvt_storage  = CUDA.zeros(Float32, D, D, N)      # slarfb W2 = V T

    # Batched pointer arrays (host-built once, valid for the whole benchmark).
    dA_array       = CUDA.CUBLAS.unsafe_strided_batch(A_storage)
    dV_array       = CUDA.CUBLAS.unsafe_strided_batch(V_storage)
    dtau_array     = CUDA.CUBLAS.unsafe_strided_batch(tau_storage)
    dT_array       = CUDA.CUBLAS.unsafe_strided_batch(T_storage)
    dwslarft_array = CUDA.CUBLAS.unsafe_strided_batch(wslarft_storage)
    dC_array       = CUDA.CUBLAS.unsafe_strided_batch(Qs)
    dwork_array    = CUDA.CUBLAS.unsafe_strided_batch(work_storage)
    dworkvt_array  = CUDA.CUBLAS.unsafe_strided_batch(workvt_storage)

    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    bench_results = @benchmark begin
        qr_q_magma!(
            $dA_array, $dV_array, $dtau_array, $dT_array, $dwslarft_array,
            $dC_array, $dwork_array, $dworkvt_array,
            $A_storage, $V_storage, $Qs,
            $tau_storage, $T_storage, $wslarft_storage,
            $work_storage, $workvt_storage,
            $info_d, $D, $N, $queue_ptr,
        )
    end setup=begin
        copyto!($A_storage, $As)             # fresh A for geqrf
        copyto!($Qs, $identity_batch)        # fresh I for slarfb
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
