using CUDA
using CUDA: i32
using BenchmarkTools
using BatchedKernels
using KernelAbstractions.Extras: @unroll
include("../config/Schedule.jl")

@inline function kernel_sqrt_kalman_pad!(
    Ss_out,  # all matrices are dim = D + 2
    Ss_in,
    A_glob,
    S_Q_glob,
    H_glob,
    S_R_glob,
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
    ::Val{:small},
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    Ddiv2 = D ÷ 2

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
    
    pad_interval = div(32i32, Ddiv2 & -Ddiv2) * Ddiv2
    shmem_size_fixed = Ddiv2 * Ddiv2 + (Ddiv2 * Ddiv2 - 1i32) ÷ pad_interval

    shmem_A = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32))
    shmem_S_Q = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + shmem_size_fixed * sizeof(Float32))
    shmem_H = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + 2 * shmem_size_fixed * sizeof(Float32))
    shmem_S_R = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + 3 * shmem_size_fixed * sizeof(Float32))

    # Load fixed matrices in parallel
    if wid == 1i32
        shared_matrix_load!(shmem_A, A_glob, Val(Ddiv2), Val(Ddiv2))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_S_Q, S_Q_glob, Val(Ddiv2), Val(Ddiv2))
    end
    if wid == 3i32
        shared_matrix_load!(shmem_H, H_glob, Val(Ddiv2), Val(Ddiv2))
    end
    if wid == 4i32
        shared_matrix_load!(shmem_S_R, S_R_glob, Val(Ddiv2), Val(Ddiv2))
    end
    sync_threads()

    B1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    B2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))

    A = SharedMatrix(shmem_A, Val(Ddiv2), Val(Ddiv2))
    S_Q = SharedMatrix(shmem_S_Q, Val(Ddiv2), Val(Ddiv2))
    H = SharedMatrix(shmem_H, Val(Ddiv2), Val(Ddiv2))
    S_R = SharedMatrix(shmem_S_R, Val(Ddiv2), Val(Ddiv2))

    # Load S (lower tri) shape = (Ddiv2,Ddiv2)
    intermediate_layout_load!(shmem_2, Ss_in, Val(Ddiv2), Val(Ddiv2), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(Ddiv2), Val(Ddiv2), Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        ######################
        #### PREDICT STEP ####
        ######################

        # X = A * S = A * B1
        batch_op!(*, B2, A, LowerTriangular(B1), d, Val(Ddiv2), Val(Ddiv2), Val(D), Val(:small))

        # Form predict pre-array in B1:
        # M_pred = [(AS)'; S_Q'] = [B2'; S_Q'] (2D x D)
        # Each thread is responsible for row d
        @inbounds @unroll for j in 1i32:Ddiv2
            # Store (AS)' in upper half, S_Q' in lower half
            if d <= Ddiv2
                B1[d, j] = B2[j, d]
            else
                B1[d, j] = S_Q[j, d - Ddiv2]
            end
        end
        # B1 contains M_pred

        # QR on M_pred
        batch_op!(qr, B2, B1, d, Val(D), Val(Ddiv2), Val(D), warp_matrix_id, Val(:small))
        # B2 contains R_pred

        # Y = H * S_pred = H * B2'
        batch_op!(*, B1, H, LowerTriangular(B2'), d, Val(Ddiv2), Val(Ddiv2), Val(D), Val(:small))
        # B1 contains Y

        # Form update pre-array:
        # M_upd =   [S_R'   0       ]
        #           [Y'     S_pred' ]
        # =
        #           [S_R'   - ]
        #           [B1'     B2]
        # Each thread is responsible for row d
        S_pred = UpperTriangular(B2)
        # Order important to allow in-place movement within B2 (upper left -> lower right)
        if d > Ddiv2
            @inbounds @unroll for j in D:(-1i32):1i32
                if j <= Ddiv2
                    B2[d, j] = B1[j, d - Ddiv2]
                else
                    B2[d, j] = S_pred[d - Ddiv2, j - Ddiv2]
                end
            end
        end
        if d <= Ddiv2
            @inbounds @unroll for j in D:(-1i32):1i32
                if j <= Ddiv2
                    B2[d, j] = S_R[j, d]
                else
                    B2[d, j] = 0.0f0
                end
            end
        end
        # B2 contains M_upd

        # Perform QR in-place on B1
        batch_op!(qr, B2, B2, d, Val(D), Val(D), Val(D), warp_matrix_id, Val(:small))
        # B2_{bottom right} = L_new' = R_22

        # Move R_22 to upper right half for standard format
        R = LowerTriangular(B2')
        if d <= Ddiv2
            @inbounds @unroll for j in 1i32:Ddiv2
                B1[d, j] = R[d + Ddiv2, j + Ddiv2]
            end
        end
        # B1 contains R_22' = L_new
    end

    sync_warp()

    # Write S_out (final output)
    dual_to_interm_transfer!(shmem_2, B1, Val(Ddiv2), Val(Ddiv2), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Ss_out, shmem_2, Val(Ddiv2), Val(Ddiv2), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

function sqrt_kalman_timing(Ss_out_cpu, Ss_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, _, ::Val, _, ::Val{:pad})
    Ddiv2, _, N = size(Ss_out_cpu)
    D = 2 * Ddiv2
    
    Ss_out = cu(Ss_out_cpu)
    Ss_in = cu(Ss_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    # Due to padding, need the nthreads for 2D
    nthreads = Schedule.best_nthreads("sqrt_kalman", D)
    
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    pad_interval = div(32, Ddiv2 & -Ddiv2) * Ddiv2

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1)
    shmem_elems = warp_shmem_size * n_warps
    shmem_size_fixed = Ddiv2 * Ddiv2 + (Ddiv2 * Ddiv2 - 1) ÷ pad_interval

    shmem_bytes = sizeof(Float32) * (
        2 * shmem_elems + 4 * shmem_size_fixed
    )

    kernel = @cuda launch=false kernel_sqrt_kalman_pad!(
        Ss_out, Ss_in, A, Q, H, R,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    bench_results = @benchmark begin
        CUDA.@sync $kernel(
            $Ss_out, $Ss_in, $A, $Q, $H, $R,
            Val(Int32($D)), Val(Int32($nthreads)), Int32($N),
            $Val(:small);
            threads=$nthreads, blocks=$nblocks, shmem=$shmem_bytes,
        )
    end

    return median(bench_results.times) / 1e9 / N
end