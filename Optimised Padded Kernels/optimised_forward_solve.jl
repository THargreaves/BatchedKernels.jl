using BenchmarkTools
using CUDA
using Unroll
using StaticArrays
using LinearAlgebra

const nthreads = 192
D = 8

# Use transpose of colmun-Cholesky from CS554 to compute upper triangular with parallel
# access patterns
function batched_forward_solve_kernel!(
    Y,
    L,
    B,
    ::Val{N},
    ::Val{D},
    ::Val{N_W},
    ::Val{N_B},
    ::Val{E_W},
    ::Val{E_B},
    ::Val{P_W},
    ::Val{Ē_W},
    ::Val{Ē_B},
    ::Val{pad_stride},
    ::Val{pad_interval},
) where {D,N,N_W,N_B,E_W,E_B,P_W,Ē_W,Ē_B,pad_stride,pad_interval}
    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32)                           # lane id
    wid = div(tid - 1, 32) + 1                    # warp id

    warp_matrix_id = div(lid - 1, D) + 1
    block_matrix_id = warp_matrix_id + (wid - 1) * N_W
    grid_matrix_id = block_matrix_id + (bid - 1) * N_B

    col_id = mod1(lid, D)

    shmem_L = CuStaticSharedArray(Float32, (Ē_B,))
    shmem_B = CuStaticSharedArray(Float32, (Ē_B,))
    shmem_Y = CuStaticSharedArray(Float32, (Ē_B,))

    # Load matrix into shared memory using coalesced reads
    base_addr = (bid - 1) * N_B * D^2 + 1
    align_offset = (base_addr - 1) % 32

    begin
        offset = 0
        while offset < E_B + align_offset  # check
            raw_idx = offset + tid - align_offset
            raw_mtrx = div(raw_idx - 1, D^2) + 1
            grid_mtrx_load = raw_mtrx + (bid - 1) * N_B

            if raw_mtrx <= N_B && grid_mtrx_load <= N && raw_idx > 0
                padded_amount = (raw_idx - 1) ÷ pad_stride

                src_idx = (bid - 1) * N_B * D^2 + raw_idx
                dest_idx = raw_idx + padded_amount

                shmem_L[dest_idx] = L[src_idx]
                shmem_B[dest_idx] = B[src_idx]
            end

            offset += nthreads
        end

        sync_threads()

        j = col_id

        # Only the first D * floor(32 / D) threads in each warp will compute the result
        if lid <= D * N_W
            # column of Y this thread is responsible for — store in registers for efficiency
            y = @MVector zeros(Float32, D)

            for i in 1:D
                master_logical_idx = (block_matrix_id - 1) * D^2 + (col_id - 1) * D + i
                master_padding = (master_logical_idx - 1) ÷ pad_stride
                y[i] = shmem_B[master_logical_idx + master_padding]

                for j in 1:(i - 1)
                    # All threads in this matrix read same value from shmem
                    logical_idx = (block_matrix_id - 1) * D^2 + (j - 1) * D + i
                    padding = (logical_idx - 1) ÷ pad_stride
                    l = shmem_L[logical_idx + padding]

                    y[i] -= l * y[j]
                end

                # Finally read the diagonal element and divide
                # TODO: could read these across threads, compute inverse in parallel then
                # sync shuffle. Div is 8–32 cycles
                logical_idx = (block_matrix_id - 1) * D^2 + (i - 1) * D + i
                padding = (logical_idx - 1) ÷ pad_stride
                l = shmem_L[logical_idx + padding]
                y[i] /= l

                # Write register back to shared memory - resuse master idx
                # TODO: confirm this is best to do now so shmem access is staggered
                shmem_Y[master_logical_idx + master_padding] = y[i]
            end
        end

        sync_threads()

        # Write result back to shared memory using coalesced writes
        offset = 0
        while offset < E_B + align_offset
            raw_idx = offset + tid - align_offset
            raw_mtrx = div(raw_idx - 1, D^2) + 1
            grid_mtrx_store = raw_mtrx + (bid - 1) * N_B

            if raw_mtrx <= N_B && grid_mtrx_store <= N && raw_idx > 0
                padded_amount = (raw_idx - 1) ÷ pad_stride

                dest_idx = (bid - 1) * N_B * D^2 + raw_idx
                src_idx = raw_idx + padded_amount

                Y[dest_idx] = shmem_Y[src_idx]
            end

            offset += nthreads
        end
    end

    return nothing
end

# N = Target 1GB with some noise
N = floor(Int, 1e9 / (D^2 * 4)) * 1 + 783
WARPS_PER_BLOCK = nthreads ÷ 32
Ls = [rand(Float32, D, D) + I for _ in 1:N];
Bs = [rand(Float32, D, D) for _ in 1:N];
L = cu(stack(Ls));
B = cu(stack(Bs));
Y = CUDA.zeros(Float32, D, D, N);

Ys = [tril(Ls[i]) \ Bs[i] for i in 1:N];
Y_truth = cu(stack(Ys));

matrices_per_warp = floor(Int, 32 / D)
matrices_per_block = matrices_per_warp * WARPS_PER_BLOCK
nblocks = ceil(Int, N / matrices_per_block)

# TODO: this can be made smaller in the non-division case. I.e. remove stride all together
pad_interval = div(32, gcd(32, D))
pad_stride = pad_interval * D
N_W = floor(Int, 32 / D)                      # matrices per warp
N_B = N_W * WARPS_PER_BLOCK                                # matrices per block
E_W = D^2 * N_W                               # elements per warp
E_B = D^2 * N_B                               # elements per block
P_W = ceil(Int, E_W / pad_interval)
Ē_W = E_W + P_W                               # elements per warp with padding
Ē_B = Ē_W * WARPS_PER_BLOCK                                # elements per block with padding

CUDA.@sync @cuda threads = nthreads blocks = nblocks batched_forward_solve_kernel!(
    Y,
    L,
    B,
    Val(N),
    Val(D),
    Val(N_W),
    Val(N_B),
    Val(E_W),
    Val(E_B),
    Val(P_W),
    Val(Ē_W),
    Val(Ē_B),
    Val(pad_stride),
    Val(pad_interval),
);

# Validate
# TODO: this fails when L is just random. Likely due to how singularities are handled
# println("Error: ", maximum(abs.(Y - Y_truth)))

using Magma
Magma.LibMagma.magma_init()
queue = Magma.LibMagma.magma_queue_t
queue_ptr = Ref{Magma.LibMagma.magma_queue_t}()
device = 0  # or get from CUDA context
Magma.LibMagma.magma_queue_create_internal(
    device,
    queue_ptr,
    C_NULL,  # func
    C_NULL,  # file
    0,        # line
)

B_copy = deepcopy(B);
B_ptrs = CUDA.CUBLAS.unsafe_strided_batch(B_copy);
L_ptrs = CUDA.CUBLAS.unsafe_strided_batch(L);

# Error in magmablas_strsm_inv_batched, cannot allocate memory on GPU device (info = -113)
# ccall(
#     (:magmablas_strsm_inv_batched, Magma.LibMagma.libmagma),
#     Cvoid,
#     (
#         Magma.LibMagma.magma_side_t,
#         Magma.LibMagma.magma_uplo_t,
#         Magma.LibMagma.magma_trans_t,
#         Magma.LibMagma.magma_diag_t,
#         Cint,
#         Cint,
#         Cfloat,
#         CuPtr{CuPtr{Float32}},
#         Cint,
#         CuPtr{CuPtr{Float32}},
#         Cint,
#         Cint,
#         Magma.LibMagma.magma_queue_t,
#     ),
#     Magma.LibMagma.MagmaLeft,
#     Magma.LibMagma.MagmaLower,
#     Magma.LibMagma.MagmaNoTrans,
#     Magma.LibMagma.MagmaNonUnit,
#     D, 
#     D,
#     1.0f0,
#     L_ptrs,
#     D,
#     B_ptrs,
#     D,
#     N,
#     queue_ptr[]
# )
