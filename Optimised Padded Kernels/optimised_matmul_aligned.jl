using BenchmarkTools
using CUDA
using Unroll
using StaticArrays

## !!!!
# Can we remove the shuffle sync by computing outer product and using that no bank conflict
# occurs when all threads read the same value?

# const nthreads = 256
const nthreads = 192
D = 5

# Matches smallsq up to D = 10 (maybe with N = 192)
# Shared memory becomes the bottleneck for D = 12
# Errors at D = 16

# Uncoalesced shared memory access reported for D = 10 (not D = 5 though). Weird

function batched_matmul_kernel!(
    C,
    A,
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

    shmem_A = CuStaticSharedArray(Float32, (Ē_B,))
    shmem_B = CuStaticSharedArray(Float32, (Ē_B,))
    shmem_C = CuStaticSharedArray(Float32, (Ē_B,))

    # Load matrices into shared memory using coalesced reads
    base_addr = (bid - 1) * N_B * D^2 + 1
    align_offset = (base_addr - 1) % 32

    @inbounds begin
        offset = 0
        while offset < E_B + align_offset  # check
            raw_idx = offset + tid - align_offset
            raw_mtrx = div(raw_idx - 1, D^2) + 1
            grid_mtrx_load = raw_mtrx + (bid - 1) * N_B

            if raw_mtrx <= N_B && grid_mtrx_load <= N && raw_idx > 0
                padded_amount = (raw_idx - 1) ÷ pad_stride

                src_idx = (bid - 1) * N_B * D^2 + raw_idx
                dest_idx = raw_idx + padded_amount

                # if bid == 1
                #     @cuprintln(mod1(src_idx, 32), " ", mod1(tid, 32))
                # end

                shmem_A[dest_idx] = A[src_idx]
                shmem_B[dest_idx] = B[src_idx]
            end

            offset += nthreads
        end

        sync_threads()

        # Debug
        # if tid == 1 && bid == 16
        #     @cuprintln(shmem_A[23])
        # end

        # if bid == 16
        #     @cuprintln(shmem_A[tid])
        # end

        # Only the first D * floor(32 / D) threads in each warp will compute the result
        if lid <= D * N_W

            # Load column of B into registers
            B_col = @MVector zeros(Float32, D)
            @unroll for row in 1:D
                logical_idx = (block_matrix_id - 1) * D^2 + (col_id - 1) * D + row
                padding = (logical_idx - 1) ÷ pad_stride
                B_col[row] = shmem_B[logical_idx + padding]
            end

            # Perform matrix multiplication
            @unroll for row in 1:D
                C_val = 0.0f0

                # Each thread loads the value from its column
                logical_idx = (block_matrix_id - 1) * D^2 + (col_id - 1) * D + row
                padding = (logical_idx - 1) ÷ pad_stride
                local_v = shmem_A[logical_idx + padding]

                # Multiply and accumulate
                for k in 1:D
                    # Share value with other threads in warp if this one was responsible
                    mask = (1 << D) - 1
                    mask = mask << ((warp_matrix_id - 1) * D)
                    v = shfl_sync(mask, local_v, k + (warp_matrix_id - 1) * D)

                    # if tid == 2 && bid == 1
                    #     @cuprintln(v, " ", B_col[k])
                    # end
                    C_val += v * B_col[k]
                end

                # Store result in shared memory
                shmem_C[logical_idx + padding] = C_val
            end
        end

        sync_threads()

        # Write result back to shared memory using coalesced writes
        offset = 0
        while offset < E_B + align_offset
            raw_idx = offset + tid - align_offset
            raw_mtrx = div(raw_idx - 1, D^2) + 1
            grid_mtrx_store = raw_mtrx + (bid - 1) * N_B

            # if bid == 16 && tid == 385
            #     @cuprintln("Thread C: ", shmem_C[385 + 12])
            # end

            if raw_mtrx <= N_B && grid_mtrx_store <= N && raw_idx > 0
                padded_amount = (raw_idx - 1) ÷ pad_stride

                dest_idx = (bid - 1) * N_B * D^2 + raw_idx
                src_idx = raw_idx + padded_amount

                C[dest_idx] = shmem_C[src_idx]
            end

            offset += nthreads
        end

        # Debug
        # if (368 <= tid <= 400)  && bid == 16
        #     @cuprintln(tid, ": ", shmem_A[tid], " ", shmem_B[tid], " ", shmem_C[tid])
        # end

        # if tid == 1 && bid == 16
        #     # @cuprintln("A: ", shmem_A[397])
        #     @cuprintln(A[1, 1, 1945])
        #     @cuprintln("A: ", shmem_A[385 + 12])
        #     @cuprintln("C: ", shmem_C[385 + 12])
        #     @cuprintln(C[1, 1, 1945])
        # end

    end

    return nothing
end

# N = Target 1GB with some noise
N = floor(Int, 1e9 / (D^2 * 4)) * 1 + 783
WARPS_PER_BLOCK = nthreads ÷ 32
# A = reshape(CuArray(Float32.(1:(D^2 * N))), D, D, N)
# B = reshape(CuArray(Float32.(1:(D^2 * N))), D, D, N)
A = CUDA.rand(Float32, D, D, N);
B = CUDA.rand(Float32, D, D, N);
C = CuArray{Float32}(undef, D, D, N);

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

CUDA.@sync @cuda threads = nthreads blocks = nblocks batched_matmul_kernel!(
    C,
    A,
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

# println("Error: ", maximum(abs.(C - CUDA.CUBLAS.gemm_strided_batched('N', 'N', A, B))))

# Compare to theoretical bound
bytes = 3 * D^2 * N * sizeof(Float32)
bandwidth = 1008 * 1e9  # B/s
theoretical_bound = bytes / bandwidth * 1e6  # μs
println("Theoretical bound: ", theoretical_bound, " μs")

# display(
#     @benchmark CUDA.@sync @cuda threads = nthreads blocks = nblocks batched_matmul_kernel!(
#         $C,
#         $A,
#         $B,
#         Val($N),
#         Val($D),
#         Val($N_W),
#         Val($N_B),
#         Val($E_W),
#         Val($E_B),
#         Val($P_W),
#         Val($Ē_W),
#         Val($Ē_B),
#         Val($pad_stride),
#         Val($pad_interval),
#     );
# )

# @benchmark CUDA.@sync CUDA.CUBLAS.gemm_strided_batched!('N', 'N', 0.0f0, A, B, 1.0f0, C)

### CONTIGUOUS BENCHMARKS ###

C_ptrs = CUDA.CUBLAS.unsafe_strided_batch(C);
A_ptrs = CUDA.CUBLAS.unsafe_strided_batch(A);
B_ptrs = CUDA.CUBLAS.unsafe_strided_batch(B);

# display(
#     @benchmark CUDA.@sync CUDA.CUBLAS.cublasSgemmStridedBatched_64(
#         CUDA.CUBLAS.handle(),
#         'N',
#         'N',
#         $D,
#         $D,
#         $D,
#         1.0f0,
#         $A_ptrs,
#         $D,
#         $D^2,
#         $B_ptrs,
#         $D,
#         $D^2,
#         0.0f0,
#         $C_ptrs,
#         $D,
#         $D^2,
#         $N,
#     );
# )

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

# ```c
# void magma_sgemm_batched( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, float alpha, float const * const * dA_array, magma_int_t ldda, float const * const * dB_array, magma_int_t lddb, float beta, float **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
# ```

# ccall(
#     (:magma_sgemm_batched, Magma.LibMagma.libmagma),
#     Cvoid,
#     (
#         Magma.LibMagma.magma_trans_t,
#         Magma.LibMagma.magma_trans_t,
#         Magma.LibMagma.magma_int_t,
#         Magma.LibMagma.magma_int_t,
#         Magma.LibMagma.magma_int_t,
#         Cfloat,
#         CuPtr{CuPtr{Cfloat}},
#         Magma.LibMagma.magma_int_t,
#         CuPtr{CuPtr{Cfloat}},
#         Magma.LibMagma.magma_int_t,
#         Cfloat,
#         CuPtr{CuPtr{Cfloat}},
#         Magma.LibMagma.magma_int_t,
#         Magma.LibMagma.magma_int_t,
#         Magma.LibMagma.magma_queue_t,
#     ),
#     Magma.LibMagma.MagmaNoTrans,
#     Magma.LibMagma.MagmaNoTrans,
#     D,
#     D,
#     D,
#     1.0f0,
#     A_ptrs,
#     D,
#     B_ptrs,
#     D,
#     0.0f0,
#     C_ptrs,
#     D,
#     N,
#     queue_ptr[],
# )

ccall(
    (:magmablas_sgemm_batched_smallsq, Magma.LibMagma.libmagma),
    Cvoid,
    (
        Magma.LibMagma.magma_trans_t,
        Magma.LibMagma.magma_trans_t,
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_int_t,
        Cfloat,
        CuPtr{CuPtr{Cfloat}},
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_int_t,
        CuPtr{CuPtr{Cfloat}},
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_int_t,
        Cfloat,
        CuPtr{CuPtr{Cfloat}},
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_int_t,
        Magma.LibMagma.magma_queue_t,
    ),
    Magma.LibMagma.MagmaNoTrans,
    Magma.LibMagma.MagmaNoTrans,
    D,
    D,
    D,
    1.0f0,
    A_ptrs,
    0,
    0,
    D,
    B_ptrs,
    0,
    0,
    D,
    0.0f0,
    C_ptrs,
    0,
    0,
    D,
    length(C_ptrs),
    queue_ptr[],
)
