using CUDA
using CUDA: i32
using Magma
using Random

function roofline_marker_kernel()
    return
end

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
            Magma.LibMagma.magma_int_t,           # Ai (row offset into each dA[k])
            Magma.LibMagma.magma_int_t,           # Aj (col offset into each dA[k])
            Magma.LibMagma.magma_int_t,           # ldda
            CuPtr{CuPtr{Cfloat}},                 # dtau_array
            Magma.LibMagma.magma_int_t,           # taui (offset into each dtau[k])
            CuPtr{Magma.LibMagma.magma_int_t},    # info_array
            Magma.LibMagma.magma_int_t,           # batchCount
            Magma.LibMagma.magma_queue_t,         # queue
        ),
        n,
        dA_array,
        Ai,
        Aj,
        ldda,
        dtau_array,
        taui,
        info_array,
        batchCount,
        queue,
    )
end

function qr_r_magma!(dR, dtau, tau_storage, info_d, D, N, queue_ptr)
    GC.@preserve tau_storage begin
        magma_sgeqrf_batched_smallsq!(
            D, dR, 0, 0, D, dtau, 0, info_d, N, queue_ptr[],
        )
        Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
        CUDA.synchronize()
    end
end

function launch_magma(Rs_cpu, As_cpu, queue_ptr, loop_count)
    D, _, N = size(Rs_cpu)

    Rs = cu(Rs_cpu)
    As = cu(As_cpu)
    copyto!(Rs, As)
    dR = CUDA.CUBLAS.unsafe_strided_batch(Rs)

    tau_storage = CUDA.zeros(Float32, D, N)
    dtau = CUDA.CUBLAS.unsafe_strided_batch(tau_storage)

    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)
        
    for i in 1:loop_count
        if i == loop_count
            CUDA.@sync @cuda threads = 1 blocks = 1 roofline_marker_kernel()
            CUDA.synchronize()
        end
        qr_r_magma!(
            dR,
            dtau,
            tau_storage,
            info_d,
            D,
            N,
            queue_ptr,
        )
    end
end

function main(D, loop_count)
    Random.seed!(1234)
    
    N = Int(ceil(1e9 / (4 * 2 * D^2)))
    As_cpu = rand(Float32, D, D, N)
    Rs_cpu = zeros(Float32, D, D, N)

    Magma.LibMagma.magma_init()
    queue_ptr = Ref{Magma.LibMagma.magma_queue_t}()
    device = 0
    Magma.LibMagma.magma_queue_create_internal(
        device,
        queue_ptr,
        C_NULL,  # func
        C_NULL,  # file
        0,       # line
    )

    launch_magma(Rs_cpu, As_cpu, queue_ptr, loop_count)
end

D = parse(Int, ARGS[1])
loop_count = parse(Int, ARGS[2])
main(D, loop_count)

# nsys profile --force-overwrite true --stats=true \
#   -o qr_r/profile_results/qr_magma_trace \
#   julia --project=../../. qr_r/kernel_count_magma.jl 8