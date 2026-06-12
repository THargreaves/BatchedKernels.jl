using CUDA
using CUDA: i32
using Random

function roofline_marker_kernel()
    return
end

function _fill_batch_ptrs_kernel!(ptrs, A, slice_elems)
    tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    gstride = gridDim().x * blockDim().x
    n = length(ptrs)
    i = tid
    while i <= n
        idx = (Int64(i) - 1) * Int64(slice_elems) + 1
        @inbounds ptrs[i] = reinterpret(Ptr{Float32}, pointer(A, idx))
        i += gstride
    end
    return
end

function _batch_ptrs(A::DenseCuArray{Float32}, N::Integer, slice_elems::Integer)
    ptrs = CuArray{Ptr{Float32}}(undef, N)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _fill_batch_ptrs_kernel!(ptrs, A, Int32(slice_elems))
    return ptrs
end

function qr_r_cublas!(
    dR, dtau,        # batched pointer arrays
    info_h, D, N,
)
    h = CUDA.CUBLAS.handle()           # cuBLAS handle, bound to current stream

    CUDA.CUBLAS.cublasSgeqrfBatched(h, D, D, dR, D, dtau, info_h, N)

    CUDA.synchronize()
    return nothing
end

function launch_cublas(Rs_cpu, As_cpu, loop_count)
    D, _, N = size(Rs_cpu)
    DD = D * D

    Rs = cu(Rs_cpu)
    As = cu(As_cpu)
    copyto!(Rs, As)

    dR = _batch_ptrs(Rs, N, DD)

    tau_storage = CUDA.zeros(Float32, D, N)
    dtau = _batch_ptrs(tau_storage, N, D)

    info_h = Ref{Cint}(0)

    for i in 1:loop_count
        if i == loop_count
            CUDA.@sync @cuda threads = 1 blocks = 1 roofline_marker_kernel()
            CUDA.synchronize()
        end
        qr_r_cublas!(
            dR,
            dtau,
            info_h,
            D,
            N,
        )
    end
end


function main(D, loop_count)
    Random.seed!(1234)
    
    N = Int(ceil(1e9 / (4 * 2 * D^2)))
    As_cpu = rand(Float32, D, D, N)
    Rs_cpu = zeros(Float32, D, D, N)

    launch_cublas(Rs_cpu, As_cpu, loop_count)
end

D = parse(Int, ARGS[1])
loop_count = parse(Int, ARGS[2])
main(D, loop_count)

# nsys profile --force-overwrite true --stats=true \
#   -o qr_r/profile_results/qr_cublas_trace \
#   julia --project=../../. qr_r/kernel_count_cublas.jl 8