# compute-sanitizer --tool {memcheck,racecheck,synccheck} --error-exitcode 1 \
#   julia --project=. benchmarking/kalman/sanitize_automatic.jl
include(joinpath(@__DIR__, "automatic_joseph.jl"))

for (T, d, m, N) in ((Float32, 3, 2, 23), (Float64, 9, 4, 7), (Float32, 16, 6, 9))
    args, reference = joseph_benchmark_args(T, d, m, N; batched_A=true)
    for shared_memory in (:static, :dynamic)
        result = fuse(joseph_kalman_step, args...; nthreads=64, shared_memory)
        values_cpu = map(x -> Array(x.data), values(getfield(result, :components)))
        tol = T === Float32 ? 5e-5 : 2e-12
        @assert isapprox(values_cpu[1][:, 1], reference[1]; rtol=tol, atol=tol)
        @assert isapprox(values_cpu[2][:, :, 1], reference[2]; rtol=tol, atol=tol)
        @assert isapprox(values_cpu[3][1], reference[3]; rtol=tol, atol=tol)
    end
end
CUDA.synchronize()
println("Automatic Joseph sanitizer cases passed")
