# compute-sanitizer --tool {memcheck,racecheck,synccheck} --error-exitcode 1 \
#   julia --project=. benchmarking/kalman/sanitize_srkf.jl
include(joinpath(@__DIR__, "automatic_srkf.jl"))

for (T, d, m, N) in ((Float32, 3, 2, 23), (Float64, 9, 4, 7), (Float32, 20, 16, 3))
    args, reference = srkf_benchmark_args(T, d, m, N; batched_A=true)
    for shared_memory in (:static, :dynamic)
        out = fuse(srkf_step, args...; nthreads=64, shared_memory)
        got = map(x -> Array(x.data), values(out.components))
        tol = T === Float32 ? 2e-4 : 2e-11
        @assert isapprox(got[1][:, 1], reference[1]; rtol=tol, atol=tol)
        @assert isapprox(got[2][:, :, 1], reference[2]; rtol=tol, atol=tol)
        @assert isapprox(got[3][1], reference[3]; rtol=tol, atol=tol)
    end
end
CUDA.synchronize()
println("Automatic SRKF sanitizer cases passed")
