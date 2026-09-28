# compute-sanitizer --tool {memcheck,racecheck,synccheck} --error-exitcode 1 \
#   julia --project=. benchmarking/kalman/sanitize_backward.jl
include(joinpath(@__DIR__, "automatic_backward.jl"))
println("Accessor diagnostics: ", BK.DEBUG_ACCESSORS)
flush(stdout)
for (T, n, m, N) in
    ((Float32, 3, 2, 23), (Float64, 5, 2, 7), (Float32, 20, 16, 3), (Float32, 32, 2, 2))
    for (stage, f, args, ref) in backward_benchmark_cases(T, n, m, N; batched_A=true)
        println("Checking ", T, " n=", n, " m=", m, " ", stage)
        flush(stdout)
        for shared_memory in (:static, :dynamic)
            out = fuse(f, args...; nthreads=64, shared_memory)
            leaves = out isa BatchedStruct ? values(out.components) : (out,)
            reference = ref isa Tuple ? ref : (ref,)
            tol = T === Float32 ? 2e-4 : 3e-11
            for (leaf, r) in zip(leaves, reference)
                cpu = Array(leaf.data)
                first = if ndims(cpu) == 3
                    cpu[:, :, 1]
                elseif ndims(cpu) == 2
                    cpu[:, 1]
                else
                    cpu[1]
                end
                @assert isapprox(first, r; rtol=tol, atol=tol)
            end
        end
    end
end
CUDA.synchronize()
println("Backward sanitizer cases passed")
