@testitem "Dynamic hybrid arena and launch contract" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra
    const BK = BatchedKernels
    # Matrix, vector, scalar-output and common-input regions in one arena;
    # D3 and a partial block exercise padding and alignment boundaries.
    f(A, v, S, w) = (A * v + w, sum(abs2, v), A * S)
    N = 23
    ah = reshape(sin.(Float32.(1:(9N))), 3, 3, N)
    vh = reshape(cos.(Float32.(1:(3N))), 3, N)
    sh = Float32[1 0 0; 0 2 0; 0 0 3]
    wh = Float32[0.1, 0.2, 0.3]
    args = (
        BK.BatchedCuMatrix(CuArray(ah)),
        BK.BatchedCuVector(CuArray(vh)),
        BK.SharedCuMatrix(CuArray(sh), N),
        BK.SharedCuVector(CuArray(wh), N),
    )
    tape = BK.trace(f, BK.InputSpec[BK.input_spec(x) for x in args])
    a = BK.Assignment(tape; nthreads=64)
    p = BK.plan_memory(tape, a; D_MAX=3)
    static = BK._ensure_compiled!(f, args; assignment=a, nthreads=64)
    dynamic = BK._ensure_compiled!(
        f, args; assignment=a, nthreads=64, shared_memory=:dynamic
    )
    @test static.fn !== dynamic.fn
    @test static.sig.dynamic_shared_bytes == 0
    @test dynamic.sig.dynamic_shared_bytes == p.shared_bytes
    @test BK._ensure_compiled!(
        f, args; assignment=a, nthreads=64, shared_memory=:dynamic
    ) === dynamic
    @test_throws ArgumentError BK._ensure_compiled!(
        f, args; assignment=a, nthreads=64, shared_memory=:invalid
    )
    @test_throws ArgumentError BK._ensure_compiled!(f, args; shared_memory=:dynamic)
    out = @inferred BK.fuse(f, args...; assignment=a, shared_memory=:dynamic)
    reference = (
        hcat((ah[:, :, b] * vh[:, b] + wh for b in 1:N)...),
        vec(sum(abs2, vh; dims=1)),
        cat((ah[:, :, b] * sh for b in 1:N)...; dims=3),
    )
    @test Array(out.components._1.data) ≈ reference[1]
    @test Array(out.components._2.data) ≈ reference[2]
    @test Array(out.components._3.data) ≈ reference[3]
    @test all(Array(x.data) == y for (x, y) in zip(args, (ah, vh, sh, wh)))
    # Reject an oversized arena before emitting or launching a kernel.
    large(A) = (A * A, A + A)
    big_args = (BK.BatchedCuMatrix(CUDA.zeros(Float64, 32, 32, 1)),)
    big_tape = BK.trace(large, BK.InputSpec[BK.input_spec(only(big_args))])
    big_assignment = BK.Assignment(big_tape; nthreads=1024)
    @test_throws ArgumentError BK._ensure_compiled!(
        large, big_args; assignment=big_assignment, nthreads=1024, shared_memory=:dynamic
    )
end
