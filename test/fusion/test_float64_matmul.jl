@testitem "Float64 hybrid matmul byte accounting" tags = [:cpu] begin
    using BatchedKernels
    const BK = BatchedKernels
    f(A, B) = (A * B) * A
    function make_plan(T)
        tape = BK.trace(
            f, BK.InputSpec[BK.LeafInput(BK.TraceMatrix{T,3,3}, BK.BATCHED) for _ in 1:2]
        )
        residences = Dict{Int,Symbol}()
        orientations = Dict{Int,Symbol}()
        variants = Dict{Int,Symbol}()
        products = Int[]
        for (id, node) in enumerate(tape.nodes)
            node isa BK.CallNode || continue
            if BK._isplacement(node.fn) || node.fn === (*)
                residences[id] = :register
                orientations[id] = :row
            end
            if node.fn === (*)
                variants[id] = :matmul_row
                push!(products, id)
            end
        end
        residences[first(products)] = :single
        a = BK.Assignment(tape; residences, orientations, variants, nthreads=64)
        return tape, a, BK.plan_memory(tape, a; D_MAX=4, T)
    end
    _, _, p32 = make_plan(Float32)
    tape, a, p64 = make_plan(Float64)
    @test p64.element_type === Float64
    @test p64.shared_bytes == 2 * p32.shared_bytes
    @test p64.peak_register_elements == p32.peak_register_elements # elements, not 32-bit registers
    @test_throws ArgumentError BK.plan_memory(
        tape, a; D_MAX=4, T=Float64, max_shared_bytes=p64.shared_bytes - 1
    )
    @test_throws ArgumentError BK.plan_memory(tape, a; D_MAX=4, T=Float32)
end

@testitem "Float64 hybrid matmul precision and collective accesses" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra, Random
    const BK = BatchedKernels
    rng = MersenneTwister(59317)
    N = 19
    chain(A, B, C) = (A * B) * C
    mirrored(A, B) = A' * B
    # Rectangular subgroups, unused output lanes, a partial final block, and a
    # true shared intermediate consumed alongside a block-common input.
    A = randn(rng, Float64, 3, 4, N)
    B = randn(rng, Float64, 4, 3, N)
    C = randn(rng, Float64, 3, 2)
    reference = cat((chain(A[:, :, n], B[:, :, n], C) for n in 1:N)...; dims=3)
    row_args = (
        BK.BatchedCuMatrix(CuArray(A)),
        BK.BatchedCuMatrix(CuArray(B)),
        BK.SharedCuMatrix(CuArray(C), N),
    )
    # The adjoint input and column output exercise the mirrored Float64 body.
    L = randn(rng, Float64, 4, 3, N)
    R = randn(rng, Float64, 4, 2, N)
    col_reference = cat((mirrored(L[:, :, n], R[:, :, n]) for n in 1:N)...; dims=3)
    col_args = (BK.BatchedCuMatrix(CuArray(L)), BK.BatchedCuMatrix(CuArray(R)))
    for (f, args, expected, originals, col) in (
        (chain, row_args, reference, (A, B, C), false),
        (mirrored, col_args, col_reference, (L, R), true),
    )
        tape = BK.trace(f, BK.InputSpec[BK.input_spec(x) for x in args])
        residences = Dict{Int,Symbol}()
        orientations = Dict{Int,Symbol}()
        variants = Dict{Int,Symbol}()
        products = Int[]
        for (id, node) in enumerate(tape.nodes)
            node isa BK.CallNode || continue
            if BK._isplacement(node.fn)
                residences[id] = :register
                orientations[id] = :row
            elseif node.fn === (*)
                residences[id] = :register
                orientations[id] = col ? :col : :row
                variants[id] = col ? :matmul_col : :matmul_row
                push!(products, id)
            elseif BK._isstage(node.fn)
                orientations[id] = col ? :col : :row
            end
        end
        !col && (residences[first(products)] = :single)
        a = BK.Assignment(tape; residences, orientations, variants, nthreads=64)
        entry = BK._ensure_compiled!(f, args; assignment=a, nthreads=64)
        out = CUDA.zeros(Float64, size(expected))
        ka = (out, (x.data for x in args)..., Int32(N))
        kernel = Base.invokelatest() do
            fn = entry.fn
            @cuda launch = false fn(ka...)
        end
        CUDA.@sync kernel(ka...; threads=64, blocks=cld(N, 16))
        @test isapprox(Array(out), expected; rtol=1e-12, atol=1e-12)
        @test all(Array(x.data) == original for (x, original) in zip(args, originals))
        BK.DEBUG_ACCESSORS || @test getproperty(CUDA.memory(kernel), :local) == 0
    end
end
