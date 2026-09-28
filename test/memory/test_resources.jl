@testitem "Hybrid register resource harness" tags = [:gpu] begin
    using CUDA
    using BatchedKernels
    using StaticArrays: MVector
    const BK = BatchedKernels
    include(joinpath(@__DIR__, "resource_checks.jl"))

    function register_probe!(output)
        lid = threadIdx().x
        group = (lid - Int32(1)) ÷ Int32(6)
        d = (lid - Int32(1)) % Int32(6) + Int32(1)
        if lid <= Int32(30)
            # Five complete six-lane groups prove the constructor metadata valid.
            A = @inbounds BK.RegisterMatrix{Float32}(
                Val(Int32(3)),
                Val(Int32(4)),
                Val(Int32(6)),
                BK.RowOriented(),
                group * Int32(6),
                d,
            )
            if d <= Int32(4)
                BK.ours_write!(A, Int32(1), d, Float32(100 * d + 1), BK.RowAccess())
                BK.ours_write!(A, Int32(2), d, Float32(100 * d + 2), BK.RowAccess())
                BK.ours_write!(A, Int32(3), d, Float32(100 * d + 3), BK.RowAccess())
            end
            first = BK.theirs(A, Int32(1), Int32(4))
            second = BK.theirs(A, Int32(2), Int32(4))
            third = BK.theirs(A, Int32(3), Int32(4))
            @inbounds output[lid] = first + second + third
        end
        return nothing
    end

    # A real local-memory fixture, not an assumption that dynamic indexing must spill.
    # Runtime indexed writes and reads preserve the array in the compiled binary.
    function local_probe!(output, indices)
        scratch = MVector{256,Float32}(undef)
        for i in 1:256
            scratch[i] = Float32(i + threadIdx().x)
        end
        scratch[indices[1]] = output[1]
        output[1] = scratch[indices[2]]
        return nothing
    end

    output = CUDA.zeros(Float32, 32)
    good = @cuda launch = false register_probe!(output)
    resources =
        BK.DEBUG_ACCESSORS ? register_resources(good) : require_register_resident(good)
    @test BK.DEBUG_ACCESSORS || resources.local_bytes == 0
    @test resources.registers > 0
    CUDA.@sync good(output; threads=32, blocks=1)
    @test Array(output) == [fill(1206.0f0, 30); zeros(Float32, 2)]

    # Explicit device inference complements the final-resource check.
    typed = CUDA.code_typed(
        register_probe!, Tuple{typeof(CUDA.cudaconvert(output))}; kernel=true
    )
    @test only(typed).second === Nothing

    indices = CuArray(Int32[13, 17])
    local_kernel = @cuda launch = false local_probe!(output, indices)
    local_resources = register_resources(local_kernel)
    @test local_resources.local_bytes > 0
    @test_throws ErrorException require_register_resident(local_kernel)
    @info "Register probe compiled resources" resources local_resources debug =
        BK.DEBUG_ACCESSORS
end
