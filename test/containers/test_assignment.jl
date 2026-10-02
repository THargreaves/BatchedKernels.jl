@testitem "Batch member assignment on host storage" tags=[:cpu] begin
    using BatchedKernels

    data = reshape(Float32.(1:15), 3, 5)
    batch = BatchedCuVector(copy(data))
    retained = batch[2]
    source = [9.0, 8.0, 7.0]
    @test setindex!(batch, source, 2) === batch
    expected = copy(data)
    expected[:, 2] = source
    @test batch.data == expected
    @test retained == Float32.(source)
    source[1] = 0
    @test retained[1] == 9
    @test_throws BoundsError batch[0] = source
    @test_throws BoundsError batch[6] = source
    @test_throws DimensionMismatch batch[1] = [1.0]
    @test_throws Exception batch[1] = 1.0
    @test batch.data == expected

    batch[2] = batch[2]
    @test batch.data == expected
    # Destination and source partially overlap, with different storage wrappers.
    backing = Float32.(1:15)
    strided = BatchedCuVector(reshape(view(backing, 3:8), 3, 2))
    strided[1] = view(backing, 1:2:5)
    @test backing[3:5] == Float32[1, 3, 5]
    @test backing[6:end] == Float32.(6:15)
    repeated = BatchedCuVector(view(batch.data, :, [1, 1]))
    @test_throws ArgumentError repeated[1] = source
    @test batch.data == expected
end

@testitem "Batch member assignment on device storage" begin
    using BatchedKernels, CUDA
    CUDA.allowscalar(false)

    data = reshape(Float32.(1:15), 3, 5)
    batch = BatchedCuVector(CuArray(data))
    trajectory = CuArray(reshape(Float64.(101:118), 6, 3))
    source = view(trajectory, 1:2:5, 2)
    retained = batch[2]
    batch[2] = source
    expected = copy(data)
    expected[:, 2] = [107, 109, 111]
    @test Array(batch.data) == expected
    @test Array(retained) == expected[:, 2]
    @test Array(trajectory) == reshape(Float64.(101:118), 6, 3)
    @test_throws BoundsError batch[0] = source
    @test_throws DimensionMismatch batch[1] = CUDA.ones(Float32, 1)
    @test_throws ArgumentError batch[1] = ones(Float32, 3)
    @test Array(batch.data) == expected
    batch[2] = batch[2]
    @test Array(batch.data) == expected

    backing = CuArray(Float32.(1:15))
    overlapping = BatchedCuVector(reshape(view(backing, 3:8), 3, 2))
    overlapping[1] = view(backing, 1:2:5)
    @test Array(backing)[3:5] == Float32[1, 3, 5]
    @test Array(backing)[6:end] == Float32.(6:15)

    storage = CuArray(reshape(Float32.(1:30), 6, 5))
    strided = BatchedCuVector(view(storage, 1:2:5, :))
    strided[3] = CuArray(Float32[90, 80, 70])
    expected_storage = reshape(Float32.(1:30), 6, 5)
    expected_storage[1:2:5, 3] = [90, 80, 70]
    @test Array(storage) == expected_storage
end
