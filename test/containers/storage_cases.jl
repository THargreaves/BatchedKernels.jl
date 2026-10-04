struct StorageBelief{M,C,F}
    mean::M
    covariance::C
    flag::F
end
struct StorageState{X,Z,I,V}
    x::X
    belief::Z
    id::I
    value::V
end
struct StorageKind{K,X}
    x::X
end

function storage_cases(array)
    n = 4
    source = BatchedStruct(
        StorageState,
        (;
            x=BatchedCuVector(array(reshape(Float32.(1:12), 3, n))),
            belief=BatchedStruct(
                StorageBelief,
                (;
                    mean=SharedCuVector(array(Float32[1, 2]), n),
                    covariance=SharedCuMatrix(array(Float32[2 0; 0 3]), n),
                    flag=SharedValue('U', n),
                ),
            ),
            id=BatchedCuScalar(array(Int64.(2^40 .+ (1:n)))),
            value=SharedValue(7.0f0, n),
        ),
    )
    dest = @inferred allocate_batch(source, 8)
    # Initialise all slots, then overwrite a reordered, non-contiguous subset.
    @test (dest[1:4] = source) === source
    setindex!(dest, source, 5:8)
    before = Array(dest.components.x.data)
    source.components.x.data .*= 2
    indices = array(Int32[8, 2, 6, 4])
    @test setindex!(dest, source, indices) === dest
    @test Array(dest.components.x.data)[:, [8, 2, 6, 4]] == Array(source.components.x.data)
    @test Array(dest.components.x.data)[:, [1, 3, 5, 7]] == before[:, [1, 3, 5, 7]]
    @test Array(dest.components.id.data)[[8, 2, 6, 4]] == 2^40 .+ (1:n)
    @test dest.components.belief.components.mean isa BatchedCuVector
    @test dest.components.belief.components.covariance isa BatchedCuMatrix
    @test dest.components.value isa BatchedCuScalar
    @test Array(dest.components.value.data) == fill(7.0f0, 8)
    source.components.belief.components.mean.data .= -5
    @test Array(dest.components.belief.components.mean.data) == repeat(Float32[1, 2], 1, 8)
    next = BatchedStruct(
        StorageState, merge(source.components, (; value=SharedValue(9.0f0, n)))
    )
    setindex!(dest, next, 1:4)
    @test Array(dest.components.value.data) == Float32[9, 9, 9, 9, 7, 7, 7, 7]

    saved = Array(dest.components.x.data)
    badbelief = BatchedStruct(
        StorageBelief,
        merge(source.components.belief.components, (; flag=SharedValue('L', n))),
    )
    bad = BatchedStruct(StorageState, merge(source.components, (; belief=badbelief)))
    @test_throws ArgumentError setindex!(dest, bad, indices)
    @test_throws ArgumentError setindex!(dest, source, array(Int32[1, 1, 2, 3]))
    @test_throws ArgumentError setindex!(dest, source, array(UInt32[1, 2, 1, 3]))
    @test_throws BoundsError setindex!(dest, source, array(Int32[0, 2, 3, 4]))
    @test_throws ArgumentError setindex!(dest, dest, 1:8)
    @test Array(dest.components.x.data) == saved

    indexdest = BatchedCuScalar(array(Int32.(1:n)))
    @test_throws ArgumentError setindex!(
        indexdest, BatchedCuScalar(array(Int32.(n:-1:1))), indexdest.data
    )
    @test Array(indexdest.data) == Int32.(1:n)

    # Value parameters carry meaning even when every field has the same shape.
    x = source.components.x
    left = BatchedStruct(StorageKind{:left,eltype(x)}, (; x))
    right = BatchedStruct(StorageKind{:right,eltype(x)}, (; x))
    @test_throws ArgumentError setindex!(allocate_batch(left, n), right, 1:n)
    aliasdest = BatchedStruct(
        StorageBelief, (; mean=x, covariance=x, flag=SharedValue('U', n))
    )
    @test_throws ArgumentError setindex!(aliasdest, allocate_batch(aliasdest, n), 1:n)
    wrong_shape = BatchedStruct(
        StorageState,
        merge(source.components, (; x=BatchedCuVector(array(zeros(Float32, 4, n))))),
    )
    @test_throws DimensionMismatch setindex!(dest, wrong_shape, indices)
    @test Array(dest.components.x.data) == saved

    # Allocation normalises views and relabels the enclosing composite.
    viewsource = BatchedStruct(
        StorageBelief,
        (;
            mean=BatchedCuVector(view(array(zeros(Float32, 4, n)), 1:2, :)),
            covariance=BatchedCuMatrix(array(zeros(Float32, 2, 2, n))),
            flag=SharedValue('U', n),
        ),
    )
    owned = allocate_batch(viewsource, n)
    setindex!(owned, viewsource, 1:n)
    @test Array(owned.components.mean.data) == zeros(Float32, 2, n)
    empty = allocate_batch(source, 0)
    @test length(empty) == 0
    setindex!(empty, source[Int[]], Int[])

    factors = SharedCuMatrix(array(Float32[2 1; 0 3]), n)
    chol = BatchedStruct(
        Cholesky{Float32,eltype(factors)},
        (; factors, uplo=SharedValue('U', n), info=SharedValue(Int64(0), n)),
    )
    owned_chol = @inferred allocate_batch(chol, 2n)
    setindex!(owned_chol, chol, (n + 1):2n)
    @test Array(owned_chol.components.factors.data)[:, :, (n + 1):2n] ==
        repeat(Float32[2 1; 0 3], 1, 1, n)
    @test Array(owned_chol.components.info.data)[(n + 1):2n] == zeros(Int64, n)
end
