@testitem "Hybrid bank-conflict guarantees" tags = [:cpu] begin
    using BatchedKernels
    const BK = BatchedKernels

    # Float32 words: repeated addresses broadcast, distinct words in one bank serialize.
    function conflict_degree(addresses)
        banks = [Set{Int}() for _ in 1:32]
        for address in addresses
            push!(banks[mod(address, 32) + 1], address)
        end
        return maximum(length, banks)
    end
    @test conflict_degree([0, 0, 0]) == 1
    @test conflict_degree([0, 32, 32]) == 2
    @test conflict_degree(Int[]) == 0

    # Region sizing is host-visible and accepts device-style Val{Int32} parameters.
    @test (@inferred BK.single_region_elems(Val(Int32(6)), Val(Int32(64)))) === Int32(362)
    @test (@inferred BK.dual_region_elems(Val(Int32(6)), Val(Int32(64)))) === Int32(430)
    @test_throws ArgumentError BK.single_region_elems(Val(6), Val(33))
    @test_throws ArgumentError BK.dual_region_elems(Val(33), Val(64))

    function single_views(D, physical)
        storage = zeros(Float32, Int(BK._single_warp_stride(Val(D))))
        return [
            BK.SingleAccessMatrix(
                storage, Val(D), Val(D), Val(D), physical, Int32(1), Int32(m)
            ) for m in 1:(32 ÷ D)
        ]
    end
    single_address(views, m, i, j) = Int(BK._single_index(views[m], Int32(i), Int32(j))) - 1
    dual_address(D, m, i, j) =
        m - 1 + (j - 1) * Int(BK._compute_stride(Val(D))) + (i - 1) * (32 ÷ D)

    # Exhaustion here checks the stated mathematical guarantee over supported D,
    # rather than repeating constructor/metadata assertions across combinations.
    violations = []
    against_free = Int[]
    for D in 2:32
        groups = 1:(32 ÷ D)
        rows = single_views(D, BK.RowOriented())
        cols = single_views(D, BK.ColOriented())
        for k in 1:D
            patterns = (
                [single_address(rows, m, k, d) for m in groups for d in 1:D],
                [single_address(cols, m, d, k) for m in groups for d in 1:D],
                [dual_address(D, m, k, d) for m in groups for d in 1:D],
                [dual_address(D, m, d, k) for m in groups for d in 1:D],
            )
            for (pattern, addresses) in enumerate(patterns)
                conflict_degree(addresses) == 1 || push!(violations, (D, k, pattern))
            end
        end
        for j in 1:D, i in 1:D
            # Replication within a group does not affect distinct-address counting.
            patterns = (
                [single_address(rows, m, i, j) for m in groups],
                [single_address(cols, m, i, j) for m in groups],
                [dual_address(D, m, i, j) for m in groups],
            )
            for (pattern, addresses) in enumerate(patterns)
                conflict_degree(addresses) == 1 || push!(violations, (D, i, j, pattern))
            end
        end
        against = maximum(
            conflict_degree([single_address(rows, m, d, j) for m in groups for d in 1:D])
            for j in 1:D
        )
        against == 1 && push!(against_free, D)
    end
    @test isempty(violations)
    @test against_free == [12; collect(17:32)]

    # The legacy packed rectangular map does not inherit the square proof.
    function packed_own_degree(M, N, D)
        interval = (32 ÷ (D & -D)) * D
        address(m, i, j) =
            let r = (m - 1) * M * N + (j - 1) * M + i - 1
                r + r ÷ interval
            end
        return maximum(
            conflict_degree([address(m, i, j) for m in 1:(32 ÷ D) for j in 1:N]) for
            i in 1:M
        )
    end
    @test packed_own_degree(3, 4, 4) == 2
    @test packed_own_degree(4, 6, 6) == 3

    # Each pass reads contiguous global words; its shared destinations are a scatter.
    # Report that cost separately: coalesced global access does not imply conflict-free shared access.
    function transfer_profile(M, N, D, physical)
        views = single_views(D, physical)
        words = (32 ÷ D) * M * N
        return [
            conflict_degree([
                begin
                    matrix, element = divrem(q, M * N)
                    j, i = divrem(element, M)
                    single_address(views, matrix + 1, i + 1, j + 1)
                end for q in first:min(first + 31, words - 1)
            ]) for first in 0:32:(words - 1)
        ]
    end
    profiles = [
        (M=M, N=N, D=D, orientation=physical, degrees=transfer_profile(M, N, D, physical))
        for (M, N, D, physical) in (
            (4, 4, 4, BK.RowOriented()),
            (3, 4, 6, BK.RowOriented()),
            (4, 3, 6, BK.ColOriented()),
            (32, 32, 32, BK.ColOriented()),
        )
    ]
    @info "Shared bank degrees per coalesced global-transfer pass" profiles
    @test profiles[1].degrees == [1, 1, 1, 1]
    @test profiles[2].degrees == [2, 2]
    @test profiles[3].degrees == [2, 2]
    @test profiles[4].degrees == ones(Int, 32)
end
