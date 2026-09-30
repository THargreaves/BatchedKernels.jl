function check_block_qr_cases(cases)
    for (T, m, n, N) in cases
        host = (randn(T, m, m, N), randn(T, n, m, N), randn(T, n, n, N))
        args = map(x -> BatchedCuMatrix(CuArray(x)), host)
        out = @inferred fuse(
            qr_upper_blocks,
            args...;
            nthreads=64,
            shared_memory=max(m, n) == 32 ? :dynamic : :static,
        )
        got = map(x -> Array(x.data), values(out.components))
        tol = T === Float32 ? 4e-5 : 2e-12
        for k in 1:N
            A, B, C = map(x -> x[:, :, k], host)
            R11, R12, R22 = map(x -> x[:, :, k], got)
            R = [R11 R12; zeros(T, n, m) R22]
            M = [A zeros(T, m, n); B C]
            @test R'R ≈ M'M rtol = tol atol = tol
            @test istriu(R)
            @test all(diag(R) .>= 0)
            ref = qr_upper_blocks(A, B, C)
            @test all(
                isapprox(x, y; rtol=tol, atol=tol) for (x, y) in zip((R11, R12, R22), ref)
            )
        end
        @test all(Array(x.data) == y for (x, y) in zip(args, host))
        # Both access contracts remain supported, including full-group masks,
        # non-power-of-two extents and logical stacks larger than a warp.
        tape = BK.trace(qr_upper_blocks, BK.InputSpec[BK.input_spec(x) for x in args])
        producer = only(
            i for (i, node) in enumerate(tape.nodes) if
            node isa BK.CallNode && node.fn === qr_upper_blocks
        )
        for variant in (:qr_blocks_row, :qr_blocks_col)
            # The all-dual D32 fixture needs one warp/block to fit the arena.
            threads = max(m, n) == 32 ? 32 : 64
            assignment = Assignment(
                tape; variants=Dict(producer => variant), nthreads=threads
            )
            forced = fuse(qr_upper_blocks, args...; assignment, shared_memory=:dynamic)
            fg = map(x -> Array(x.data), values(forced.components))
            @test all(isapprox(x, y; rtol=tol, atol=tol) for (x, y) in zip(got, fg))
        end
        if m == 2 && n == 3
            t = BK.trace(qr_upper_blocks, BK.InputSpec[BK.input_spec(x) for x in args])
            a = Assignment(t; nthreads=64)
            forced = fuse(qr_upper_blocks, args...; assignment=a, shared_memory=:dynamic)
            fg = map(x -> Array(x.data), values(forced.components))
            @test all(isapprox(x, y; rtol=tol, atol=tol) for (x, y) in zip(got, fg))
            auto = automatic_assignment(t; nthreads=64)
            residences = copy(auto.residences)
            for (i, node) in enumerate(t.nodes)
                node isa BK.ResultNode && (residences[i] = :single)
            end
            singles = Assignment(
                t;
                residences,
                orientations=auto.orientations,
                variants=auto.variants,
                nthreads=64,
            )
            sg = map(
                x -> Array(x.data),
                values(fuse(qr_upper_blocks, args...; assignment=singles).components),
            )
            @test all(isapprox(x, y; rtol=tol, atol=tol) for (x, y) in zip(got, sg))
            @test_throws ArgumentError fuse(
                qr_upper_blocks, args...; policy=:legacy, nthreads=64
            )
            onlyposterior(A, B, C) = last(qr_upper_blocks(A, B, C))
            @test Array(fuse(onlyposterior, args...; nthreads=64).data) ≈ got[3]
        end
    end
end
