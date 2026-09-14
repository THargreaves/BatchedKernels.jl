@testitem "Hybrid out-of-place factorizations" tags = [:gpu] begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra
    using Random
    using KernelAbstractions.Extras: @unroll
    const BK = BatchedKernels
    include("../memory/resource_checks.jl")

    @inline function factor_view(
        raw, ::Val{N}, ::Val{P}, ::Val{D}, ::Val{layout}, o, mid, d, base
    ) where {N,P,D,layout}
        if layout === :register
            return @inbounds BK.RegisterMatrix{eltype(raw)}(
                Val(N), Val(P), Val(D), o, base, d
            )
        elseif layout === :single
            return @inbounds BK.SingleAccessMatrix(
                raw, Val(N), Val(P), Val(D), o, Int32(1), mid
            )
        else
            return BK.DualAccessMatrix(raw, Val(D), Int32(1), mid)
        end
    end
    @inline factor_wrap(A, ::Val{:upper}) = UpperTriangular(A)
    @inline factor_wrap(A, ::Val{:lower}) = LowerTriangular(A)
    @inline factor_wrap(A, ::Val{:unit_upper}) = UnitUpperTriangular(A)
    @inline factor_wrap(A, ::Val{:unit_lower}) = UnitLowerTriangular(A)
    @inline factor_wrap(A, ::Val{:adjoint_upper}) = adjoint(UpperTriangular(A))
    @inline factor_wrap(A, ::Val{:transpose_lower}) = transpose(LowerTriangular(A))
    @inline factor_wrap(A, ::Val{:adjoint_unit_upper}) = adjoint(UnitUpperTriangular(A))
    @inline factor_wrap(A, ::Val{:transpose_unit_lower}) = transpose(UnitLowerTriangular(A))

    function factor_kernel!(
        output,
        input,
        rhs,
        ::Val{N},
        ::Val{P},
        ::Val{D},
        ::Val{layout},
        ::Val{operation},
        wrap,
    ) where {N,P,D,layout,operation}
        nwords = max(
            BK.single_region_elems(Val(D), Val(Int32(32))),
            BK.dual_region_elems(Val(D), Val(Int32(32))),
        )
        rawA = CuStaticSharedArray(eltype(input), (nwords,))
        rawB = CuStaticSharedArray(eltype(input), (nwords,))
        rawC = CuStaticSharedArray(eltype(input), (nwords,))
        lid = threadIdx().x
        mid = (lid - Int32(1)) ÷ Int32(D) + Int32(1)
        d = (lid - Int32(1)) % Int32(D) + Int32(1)
        base = lid - d
        if mid <= Int32(32 ÷ D)
            transposed =
                wrap isa Union{
                    Val{:adjoint_upper},
                    Val{:transpose_lower},
                    Val{:adjoint_unit_upper},
                    Val{:transpose_unit_lower},
                }
            # The *wrapped* factor must be ColOriented for solve_col. Adjoint and
            # transpose flip the raw parent's physical ownership.
            physical = if operation === :solve_col && transposed
                BK.RowOriented()
            elseif operation in (:solve, :solve_col)
                BK.ColOriented()
            else
                BK.RowOriented()
            end
            convention = physical isa BK.ColOriented ? BK.ColAccess() : BK.RowAccess()
            rhs_physical = operation === :solve_col ? BK.ColOriented() : BK.RowOriented()
            # Force factory inlining so mutable register backing cannot escape a
            # returned wrapper; the production variant call itself needs no override.
            A = @inline factor_view(
                rawA, Val(N), Val(N), Val(D), Val(layout), physical, mid, d, base
            )
            B = @inline factor_view(
                rawB, Val(N), Val(P), Val(D), Val(layout), rhs_physical, mid, d, base
            )
            C = @inline factor_view(
                rawC, Val(N), Val(P), Val(D), Val(layout), rhs_physical, mid, d, base
            )
            if d <= Int32(N)
                @inbounds @unroll for k in Int32(1):Int32(N)
                    value =
                        physical isa BK.ColOriented ? input[d, k, mid] : input[k, d, mid]
                    BK.ours_write!(A, k, d, value, convention)
                end
            end
            if operation === :solve_col
                if d <= Int32(N)
                    @inbounds @unroll for k in Int32(1):Int32(P)
                        BK.ours_write!(B, k, d, rhs[d, k, mid], BK.ColAccess())
                        BK.ours_write!(C, k, d, 0.0f0, BK.ColAccess())
                    end
                end
            elseif d <= Int32(P)
                @inbounds @unroll for k in Int32(1):Int32(N)
                    BK.ours_write!(B, k, d, rhs[k, d, mid], BK.RowAccess())
                    BK.ours_write!(C, k, d, 0.0f0, BK.RowAccess())
                end
            end
            mask = (typemax(UInt32) >>> (Int32(32) - Int32(D))) << base
            sync_warp(mask)
            if operation === :cholesky
                BK.variant_op!(Val(:cholesky_row), C, A, d, Val(N), Val(D))
            elseif operation === :legacy_cholesky
                BK.batch_op!(cholesky, C, A, d, Val(N), Val(D), mid)
            elseif operation === :solve
                BK.variant_op!(
                    Val(:solve_row), C, factor_wrap(A, wrap), B, d, Val(N), Val(P), Val(D)
                )
            elseif operation === :solve_col
                BK.variant_op!(
                    Val(:solve_col), C, factor_wrap(A, wrap), B, d, Val(N), Val(P), Val(D)
                )
            else
                BK.batch_op!(\, C, factor_wrap(A, wrap), B, d, Val(N), Val(P), Val(D))
            end
            sync_warp(mask)
            if operation === :solve_col
                if d <= Int32(N)
                    @inbounds @unroll for k in Int32(1):Int32(P)
                        output[d, k, mid] = BK.ours(C, k, d, BK.ColAccess())
                    end
                end
            elseif d <= Int32(P)
                @inbounds @unroll for k in Int32(1):Int32(N)
                    output[k, d, mid] = BK.ours(C, k, d, BK.RowAccess())
                end
            end
        end
        return nothing
    end

    function launch_factor(input, rhs, N, P, D, layout, operation, wrap)
        output = CUDA.zeros(eltype(rhs), size(rhs))
        args = (
            output,
            CuArray(input),
            CuArray(rhs),
            Val(Int32(N)),
            Val(Int32(P)),
            Val(Int32(D)),
            Val(layout),
            Val(operation),
            Val(wrap),
        )
        kernel = @cuda launch = false factor_kernel!(args...)
        resources = register_resources(kernel)
        if layout === :register && !BK.DEBUG_ACCESSORS
            @test require_register_resident(kernel).local_bytes == 0
        end
        if layout === :register || N == 32
            @info "Factor variant resources" N P D layout operation wrap resources
        end
        CUDA.@sync kernel(args...; threads=32)
        return Array(output)
    end

    rng = MersenneTwister(71)
    for (N, D, layout) in (
        (2, 2, :register),
        (3, 6, :register),
        (3, 6, :single),
        (8, 8, :dual),
        (32, 32, :register),
    )
        nm = 32 ÷ D
        inputs = Array{Float32}(undef, N, N, nm)
        for m in 1:nm
            X = randn(rng, Float32, N, N)
            inputs[:, :, m] = X' * X + N * I
        end
        rhs = zeros(Float32, N, N, nm)
        reference = cat(
            (Matrix(cholesky(Hermitian(inputs[:, :, m])).U) for m in 1:nm)...; dims=3
        )
        legacy = launch_factor(inputs, rhs, N, N, D, :dual, :legacy_cholesky, :upper)
        @test legacy ≈ reference rtol = 4.0f-5 atol = 4.0f-5
        output = launch_factor(inputs, rhs, N, N, D, layout, :cholesky, :upper)
        @test output ≈ reference rtol = 4.0f-5 atol = 4.0f-5
        @test output ≈ legacy rtol = 4.0f-5 atol = 4.0f-5
        @test all(iszero(output[i, j, m]) for m in 1:nm for j in 1:N for i in (j + 1):N)
    end
    for (N, P, D, wrap, layout) in (
        (3, 2, 6, :upper, :register),
        (4, 3, 6, :lower, :single),
        (4, 2, 6, :unit_upper, :register),
        (3, 2, 6, :unit_lower, :dual),
        (4, 2, 6, :adjoint_upper, :single),
        (3, 2, 6, :transpose_lower, :register),
        (32, 2, 32, :upper, :register),
    )
        nm = 32 ÷ D
        inputs = 0.04f0 .* randn(rng, Float32, N, N, nm)
        for m in 1:nm, i in 1:N
            inputs[i, i, m] = 2.0f0 + Float32(i) / N
        end
        rhs = randn(rng, Float32, N, P, nm)
        reference = cat(
            (factor_wrap(inputs[:, :, m], Val(wrap)) \ rhs[:, :, m] for m in 1:nm)...;
            dims=3,
        )
        output = launch_factor(inputs, rhs, N, P, D, layout, :solve, wrap)
        @test output ≈ reference rtol = 3.0f-5 atol = 3.0f-5
        if wrap in (:upper, :lower)
            legacy = launch_factor(inputs, rhs, N, P, D, :dual, :legacy_solve, wrap)
            @test legacy ≈ reference rtol = 3.0f-5 atol = 3.0f-5
        end
    end

    # Four column-convention cases isolate wide pressure, P>N with padded lanes,
    # and unit-diagonal semantics through each transposing wrapper.
    for (N, P, D, wrap, layout) in (
        (32, 2, 32, :upper, :register),
        (3, 5, 6, :lower, :register),
        (4, 2, 6, :adjoint_unit_upper, :single),
        (3, 2, 6, :transpose_unit_lower, :register),
    )
        nm = 32 ÷ D
        inputs = 0.04f0 .* randn(rng, Float32, N, N, nm)
        is_unit = wrap in (:adjoint_unit_upper, :transpose_unit_lower)
        for m in 1:nm, i in 1:N
            inputs[i, i, m] = is_unit ? NaN32 : 2.0f0 + Float32(i) / N
        end
        rhs = randn(rng, Float32, N, P, nm)
        reference = cat(
            (factor_wrap(inputs[:, :, m], Val(wrap)) \ rhs[:, :, m] for m in 1:nm)...;
            dims=3,
        )
        output = launch_factor(inputs, rhs, N, P, D, layout, :solve_col, wrap)
        @test output ≈ reference rtol = 3.0f-5 atol = 3.0f-5
        if !is_unit
            legacy = launch_factor(inputs, rhs, N, P, D, :dual, :legacy_solve, wrap)
            @test output ≈ legacy rtol = 3.0f-5 atol = 3.0f-5
        end
    end
    # Focused double-precision cases: Cholesky and both solve orientations,
    # including inactive subgroup lanes and an adjointed triangular factor.
    for (operation, layout, wrap) in (
        (:cholesky, :register, :upper),
        (:solve, :single, :adjoint_upper),
        (:solve_col, :register, :upper),
    )
        N, P, D = 3, 2, 6
        nm = 32 ÷ D
        inputs = Array{Float64}(undef, N, N, nm)
        for m in 1:nm
            x = randn(rng, N, N)
            inputs[:, :, m] = x' * x + N * I
        end
        width = operation === :cholesky ? N : P
        rhs = randn(rng, Float64, N, width, nm)
        expected = cat(
            (
                if operation === :cholesky
                    Matrix(cholesky(Hermitian(inputs[:, :, m])).U)
                else
                    factor_wrap(inputs[:, :, m], Val(wrap)) \ rhs[:, :, m]
                end for m in 1:nm
            )...;
            dims=3,
        )
        output = launch_factor(inputs, rhs, N, width, D, layout, operation, wrap)
        @test isapprox(output, expected; rtol=1e-12, atol=1e-12)
    end
end
