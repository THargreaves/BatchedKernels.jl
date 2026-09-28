@testitem "Hybrid layout transfers and structured output" tags = [:gpu] begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    const BK = BatchedKernels

    @inline output_view(A, ::Val{false}) = A
    @inline output_view(A, ::Val{true}) = LowerTriangular(A)

    function transfer_kernel!(
        output,
        loaded_raw,
        dual_raw,
        staged_raw,
        input,
        ::Val{M},
        ::Val{N},
        ::Val{D},
        ::Val{THREADS},
        batches,
        input_orientation,
        output_orientation,
        structured,
    ) where {M,N,D,THREADS}
        single_count = BK.single_region_elems(Val(D), Val(THREADS))
        dual_count = BK.dual_region_elems(Val(D), Val(THREADS))
        single = CuStaticSharedArray(Float32, (single_count,))
        dual = CuStaticSharedArray(Float32, (dual_count,))
        tid = threadIdx().x
        warp = (tid - Int32(1)) ÷ Int32(32) + Int32(1)
        matrix = ((tid - Int32(1)) % Int32(32)) ÷ Int32(D) + Int32(1)
        for index in tid:Int32(THREADS):single_count
            single[index] = -1.0f0
        end
        for index in tid:Int32(THREADS):dual_count
            dual[index] = -1.0f0
        end
        sync_threads()
        BK.intermediate_layout_load!(
            single, input, Val(M), Val(N), Val(D), Val(THREADS), batches, input_orientation
        )
        sync_threads()
        for index in tid:Int32(THREADS):single_count
            loaded_raw[index, blockIdx().x] = single[index]
        end
        BK.interm_to_dual_transfer!(
            dual, single, Val(M), Val(N), Val(D), Val(THREADS), batches, input_orientation
        )
        # Protect the final single reads before recycling the buffer for output.
        sync_threads()
        for index in tid:Int32(THREADS):dual_count
            dual_raw[index, blockIdx().x] = dual[index]
        end
        for index in tid:Int32(THREADS):single_count
            single[index] = -1.0f0
        end
        sync_threads()
        A = BK.DualAccessMatrix(dual, Val(D), warp, matrix)
        BK.dual_to_interm_transfer!(
            single,
            output_view(A, structured),
            Val(M),
            Val(N),
            Val(D),
            Val(THREADS),
            batches,
            output_orientation,
        )
        sync_threads()
        for index in tid:Int32(THREADS):single_count
            staged_raw[index, blockIdx().x] = single[index]
        end
        BK.intermediate_layout_write!(
            output,
            single,
            Val(M),
            Val(N),
            Val(D),
            Val(THREADS),
            batches,
            output_orientation,
        )
        return nothing
    end

    # Each case isolates a distinct risk; mixed orientations prevent inverse-map
    # errors cancelling out, and the final case checks logical wrapper materialization.
    cases = (
        (4, 4, 4, BK.RowOriented(), BK.ColOriented(), false),
        (3, 4, 6, BK.ColOriented(), BK.RowOriented(), false),
        (32, 32, 32, BK.RowOriented(), BK.ColOriented(), true),
    )
    for (M, N, D, input_orientation, output_orientation, structured) in cases
        threads = Int32(64)
        matrices_per_warp = 32 ÷ D
        batches = 2 * matrices_per_warp + 1
        blocks = 2
        # Independent formulas assert exact footprint and every touched/untouched word.
        interval = (32 ÷ (D & -D)) * D
        words = matrices_per_warp * D^2
        single_stride = words + (words - 1) ÷ interval
        dual_padding = mod(matrices_per_warp - mod(matrices_per_warp * D, 32), 32)
        dual_column_stride = matrices_per_warp * D + dual_padding
        dual_stride = words + dual_padding * (D - 1)
        @test BK.single_region_elems(Val(Int32(D)), Val(threads)) == 2 * single_stride
        @test BK.dual_region_elems(Val(Int32(D)), Val(threads)) == 2 * dual_stride
        # Extra global matrix is a canary for batch-tail guards.
        input = reshape(Float32.(1:(M * N * (batches + 1))), M, N, batches + 1)
        expected_output = fill(-1.0f0, size(input))
        expected_loaded = fill(-1.0f0, 2 * single_stride, blocks)
        expected_dual = fill(-1.0f0, 2 * dual_stride, blocks)
        expected_staged = fill(-1.0f0, 2 * single_stride, blocks)
        for batch in 1:batches, j in 1:N, i in 1:M
            block = (batch - 1) ÷ (2 * matrices_per_warp) + 1
            within = (batch - 1) % (2 * matrices_per_warp)
            warp = within ÷ matrices_per_warp
            matrix = within % matrices_per_warp
            value = input[i, j, batch]
            result = structured && i < j ? 0.0f0 : value
            expected_output[i, j, batch] = result
            r_in = matrix * D^2 + (
                if input_orientation isa BK.RowOriented
                    (j - 1) * D + i - 1
                else
                    (i - 1) * D + j - 1
                end
            )
            r_out = matrix * D^2 + (
                if output_orientation isa BK.RowOriented
                    (j - 1) * D + i - 1
                else
                    (i - 1) * D + j - 1
                end
            )
            expected_loaded[warp * single_stride + r_in + r_in ÷ interval + 1, block] =
                value
            expected_staged[warp * single_stride + r_out + r_out ÷ interval + 1, block] =
                result
            expected_dual[
                warp * dual_stride + matrix + (j - 1) * dual_column_stride + (i - 1) * matrices_per_warp + 1,
                block,
            ] = value
        end
        output = CuArray(fill(-1.0f0, size(input)))
        loaded_raw = CUDA.zeros(Float32, size(expected_loaded))
        dual_raw = CUDA.zeros(Float32, size(expected_dual))
        staged_raw = CUDA.zeros(Float32, size(expected_staged))
        device_input = CuArray(input)
        CUDA.@sync @cuda threads = threads blocks = blocks transfer_kernel!(
            output,
            loaded_raw,
            dual_raw,
            staged_raw,
            device_input,
            Val(Int32(M)),
            Val(Int32(N)),
            Val(Int32(D)),
            Val(threads),
            Int32(batches),
            input_orientation,
            output_orientation,
            Val(structured),
        )
        @test Array(loaded_raw) == expected_loaded
        @test Array(dual_raw) == expected_dual
        @test Array(staged_raw) == expected_staged
        @test Array(output) == expected_output
    end
end
